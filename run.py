
import os
import gc
import sys
import time
import json
import pickle
import random
import numpy as np
from collections import deque, namedtuple
from typing import Any, Callable, Optional, Sequence, TypeVar, Union

import torch
from copy import deepcopy
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader

from vllm import LLM, SamplingParams
from transformers import PreTrainedTokenizer, set_seed, GenerationConfig, AutoTokenizer, AutoModelForCausalLM, AutoConfig

from utils.fsdp_worker import fsdp_worker
from utils.experience import Experience
from utils.actor_worker import torch_dist_worker
from utils.utils import load_config, load_policy, get_workdir, setup_logger
from utils.comms import actor_send, buffer_recv, buffer_send, learner_recv, learner_send, actor_recv
from utils.review_only import REVIEW_ONLY_ARTIFACT, review_only_runtime_unavailable
from utils.load_gsm8k_al_05B_vllm import get_task_spec


REVIEW_ONLY_ARTIFACT = True


class Actor:
    def __init__(self, comm, rank, learner_rank, actor_rank, buffer_rank, work_dir, config):
        self.comm = comm
        self.rank = rank 
        self.learner_rank = learner_rank   
        self.actor_rank = actor_rank
        self.buffer_rank = buffer_rank
        self.work_dir = work_dir
        self.base_conf = config['base_conf']
        self.actor_conf = config['actor_conf']
        self.logging_conf = config['logging_conf']
        self.task_type = self.base_conf.get("task_type", "gsm8k")
        self.task_spec = get_task_spec(self.task_type)
        self.logger = setup_logger(f"actor{self.rank}", self.work_dir, self.logging_conf['log_level'], filename=f"actor{self.rank}.log")
        self.logger.info(f"actor_conf: {self.actor_conf}")
        self.logger.info(f"task_type: {self.task_type}")

        data_paths = self.base_conf['data_path'].split(', ')
        self.dataset = []
        for data_path in data_paths:
            self.dataset.extend(list(self.task_spec["loader"](data_path)))

        self.batchsize = self.actor_conf["batchsize"]
        self.epoch_num = self.actor_conf["epoch_num"]
        set_seed(int(self.base_conf['seed']+self.rank))
        self.max_prompt_length = self.actor_conf["max_prompt_length"]
        self.max_new_tokens = self.actor_conf["max_new_tokens"]
        self.num_generations = self.actor_conf["num_generations"]
        self.vllm_flag = True

        self.device = self.actor_conf["device"]
        
        model_path = self.base_conf['model_path']
        actor_model_config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=False, attn_implementation="flash_attention_2"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id 
        actor_model_config.eos_token_id = self.tokenizer.eos_token_id
        actor_model_config.pad_token_id = self.tokenizer.pad_token_id
        actor_model_config.bos_token_id = self.tokenizer.bos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype='float32',
            config=actor_model_config,
            trust_remote_code=False
        ).to('cpu')
        
        self.batch_queue = None
        self.model_queue = None
        self.result_queue = None
        self.processes = []
        self.worker_runtime_available = False
        self.sampling_params = SamplingParams(
            n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1.0, top_p=1, top_k=-1, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False,
            max_tokens=self.max_new_tokens, min_tokens=0, logprobs=0, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None
        )
        self.logger.info("vLLM worker bootstrap omitted in the anonymous review artifact.")

        self.exp = Experience()
        self.send_size = self.actor_conf["first_send_size"]
        self.filter_flag = self.actor_conf["filter"]
        self.send_data_req = None
        self.per_start = self.actor_conf["per_start"]
        self.update_iter = 0
        
        self.recv_model_flag = False

        self.start_time = time.time()
        self.send_count = 0
        self.actor_metrics = {
            "reward": [],
            "reward_std": [],
            "completion_ids_length": [],
            "completion_text_length": [],
            "entropy_mean": [],
            "top_entropy_mean": [],
        }

    def get_state(self, dataset):
        states = []
        for _ in range(self.batchsize):
            states.append(dataset.pop(0))
        return states
    
    def apply_chat_template(self, example: dict[str, list[dict[str, str]]],
            tokenizer: PreTrainedTokenizer,
            tools: Optional[list[Union[dict, Callable]]] = None,
            ) -> dict[str, str]:
        supported_keys = ["prompt", "chosen", "rejected", "completion", "messages", "label"]
        example_keys = {key for key in example.keys() if key in supported_keys}
        if example_keys not in [
            {"messages"},
            {"prompt"},
            {"prompt", "completion"},
            {"prompt", "chosen", "rejected"},
            {"chosen", "rejected"},
            {"prompt", "completion", "label"},
        ]:
            raise KeyError(f"Invalid keys in the example: {example_keys}")

        if "messages" in example:
            messages = tokenizer.apply_chat_template(example["messages"], tools=tools, tokenize=False)

        if "prompt" in example:
            last_role = example["prompt"][-1]["role"]
            if last_role == "user":
                add_generation_prompt = True
                continue_final_message = False
            elif last_role == "assistant":
                add_generation_prompt = False
                continue_final_message = True
            else:
                raise ValueError(f"Invalid role in the last message: {last_role}")
            prompt = tokenizer.apply_chat_template(
                example["prompt"],
                tools=tools,
                continue_final_message=continue_final_message,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

        if "prompt" in example:
            if "chosen" in example:
                prompt_chosen = tokenizer.apply_chat_template(
                    example["prompt"] + example["chosen"], tools=tools, tokenize=False
                )
                chosen = prompt_chosen[len(prompt) :]
            if "rejected" in example and "prompt" in example:
                prompt_rejected = tokenizer.apply_chat_template(
                    example["prompt"] + example["rejected"], tools=tools, tokenize=False
                )
                rejected = prompt_rejected[len(prompt) :]
            if "completion" in example:
                prompt_completion = tokenizer.apply_chat_template(
                    example["prompt"] + example["completion"], tools=tools, tokenize=False
                )
                completion = prompt_completion[len(prompt) :]
        else:
            if "chosen" in example:
                chosen = tokenizer.apply_chat_template(example["chosen"], tools=tools, tokenize=False)
            if "rejected" in example:
                rejected = tokenizer.apply_chat_template(example["rejected"], tools=tools, tokenize=False)

        if "prompt" in example:
            error_message = (
                "The chat template applied to the prompt + completion does not start with the chat template applied to "
                "the prompt alone. This can indicate that the chat template is not supported by TRL."
                "\n**Prompt**:\n{}\n\n**Prompt + Completion**:\n{}"
            )
            if "chosen" in example and not prompt_chosen.startswith(prompt):
                raise ValueError(error_message.format(prompt, prompt_chosen))
            if "rejected" in example and not prompt_rejected.startswith(prompt):
                raise ValueError(error_message.format(prompt, prompt_rejected))
            if "completion" in example and not prompt_completion.startswith(prompt):
                raise ValueError(error_message.format(prompt, prompt_completion))

        output = {}
        if "messages" in example:
            output["text"] = messages
        if "prompt" in example:
            output["prompt"] = prompt
        if "chosen" in example:
            output["chosen"] = chosen
        if "rejected" in example:
            output["rejected"] = rejected
        if "completion" in example:
            output["completion"] = completion
        if "label" in example:
            output["label"] = example["label"]

        return output

    def process_state(self, inputs):
        prompts_text = [self.apply_chat_template(example, self.tokenizer)["prompt"] for example in inputs]

        batch_prompt_ids, batch_prompt_mask = [], []
        for i in range(self.batchsize):
            prompt_inputs = self.tokenizer(prompts_text[i], return_tensors="pt", padding=False, padding_side="left", add_special_tokens=False)
            prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

            if self.max_prompt_length is not None:
                prompt_ids = prompt_ids[:, -self.max_prompt_length :]
                prompt_mask = prompt_mask[:, -self.max_prompt_length :]
            for i in range(self.num_generations):
                batch_prompt_ids.append(prompt_ids)
                batch_prompt_mask.append(prompt_mask)
        return batch_prompt_ids, batch_prompt_mask

    def pad_batch(self, batch_completion_ids, pad_token_id):
        padded_tensors = []
        attention_masks = []
        for i in range(len(batch_completion_ids)):
            sequences = [torch.tensor(seq) for seq in batch_completion_ids[i]]
            sequence_lengths = [len(seq) for seq in batch_completion_ids[i]]
            padded = pad_sequence(sequences, batch_first=True, padding_value=pad_token_id)
            padded_tensors.append(padded)
            batch_size, max_len = padded.shape
            mask = torch.zeros((batch_size, max_len), dtype=torch.int)
            for j, length in enumerate(sequence_lengths):
                mask[j, :length] = 1
            attention_masks.append(mask)

        return padded_tensors, attention_masks

    def put_prompt(self, prompts):
        if not self.worker_runtime_available:
            review_only_runtime_unavailable("actor generation worker")
        self.batch_queue.put(prompts)

    def get_result(self):
        if not self.worker_runtime_available:
            review_only_runtime_unavailable("actor generation result queue")
        result = self.result_queue.get()
        batch_completion_text, batch_completion_ids, old_logprob, entropy, ref_logprob, batch_time = result
        print(f"\n=== Batch inference result (elapsed: {batch_time:.2f}s) ===", flush=True)
        
        return (batch_completion_text, batch_completion_ids, old_logprob, entropy, ref_logprob)

    def model_generate(self, batch_prompt_ids):
        batch_prompt_texts = []
        for batch in batch_prompt_ids:
            decoded_batch = self.tokenizer.batch_decode(batch)
            batch_prompt_texts.extend(decoded_batch)
        
        self.put_prompt((batch_prompt_ids, batch_prompt_texts))

        batch_completion_text, batch_completion_ids, batch_old_logprob, batch_entropy, batch_ref_logprob = self.get_result()
        batch_completion_tensor = [torch.tensor(ids) for ids in batch_completion_ids]
        gc.collect()
        torch.cuda.empty_cache()
        return batch_completion_text, batch_completion_tensor, batch_old_logprob, batch_entropy, batch_ref_logprob

    def compute_advantages(self, batch_input, batch_completion_text):
        batch_completion_text_2d = [
            batch_completion_text[i * self.num_generations : (i + 1) * self.num_generations] 
            for i in range(self.batchsize)
        ]

        multi_advantages, multi_rewards = [], []
        for i in range(self.batchsize):
            completions = batch_completion_text_2d[i]
            answer = self.task_spec["answer_getter"](batch_input[i])
            reward_funcs = self.task_spec["reward_funcs"]
            rewards_per_func = torch.zeros(len(completions), len(reward_funcs))
            for reward_idx, reward_func in enumerate(reward_funcs):
                output_reward_func = reward_func(completions=completions, answer=answer)
                rewards_per_func[:, reward_idx] = torch.tensor(output_reward_func, dtype=torch.float32)
            rewards = rewards_per_func.sum(dim=1) 
            self.logger.info(f"rewards_per_func {rewards_per_func} \n rewards {rewards}")
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
            multi_rewards.append(rewards)
            multi_advantages.append(advantages)
        batch_reward = [torch.tensor(x) for reward_tensor in multi_rewards for x in reward_tensor]
        batch_advantage = [torch.tensor(x) for reward_tensor in multi_advantages for x in reward_tensor]
        return batch_reward, batch_advantage
        
    def calculate_entropy(self, batch_logprobs_tensor, mask_logprobs_tensors):
        batch_entropy = []
        for i in range(len(batch_logprobs_tensor)):
            logprobs = batch_logprobs_tensor[i]
            mask = mask_logprobs_tensors[i]
            entropy = -logprobs
            entropy_masked = entropy * mask.float() 
            topk = int(min(self.max_new_tokens*0.2, len(entropy_masked[0])))
            topk_values, _ = entropy_masked.topk(k=topk, dim=1)
            top_entropy_mean = topk_values.mean(dim=1, keepdim=True).squeeze(1)
            batch_entropy.append(top_entropy_mean)
        return batch_entropy

    def compute_weight(self, batch_prompt_ids, batch_entropy, batch_advantage):
        batch_weight = []
        for i, entropy in enumerate(batch_entropy):
            prompt_length = len(batch_prompt_ids[i].squeeze(0))
            entropy_completion = entropy[prompt_length:]
            complete_length = len(entropy_completion)
            topk = max(1, int(min(self.max_new_tokens*0.1, complete_length*0.2)))
            self.logger.info(f"debug prompt_length {prompt_length}, complete_length {complete_length}, topk {topk}")
            topk_values, _ = entropy_completion.topk(k=topk)
            top_entropy_mean = topk_values.mean(dim=0)
            
            self.actor_metrics["completion_ids_length"].append(complete_length)
            self.actor_metrics["entropy_mean"].append(entropy_completion.mean(dim=0).item())
            self.actor_metrics["top_entropy_mean"].append(top_entropy_mean.item())

            advantage = batch_advantage[i]
            norm_entropy = top_entropy_mean / (1.0 + top_entropy_mean)
            weight = (1 - norm_entropy) * torch.exp(-advantage) + norm_entropy * torch.exp(advantage)
            weight += 120 
            batch_weight.append(torch.as_tensor(weight))


            
        return batch_weight

    def filter_sample(self, batch_advantage, batch_prompt_ids, batch_completion_ids, batch_old_logprob, batch_ref_logprob, batch_weight):
        batch_prompt_ids_squeezed = [torch.squeeze(prompt_tensor, dim=0) for prompt_tensor in batch_prompt_ids]

        dic = {"prompt_ids": batch_prompt_ids_squeezed, 
                "completion_ids": batch_completion_ids, 
                "old_logps": batch_old_logprob,
                "ref_logps": batch_ref_logprob,
                "advantage": batch_advantage,
                "weight": batch_weight,
            } 
        
        if self.filter_flag:
            zero_indices = [i for i, adv in enumerate(batch_advantage) if adv.item() == 0]
            self.logger.info(f"del indexs {zero_indices}")
            for key in dic.keys():
                for i in sorted(zero_indices, reverse=True):
                    del dic[key][i]

            self.logger.info(f"after del, dic.length {len(dic['weight'])} \n dic['weight'] {dic['weight']}")
        dic['count'] = [0 for _ in range(len(dic['weight']))]
        dic['other_args'] = ["null" for _ in range(len(dic['weight']))]

        return dic
    
    def pad_tensor(self, t, max_len, pad_token_id=151643, padding_side="right"):
        pad_len = max_len - t.shape[1]
        if pad_len <= 0:
            return t
        pad_tensor = torch.full((t.shape[0], pad_len), pad_token_id, dtype=t.dtype, device=t.device)
        if padding_side == "right":
            padded_t = torch.cat([t, pad_tensor], dim=1)
        else:
            padded_t = torch.cat([pad_tensor, t], dim=1)
        return padded_t
    
    def process_train(self, dic):
        new_dic = {
            "prompt_ids": [],
            "prompt_mask": [], 
            "completion_ids": [],
            "completion_mask": [],
            "advantages": [],
            "old_logps": [],
            "weight": [],
            "count": [],
            "other_args": []
        }
        comp_keys = ['completion_ids', 'completion_mask', 'old_logps']
        prompt_keys = ['prompt_ids', 'prompt_mask']
        adv_keys = ['advantages', 'weight']
        for key, value in dic.items():
            if key in comp_keys:
                padid = 0 if key == 'completion_mask' else self.tokenizer.pad_token_id
                completion_ids_list = value
                max_seq_len = max(t.shape[1] for t in completion_ids_list)
                padded_completion_ids = [
                    self.pad_tensor(t, max_seq_len, pad_token_id=padid, padding_side="right")
                    for t in completion_ids_list
                ]
                completion_ids_tensor = torch.stack(padded_completion_ids, dim=0)
                reshape_tensor = completion_ids_tensor.reshape(-1, completion_ids_tensor.shape[-1])
                new_dic[key] = reshape_tensor.tolist()
            elif key in prompt_keys:
                padid = 0 if key == 'prompt_mask' else self.tokenizer.pad_token_id 
                prompt_ids_list = value
                expanded_prompt_ids = []
                for t in prompt_ids_list:
                    expanded_t = t.repeat(self.num_generations, 1)
                    expanded_prompt_ids.append(expanded_t)
                max_seq_len = max(t.shape[1] for t in expanded_prompt_ids)
                padded_prompt = [
                    self.pad_tensor(t, max_seq_len, pad_token_id=padid, padding_side="right")
                    for t in expanded_prompt_ids
                ]
                prompt_ids_tensor = torch.stack(padded_prompt, dim=0)
                reshape_tensor = prompt_ids_tensor.reshape(-1, prompt_ids_tensor.shape[-1])
                new_dic[key] = reshape_tensor.tolist()
            elif key in adv_keys:
                advantages = value
                advantages_tensor = torch.stack(advantages, dim=0)
                reshape_tensor = advantages_tensor.reshape(-1)
                new_dic[key] = reshape_tensor.tolist()
        new_dic['count'] = [0 for _ in range(len(reshape_tensor))]
        new_dic['other_args'] = ["null" for _ in range(len(reshape_tensor))]
        return new_dic
            
    def push_sample_filter(self, dic):
        self.exp.push_sample_batch(*dic.values())

    def pull_sample(self):
        exp_pulled = self.exp.pull_sample(self.send_size)
        self.exp.clear()
        self.logger.info(f"after pull, pull size {self.send_size}, exp size {self.exp.length()}")
        return exp_pulled

    def send_sample(self, send_data, recv_rank):
        packed_data = pickle.dumps(send_data)
        self.send_data_req = actor_send(
            channel="actor_to_buffer",
            payload=packed_data,
            source_rank=self.rank,
            target_rank=recv_rank,
            logger=self.logger,
        )
        self.logger.info(f"rank {self.rank} queued samples for rank {recv_rank}")

    def recv_model(self, src_rank):
        while 1:
            if self.recv_model_flag == True: break
            else:
                received_data = actor_recv(
                    channel="learner_to_actor",
                    source_rank=src_rank,
                    target_rank=self.rank,
                    logger=self.logger,
                )
                if received_data:
                    if received_data == 'stop':
                        self.logger.info("recv stop, send stop to buffer")
                        self.send_sample('stop', self.buffer_rank[0])
                        time.sleep(20)
                        sos._exit(-1)
                    else:
                        self.update_iter += 1
                        self.logger.info(f"rank {self.rank} received model from {src_rank}, self.update_iter {self.update_iter}")
                        self.model = received_data
                        if self.vllm_flag:
                            if not self.worker_runtime_available:
                                review_only_runtime_unavailable("actor model update worker")
                            self.model_queue.put(self.model)
                            time.sleep(2)
                        self.recv_model_flag = True

    def run(self):
        self.logger.info(f"actor is running, rank id {self.rank}")
        try:
            for i in range(self.epoch_num):
                shuffled_dataset = self.dataset.copy()
                random.shuffle(shuffled_dataset)
                self.logger.info(f"------ epoch {i} ------")
                while True:
                    self.logger.info(f"len(shuffled_dataset) {len(shuffled_dataset)}")
                    if len(shuffled_dataset) < self.batchsize:
                        break
                    batch_inputs = self.get_state(shuffled_dataset)
                    batch_prompt_ids, _ = self.process_state(batch_inputs)
                    batch_completion_text, batch_completion_ids, batch_old_logprob, batch_entropy, batch_ref_logprob = self.model_generate(batch_prompt_ids)
                    _, batch_advantage = self.compute_advantages(batch_inputs, batch_completion_text)
                    batch_weight = self.compute_weight(batch_prompt_ids, batch_entropy, batch_advantage)
                    dic = self.filter_sample(batch_advantage, batch_prompt_ids, batch_completion_ids, batch_old_logprob, batch_ref_logprob, batch_weight)
                    if len(dic['weight']) > 0: 
                        self.push_sample_filter(dic)
                        gc.collect()
                        torch.cuda.empty_cache()
                    
                    if self.exp.length() >= self.send_size: 
                        pulled = self.pull_sample()
                        self.send_sample(pulled, self.buffer_rank[0])
                        self.send_count += len(pulled['completion_ids'])
                        thr = self.send_count / (time.time()-self.start_time)
                        self.logger.info(f"send samples {self.send_count}, duration {time.time()-self.start_time:.2f}, throughput {thr:.2f}")
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                        self.recv_model_flag = False
                        self.logger.info("before recv_model")
                        self.recv_model(self.learner_rank[0])
                        if self.update_iter == self.per_start:
                            self.send_size = self.actor_conf["send_size"]
                    else:
                        self.recv_model_flag = True
                    

                    if self.update_iter % self.actor_conf['save_iter'] == 0:
                        metrics_file = f"{self.work_dir}/actor{self.rank}_metrics.json"
                        with open(metrics_file, "w") as f:
                            json.dump(self.actor_metrics, f, indent=4)

                    gc.collect()
                    torch.cuda.empty_cache()
            self.logger.info("actor complete")
        except Exception as e:
            self.logger.info(f"Process {self.rank} encountered an error: {str(e)}")
            import traceback
            error_traceback = traceback.format_exc()
            self.logger.info(f"Full traceback:\n{error_traceback}")
            os._exit(-1)


class Buffer:
    def __init__(self, comm, rank, learner_rank, actor_rank, buffer_rank, work_dir, config):
        self.comm = comm
        self.rank = rank 
        self.learner_rank = learner_rank
        self.actor_rank = actor_rank
        self.buffer_rank = buffer_rank
        self.work_dir = work_dir
        self.base_conf = config['base_conf']
        self.buffer_conf = config['buffer_conf']
        self.logging_conf = config['logging_conf']
        self.logger = setup_logger(f"buffer{self.rank}", self.work_dir, self.logging_conf['log_level'], filename=f"buffer{self.rank}.log")
        self.logger.info(f"buffer_conf: {self.buffer_conf}")

        self.device = self.buffer_conf["device"]
        self.exp = Experience(max_size=self.buffer_conf["expsize"])
        self.send_size = self.buffer_conf["first_send_size"]
        self.send_data_req = None
        self.per_start = self.buffer_conf["per_start"]
        self.update_iter = 0
        self.start_time = time.time()

    def recv_sample(self):
        for src_rank in self.actor_rank:
            received_data = buffer_recv(
                channel="actor_to_buffer",
                source_rank=src_rank,
                target_rank=self.rank,
                logger=self.logger,
            )
            
            if received_data is not None:
                try:
                    received_data = pickle.loads(received_data)
                except Exception as e:
                    self.logger.info(f"Process {self.rank} encountered an error: {str(e)}")
                    return None
                if received_data == 'stop':
                    self.logger.info("recv stop, stop")
                    os._exit(-1)
                else:
                    self.push_sample(received_data)
        
    def push_sample(self, received_data):
        keys = list(received_data.keys())
        for key in keys:
            key = received_data[f'{key}']
        self.exp.push_sample_batch(*received_data.values())

    def pull_sample(self, pull_size):
        exp_pulled = self.exp.pull_sample(pull_size)
        self.logger.info(f"after pull, pull size {pull_size}, exp size {self.exp.length()}")
        return exp_pulled

    def send_sample(self, send_data, recv_rank):
        self.logger.info(f"[before] rank {self.rank} send samples to rank {recv_rank}")
        packed_data = pickle.dumps(send_data)
        self.send_data_req = buffer_send(
            channel="buffer_to_learner",
            payload=packed_data,
            source_rank=self.rank,
            target_rank=recv_rank,
            logger=self.logger,
        )
        self.logger.info(f"rank {self.rank} send samples to rank {recv_rank}")

    def run(self):
        self.logger.info(f"buffer is running, rank id {self.rank}")

        while True:
            try:
                self.recv_sample()
                if self.exp.length() >= self.send_size: 
                    self.update_iter += 1
                    pulled = self.pull_sample(self.send_size)
                    self.send_sample(pulled, self.learner_rank[0])
                    if self.update_iter == self.per_start:
                            self.send_size = self.buffer_conf["send_size"]
                    
                    gc.collect()
                    torch.cuda.empty_cache()
            except Exception as e:
                self.logger.info(f"Process {self.rank} encountered an error: {str(e)}")
                import traceback
                error_traceback = traceback.format_exc()
                self.logger.info(f"Full traceback:\n{error_traceback}")
                os._exit(-1)
        

class Learner:
    def __init__(self, comm, rank, learner_rank, actor_rank, buffer_rank, work_dir, config):
        self.comm = comm
        self.rank = rank 
        self.learner_rank = learner_rank
        self.actor_rank = actor_rank
        self.buffer_rank = buffer_rank
        self.work_dir = work_dir
        self.base_conf = config['base_conf']
        self.learner_conf = config['learner_conf']
        self.logging_conf = config['logging_conf']
        self.logger = setup_logger(f"learner{self.rank}", self.work_dir, self.logging_conf['log_level'], filename=f"learner{self.rank}.log")
        self.logger.info(f"learner_conf: {self.learner_conf}")
        set_seed(self.base_conf['seed'])
        
        self.n_gpus = self.learner_conf["n_gpus"]
        self.worker_runtime_available = False
        
        self.learner_metrics = {
            "loss": [],
            "kl": [],
            "entropy": [],
            "perplexity": []
        }

        actor_model_config = AutoConfig.from_pretrained(
            self.base_conf['model_path'], trust_remote_code=False, attn_implementation="flash_attention_2"
            )
        self.tokenizer = AutoTokenizer.from_pretrained( self.base_conf['model_path'])
        actor_model_config.eos_token_id = self.tokenizer.eos_token_id
        actor_model_config.pad_token_id = self.tokenizer.pad_token_id
        actor_model_config.bos_token_id = self.tokenizer.bos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_conf['model_path'],
            torch_dtype='float32',
            config=actor_model_config,
            trust_remote_code=False
        ).to('cpu')

        self.logger.info("FSDP learner worker bootstrap omitted in the anonymous review artifact.")
        self.sample_queue = None
        self.model_queue = None
        self.processes = []

        self.device = self.learner_conf["device"]
        self.model = self.model.to(self.device)

        self.ref_model, self.tokenizer = load_policy(self.base_conf['model_path'], dtyp=torch.bfloat16, is_vllm=False, device=self.device)
        self.ref_model = self.ref_model.to(self.device)
        
        self.batchsize = self.learner_conf["batchsize"]
        self.send_iter = self.learner_conf["send_iter"]
        self.per_flag = self.learner_conf["per"]
        self.per_gamma = self.learner_conf['per_gamma']
        self.a_decay = self.learner_conf['a_decay']
        self.exp = Experience(max_size=self.learner_conf["expsize"])
        self.send_params_req = [None for _ in range(len(self.actor_rank))]
        self.update_iter = 0

        self.recv_sample_flag = False
        self.start_time = time.time()

    def recv_sample(self, src_rank):
        recv_count = 0
        while 1:
            if self.recv_sample_flag == True: break
            else:
                received_data = learner_recv(
                    channel="buffer_to_learner",
                    source_rank=src_rank,
                    target_rank=self.rank,
                    logger=self.logger,
                )
                if received_data is not None:
                    try:
                        received_data = pickle.loads(received_data)
                    except Exception as e:
                        self.logger.info(f"Process {self.rank} encountered an error: {str(e)}")
                        return None
                    self.logger.info(f"received_data from rank {src_rank}\n len(received_data['completion_ids']) {len(received_data['completion_ids'])}")
                    self.push_sample(received_data)
                    recv_count += len(received_data['completion_ids'])
                    total_memory = self.exp.get_memory_usage()
                    self.logger.info(f"update_iter {self.update_iter}, total_memory {total_memory:2f}")
                    if recv_count > 0:
                        self.recv_sample_flag = True

    def push_sample(self, received_data):
        self.exp.push_sample_batch(*received_data.values())

    def pull_sample(self, pull_size):
        exp_pulled = self.exp.pull_sample(pull_size, reuse=True, random=True, per=self.per_flag, per_gamma=self.per_gamma, a_decay=self.a_decay, logger=self.logger)
        self.logger.info(f"after pull, pull size {pull_size}, exp size {self.exp.length()}")
        return exp_pulled

    def selective_log_softmax(self, logits, index):
        if logits.dtype in [torch.float32, torch.float64]:
            selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
            logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
            per_token_logps = selected_logits - logsumexp_values
        else:
            per_token_logps = []
            for row_logits, row_labels in zip(logits, index):
                row_logps = F.log_softmax(row_logits, dim=-1)
                row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
                per_token_logps.append(row_per_token_logps)
            per_token_logps = torch.stack(per_token_logps)
        return per_token_logps

    def get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:]
        return logits, self.selective_log_softmax(logits, input_ids)

    def calculate_entropy_and_perplexity(self, logits):
        seq_len = logits.size(1)
        entropy_list = []
        perplexity_list = []

        chunk_size = 30
        for i in range(0, seq_len, chunk_size):
            chunk_logits = logits[:, i:min(i+chunk_size, seq_len), :]
            
            with torch.no_grad():
                log_probs = F.log_softmax(chunk_logits, dim=-1)
                
                probs = log_probs.exp()
                
                token_entropy = -torch.sum(probs * log_probs, dim=-1)
                
                token_perplexity = torch.exp(token_entropy)
                
            entropy_list.append(token_entropy)
            perplexity_list.append(token_perplexity)
        
        entropy = torch.cat(entropy_list, dim=1)
        perplexity = torch.cat(perplexity_list, dim=1)
        return entropy, perplexity

    def compute_loss(self, prompt_comp_ids, attention_mask, completion_mask, logits_to_keep, advantages, old_per_token_logps, ref_per_token_logps):
        logits_to_keep = int(logits_to_keep[0])
        self.logger.info(f"prompt_comp_ids {prompt_comp_ids.shape} \n logits_to_keep {logits_to_keep}")

        logits, per_token_logps = self.get_per_token_logps(
            self.model, prompt_comp_ids, attention_mask, logits_to_keep
            )
        self.logger.info(f"per_token_logps {per_token_logps.shape}")

        ref_per_token_logps = ref_per_token_logps * completion_mask
        per_token_logps = per_token_logps * completion_mask
        old_per_token_logps = old_per_token_logps * completion_mask

        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        beta = self.learner_conf['beta']
        epsilon = self.learner_conf['epsilon']
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - epsilon, 1 + epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        per_token_loss = per_token_loss + beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        self.logger.info(f"loss {loss} \n per_token_logps {per_token_logps.mean()} \n old_per_token_logps {old_per_token_logps.mean()}")

        self.learner_metrics["loss"].append(float(loss.item()))
        self.learner_metrics["kl"].append(float(per_token_kl.mean()))

        del prompt_comp_ids, attention_mask, completion_mask, logits_to_keep, advantages, old_per_token_logps, ref_per_token_logps
        del logits, per_token_logps, per_token_loss, per_token_kl
        gc.collect()
        torch.cuda.empty_cache()
        return loss
        
    def send_model(self, send_data, actor_ranks):
        packed_data = pickle.dumps(send_data)
        self.send_params_req = learner_send(
            channel="learner_to_actor",
            payload=packed_data,
            source_rank=self.rank,
            target_ranks=actor_ranks,
            logger=self.logger,
        )
        self.logger.info(f"rank {self.rank} send model to rank {actor_ranks}")
    
    def list2tensor(self, tensor_list, pad_id=151643):
        if len(tensor_list) == 0:
            return torch.empty((0, 0), dtype=torch.long), torch.empty((0, 0), dtype=torch.int)
        
        max_len = max(len(t) for t in tensor_list)
        padded_list = []
        mask_list = []  
        
        for t in tensor_list:
            pad_len = max_len - len(t)
            padding = torch.full((pad_len,), pad_id, dtype=t.dtype, device=t.device)
            padded_tensor = torch.cat((t, padding))
            mask = torch.ones(len(t), dtype=torch.int)
            if pad_len > 0:
                mask = torch.cat((mask, torch.zeros(pad_len, dtype=torch.int)))
            padded_list.append(padded_tensor)
            mask_list.append(mask)
        
        completion_ids_tensor = torch.stack(padded_list)
        mask_tensor = torch.stack(mask_list)
        return completion_ids_tensor, mask_tensor

    def remove_pad(self, tensor, mask):
        result_list = []
        for i in range(tensor.size(0)):
            non_pad = tensor[i][mask[i] == 1]
            result_list.append(non_pad)
        return result_list

    def process_train(self, exp_pulled):
        prompt_ids_tensor, prompt_mask_tensor = self.list2tensor(exp_pulled['prompt_ids'], pad_id=self.tokenizer.pad_token_id)
        completion_ids_tensor, completion_mask_tensor = self.list2tensor(exp_pulled['completion_ids'], pad_id=self.tokenizer.pad_token_id)
        oldlogp_tensor, oldlogp_mask_tensor = self.list2tensor(exp_pulled['old_logps'], pad_id=-20)
        reflogp_tensor, reflogp_mask_tensor = self.list2tensor(exp_pulled['ref_logps'], pad_id=-20)
        advantage_tensor = torch.tensor(exp_pulled['advantage'])

        train_tensor = (prompt_ids_tensor, prompt_mask_tensor, completion_ids_tensor, completion_mask_tensor, oldlogp_tensor, oldlogp_mask_tensor, reflogp_tensor, reflogp_mask_tensor, advantage_tensor)
        return train_tensor

    def ref_logprobs(self, train_tensor):
        prompt_comp_ids, attention_mask, _, logits_to_keep, _, _ = train_tensor
        logits_to_keep = int(logits_to_keep[0])

        cut_num = 4
        batch_size = prompt_comp_ids.shape[0]
        split_sizes = [batch_size // cut_num] * (cut_num - 1) + [batch_size - (batch_size // cut_num) * (cut_num - 1)]
        prompt_comp_ids_chunks = torch.split(prompt_comp_ids, split_sizes, dim=0)
        attention_mask_chunks = torch.split(attention_mask, split_sizes, dim=0)

        ref_per_token_logps_chunks = []
        for i in range(cut_num):
            chunk_prompt_comp_ids = prompt_comp_ids_chunks[i].to(self.device)
            chunk_attention_mask = attention_mask_chunks[i].to(self.device)
            
            with torch.no_grad():
                logits = self.ref_model(input_ids=chunk_prompt_comp_ids, attention_mask=chunk_attention_mask, logits_to_keep=logits_to_keep + 1).logits
                logits = logits[:, :-1, :]
                chunk_prompt_comp_ids = chunk_prompt_comp_ids[:, -logits_to_keep:]
                logits = logits[:, -logits_to_keep:]
                chunk_ref_logps = self.selective_log_softmax(logits, chunk_prompt_comp_ids)
            
            ref_per_token_logps_chunks.append(chunk_ref_logps)
            del logits
            gc.collect()
            torch.cuda.empty_cache()

        ref_per_token_logps = torch.cat(ref_per_token_logps_chunks, dim=0).to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        return ref_per_token_logps
 
    def run(self):
        self.logger.info(f"learner is running, rank id {self.rank}")
        while True:
            try:
                self.recv_sample(self.buffer_rank[0])
                if self.exp.length() >= self.learner_conf['start_samples']: 
                    exp_pulled = self.pull_sample(self.batchsize)

                    self.recv_sample_flag = False
                    self.process_tokens = sum(len(tensor) for tensor in exp_pulled['completion_ids'])    
                    train_time = time.time()

                    train_tensor = self.process_train(exp_pulled)
                    dataset = TensorDataset(
                        train_tensor[0],
                        train_tensor[1],   
                        train_tensor[2],
                        train_tensor[3],
                        train_tensor[4],
                        train_tensor[5],
                        train_tensor[6],
                        train_tensor[7],
                        train_tensor[8],
                    )
                    if not self.worker_runtime_available:
                        review_only_runtime_unavailable("FSDP learner update worker")
                    for _ in range(self.n_gpus):
                        self.sample_queue.put(dataset)
                    

                    new_model_state = pickle.loads(self.model_queue.get())
                    self.model.load_state_dict(new_model_state, strict=True)
                    
                    total_sum = 0.0
                    total_elements = 0
                    for param in self.model.parameters():
                        total_sum += param.data.sum().item()
                        total_elements += param.data.numel()
                    mean_weighted = total_sum / total_elements
                    self.logger.info(f"model mean: {mean_weighted}")
                    total_sum = 0.0
                    total_elements = 0
                    for param in self.ref_model.parameters():
                        total_sum += param.data.sum().item()
                        total_elements += param.data.numel()
                    ref_mean_weighted = total_sum / total_elements
                    self.logger.info(f"ref_model mean: {ref_mean_weighted}")
                    self.logger.info(f"diff mean: {mean_weighted - ref_mean_weighted}")

                    self.update_iter += 1                
                    thr_step = (self.update_iter * self.batchsize) / (time.time()-self.start_time)
                    thr_token = self.process_tokens / (time.time()-self.start_time)
                    self.logger.info(f"self.update_iter {self.update_iter}, duration {time.time()-self.start_time:.2f}, throughput_step {thr_step:.4f}, throughput_token {thr_token:.4f} tokens/s")
                    if self.update_iter >= self.learner_conf['max_trainstep']:
                        self.logger.info(f"update iter {self.update_iter}, stop run, duration {time.time()-self.start_time:.2f} seconds")
                        self.send_model('stop', self.actor_rank)
                        time.sleep(30)
                        os._exit(-1)
                    elif self.update_iter % self.send_iter == 0:
                        self.logger.info("start send model")
                        self.send_model(self.model, self.actor_rank)

                if self.update_iter % self.learner_conf['kl_iter'] == 0 and self.update_iter!= 0:
                    self.logger.info(f"update ref_model")
                    self.ref_model = deepcopy(self.model)
                
                if (self.update_iter-1) % self.learner_conf['save_iter'] == 0:
                    save_model_dir = self.work_dir / f"model-iter{self.update_iter}"
                    if not os.path.exists(save_model_dir):
                        with open(f"{self.work_dir}/model_time.txt", "a") as f:
                            f.write(f"{self.update_iter}    {time.time() - self.start_time:.2f}\n")
                        with open(self.work_dir / "learner_metrics.json", "w") as f:
                            json.dump(self.learner_metrics, f, indent=4)
                        self.logger.info(f"model saved, iter {self.update_iter}, running {time.time() - self.start_time:.2f} seconds")
                        self.model.save_pretrained(save_model_dir)
                        self.tokenizer.save_pretrained(save_model_dir)
            except Exception as e:
                print(f"Process {self.rank} encountered an error: {str(e)}", flush=True)
                import traceback
                error_traceback = traceback.format_exc()
                print(f"Full traceback:\n{error_traceback}", flush=True)
                os._exit(-1)
            

def main(config):
    review_only_runtime_unavailable("MPI communicator initialization")


if __name__ == "__main__":
    config = load_config('./configs/run_gsm8k.yaml')
    main(config)
 

# -*- coding: utf-8 -*-

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

from utils.fsdp_worker_lora import fsdp_worker
from utils.experience import Experience
from utils.actor_lora_worker import torch_dist_worker
from utils.utils import load_config, load_policy, get_workdir, setup_logger
from utils.comms import actor_send, buffer_recv, buffer_send, learner_recv, learner_send, actor_recv
from utils.review_only import REVIEW_ONLY_ARTIFACT, review_only_runtime_unavailable
from utils.load_math_al_15B import (
    correctness_reward_func,
    get_math_questions,
    int_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    xmlcount_reward_func,
    xml_tag_excess_penalty_func
)

from peft import LoraConfig, get_peft_model
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


REVIEW_ONLY_ARTIFACT = True


class Actor:
    """
    1) Load the model and sample responses
    2) rollout
    3) Verify answers
    4) Compute rewards, advantages, and sample weights
    5) Merge samples and store them in the local replay pool
    6) Pull samples from the replay pool and send them to Buffer
    """
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
        self.logger = setup_logger(f"actor{self.rank}", self.work_dir, self.logging_conf['log_level'], filename=f"actor{self.rank}.log")
        self.logger.info(f"actor_conf: {self.actor_conf}")

        data_paths = self.base_conf['data_path'].split(', ')
        self.dataset = []
        for data_path in data_paths:
            self.dataset.extend(list(get_math_questions(data_path)))

        self.batchsize = self.actor_conf["batchsize"]
        self.epoch_num = self.actor_conf["epoch_num"]
        set_seed(int(self.base_conf['seed']+self.rank))
        self.max_prompt_length = self.actor_conf["max_prompt_length"]
        self.max_new_tokens = self.actor_conf["max_new_tokens"]
        self.num_generations = self.actor_conf["num_generations"]
        self.mini_batchsize = self.actor_conf["mini_batchsize"]
        self.vllm_flag = True

        self.device = self.actor_conf["device"]
        
        model_path = self.base_conf['model_path']
        actor_model_config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=False, attn_implementation="flash_attention_2"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id 
        actor_model_config.eos_token_id = self.tokenizer.eos_token_id    # 151645
        actor_model_config.pad_token_id = self.tokenizer.pad_token_id    # 151643
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
        self.update_iter = 0
        
        # sync
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
        r"""
        Apply a chat template to a conversational example along with the schema for a list of functions in `tools`.

        For more details, see [`maybe_apply_chat_template`].
        """
        # Check that the example has the correct keys
        supported_keys = ["prompt", "chosen", "rejected", "completion", "messages", "label"]
        example_keys = {key for key in example.keys() if key in supported_keys}
        if example_keys not in [
            {"messages"},  # language modeling
            {"prompt"},  # prompt-only
            {"prompt", "completion"},  # prompt-completion
            {"prompt", "chosen", "rejected"},  # preference
            {"chosen", "rejected"},  # preference with implicit prompt
            {"prompt", "completion", "label"},  # unpaired preference
        ]:
            raise KeyError(f"Invalid keys in the example: {example_keys}")

        # Apply the chat template to the whole conversation
        if "messages" in example:
            messages = tokenizer.apply_chat_template(example["messages"], tools=tools, tokenize=False)

        # Apply the chat template to the prompt, adding the generation prompt
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

        # Apply the chat template to the entire prompt + completion
        if "prompt" in example:  # explicit prompt and prompt-completion case
            if "chosen" in example:
                prompt_chosen = tokenizer.apply_chat_template(
                    example["prompt"] + example["chosen"], tools=tools, tokenize=False
                )
                chosen = prompt_chosen[len(prompt) :]
            if "rejected" in example and "prompt" in example:  # explicit prompt
                prompt_rejected = tokenizer.apply_chat_template(
                    example["prompt"] + example["rejected"], tools=tools, tokenize=False
                )
                rejected = prompt_rejected[len(prompt) :]
            if "completion" in example:
                prompt_completion = tokenizer.apply_chat_template(
                    example["prompt"] + example["completion"], tools=tools, tokenize=False
                )
                completion = prompt_completion[len(prompt) :]
        else:  # implicit prompt case
            if "chosen" in example:
                chosen = tokenizer.apply_chat_template(example["chosen"], tools=tools, tokenize=False)
            if "rejected" in example:
                rejected = tokenizer.apply_chat_template(example["rejected"], tools=tools, tokenize=False)

        # Ensure that the prompt is the initial part of the prompt-completion string
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

        # Extract the completion by removing the prompt part from the prompt-completion string
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
            # prompt_inputs = self.tokenizer(prompts_text[i], return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
            prompt_inputs = self.tokenizer(prompts_text[i], return_tensors="pt", padding=False, padding_side="left", add_special_tokens=False)
            prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

            if self.max_prompt_length is not None:
                prompt_ids = prompt_ids[:, -self.max_prompt_length :]
                prompt_mask = prompt_mask[:, -self.max_prompt_length :]
            for _ in range(self.num_generations):
                batch_prompt_ids.append(prompt_ids)
                batch_prompt_mask.append(prompt_mask)
        return batch_prompt_ids, batch_prompt_mask

    def put_prompt(self, prompts):
        if not self.worker_runtime_available:
            review_only_runtime_unavailable("LoRA actor generation worker")
        self.batch_queue.put(prompts)

    def get_result(self):
        if not self.worker_runtime_available:
            review_only_runtime_unavailable("LoRA actor generation result queue")
        result = self.result_queue.get()
        batch_completion_text, batch_completion_ids, old_logprob, entropy, ref_logprob, batch_time = result
        print(f"\n=== Batch inference result (elapsed: {batch_time:.2f}s) ===", flush=True)
        return (batch_completion_text, batch_completion_ids, old_logprob, entropy, ref_logprob)

    def model_generate(self, batch_prompt_ids):
        """
        Convert truncated prompts to text and send them to the generation worker.

        Returns:
            batch_completion_text: generated text list.
            batch_completion_tensor: completion token-id tensor list.
            batch_old_logprob: per-token log probability under the actor.
            batch_entropy: per-token entropy.
            batch_ref_logprob: per-token log probability under the reference model.
        """
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
        """
        Args:
            batch_input: input list with length batchsize.
            batch_completion_text: generated text list with length batchsize * group_size.
        Returns:
            batch_reward: reward for each sequence.
            batch_advantage: advantage for each sequence.
        """
        batch_completion_text_2d = [
            batch_completion_text[i * self.num_generations : (i + 1) * self.num_generations] 
            for i in range(self.batchsize)
        ]

        multi_advantages, multi_rewards = [], []
        for i in range(self.batchsize):
            completions = batch_completion_text_2d[i]
            answer = batch_input[i]['answer']
            reward_funcs=[
                    strict_format_reward_func,
                    correctness_reward_func,
                    xml_tag_excess_penalty_func
                ]
            rewards_per_func = torch.zeros(len(completions), len(reward_funcs))
            for i, reward_func in enumerate(reward_funcs):
                output_reward_func = reward_func(completions=completions, answer=answer)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32)
            rewards = rewards_per_func.sum(dim=1) 
            # for j in range(self.num_generations):
            #     self.logger.info(f"completion {j}: \n {completions[j]}")
            self.logger.info(f"rewards_per_func {rewards_per_func} \n rewards {rewards}")
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
            multi_rewards.append(rewards)   # eg: [tensor([ 2.0000, -0.5000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]), tensor([0., 0., 2., 0., 0., 2., 0., 0.])]
            multi_advantages.append(advantages)
        batch_reward = [torch.tensor(x) for reward_tensor in multi_rewards for x in reward_tensor]
        batch_advantage = [torch.tensor(x) for reward_tensor in multi_advantages for x in reward_tensor]
        return batch_reward, batch_advantage
    
    def compute_weight(self, batch_prompt_ids, batch_entropy, batch_advantage):
        """
        Compute the CSER sampling weight for each sequence.

        Args:
            batch_prompt_ids: prompt token tensor list.
            batch_entropy: entropy tensor list for each sequence.
            batch_advantage: advantage for each sequence.
        Returns:
            batch_weight: replay weight for each sequence.
        """
        batch_weight = []
        for i, entropy in enumerate(batch_entropy):
            prompt_length = len(batch_prompt_ids[i].squeeze(0))
            entropy_completion = entropy[prompt_length:]
            topk = int(min(self.max_new_tokens*0.2, len(entropy_completion)))
            topk_values, _ = entropy_completion.topk(k=topk)
            top_entropy_mean = topk_values.mean(dim=0)
            
            self.actor_metrics["completion_ids_length"].append(len(entropy_completion))
            self.actor_metrics["entropy_mean"].append(entropy_completion.mean(dim=0).item())
            self.actor_metrics["top_entropy_mean"].append(top_entropy_mean.item())

            advantage = batch_advantage[i]
            norm_entropy = top_entropy_mean / (1.0 + top_entropy_mean)
            weight = (1 - norm_entropy) * torch.exp(-advantage) + norm_entropy * torch.exp(advantage)
            weight += 120
            batch_weight.append(torch.as_tensor(weight))
            
        return batch_weight

    def filter_sample(self, batch_advantage, batch_prompt_ids, batch_completion_ids, batch_old_logprob, batch_ref_logprob, batch_weight):
        """
        Filter out sequences with zero advantage and pack Experience fields.

        Args:
            batch_prompt_ids: prompt token tensor list.
            batch_advantage: advantage for each sequence.
        """
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
        """
        Pad a 2D token tensor to the target length.
        """
        pad_len = max_len - t.shape[1]
        if pad_len <= 0:
            return t
        pad_tensor = torch.full((t.shape[0], pad_len), pad_token_id, dtype=t.dtype, device=t.device)
        if padding_side == "right":
            padded_t = torch.cat([t, pad_tensor], dim=1)
        else:
            padded_t = torch.cat([pad_tensor, t], dim=1)
        return padded_t
    
    def push_sample_filter(self, dic):
        self.exp.push_sample_batch(*dic.values())

    def pull_sample(self):
        # send_size = max(self.exp.length(), self.send_size)
        exp_pulled = self.exp.pull_sample(self.send_size)
        # sync
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
                        sys.exit(1)
                    else:
                        self.update_iter += 1
                        self.logger.info(f"rank {self.rank} received model from {src_rank}, self.update_iter {self.update_iter}")
                        self.model = received_data
                        # self.model.load_state_dict(new_model_state, strict=True)
                        # self.model.cpu().to(self.device)
                        if self.vllm_flag:
                            if not self.worker_runtime_available:
                                review_only_runtime_unavailable("LoRA actor model update worker")
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
                        
                        # sync
                        self.recv_model_flag = False
                        self.send_size = self.actor_conf["send_size"]
                        self.logger.info("before recv_model")
                        self.recv_model(self.learner_rank[0])
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
    """
    1) Receive samples from each Actor
    2) Send samples to Learner
    """
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
                # received_data = b'\x80\x04\x95\xff\xff\xff\xff\x00\x00\x00\x00\x8c\x04test\x94.'
                try:
                    received_data = pickle.loads(received_data)
                except Exception as e:
                    self.logger.info(f"Process {self.rank} encountered an error: {str(e)}")
                    return None
                if received_data == 'stop':
                    sys.exit(1)
                else:
                    # self.logger.info(f"rank {self.rank} received samples from rank {src_rank}, received_data['advantages']: \n {received_data['advantages']}")
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
                    pulled = self.pull_sample(self.send_size)
                    self.send_sample(pulled, self.learner_rank[0])
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
    """
    1) Receive samples
    2) Check sample count and update the model
    3) Send the model to Actor
    """
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

        self.lora_conf = config.get('lora_conf', {
            'r': self.learner_conf.get('lora_r', 32),
            'alpha': self.learner_conf.get('lora_alpha', 32),
            'target_modules': self.learner_conf.get(
                'lora_target_modules',
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
            'dropout': self.learner_conf.get('lora_dropout', 0.00)
        })

        self.learner_conf['model_path'] = self.base_conf['model_path']
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_conf['model_path'])
        # actor_model_config = AutoConfig.from_pretrained(
        #     self.base_conf['model_path'], trust_remote_code=False, attn_implementation="flash_attention_2"
        #     )
        # self.tokenizer = AutoTokenizer.from_pretrained( self.base_conf['model_path'])
        # actor_model_config.eos_token_id = self.tokenizer.eos_token_id    # 151645
        # actor_model_config.pad_token_id = self.tokenizer.pad_token_id    # 151643
        self.logger.info("FSDP learner worker bootstrap omitted in the anonymous review artifact.")
        self.sample_queue = None
        self.model_queue = None
        self.processes = []

        self.device = self.learner_conf["device"]
        # self.model, self.tokenizer = load_policy(self.base_conf['model_path'], dtyp=torch.bfloat16, is_vllm=False, utilization=0.5, device=self.device)
        # self.model = self.model.to(self.device)

        # self.ref_model, self.tokenizer = load_policy(self.base_conf['model_path'], dtyp=torch.bfloat16, is_vllm=False, device=self.device)
        # self.ref_model = self.ref_model.to(self.device)
        
        self.batchsize = self.learner_conf["batchsize"]
        self.samples_per_step = self.learner_conf.get("samples_per_step", self.batchsize)
        self.send_iter = self.learner_conf["send_iter"]
        self.per_flag = self.learner_conf["per"]
        self.per_gamma = self.learner_conf['per_gamma']
        self.a_decay = self.learner_conf['a_decay']
        self.exp = Experience(max_size=self.learner_conf["expsize"])
        self.send_params_req = [None for _ in range(len(self.actor_rank))]
        self.update_iter = 0

        # sync
        self.recv_sample_flag = False
        self.start_time = time.time()

    def recv_sample(self, src_rank):
        # sync
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
                    if recv_count > 0:
                        self.recv_sample_flag = True

    def push_sample(self, received_data):
        self.exp.push_sample_batch(*received_data.values())
        # self.logger.info(f"self.exp {self.exp} \n self.exp['completion_ids'] {self.exp.completion_ids}")
        # self.logger.info(f"push sample success. exp size {self.exp.length()} , single sample shape {self.exp.completion_ids[0].shape}")

    def pull_sample(self, pull_size):
        exp_pulled = self.exp.pull_sample(pull_size, reuse=True, random=False, per=self.per_flag, per_gamma=self.per_gamma, a_decay=self.a_decay, samples_per_step=self.samples_per_step, logger=self.logger)
        self.logger.info(f"after pull, pull size {pull_size}, exp size {self.exp.length()}")
        return exp_pulled

    def selective_log_softmax(self, logits, index):
        if logits.dtype in [torch.float32, torch.float64]:
            selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
            # loop to reduce peak mem consumption
            logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
            per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
        else:
            # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
            per_token_logps = []
            for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
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
        return logits, self.selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens
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
        """Pad one-dimensional tensors into a 2D tensor and return its mask."""
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
    def process_train(self, exp_pulled):
        prompt_ids_tensor, prompt_mask_tensor = self.list2tensor(exp_pulled['prompt_ids'], pad_id=self.tokenizer.pad_token_id)
        completion_ids_tensor, completion_mask_tensor = self.list2tensor(exp_pulled['completion_ids'], pad_id=self.tokenizer.pad_token_id)
        oldlogp_tensor, oldlogp_mask_tensor = self.list2tensor(exp_pulled['old_logps'], pad_id=-20)
        reflogp_tensor, reflogp_mask_tensor = self.list2tensor(exp_pulled['ref_logps'], pad_id=-20)
        advantage_tensor = torch.tensor(exp_pulled['advantage'])

        train_tensor = (prompt_ids_tensor, prompt_mask_tensor, completion_ids_tensor, completion_mask_tensor, oldlogp_tensor, oldlogp_mask_tensor, reflogp_tensor, reflogp_mask_tensor, advantage_tensor)
        return train_tensor
    def run(self):
        self.logger.info(f"learner is running, rank id {self.rank}")
        while True:
            try:
                self.recv_sample(self.buffer_rank[0])
                if self.exp.length() >= self.learner_conf['start_samples']: 
                    # self.logger.info(f"collect {self.exp.length()} samples, duration {time.time()-self.start_time:.2f} seconds")
                    exp_pulled = self.pull_sample(self.batchsize)
                    # sync
                    self.recv_sample_flag = False
                    self.process_tokens = sum(len(tensor) for tensor in exp_pulled['completion_ids'])    
                    train_time = time.time()

                    train_tensor = self.process_train(exp_pulled)
                    dataset = TensorDataset(
                        train_tensor[0],    # prompt_ids_tensor
                        train_tensor[1],   
                        train_tensor[2],    # completion_ids_tensor
                        train_tensor[3],
                        train_tensor[4],    # oldlogp_tensor
                        train_tensor[5],
                        train_tensor[6],    # reflogp_tensor
                        train_tensor[7],
                        train_tensor[8],    # advantages_tensor
                    )
                    for _ in range(self.n_gpus):
                        if not self.worker_runtime_available:
                            review_only_runtime_unavailable("LoRA FSDP learner update worker")
                        self.sample_queue.put(dataset)
                    
                    lora_state_dict_bytes = self.model_queue.get()
                    lora_state_dict = pickle.loads(lora_state_dict_bytes)

                    # formatted_dict = {}
                    # for k, v in lora_state_dict.items():
                    #     if not k.startswith("base_model.model."):
                    #         formatted_dict[f"base_model.model.{k}"] = v.to(self.device)
                    #     else:
                    #         formatted_dict[k] = v.to(self.device)

                    # new_model_state = pickle.loads(self.model_queue.get())
                    # self.model.load_state_dict(formatted_dict, strict=True)
                    # self.logger.info(f"train epoch time: {time.time()-train_time:.2f}")
                    
                    # total_sum = 0.0
                    # total_elements = 0
                    # for param in self.model.parameters():
                    #     total_sum += param.data.sum().item()
                    #     total_elements += param.data.numel()
                    # mean_weighted = total_sum / total_elements
                    # self.logger.info(f"model mean: {mean_weighted}")
                    # total_sum = 0.0
                    # total_elements = 0
                    # for param in self.ref_model.parameters():
                    #     total_sum += param.data.sum().item()
                    #     total_elements += param.data.numel()
                    # ref_mean_weighted = total_sum / total_elements
                    # self.logger.info(f"ref_model mean: {ref_mean_weighted}")
                    # self.logger.info(f"diff mean: {mean_weighted - ref_mean_weighted}")

                    self.update_iter += 1                
                    thr_step = (self.update_iter * self.batchsize) / (time.time()-self.start_time)
                    thr_token = self.process_tokens / (time.time()-self.start_time)
                    self.logger.info(f"self.update_iter {self.update_iter}, duration {time.time()-self.start_time:.2f}, throughput_step {thr_step:.4f}, throughput_token {thr_token:.4f} tokens/s")
                    if self.update_iter >= self.learner_conf['max_trainstep']:
                        self.logger.info(f"update iter {self.update_iter}, stop run, duration {time.time()-self.start_time:.2f} seconds")
                        self.send_model('stop', self.actor_rank)
                        time.sleep(30)
                        sys.exit(1)
                    elif self.update_iter % self.send_iter == 0:
                        self.logger.info("start send model")
                        # self.send_model(deepcopy(self.model.module).to("cpu"), self.actor_rank)
                        # self.send_model(new_model_state, self.actor_rank)
                        self.send_model(pickle.dumps(lora_state_dict), self.actor_rank)

                # if self.update_iter % self.learner_conf['kl_iter'] == 0 and self.update_iter!= 0:
                #     self.logger.info(f"update ref_model")
                #     self.ref_model = deepcopy(self.model)
                
                if (self.update_iter-1) % self.learner_conf['save_iter'] == 0:
                    save_model_dir = self.work_dir / f"model-iter{self.update_iter}"
                    if not os.path.exists(save_model_dir):
                        # with open(f"{self.work_dir}/model_time.txt", "a") as f:
                        #     f.write(f"{self.update_iter}    {time.time() - self.start_time:.2f}\n")
                        # with open(self.work_dir / "learner_metrics.json", "w") as f:
                        #     json.dump(self.learner_metrics, f, indent=4)
                        # self.logger.info(f"model saved, iter {self.update_iter}, running {time.time() - self.start_time:.2f} seconds")
                        # self.model.save_pretrained(save_model_dir)
                        # self.tokenizer.save_pretrained(save_model_dir)
                        os.makedirs(save_model_dir, exist_ok=True)

                        lora_save_dict = pickle.loads(lora_state_dict_bytes)
                        torch.save(lora_save_dict, save_model_dir / "adapter_model.bin")

                        adapter_config = {
                                "r": self.lora_conf['r'],
                                "lora_alpha": self.lora_conf['alpha'],
                                "target_modules": self.lora_conf['target_modules'],
                                "lora_dropout": self.lora_conf['dropout'],
                                "bias": "none",
                                "task_type": "CAUSAL_LM"
                            }
                        with open(save_model_dir / "adapter_config.json", "w") as f:
                                json.dump(adapter_config, f, indent=2)
                        
                        self.tokenizer.save_pretrained(save_model_dir)
                        self.logger.info(f"Checkpoint saved: {save_model_dir}")

                    # torch.distributed.barrier()
            except Exception as e:
                print(f"Process {self.rank} encountered an error: {str(e)}", flush=True)
                import traceback
                error_traceback = traceback.format_exc()
                print(f"Full traceback:\n{error_traceback}", flush=True)
                os._exit(-1)
            

def main(config):
    review_only_runtime_unavailable("LoRA MPI communicator initialization")


if __name__ == "__main__":
    config = load_config('./configs/run_lora_math.yaml')
    main(config)
 

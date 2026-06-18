
import os
import re
import sys
import time
import argparse
sys.path.append("../")

import torch
import random
import numpy as np
from datasets import load_dataset
import torch.multiprocessing as mp
from vllm import LLM, SamplingParams

from utils.answer_match import compute_score
from utils.vllm_mpworker_eval import torch_dist_worker
from utils.utils import load_config, load_policy, get_workdir, setup_logger


def load_data(data_path, data_type='parquet'):
    if data_type == 'json':
        ds = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  
                    continue
                item = json.loads(line)
                ds.append(item)
        problems = [{"problem": p['problem'], "ground_truth": p['answer']} for p in ds]
    elif data_type == "parquet":
        ds = load_dataset("parquet", data_files=data_path)['train']
        problems = [{"problem": p['question'], "ground_truth": p['answer']} for p in ds]
    print(f"got data from {data_path}", flush=True)
    return problems

def get_prompt(questions):
    prompts = []
    for ques in questions:
        system_prompt = "<|im_start|>system\nYou are a math problem solver. You will receive a math problem and need to provide the final answer. Format your response with the solution process in <think> </think> tags and the final answer in <answer> </answer> tags. Please output in format: <number> in the <answer> </answer> tags. And <number> is the final answer. For example <answer> 30 </answer>.\n<|im_end|>\n"
        user_part = f'<|im_start|>user\n{ques}<|im_end|>\n'
        assistant_start = '<|im_start|>assistant\n'
        new_question = system_prompt + user_part + assistant_start
        prompts.append(new_question)
    return prompts

    

def rollout(batch_queue, result_queue, data, logger, group_k=1):
    questions = [problem['problem'] for problem in data]
    batch_prompts = get_prompt(questions)
    batch_queue.put(batch_prompts)
    
    _, outputs, _ = result_queue.get()
    for i, output in enumerate(outputs):
        for k in range(group_k):
            answer = output.outputs[k].text
            data[i][f'model_answer{k}'] = answer
    
    return data
    
def judge(data, group_k):
    rewards = [0 for i in range(len(data))]
    for index, item in enumerate(data):
        ground_truth = item['ground_truth']
        gt_match = re.search(r'####\s*(\d+)', ground_truth)
        if gt_match:
            gt_number = float(gt_match.group(1))
        else:
            gt_number = None  
            
        for k in range(group_k):
            model_answer = item[f'model_answer{k}']
            boxed_match = re.search(r'\\boxed{(\d+)}', model_answer)
            sharp_match = re.search(r'####\s*(\d+)', model_answer)
            numbers_match = re.findall(r'\d+\.?\d*', model_answer)
            matchs = []
            if boxed_match:
                boxed = float(boxed_match.group(1))
                matchs.append(boxed)
            if sharp_match:
                sharp = float(sharp_match.group(1))
                matchs.append(sharp)
            if numbers_match:
                number = float(numbers_match[-1])
                matchs.append(number)


            for match in matchs:
                if match == gt_number:
                    rewards[index] = 1
                    break
            
            if rewards[index] == 1:
                break

    return rewards

def eval(model_path, data, group_k=1):
    rollout_data = rollout(model_path, data=data, group_k=group_k)
    rewards = judge(data=rollout_data, group_k=group_k)
    win_rate = sum(rewards) / ( len(rewards) + 0.0000001)
    print(f"{win_rate:.4f}")
    return win_rate

def save(txt_path, model_name, win_rate):
    save_txt = model_name + '   ' + f"{win_rate:.4f} \n"
    with open(txt_path, 'a', encoding='utf-8') as file:
        file.write(save_txt)

def draw(txt_path):
    pass

def get_model_folders(proj_path):
    model_folders = []
    if not os.path.exists(proj_path):
        print(f"Path {proj_path} does not exist")
        os._exit(-1)
    
    for item in os.listdir(proj_path):
        item_path = os.path.join(proj_path, item)
        if os.path.isdir(item_path) and re.match(r'model-iter\d+', item):
            model_folders.append(item)
        if os.path.isdir(item_path) and re.match(r'global_step_\d+', item):
            model_folders.append(item)

    def extract_number(folder_name):
        match = re.search(r'\d+', folder_name)
        return int(match.group()) if match else 0
    model_folders.sort(key=extract_number)
    return model_folders

def vllm_worker(model_path, local_port, gpu_id, group_k, logger, batch_queue, model_queue, result_queue):
    torch_dist_worldsize = 1
    rank = 0
    processes = []
    if group_k > 1:
        sampling_params = SamplingParams(temperature=0.6, 
                                        max_tokens=800, 
                                        top_p=1.0,        
                                        n=group_k,
                                        logprobs=None)
    else:
        sampling_params = SamplingParams(
                                        temperature=0.0,
                                        top_p=1.0,        
                                        top_k=-1,         
                                        max_tokens=800,
                                        n=group_k,
                                        seed=42,
                                        logprobs=None)

    for local_rank in range(torch_dist_worldsize):
        p = mp.Process(
            target=torch_dist_worker,
            args=(local_port,
                    rank,
                    gpu_id, 
                    sampling_params,
                    local_rank,
                    torch_dist_worldsize, 
                    model_path, 
                    batch_queue, 
                    model_queue, 
                    result_queue,
                    logger
                )
        )
        p.start()
        processes.append(p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj", type=str, default='0518-bs24-per8-grpo-lr2e6-random-st128', help="project name")
    parser.add_argument("--model_type", type=str, default='qwen', help="qwen or llama")
    parser.add_argument("--data_path", type=str, default="<gsm8k-test-parquet>", help="data")
    parser.add_argument("--group_k", type=int, default=1, help="group_ks")
    parser.add_argument("--gpu_id", type=int, default=3, help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--local_port", type=int, default=10256, help="port for vllm worker")
    parser.add_argument("--is_verl", action="store_true", default=False)
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    proj_path = f"../experiments/{args.proj}/"
    logger = setup_logger(f"eval", proj_path, "INFO", filename=f"eval.log")
    group_ks = [args.group_k]

    
    if args.model_type == 'qwen':
        model_path = "<qwen2.5-0.5b-instruct>"
    elif args.model_type == 'llama':
        model_path = "<llama-3.2-3b-instruct>"
    batch_queue = mp.Queue()
    model_queue = mp.Queue()
    result_queue = mp.Queue()
    
    
    for group_k in group_ks:
        vllm_worker(model_path, args.local_port, args.gpu_id, group_k, logger, batch_queue, model_queue, result_queue)
        eval_data = load_data(args.data_path, data_type='parquet')
        model_names = get_model_folders(proj_path)
        save_path = proj_path + f'temp0-pass@{group_k}.txt'
        
        for name in model_names:
            model_path = os.path.join(proj_path, name, "actor/huggingface") if args.is_verl else os.path.join(proj_path, name)
            new_model, tokenizer = load_policy(model_path, is_vllm=False, device="cuda", dtyp='float32')
            logger.info(f"Loaded new model successfully: {model_path}")
            model_queue.put(new_model)
            time.sleep(1)
            logger.info(f"vLLM worker model update completed: {model_path}")

            rollout_data = rollout(batch_queue, result_queue, eval_data, logger, group_k)
            rewards = judge(data=rollout_data, group_k=group_k)
            win_rate = sum(rewards) / ( len(rewards) + 0.0000001)
            print(f"{model_path}: {win_rate:.4f}")
            save(save_path, name, win_rate)
    

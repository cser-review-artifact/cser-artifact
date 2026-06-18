import os
import sys
import yaml
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_workdir(project_name):
    path = Path("./experiments") / project_name
    path = Path(path)
    if path.exists():
        path, suffix = (
            (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        )
        for n in range(2, 1024):
            p = f"{path}{n}{suffix}"
            if not os.path.exists(p):
                break
        path = Path(p)
    os.makedirs(path, exist_ok=True)  
    return path  


def load_policy(model_path, dtyp=torch.bfloat16, is_vllm=False, device="cuda:0", utilization=0.7, max_tokens=800):
    if is_vllm:
        model = LLM(model=model_path, 
                    device=device, 
                    gpu_memory_utilization=utilization,
                    dtype=dtyp,
                    tensor_parallel_size=1)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"vllm, got policy from {model_path}", flush=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtyp,
            use_cache=True,
            device_map=device
        )
        print(f"transformer, got policy from {model_path}", flush=True)
    return model, tokenizer

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

def extract_answer(text):
    if text.startswith("####"):
        return text.strip().split("####")[-1].strip()
    else:
        import re
        match = re.search(r'\d+', text)
        if match:
            return match.group()
        else:
            return None

def setup_logger(name='default', path=None, log_level='INFO', filename="train.log"):
    logger = logging.getLogger(name)
    os.makedirs(path, exist_ok=True)
    log_filename = os.path.join(path, filename)
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    log_level = getattr(logging, log_level, logging.INFO)

    if not logger.handlers:
        file_handler = RotatingFileHandler(
            log_filename,
            maxBytes=1024 * 1024 * 1024
        )
        file_handler.setFormatter(logging.Formatter(log_format, date_format))

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format, date_format))

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    logger.setLevel(log_level)
    logger.propagate = False

    return logger

def print_memory_usage(logger):
    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    logger.info(f"Memory allocated: {allocated} MB, Memory reserved: {reserved} MB")
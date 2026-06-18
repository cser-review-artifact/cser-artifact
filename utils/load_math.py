import re
import json
from datasets import load_dataset, Dataset

from utils.answer_match import compute_score







SYSTEM_PROMPT = """Let's think step by step and output the final answer within \\boxed{}. """




def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions(gsm8k_path, split = "train") -> Dataset:
    data = load_dataset("parquet", data_files=gsm8k_path)['train']
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })
    return data

def get_math_questions(math_path, split = "train") -> Dataset:
    print(f"math_path {math_path}", flush=True)
    data = load_dataset("parquet", data_files={math_path})['train']
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['problem']}
        ],
        'answer': x['answer']
    })
    return data

def correctness_reward_func(completions, answer, **kwargs) -> list[float]:
    answer = [answer] * len(completions)
    rewards = []
    for i, a in enumerate(answer):
        if compute_score(a, completions[i]):
            rewards.append(2.0)
        else:
            rewards.append(0.0)
    return rewards


def int_reward_func(completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, answer, **kwargs) -> list[float]:
    pattern = r"^<think>[\s\S]*</think>\s*<answer>[\s\S]*</answer>$"    
    rewards = []
    for r in completions:
        match = re.match(pattern, r, re.DOTALL)
        if match:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def soft_format_reward_func(completions, answer, **kwargs) -> list[float]:
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, answer, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def xml_tag_excess_penalty_func(completions, answer, **kwargs) -> list[float]:
    rewards = []
    for completion in completions:
        reasoning_open = completion.count("<think>")
        reasoning_close = completion.count("</think>")
        answer_open = completion.count("<answer>")
        answer_close = completion.count("</answer>")
        
        if (
            reasoning_open > 1 
            or reasoning_close > 1 
            or answer_open > 1 
            or answer_close > 1
        ):
            rewards.append(-0.5)
        else:
            rewards.append(0.0)
    return rewards
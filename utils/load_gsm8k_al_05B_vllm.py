import os
import re
from datasets import load_dataset, Dataset




SYSTEM_PROMPT = """You are a math problem solver. You will receive a math problem and need to provide both the solution process and the final answer. Format your response with the solution process in <think> </think> tags and the final answer in <answer> </answer> tags. Please output in format: <number> in the <answer> </answer> tags. And <number> is the final answer. For example <answer> 30 </answer>.
"""





def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


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


def correctness_reward_func(completions, answer, **kwargs) -> list[float]:
    extracted_responses = [extract_xml_answer(r) for r in completions]
    answer = [answer] * len(extracted_responses)
    rewards = [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]
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


def extract_countdown_answer(text: str) -> str | None:
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def validate_countdown_equation(equation_str, available_numbers) -> bool:
    try:
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
        return sorted(numbers_in_eq) == sorted(int(n) for n in available_numbers)
    except Exception:
        return False


def evaluate_countdown_equation(equation_str):
    try:
        if not re.match(r"^[\d+\-*/().\s]+$", equation_str):
            return None
        return eval(equation_str, {"__builtins__": None}, {})
    except Exception:
        return None


def countdown_func(completions, answer, **kwargs) -> list[float]:
    rewards = []
    for completion in completions:
        equation = extract_countdown_answer(completion)
        if equation is None:
            rewards.append(0.0)
            continue

        target = answer["target"]
        numbers = answer["numbers"]
        if not validate_countdown_equation(equation, numbers):
            rewards.append(0.1)
            continue

        result = evaluate_countdown_equation(equation)
        if result is not None and abs(result - target) < 1e-5:
            rewards.append(1.0)
        else:
            rewards.append(0.1)
    return rewards


def get_countdown_questions(countdown_path, split = "train") -> Dataset:
    data_file = countdown_path if countdown_path.endswith(".parquet") else os.path.join(countdown_path, f"{split}.parquet")
    data = load_dataset("parquet", data_files=data_file)["train"]
    column_names = set(getattr(data, "column_names", []))
    required_columns = {"prompt", "reward_model"}
    if column_names and not required_columns.issubset(column_names):
        missing = sorted(required_columns - column_names)
        raise KeyError(f"Countdown dataset missing required columns: {missing}")
    if len(data) > 0:
        first = data[0]
        if "prompt" not in first or "reward_model" not in first:
            raise KeyError("Countdown dataset rows must contain prompt and reward_model")
        reward_model = first["reward_model"]
        if not isinstance(reward_model, dict) or "ground_truth" not in reward_model:
            raise KeyError("Countdown reward_model must contain ground_truth")
    return data


def _gsm8k_answer_getter(example):
    return example["answer"]


def _countdown_answer_getter(example):
    return example["reward_model"]["ground_truth"]


TASK_SPECS = {
    "gsm8k": {
        "loader": get_gsm8k_questions,
        "answer_getter": _gsm8k_answer_getter,
        "reward_funcs": [
            strict_format_reward_func,
            correctness_reward_func,
            xml_tag_excess_penalty_func,
        ],
    },
    "countdown": {
        "loader": get_countdown_questions,
        "answer_getter": _countdown_answer_getter,
        "reward_funcs": [countdown_func],
    },
}


def get_task_spec(task_type: str):
    normalized = task_type.lower()
    if normalized not in TASK_SPECS:
        supported = ", ".join(sorted(TASK_SPECS))
        raise ValueError(f"Unsupported task_type '{task_type}'. Supported tasks: {supported}")
    return TASK_SPECS[normalized]

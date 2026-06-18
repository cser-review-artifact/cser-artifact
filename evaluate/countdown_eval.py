import argparse
import os
import random
import time

from utils.load_gsm8k_al_05B_vllm import countdown_func


def countdown_judge_single(completion, ground_truth):
    return 1 if countdown_func([completion], ground_truth)[0] == 1.0 else 0


def _resolve_countdown_data_path(data_path):
    if data_path.endswith(".parquet"):
        return data_path
    return os.path.join(data_path, "test.parquet")


def load_countdown_data(data_path):
    from datasets import load_dataset

    parquet_path = _resolve_countdown_data_path(data_path)
    ds = load_dataset("parquet", data_files=parquet_path)["train"]
    problems = []
    for item in ds:
        if "prompt" not in item or "reward_model" not in item:
            raise KeyError("Countdown rows must contain prompt and reward_model")
        reward_model = item["reward_model"]
        if not isinstance(reward_model, dict) or "ground_truth" not in reward_model:
            raise KeyError("Countdown reward_model must contain ground_truth")
        prompt = item["prompt"]
        if isinstance(prompt, list):
            prompt_text = prompt[0]["content"]
        else:
            prompt_text = prompt
        problems.append({
            "prompt": prompt_text,
            "ground_truth": reward_model["ground_truth"],
        })
    print(f"Loaded {len(problems)} countdown problems from {parquet_path}", flush=True)
    return problems


def rollout(batch_queue, result_queue, data, logger, group_k=1):
    prompts = [problem["prompt"] for problem in data]
    batch_queue.put(prompts)

    _, outputs, _ = result_queue.get()
    for i, output in enumerate(outputs):
        for k in range(group_k):
            data[i][f"model_answer{k}"] = output.outputs[k].text

    for i in range(min(3, len(data))):
        logger.info(f"\n--- data[{i}] ---")
        logger.info(f"ground_truth: {data[i]['ground_truth']}")
        for k in range(min(group_k, 2)):
            logger.info(f"model_answer{k}: {data[i][f'model_answer{k}'][:300]}")
    return data


def judge(data, group_k):
    rewards = [0] * len(data)
    for index, item in enumerate(data):
        ground_truth = item["ground_truth"]
        for k in range(group_k):
            if countdown_judge_single(item[f"model_answer{k}"], ground_truth) == 1:
                rewards[index] = 1
                break
    return rewards


def iter_batches(data, batch_size):
    if batch_size is None or batch_size <= 0:
        yield data
        return
    for start in range(0, len(data), batch_size):
        yield data[start:start + batch_size]


def vllm_worker(model_path, local_port, gpu_id, group_k, logger, batch_queue, model_queue, result_queue):
    import torch.multiprocessing as mp
    from vllm import SamplingParams
    from utils.vllm_mpworker_eval import torch_dist_worker

    torch_dist_worldsize = 1
    rank = 0
    sampling_params = SamplingParams(
        temperature=0.6 if group_k > 1 else 0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=2048,
        n=group_k,
        seed=None if group_k > 1 else 42,
        logprobs=None,
    )

    processes = []
    for local_rank in range(torch_dist_worldsize):
        process = mp.Process(
            target=torch_dist_worker,
            args=(
                local_port,
                rank,
                gpu_id,
                sampling_params,
                local_rank,
                torch_dist_worldsize,
                model_path,
                batch_queue,
                model_queue,
                result_queue,
                logger,
            ),
        )
        process.start()
        processes.append(process)
    return processes


def get_model_folders(proj_path):
    import re

    if not os.path.exists(proj_path):
        raise FileNotFoundError(f"Project path does not exist: {proj_path}")

    model_folders = []
    for item in os.listdir(proj_path):
        item_path = os.path.join(proj_path, item)
        if os.path.isdir(item_path) and (re.match(r"model-iter\d+", item) or re.match(r"global_step_\d+", item)):
            model_folders.append(item)

    def extract_number(folder_name):
        match = re.search(r"\d+", folder_name)
        return int(match.group()) if match else 0

    model_folders.sort(key=extract_number)
    return model_folders


def save_result(txt_path, model_name, win_rate):
    with open(txt_path, "a", encoding="utf-8") as file:
        file.write(model_name + "   " + f"{win_rate:.4f}\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj", type=str, default="0611-vespo-bs24-countdown-cser8-st128-lr2e6")
    parser.add_argument("--proj_root", type=str, default="../experiments")
    parser.add_argument("--data_path", type=str, default="<countdown-test-parquet>")
    parser.add_argument("--base_model", "--model_path", dest="base_model", type=str, default="<qwen2.5-1.5b-instruct>")
    parser.add_argument("--group_k", type=int, default=None)
    parser.add_argument("--group_ks", type=int, nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--local_port", "--port", dest="local_port", type=int, default=19001)
    parser.add_argument("--is_verl", "--use_lora", dest="is_verl", action="store_true", default=False)
    return parser.parse_args()


def main():
    import numpy as np
    import torch
    import torch.multiprocessing as mp
    from utils.utils import load_policy, setup_logger

    args = parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    proj_path = os.path.join(args.proj_root, args.proj)
    logger = setup_logger("eval_countdown", proj_path, "INFO", filename="countdown_eval.log")
    group_ks = args.group_ks if args.group_ks is not None else [args.group_k or 1]

    batch_queue = mp.Queue()
    model_queue = mp.Queue()
    result_queue = mp.Queue()

    for group_k in group_ks:
        logger.info(f"start countdown pass@{group_k}")
        vllm_worker(args.base_model, args.local_port, args.gpu_id, group_k, logger, batch_queue, model_queue, result_queue)
        eval_data = load_countdown_data(args.data_path)
        model_names = get_model_folders(proj_path)
        save_path = os.path.join(proj_path, f"countdown_pass@{group_k}.txt")

        for name in model_names:
            checkpoint_path = os.path.join(proj_path, name, "actor", "huggingface") if args.is_verl else os.path.join(proj_path, name)
            new_model, _ = load_policy(checkpoint_path, is_vllm=False, device="cuda", dtyp="float32")
            logger.info(f"loaded model {checkpoint_path}")
            model_queue.put(new_model)
            time.sleep(1)

            rewards = []
            for batch in iter_batches(eval_data, args.batch_size):
                rollout_data = rollout(batch_queue, result_queue, batch, logger, group_k)
                rewards.extend(judge(data=rollout_data, group_k=group_k))
            win_rate = sum(rewards) / (len(rewards) + 1e-7)
            print(f"{checkpoint_path}: {win_rate:.4f}", flush=True)
            logger.info(f"{checkpoint_path}: pass@{group_k}={win_rate:.4f}")
            save_result(save_path, name, win_rate)


if __name__ == "__main__":
    main()

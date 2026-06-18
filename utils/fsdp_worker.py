import functools
import gc
import math
import os
import pickle
import time
import traceback
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast
from flash_attn.bert_padding import pad_input
from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, CPUOffload
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.fsdp.api import StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, get_peft_model_state_dict
from utils.review_only import review_only_runtime_unavailable


REVIEW_ONLY_ARTIFACT = True



def param_mean(model):
    total_sum = 0.0
    total_elements = 0
    for param in model.parameters():
        total_sum += param.data.sum().item()
        total_elements += param.data.numel()
    mean_weighted = total_sum / total_elements
    return mean_weighted

def init_fn(module):
    review_only_runtime_unavailable("FSDP parameter initialization")

def get_qwen2_5b_fsdp_wrap_policy():
    review_only_runtime_unavailable("FSDP wrapping policy")

def split_tensor_by_lengths(tensor, lengths):
    segments = []
    start = 0
    for length in lengths:
        segments.append(tensor[start:start+length])
        start += length
    return segments

def agg_loss(per_token_loss, completion_mask):
    valid_values = torch.where(completion_mask.bool(), per_token_loss, 0.0)
    s = (valid_values * completion_mask).sum()
    policy_loss = s / (completion_mask.sum() + 1e-8)
    return policy_loss
    
def remove_pad(tensor, mask):
    result_list = []
    for i in range(tensor.size(0)):
        non_pad = tensor[i][mask[i] == 1]
        result_list.append(non_pad)
    return result_list

def remove_prompt(batch_logprob, batch_prompt_ids):
    batch_logprob_rmp = []
    for i in range(len(batch_logprob)):
        prompt_length = len(batch_prompt_ids[i])
        remaining = batch_logprob[i][prompt_length:]
        batch_logprob_rmp.append(remaining)
    return batch_logprob_rmp

def list2tensor(tensor_list, pad_id=151643):
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

def expand_advantage(batch_new, advantage):
    batch_advantage = []
    for i in range(len(batch_new)):
        n = batch_new[i].numel()
        expanded_tensor = torch.full((n,), advantage[i], dtype=advantage.dtype)
        batch_advantage.append(expanded_tensor)
    return batch_advantage

def logprobs_from_logits_flash_attn(logits, labels, inplace_backward=True):
    output = cross_entropy_loss(logits, labels, inplace_backward=inplace_backward)
    assert isinstance(output, tuple), (
        "please make sure flash-attn>=2.4.3 where cross_entropy_loss returns Tuple[losses, z_losses]."
    )
    return -output[0]

def logprobs_from_logits(logits, labels, inplace_backward=True): 
    batch_dim = logits.shape[:-1]
    last_dim = logits.shape[-1]
    logits = logits.reshape(-1, last_dim)
    labels = labels.reshape(-1)

    output = logprobs_from_logits_flash_attn(logits, labels, inplace_backward=inplace_backward)
    output = output.view(*batch_dim)
    return output

def compute_logprob(model, batch_prompt_ids, batch_completion_ids, device, logger, temperature=1.0):
    batch_prompt_completion_ids = []
    for i in range(len(batch_prompt_ids)):
        prompt_list = batch_prompt_ids[i].tolist()  
        completion_list = batch_completion_ids[i].tolist()  
        combined_list = prompt_list + completion_list
        batch_prompt_completion_ids.append(combined_list)
    
    flat_ids = [token_id for seq in batch_prompt_completion_ids for token_id in seq]
    input_ids_tensor = torch.tensor(flat_ids, dtype=torch.int32).unsqueeze(0).to(device)

    position_ids = []
    seq_length = []
    for seq in batch_prompt_completion_ids:
        seq_positions = list(range(len(seq)))  
        seq_length.append(len(seq))   
        position_ids.extend(seq_positions)
    position_ids_tensor = torch.tensor(position_ids, dtype=torch.int32).unsqueeze(0).to(device)

    output = model(
        input_ids=input_ids_tensor,
        attention_mask=None,
        position_ids=position_ids_tensor,
        use_cache=False,
    )
    logits = output.logits.squeeze(0)
    input_ids_rmpad_rolled = torch.roll(input_ids_tensor, shifts=-1, dims=1)
    inplace_backward = False
    log_probs = logprobs_from_logits(
        logits=logits,
        labels=input_ids_rmpad_rolled,
        inplace_backward=inplace_backward,
    )
    
    batch_logprob = split_tensor_by_lengths(log_probs, seq_length)
    logger.info(f"len(logprob) {len(batch_logprob)}, logprob[0] {batch_logprob[0].shape}")

    return batch_logprob

def compute_vespo_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    rollout_is_weights=None,
):
    k_pos = 2.0
    lambda_pos = 3.0
    k_neg = 3.0
    lambda_neg = 2.0

    log_ratio = log_prob - old_log_prob
    log_ratio = torch.clamp(log_ratio, -20.0, 20.0)

    seq_log_ratio = torch.sum(log_ratio * response_mask, dim=-1)

    seq_log_tis = None
    if rollout_is_weights is not None:
        log_tis = torch.log(rollout_is_weights.clamp(min=1e-8))
        log_tis = torch.clamp(log_tis, -20.0, 20.0)
        seq_log_tis = torch.sum(log_tis * response_mask, dim=-1)
        seq_log_ratio_combined = seq_log_ratio + seq_log_tis
    else:
        seq_log_ratio_combined = seq_log_ratio

    seq_log_ratio_clamped = torch.clamp(seq_log_ratio_combined, -20.0, 20.0)
    w_seq = torch.exp(seq_log_ratio_clamped.detach())

    seq_adv = advantages[:, 0] if advantages.dim() > 1 else advantages

    pos_mask_seq = (seq_adv >= 0).float()
    neg_mask_seq = 1.0 - pos_mask_seq

    k_seq = pos_mask_seq * k_pos + neg_mask_seq * k_neg
    lam_seq = pos_mask_seq * lambda_pos + neg_mask_seq * lambda_neg

    lam_safe = torch.clamp(lam_seq, min=1e-4)
    log_w_seq = torch.log(w_seq.clamp(min=1e-8))

    log_phi = lam_safe + k_seq * log_w_seq - lam_safe * w_seq

    phi_seq = torch.exp(log_phi).detach()
    phi_seq = torch.nan_to_num(phi_seq, nan=0.0, posinf=0.0, neginf=0.0)

    phi_token = phi_seq.unsqueeze(-1)

    loss_mat = -phi_token * advantages * log_prob

    pg_loss = agg_loss(loss_mat, response_mask)

    with torch.no_grad():
        seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)

        seq_log_ratio_normalized = seq_log_ratio / seq_lengths

        metrics = {
            "actor/approx_kl": (-seq_log_ratio_normalized).mean().item(),
            "vespo/w_seq_mean": w_seq.mean().item(),
            "vespo/w_seq_max": w_seq.max().item(),
            "vespo/w_seq_min": w_seq.min().item(),
            "vespo/w_seq_std": w_seq.std().item(),
            "vespo/log_w_seq_mean": seq_log_ratio_clamped.mean().item(),
            "vespo/log_w_seq_max": seq_log_ratio_clamped.max().item(),
            "vespo/log_w_seq_min": seq_log_ratio_clamped.min().item(),
            "vespo/phi_mean": phi_seq.mean().item(),
            "vespo/phi_max": phi_seq.max().item(),
            "vespo/phi_min": phi_seq.min().item(),
            "vespo/k_pos": k_pos,
            "vespo/lambda_pos": lambda_pos,
            "vespo/k_neg": k_neg,
            "vespo/lambda_neg": lambda_neg,
        }

        pos_seq_mask = seq_adv > 0
        neg_seq_mask = seq_adv < 0

        if pos_seq_mask.any():
            metrics["vespo/w_seq_pos_mean"] = w_seq[pos_seq_mask].mean().item()
            metrics["vespo/phi_pos_mean"] = phi_seq[pos_seq_mask].mean().item()
            metrics["vespo/n_pos_seq"] = pos_seq_mask.sum().item()
        else:
            metrics["vespo/w_seq_pos_mean"] = 0.0
            metrics["vespo/phi_pos_mean"] = 0.0
            metrics["vespo/n_pos_seq"] = 0

        if neg_seq_mask.any():
            metrics["vespo/w_seq_neg_mean"] = w_seq[neg_seq_mask].mean().item()
            metrics["vespo/phi_neg_mean"] = phi_seq[neg_seq_mask].mean().item()
            metrics["vespo/n_neg_seq"] = neg_seq_mask.sum().item()

            neg_w = w_seq[neg_seq_mask]
            noise_mask = neg_w > 100.0
            if noise_mask.any():
                metrics["vespo/neg_noise_count"] = noise_mask.sum().item()
                metrics["vespo/neg_noise_max_w"] = neg_w[noise_mask].max().item()
                metrics["vespo/neg_noise_phi_mean"] = phi_seq[neg_seq_mask][noise_mask].mean().item()
            else:
                metrics["vespo/neg_noise_count"] = 0
                metrics["vespo/neg_noise_max_w"] = 0.0
                metrics["vespo/neg_noise_phi_mean"] = 0.0
        else:
            metrics["vespo/w_seq_neg_mean"] = 0.0
            metrics["vespo/phi_neg_mean"] = 0.0
            metrics["vespo/n_neg_seq"] = 0
            metrics["vespo/neg_noise_count"] = 0
            metrics["vespo/neg_noise_max_w"] = 0.0
            metrics["vespo/neg_noise_phi_mean"] = 0.0

        if rollout_is_weights is not None:
            seq_tis_for_metrics = torch.exp(torch.clamp(seq_log_tis, -20.0, 20.0))
            metrics["vespo/tis_enabled"] = 1.0
            metrics["vespo/seq_log_tis_mean"] = seq_log_tis.mean().item()
            metrics["vespo/seq_log_tis_max"] = seq_log_tis.max().item()
            metrics["vespo/seq_log_tis_min"] = seq_log_tis.min().item()
            metrics["vespo/seq_tis_mean"] = seq_tis_for_metrics.mean().item()
            metrics["vespo/seq_tis_max"] = seq_tis_for_metrics.max().item()
        else:
            metrics["vespo/tis_enabled"] = 0.0
            metrics["vespo/seq_log_tis_mean"] = 0.0
            metrics["vespo/seq_log_tis_max"] = 0.0
            metrics["vespo/seq_log_tis_min"] = 0.0
            metrics["vespo/seq_tis_mean"] = 0.0
            metrics["vespo/seq_tis_max"] = 0.0

    return pg_loss, metrics


def compute_loss(config, learner_metrics, logger, device, model, batch):
    prompt_ids, prompt_mask, completion_ids, completion_mask, oldlogp, oldlogp_mask, reflogp, reflogp_mask, advantages = batch


    batch_prompt_ids = remove_pad(prompt_ids, prompt_mask)
    batch_completion_ids = remove_pad(completion_ids, completion_mask)
    batch_oldlogp = remove_pad(oldlogp, oldlogp_mask)
    batch_reflogp = remove_pad(reflogp, reflogp_mask)
    batch_logprob = compute_logprob(model, batch_prompt_ids, batch_completion_ids, device, logger)

    batch_logprob_rmp = remove_prompt(batch_logprob, batch_prompt_ids)
    batch_oldlogp_rmp = remove_prompt(batch_oldlogp, batch_prompt_ids)
    batch_reflogp_rmp = remove_prompt(batch_reflogp, batch_prompt_ids)

    logprob_tensor, logprob_mask_tensor = list2tensor(batch_logprob_rmp, pad_id=-20)
    oldlogprob_tensor, oldlogprob_mask_tensor = list2tensor(batch_oldlogp_rmp, pad_id=-20)
    reflogprob_tensor, reflogprob_mask_tensor = list2tensor(batch_reflogp_rmp, pad_id=-20)
    adv_tensor = torch.tensor([item.item() for item in advantages])
    adv_expanded = adv_tensor.view(-1, 1).expand(-1, logprob_tensor.shape[1])

    logprob_tensor = logprob_tensor.to(device)
    oldlogprob_tensor = oldlogprob_tensor.to(device)
    reflogprob_tensor = reflogprob_tensor.to(device)
    adv_expanded = adv_expanded.to(device)
    completion_mask = logprob_mask_tensor.to(device)

    per_token_logps = logprob_tensor * completion_mask
    old_per_token_logps = oldlogprob_tensor * completion_mask
    ref_per_token_logps = reflogprob_tensor * completion_mask
    advantages = adv_expanded * completion_mask
    
    with torch.no_grad():
        kl_token = ref_per_token_logps - per_token_logps
        kl_mean = (kl_token * completion_mask).sum() / (completion_mask.sum() + 1e-8)

        log_ratio_token = (per_token_logps - old_per_token_logps) * completion_mask
        seq_lengths = completion_mask.sum(dim=-1)
        seq_log_ratio = log_ratio_token.sum(dim=-1) / (seq_lengths + 1e-8)
        is_mean = seq_log_ratio.mean()
        is_var = seq_log_ratio.var()

        seq_delta_logprob = ((per_token_logps - old_per_token_logps) * completion_mask).sum(dim=-1)
        align = (adv_tensor.to(seq_delta_logprob.device) * seq_delta_logprob).mean()

        seq_logprob_current = (per_token_logps * completion_mask).sum(dim=-1).detach()
        seq_logprob_old = (old_per_token_logps * completion_mask).sum(dim=-1)
        seq_lengths = completion_mask.sum(dim=-1).clamp(min=1)
        delta_logprob_sum = seq_logprob_current - seq_logprob_old
        delta_logprob_norm = delta_logprob_sum / seq_lengths
        delta_logprob = delta_logprob_norm

        delta_abs_mean = delta_logprob.abs().mean()
        delta_mean = delta_logprob.mean()
        delta_std = delta_logprob.std(unbiased=False)
        delta_var = delta_logprob.var(unbiased=False)
        delta_abs_p95 = torch.quantile(delta_logprob.abs(), 0.95)

        from collections import defaultdict
        groups = defaultdict(list)
        for b in range(prompt_ids.size(0)):
            mask = prompt_mask[b].bool()
            valid_prompt = prompt_ids[b][mask]
            key = tuple(valid_prompt.cpu().tolist())
            groups[key].append(b)

        correct = 0
        total = 0
        tie_adv = 0
        tie_delta = 0
        delta_gap_abs_sum = 0.0
        delta_gap_count = 0
        eps = 1e-6
        for idx_list in groups.values():
            m = len(idx_list)
            if m < 2:
                continue
            for p in range(m):
                for q in range(p + 1, m):
                    i_idx, j_idx = idx_list[p], idx_list[q]
                    adv_diff = (adv_tensor[i_idx] - adv_tensor[j_idx]).item()
                    if abs(adv_diff) < eps:
                        tie_adv += 1
                        continue
                    delta_diff = (delta_logprob[i_idx] - delta_logprob[j_idx]).item()
                    
                    delta_gap_abs_sum += abs(delta_diff)
                    delta_gap_count += 1

                    if abs(delta_diff) < eps:
                        tie_delta += 1
                        continue
                    if adv_diff * delta_diff > 0:
                        correct += 1
                    total += 1

        delta_pair_gap_abs_mean = (
            delta_gap_abs_sum / delta_gap_count if delta_gap_count > 0 else float("nan")
        )

        pref_consistency = correct / total if total > 0 else float('nan')
        pref_inconsistency = (total - correct) / total if total > 0 else float('nan')
        wrong = total - correct
        logger.info(
            f"[FSDP WORKER] Pref stats - tie_adv: {tie_adv}, tie_delta: {tie_delta}, "
            f"correct: {correct}, wrong: {wrong}, total: {total}"
        )
        logger.info(
            f"[FSDP WORKER] Delta stats - "
            f"mean: {delta_mean.item():.6e}, "
            f"abs_mean: {delta_abs_mean.item():.6e}, "
            f"std: {delta_std.item():.6e}, "
            f"var: {delta_var.item():.6e}, "
            f"abs_p95: {delta_abs_p95.item():.6e}, "
            f"pair_gap_abs_mean: {delta_pair_gap_abs_mean:.6e}"
        )

        adv_tensor_dev = adv_tensor.to(delta_logprob.device)
        adv_centered = adv_tensor_dev - adv_tensor_dev.mean()
        delta_centered = delta_logprob - delta_logprob.mean()
        adv_var = adv_centered.pow(2).sum()
        delta_var = delta_centered.pow(2).sum()
        if adv_var > 1e-8 and delta_var > 1e-8 and adv_tensor.numel() >= 2:
            pearson_corr = (adv_centered * delta_centered).sum() / (adv_var.sqrt() * delta_var.sqrt())
        else:
            pearson_corr = torch.tensor(float('nan'), device=adv_tensor.device)
    logger.info(f"logprob_tensor {per_token_logps.shape}")
    logger.info(f"oldlogprob_tensor {oldlogprob_tensor.shape}")
    logger.info(f"reflogprob_tensor {reflogprob_tensor.shape}")
    logger.info(f"advantages {advantages.shape}")
    

    loss_type = config['loss_type']
    if loss_type == 'trl':
        kl_loss_coef = config['kl_loss_coef']
        cliprange = config['cliprange']
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        per_token_kl = agg_loss(per_token_kl, completion_mask)
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)

        no_tr = config.get('no_tr', False)
        if no_tr:
            per_token_loss = -coef_1 * advantages
        else:
            coef_2 = torch.clamp(coef_1, 1 - cliprange, 1 + cliprange)
            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        per_token_loss = agg_loss(per_token_loss, completion_mask)
        loss = per_token_loss + kl_loss_coef * per_token_kl
        logger.info(f"[FSDP WORKER] trl grpo/policy/kl loss {loss.item()}, {per_token_loss.item()}, {per_token_kl.item()}, no_tr={no_tr}")

    elif loss_type == 'dual':
        kl_loss_coef = config['kl_loss_coef']
        cliprange = config['cliprange']
        clip_ratio_c = config['clip_ratio_c']
        negative_approx_kl = per_token_logps - old_per_token_logps
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl)

        no_tr = config.get('no_tr', False)
        if no_tr:
            per_token_loss = -advantages * ratio
        else:
            pg_losses1 = -advantages * ratio
            pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
            clip_pg_losses1 = torch.maximum(
                pg_losses1, pg_losses2
            )
            pg_losses3 = -advantages * clip_ratio_c
            clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
            per_token_loss = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

        per_token_loss = agg_loss(per_token_loss, completion_mask)

        kl = ref_per_token_logps - per_token_logps
        kl = torch.clamp(kl, min=-20, max=20)
        kl_ratio = torch.exp(kl)
        kld = (kl_ratio - kl - 1).contiguous()
        per_token_kl = torch.clamp(kld, min=-10, max=10)
        per_token_kl = agg_loss(per_token_kl, completion_mask)

        loss = per_token_loss + per_token_kl * kl_loss_coef
        logger.info(f"[FSDP WORKER] verl dual/policy/kl loss {loss.item()}, {per_token_loss.item()}, {per_token_kl.item()}, no_tr={no_tr}")
        

        C_raw = 0.0


    elif loss_type == 'vespo':
        kl_loss_coef = config.get('kl_loss_coef', 0)

        per_token_loss, vespo_metrics = compute_vespo_loss(
            old_log_prob=old_per_token_logps,
            log_prob=per_token_logps,
            advantages=advantages,
            response_mask=completion_mask,
            rollout_is_weights=None,
        )

        if kl_loss_coef > 0:
            kl = ref_per_token_logps - per_token_logps
            kl = torch.clamp(kl, min=-20, max=20)
            kl_ratio = torch.exp(kl)
            kld = (kl_ratio - kl - 1).contiguous()
            per_token_kl = torch.clamp(kld, min=-10, max=10)
            per_token_kl = agg_loss(per_token_kl, completion_mask)
            loss = per_token_loss + per_token_kl * kl_loss_coef
        else:
            per_token_kl = torch.tensor(0.0, device=device)
            loss = per_token_loss
        
        C_raw = 0.0
        logger.info(
            f"[FSDP WORKER] vespo loss {loss.item()}, policy_loss {per_token_loss.item()}, "
            f"approx_kl {vespo_metrics['actor/approx_kl']:.4f}, "
            f"w_seq_mean {vespo_metrics['vespo/w_seq_mean']:.4f}, "
            f"phi_mean {vespo_metrics['vespo/phi_mean']:.4f}"
        )

    elif loss_type == 'gspo':
        clip_range_left = config.get('clip_range_left', 3e-4)
        clip_range_right = config.get('clip_range_right', 4e-4)

        log_ratio_sum = (per_token_logps - ref_per_token_logps).sum(dim=-1)
        response_lengths = completion_mask.sum(dim=-1)
        sequence_log_ratio = log_ratio_sum / (response_lengths + 1e-8)

        importance_ratio = torch.exp(sequence_log_ratio)

        clipped_ratios = torch.clamp(
            importance_ratio,
            min=1.0 - clip_range_left,
            max=1.0 + clip_range_right
        )

        adv_seq = adv_tensor.to(device)
        obj1 = importance_ratio * adv_seq
        obj2 = clipped_ratios * adv_seq
        per_token_loss = -torch.min(obj1, obj2)
        loss = per_token_loss.mean()
        per_token_kl = torch.tensor(0.0, device=device)

        clipped_fraction = (
            (importance_ratio < (1.0 - clip_range_left)) |
            (importance_ratio > (1.0 + clip_range_right))
        ).float().mean()

        logger.info(
            f"[FSDP WORKER] gspo loss {loss.item()}, policy_loss {per_token_loss.mean().item()}, "
            f"importance_ratio_mean {importance_ratio.mean().item():.4f}, "
            f"importance_ratio_std {importance_ratio.std().item():.4f}, "
            f"clipped_fraction {clipped_fraction.item():.4f}"
        )

    learner_metrics["loss"].append(float(loss.item()))
    learner_metrics["kl"].append(float(per_token_kl.mean()))
    learner_metrics.setdefault("is_var", []).append(float(is_var.item()))
    learner_metrics.setdefault("align", []).append(float(align.item()))
    pref_consistency_val = float(pref_consistency) if not math.isnan(pref_consistency) else 0.0
    learner_metrics.setdefault("pref_consistency", []).append(pref_consistency_val)
    pearson_corr_val = float(pearson_corr.item()) if not math.isnan(pearson_corr.item()) else 0.0
    learner_metrics.setdefault("pearson_corr", []).append(pearson_corr_val)
    learner_metrics.setdefault("mismatch_grad_contrib", []).append(float(C_raw))

    gc.collect()
    torch.cuda.empty_cache()
    return loss, {"kl_mean": kl_mean.item(), "is_var": is_var.item(), "align": align.item(), "pref_consistency": pref_consistency_val, "pearson_corr": pearson_corr_val, "mismatch_grad_contrib": float(C_raw)}

def fsdp_worker(*args, **kwargs):
    review_only_runtime_unavailable('FSDP learner worker')


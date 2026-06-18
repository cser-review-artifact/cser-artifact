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
# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, CPUOffload
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.fsdp.api import StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, lambda_auto_wrap_policy
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
    review_only_runtime_unavailable("LoRA FSDP parameter initialization")

def get_qwen2_5b_fsdp_wrap_policy():
    review_only_runtime_unavailable("LoRA FSDP wrapping policy")

def split_tensor_by_lengths(tensor, lengths):
    """Split a flat tensor into segments with the provided lengths."""
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
    """Remove padding according to the mask and return variable-length tensors."""
    result_list = []
    for i in range(tensor.size(0)):
        non_pad = tensor[i][mask[i] == 1]
        result_list.append(non_pad)
    return result_list

def remove_prompt(batch_logprob, batch_prompt_ids):
    """Remove the prompt prefix from each log-probability sequence."""
    batch_logprob_rmp = []
    for i in range(len(batch_logprob)):
        prompt_length = len(batch_prompt_ids[i])
        remaining = batch_logprob[i][prompt_length:]
        batch_logprob_rmp.append(remaining)
    return batch_logprob_rmp

def list2tensor(tensor_list, pad_id=151643):
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

def expand_advantage(batch_new, advantage):
    """Expand sequence-level advantages to token-level tensor shapes."""
    batch_advantage = []
    for i in range(len(batch_new)):
        n = batch_new[i].numel()
        expanded_tensor = torch.full((n,), advantage[i], dtype=advantage.dtype)
        batch_advantage.append(expanded_tensor)
    return batch_advantage

# Defensive note: the log-probability helpers below are conventional
# autoregressive Transformer bookkeeping. They only align next-token labels
# with logits and convert cross entropy into token log-probabilities. The
# flash-attn path is an implementation optimization, not a contribution of the
# proposed method in the paper.
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
    """Compute standard next-token log-probabilities for prompt+completion ids.

    This is routine causal-LM/Transformer computation used to provide policy
    log-probabilities to the loss; method-specific logic starts in
    ``compute_vespo_loss``.
    """
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
    position_ids_tensor = torch.tensor(position_ids, dtype=torch.int32).unsqueeze(0).to(device) # torch.Size([1, 5897])

    output = model(
        input_ids=input_ids_tensor,
        attention_mask=None,
        position_ids=position_ids_tensor,
        use_cache=False,
    )
    logits = output.logits.squeeze(0)  # torch.Size([5897, 151936])

    del output

    # logits.div_(temperature)
    # labels = position_ids_tensor.reshape(-1)    # torch.Size([5897])
    # logprob = -cross_entropy_loss(logits, labels, inplace_backward=False)[0]  # torch.Size([5897])
    input_ids_rmpad_rolled = torch.roll(input_ids_tensor, shifts=-1, dims=1)
    inplace_backward = False
    log_probs = logprobs_from_logits(
        logits=logits,
        labels=input_ids_rmpad_rolled,
        inplace_backward=inplace_backward,
    )

    # Release logits as soon as possible to reduce peak memory.
    del logits, input_ids_rmpad_rolled
    
    batch_logprob = split_tensor_by_lengths(log_probs, seq_length)
    # batch_logprob = torch.split(logprob, seq_length, dim=0)
    logger.info(f"len(logprob) {len(batch_logprob)}, logprob[0] {batch_logprob[0].shape}")
    del log_probs

    return batch_logprob


def compute_vespo_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    rollout_is_weights=None,
):
    """
    Sequence-Level Gamma-IS Policy Loss (REINFORCE-style gradient).
    Gradient Form:
        grad J = E[phi(w_seq) * A * grad log pi]

    where:
        - w_seq is the sequence-level importance ratio product.
        - phi(w) = w^k * exp(-lambda * w), normalized so phi(1)=1.
    """
    # 1. Config Extraction (hard-coded defaults, same as VESPO official recipe)
    k_pos = 2.0
    lambda_pos = 3.0
    k_neg = 3.0
    lambda_neg = 2.0

    # 2. Compute TRUE Sequence-Level IS Ratio (product, not geometric mean)
    log_ratio = log_prob - old_log_prob
    log_ratio = torch.clamp(log_ratio, -20.0, 20.0)

    seq_log_ratio = torch.sum(log_ratio * response_mask, dim=-1)  # (batch_size,)

    # 3. Apply TIS Correction in LOG space (if provided)
    seq_log_tis = None
    if rollout_is_weights is not None:
        log_tis = torch.log(rollout_is_weights.clamp(min=1e-8))
        log_tis = torch.clamp(log_tis, -20.0, 20.0)
        seq_log_tis = torch.sum(log_tis * response_mask, dim=-1)
        seq_log_ratio_combined = seq_log_ratio + seq_log_tis
    else:
        seq_log_ratio_combined = seq_log_ratio

    # 4. Compute w_seq for gamma weighting (detached, not in gradient graph)
    seq_log_ratio_clamped = torch.clamp(seq_log_ratio_combined, -20.0, 20.0)
    w_seq = torch.exp(seq_log_ratio_clamped.detach())  # (batch_size,)

    # 5. Get sequence-level advantage
    seq_adv = advantages[:, 0] if advantages.dim() > 1 else advantages  # (batch_size,)

    # 6. Select k and lambda based on advantage sign (sequence level)
    pos_mask_seq = (seq_adv >= 0).float()
    neg_mask_seq = 1.0 - pos_mask_seq

    k_seq = pos_mask_seq * k_pos + neg_mask_seq * k_neg
    lam_seq = pos_mask_seq * lambda_pos + neg_mask_seq * lambda_neg

    # 7. Compute gamma weight phi(w_seq).
    # phi(w) = Z * w^k * exp(-lambda * w), normalized so phi(1)=1.
    # At w=1: phi(1) = Z * exp(-lambda) = 1, so Z = exp(lambda).
    # In log space: log phi(w) = lambda + k * log(w) - lambda * w.
    lam_safe = torch.clamp(lam_seq, min=1e-4)
    log_w_seq = torch.log(w_seq.clamp(min=1e-8))

    log_phi = lam_safe + k_seq * log_w_seq - lam_safe * w_seq

    phi_seq = torch.exp(log_phi).detach()  # (batch_size,)
    phi_seq = torch.nan_to_num(phi_seq, nan=0.0, posinf=0.0, neginf=0.0)

    # 8. Broadcast phi to token level.
    phi_token = phi_seq.unsqueeze(-1)  # (batch_size, 1)

    # 9. Compute loss: L = -phi(w).detach() * A * log_prob.
    loss_mat = -phi_token * advantages * log_prob

    pg_loss = agg_loss(loss_mat, response_mask)

    # 10. Metrics
    with torch.no_grad():
        seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)

        # Normalized seq_log_ratio (geometric mean) for interpretable KL
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

    # torch.save(prompt_ids, "save_pt/prompt_ids.pt")
    # torch.save(prompt_mask, "save_pt/prompt_mask.pt")
    # torch.save(completion_ids, "save_pt/completion_ids.pt")
    # torch.save(completion_mask, "save_pt/completion_mask.pt")

    # Standard LM log-probability computation; unrelated to the proposed
    # method except for providing inputs to the loss below.
    batch_prompt_ids = remove_pad(prompt_ids, prompt_mask)
    batch_completion_ids = remove_pad(completion_ids, completion_mask)
    batch_oldlogp = remove_pad(oldlogp, oldlogp_mask)
    batch_reflogp = remove_pad(reflogp, reflogp_mask)
    batch_logprob = compute_logprob(model, batch_prompt_ids, batch_completion_ids, device, logger)    # torch.Size([seq_len])

    # remove prompt
    batch_logprob_rmp = remove_prompt(batch_logprob, batch_prompt_ids)
    batch_oldlogp_rmp = remove_prompt(batch_oldlogp, batch_prompt_ids)
    batch_reflogp_rmp = remove_prompt(batch_reflogp, batch_prompt_ids)

    # Restore variable-length sequences to padded 2D tensors and masks.
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

    logger.info(f"logprob_tensor {per_token_logps.shape}")
    logger.info(f"oldlogprob_tensor {oldlogprob_tensor.shape}")
    logger.info(f"reflogprob_tensor {reflogprob_tensor.shape}")
    logger.info(f"advantages {advantages.shape}")

    # Delete intermediate tensors that are no longer needed.
    del logprob_tensor, oldlogprob_tensor, reflogprob_tensor, adv_expanded
    
    # torch.save(completion_mask, "save_pt/completion_mask.pt")
    # torch.save(per_token_logps, "save_pt/per_token_logps.pt")
    # torch.save(old_per_token_logps, "save_pt/old_per_token_logps.pt")
    # torch.save(ref_per_token_logps, "save_pt/ref_per_token_logps.pt")
    # torch.save(advantages, "save_pt/advantages.pt")

    loss_type = config['loss_type']
    if loss_type == 'trl':
        # TRL GRPO
        kl_loss_coef = config['kl_loss_coef']
        cliprange = config['cliprange']
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - cliprange, 1 + cliprange)
        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        per_token_loss = per_token_loss + kl_loss_coef * per_token_kl
        loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        logger.info(f"[FSDP WORKER] trl loss {loss} \n per_token_logps {per_token_logps.mean()} \n old_per_token_logps {old_per_token_logps.mean()}")

    elif loss_type == 'dual':
        # VERL GRPO
        kl_loss_coef = config['kl_loss_coef']
        cliprange = config['cliprange']
        clip_ratio_c = config['clip_ratio_c']
        negative_approx_kl = per_token_logps - old_per_token_logps
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl)

        no_tr = config.get('no_tr', False)
        if no_tr:
            # No Trust Region: skip cliprange and clip_ratio_c constraints
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

    elif loss_type == 'gspo':
        # GSPO
        epsilon = config.get('epsilon', 0.2)
        epsilon_low = config.get('epsilon_low', epsilon)
        epsilon_high = config.get('epsilon_high', epsilon)
        kl_loss_coef = config['kl_loss_coef']

        seq_lengths = torch.sum(completion_mask, dim=-1).clamp(min=1)  # (B,)
        log_ratio = per_token_logps - old_per_token_logps  # (B, L)
        seq_log_ratio = torch.sum(log_ratio * completion_mask, dim=-1) / seq_lengths  # (B,)

        # Detach sequence-level ratio to avoid backprop through it
        log_seq_ratio_detached = seq_log_ratio.detach().unsqueeze(-1)  # (B, 1)
        log_token_prob_detached = per_token_logps.detach()  # (B, L)
        log_importance_ratio = log_seq_ratio_detached + (per_token_logps - log_token_prob_detached)
        importance_ratio = torch.exp(torch.clamp(log_importance_ratio, max=10.0))

        # Clipped PPO-style loss
        adv = advantages  # already (B, L), but we treat it as broadcasted from (B, 1)
        loss1 = -adv * importance_ratio
        clipped_ratio = torch.clamp(importance_ratio, 1 - epsilon_low, 1 + epsilon_high)
        loss2 = -adv * clipped_ratio
        per_token_loss = torch.maximum(loss1, loss2)

        # Aggregate: mean over tokens per sequence, then mean over batch
        seq_loss = torch.sum(per_token_loss * completion_mask, dim=-1) / torch.sum(completion_mask, dim=-1)
        loss = torch.mean(seq_loss)

        kl_val = ref_per_token_logps - per_token_logps
        kl_val = torch.clamp(kl_val, min=-20, max=20)
        kl_ratio = torch.exp(kl_val)
        kld = (kl_ratio - kl_val - 1).contiguous()
        kl_loss = torch.clamp(kld, min=-10, max=10)
        kl_loss = torch.sum(kl_loss * completion_mask, dim=-1) / torch.sum(completion_mask, dim=-1)
        kl_loss = torch.mean(kl_loss)

        loss = loss + kl_loss * kl_loss_coef

        logger.info(f"[FSDP WORKER] gspo loss {loss.item():.4f}, "
                    f"per_token_logps mean: {per_token_logps.mean().item():.4f}, "
                    f"old_per_token_logps mean: {old_per_token_logps.mean().item():.4f}")
        
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        # kl_mean = ((per_token_kl * completion_mask).sum(dim=1) / seq_lengths).mean()
    elif loss_type == 'vespo':
        # kl_loss_coef = config.get('kl_loss_coef', 0)
        kl_loss_coef = 0

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

        logger.info(
            f"[FSDP WORKER] vespo loss {loss.item()}, policy_loss {per_token_loss.item()}, "
            f"approx_kl {vespo_metrics['actor/approx_kl']:.4f}, "
            f"w_seq_mean {vespo_metrics['vespo/w_seq_mean']:.4f}, "
            f"phi_mean {vespo_metrics['vespo/phi_mean']:.4f}"
        )

    learner_metrics["loss"].append(float(loss.item()))
    learner_metrics["kl"].append(float(per_token_kl.mean()))

    del per_token_logps, old_per_token_logps, ref_per_token_logps, advantages

    gc.collect()
    torch.cuda.empty_cache()
    return loss
        
def get_lora_config(config):
    lora_r = config.get('lora_r', 32)
    lora_alpha = config.get('lora_alpha', 32)
    lora_dropout = config.get('lora_dropout', 0.00)
    target_modules = config.get('lora_target_modules', [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
    )

def fsdp_worker(*args, **kwargs):
    review_only_runtime_unavailable('LoRA FSDP learner worker')


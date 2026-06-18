# CSER Review Artifact

[Chinese version](README.zh-CN.md)

This repository contains the anonymous review artifact for the CSER paper and is intended only for inspection during peer review. The purpose of this repository is to help reviewers examine the main method implementation, especially the logic for staleness-aware replay, sequence-level scoring, temporal credit calibration, sampling, and loss-function updates.

This artifact is intentionally not provided as a complete runnable training system. Concrete implementation details for distributed runtime bootstrapping of FSDP/vLLM workers, GPU placement, NCCL configuration, ports, and private cluster paths have been omitted or replaced with review-only placeholders.

## Supplementary Materials

The following sections are intended as reviewer-facing supplementary materials.

These supplementary materials are organized around verifiability and additional evidence. They first document the software and hardware environment and key hyperparameters used by the distributed RLVR experiments, so that reviewers can interpret the experimental configuration. They then provide supplementary results beyond the main paper, covering an additional GRPO-style base objective, a verifiable task, and efficiency comparisons. Finally, they expand the derivation of the frozen-credit residual and provide a suggested reading order to help reviewers quickly locate the CSER implementation details related to experience replay, temporal credit calibration, and loss-function updates.

### Experimental Setup and Environment

The experiments were conducted in an OpenMPI-based distributed RLVR training framework. Four actor processes perform rollouts concurrently and use vLLM as the inference acceleration library. Four learner processes perform policy updates in parallel and use Fully Sharded Data Parallel (FSDP) as the training acceleration library. This review artifact preserves the method-related orchestration and loss logic, while cluster-specific launch details have been omitted as described above.

The software and hardware environment is configured as follows:

| Category | Configuration |
| --- | --- |
| Operating system | Ubuntu 18.04.4 LTS, Linux kernel 4.15.0-76-generic |
| CPU | 2 x Intel(R) Xeon(R) Gold 6242R CPU @ 3.10GHz |
| Memory | 376 GB RAM, 975 MB swap |
| GPU | 4 x NVIDIA A800 80GB; 8 x NVIDIA A40 40GB |
| GPU interconnect | PCIe |
| GPU driver | NVIDIA driver 545.29.06 |
| MPI | Open MPI 5.0.7, MPI API 3.1.0 |
| Random seed | 42 |

Hyperparameter settings vary across algorithms and applications; see `configs/` for details. Representative hyperparameter values are listed below:

| Parameter | Description | Value |
| --- | --- | ---: |
| `max prompt length` | Maximum query token length. | 1000 |
| `max response length` | Maximum response token length. | 2500 |
| `query_batchsize` | Number of queries sampled per batch. | 24 |
| `group` | Number of rollouts per prompt in GRPO. | 8 |
| `rollout_mini_batchsize` | Number of samples processed in each rollout mini-batch. | 48 |
| `temperature` | Rollout sampling temperature. | 1.0 |
| `filter` | Whether to discard samples whose responses are all correct or all incorrect. | True |
| `er_lambda` | Ratio of replayed experiences to `train_batchsize`. | 0.25 |
| `train_batchsize` | Global training batch size for each optimization step. | 192 |
| `train_mini_batchsize` | Number of samples processed per backward pass. | 16 |
| `accumulation_steps` | Number of gradient accumulation steps. | 3 |
| `lr` | Learning rate. | 1e-6 |
| `adv_gamma` | Decay coefficient for advantage correction. | 0.95 |
| `clip_range` | Clipping range for the importance ratio. | 0.2 |

### Experimental Results

This section reports supplementary results beyond the main experiments in the paper, with the goal of further examining the applicability and practical training benefits of CSER. While the main paper focuses on GRPO/VESPO, GSM8K/MATH12K, and the core stability analysis, this section evaluates CSER from three additional perspectives. First, we integrate CSER into GSPO to test its plug-in adaptability to a different GRPO-style base objective. Second, we evaluate training dynamics on the CountDown combinatorial arithmetic task with Qwen2.5-1.5B-Instruct, testing stability on a different verifiable task and model scale. Third, we compare sample consumption and per-step training time required to reach the same target performance, highlighting the efficiency gains enabled by experience replay.

#### GSPO Algorithm

![Performance comparison between GSPO-ST128 (with CSER) and on-policy GSPO.](./figs/gspo.png)

To examine whether CSER remains effective for GRPO-style base objectives beyond the main experiments in the paper, we integrate it into GSPO and conduct a supplementary evaluation. In the main paper, GSPO is categorized as a sequence-level GRPO method: it moves the importance ratio, clipping, and optimization process from the token level to the sequence level, and therefore serves as a representative setting for testing the plug-in adaptability of CSER.

As shown in Fig. (a), under the high-staleness ST128 replay setting, GSPO-ST128 (w CSER) achieves a peak validation accuracy of 51.33%, improving over on-policy GSPO-ST1 by 7.97 percentage points and maintaining higher validation accuracy in the second half of training. As shown in Fig. (b), CSER also reduces per-step training time: the average step time of GSPO-ST128 (w CSER) is 171 seconds, compared with 193 seconds for GSPO-ST1, corresponding to an 11.4% speedup. These results show that CSER's temporal credit calibration and sequence-level experience replay are not limited to the GRPO/VESPO settings in the main paper. They can also be combined with sequence-level objectives such as GSPO to improve both validation performance and training efficiency under high-staleness conditions.

#### CountDown Task

![Training dynamics of Qwen2.5-1.5B on the CountDown task using GRPO-ST128 with CSER.](./figs/G-CountDown.png)

CountDown is an automatically verifiable combinatorial arithmetic task. Given a target number and a set of available numbers, the model must construct an expression using basic operations such as addition, subtraction, multiplication, and division so that the expression evaluates to the target, typically with each given number used at most once. Because the final answer can be checked programmatically, CountDown is suitable for RLVR training and tests a model's ability to perform search, planning, and exact arithmetic composition.

We use CountDown as a supplementary application beyond the mathematical reasoning datasets studied in the main paper, in order to evaluate CSER on a different verifiable task and other model scale. This experiment trains Qwen2.5-1.5B-Instruct under the high-staleness GRPO-ST128 (w CSER) setting. The figure reports validation performance, generated length, and optimization dynamics during training. As shown in Fig. (a), validation accuracy increases with training and gradually stabilizes, improving from 4.79% to 53.22%. This indicates that the model continues to receive effective learning signals on the combinatorial arithmetic search task. As shown in Fig. (b), the generated experience length grows from less than 100 tokens in the early stage to longer responses later in training, and the length variance is relatively large on CountDown. This suggests that the model explores longer and more diverse reasoning paths as training progresses. Fig. (c) and Fig. (d) show that $\ln(g+1)$ and loss both enter a more stable fluctuation range over training steps, indicating that high-staleness replay does not cause obvious gradient explosion or optimization collapse. Overall, these results show that CSER can also support stable replay-based training on combinatorial verifiable tasks such as CountDown.

#### Efficiency Comparison

<img src="./figs/grpo-cost.png" alt="Performance comparison between GRPO-ST128 (with CSER) and on-policy GRPO." width="550">

This section further compares the sample consumption and per-step training time required by CSER to reach the same target performance. The table reports the training cost of on-policy GRPO-ST1 and GRPO-ST128 (w CSER) when converging to the target validation accuracy. The number of rollout samples denotes the number of new samples that must be generated online, while the number of training samples denotes the total number of samples used by learner updates, including both newly generated and replayed samples.

| Metric | GRPO-ST1 | GRPO-ST128 (w CSER) | Change relative to GRPO-ST1 |
| --- | ---: | ---: | ---: |
| Training steps | 511 | 331 | -35.24% |
| Total rollout samples | 98112 | 42432 | -56.75% |
| Total training samples | 98112 | 63552 | -35.22% |
| Replay ratio | 0 | 33.23% | - |

The results show that GRPO-ST128 (w CSER) reduces the number of training steps required to reach the target performance from 511 to 331, and reduces the number of online-generated rollout samples from 98112 to 42432, corresponding to a 56.75% reduction in new-sample generation cost. Although CSER additionally introduces 33.23% replayed samples during training, the total number of training samples still decreases from 98112 to 63552. This indicates that the replay mechanism improves the utilization of existing samples while reducing the demand for fresh rollouts.

The figure further reports the per-step training time under the two settings. Compared with GRPO-ST1, the minimum, maximum, and average step times of GRPO-ST128 (w CSER) decrease from 72.8, 98.6, and 85.5 seconds to 67.8, 84.6, and 76.4 seconds, respectively; the average step time is reduced by 10.64%. This result indicates that, in the current implementation, the overhead introduced by replay retrieval and cache management in CSER does not offset the benefit of reducing online rollouts. Because the current implementation still contains noticeable communication and scheduling overhead, further system optimization is expected to leave room for additional end-to-end efficiency gains.

### Theoretical Analysis

#### Derivation of the frozen-credit residual

This section expands the derivation of the frozen-credit residual used in the main paper. The purpose is to separate two quantities that are coupled in an on-policy GRPO update: the response distribution and the group-relative credit assigned to a response.

For a fixed query $q$ and a fixed response $o$, let $O_{-o}$ denote the $G-1$ peer responses used together with $o$ to form a GRPO response group. When the peer responses are sampled from the historical policy $\pi_{\theta_{t-k}}$, define the expected group-relative credit of $o$ as

$$
\begin{aligned}
\bar A_{t-k}(o;q)
&= \mathbb{E}_{O_{-o}\sim\pi_{\theta_{t-k}}(\cdot\mid q)}
\left[
\frac{\rho(q,o)-\mu(q;o,O_{-o})}
{\sigma(q;o,O_{-o})+\epsilon_A}
\right],
\end{aligned}
$$

where $\mu(q;o,O_{-o})$ and $\sigma(q;o,O_{-o})$ are the reward mean and standard deviation of the group formed by $o$ and $O_{-o}$. Similarly, when the peer responses are sampled from the current policy $\pi_{\theta_t}$, define

$$
\begin{aligned}
\bar A_t(o;q)
&= \mathbb{E}_{O_{-o}\sim\pi_{\theta_t}(\cdot\mid q)}
\left[
\frac{\rho(q,o)-\mu(q;o,O_{-o})}
{\sigma(q;o,O_{-o})+\epsilon_A}
\right].
\end{aligned}
$$

Thus, $\bar A_{t-k}(o;q)$ and $\bar A_t(o;q)$ differ only in the policy that generates the peer responses used for group-relative normalization.

Now consider replaying a response sampled from the historical policy. Assume that the current response distribution is absolutely continuous with respect to the historical response distribution on the replay support. With exact sequence-level importance weighting, the replay gradient using historical group-relative credit is

$$
\begin{aligned}
g_{t-k}(\theta_t)
&= \mathbb{E}_{q\sim\mathcal D}
\mathbb{E}_{o\sim\pi_{\theta_{t-k}}(\cdot\mid q)}
\left[
\frac{\pi_{\theta_t}(o\mid q)}
{\pi_{\theta_{t-k}}(o\mid q)}
\bar A_{t-k}(o;q)
\nabla_{\theta_t}\log\pi_{\theta_t}(o\mid q)
\right] \\
&= \mathbb{E}_{q\sim\mathcal D}
\mathbb{E}_{o\sim\pi_{\theta_t}(\cdot\mid q)}
\left[
\bar A_{t-k}(o;q)
\nabla_{\theta_t}\log\pi_{\theta_t}(o\mid q)
\right].
\end{aligned}
$$

The importance ratio changes the sampling measure from $\pi_{\theta_{t-k}}$ to $\pi_{\theta_t}$, but the credit term remains $\bar A_{t-k}(o;q)$ because it is determined by historical peer responses. The corresponding fresh current-policy group reference is

$$
\begin{aligned}
g_t(\theta_t)
&= \mathbb{E}_{q\sim\mathcal D}
\mathbb{E}_{o\sim\pi_{\theta_t}(\cdot\mid q)}
\left[
\bar A_t(o;q)
\nabla_{\theta_t}\log\pi_{\theta_t}(o\mid q)
\right].
\end{aligned}
$$

Using the compact notation $\mathbb{E}_{q,o}$ for $q\sim\mathcal D$ and $o\sim\pi_{\theta_t}(\cdot\mid q)$, the residual between exact-ratio replay with frozen credit and the fresh current-policy group reference is

$$
\begin{aligned}
B_A
&= g_{t-k}(\theta_t)-g_t(\theta_t) \\
&= \mathbb{E}_{q,o}
\left[
\bigl(\bar A_{t-k}(o;q)-\bar A_t(o;q)\bigr)
\nabla_{\theta_t}\log\pi_{\theta_t}(o\mid q)
\right].
\end{aligned}
$$

This is the frozen-credit residual used in the main text.

#### Bound on the frozen-credit residual

Starting from the frozen-credit residual derived above,

$$
\begin{aligned}
B_A
&= \mathbb{E}_{q,o}
\left[
\bigl(\bar A_{t-k}(o;q)-\bar A_t(o;q)\bigr)
\nabla_{\theta_t}\log\pi_{\theta_t}(o\mid q)
\right],
\end{aligned}
$$

where $\mathbb{E}_{q,o}$ denotes $q\sim\mathcal D$ and $o\sim\pi_{\theta_t}(\cdot\mid q)$, we derive the illustrative bound used in the main paper. Suppose the conditional credit mismatch changes smoothly with the policy distance,

$$
\begin{aligned}
\left|
\bar A_{t-k}(o;q)-\bar A_t(o;q)
\right|
&\le L_A d(\pi_{\theta_{t-k}},\pi_{\theta_t}),
\end{aligned}
$$

and suppose the score norm is bounded by

$$
\begin{aligned}
\left\|
\nabla_{\theta_t}\log\pi_{\theta_t}(o\mid q)
\right\|_2
&\le G_s.
\end{aligned}
$$

Then

$$
\begin{aligned}
\|B_A\|_2
&= \left\|
\mathbb{E}_{q,o}
\left[
\bigl(\bar A_{t-k}(o;q)-\bar A_t(o;q)\bigr)
\nabla_{\theta_t}\log\pi_{\theta_t}(o\mid q)
\right]
\right\|_2 \\
&\le \mathbb{E}_{q,o}
\left[
\left|
\bar A_{t-k}(o;q)-\bar A_t(o;q)
\right|
\left\|
\nabla_{\theta_t}\log\pi_{\theta_t}(o\mid q)
\right\|_2
\right] \\
&\le L_A G_s d(\pi_{\theta_{t-k}},\pi_{\theta_t}).
\end{aligned}
$$

This is the frozen-credit residual bound stated in the main text.

### Reviewer Reading Guide

We suggest that reviewers read the artifact in the following order:

1. Read `run.py` and `run_lora.py` to understand Actor/Buffer/Learner orchestration.
2. Read `utils/experience.py` to understand replay, temporal decay, and sampling.
3. Read `utils/fsdp_worker.py` and `utils/fsdp_worker_lora.py` to understand the objective function and update logic.
4. Read `configs/` to inspect the example configurations.

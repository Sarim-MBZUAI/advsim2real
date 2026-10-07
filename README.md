# AdvSim2Real: Training Web Agents Against Adaptive Prompt Injection in a Web World Model

[Sarim Hashmi](https://sarim-mbzuai.github.io/)<sup>1</sup>,
[Mukul Ranjan](https://mukul54.github.io/)<sup>1</sup>,
[Kshitij Mishra](https://mishrakshitij.github.io/)<sup>1</sup>,
[Mikhail Kuznetsov](https://mikkuzne.github.io/)<sup>2</sup>,
[Praneeth Vepakomma](https://praneeth.mit.edu/)<sup>1,3</sup>,
[Nils Lukas](https://nilslukas.github.io/)<sup>1</sup>

<sup>1</sup>Mohamed bin Zayed University of Artificial Intelligence &nbsp;
<sup>2</sup>Amazon &nbsp;
<sup>3</sup>Massachusetts Institute of Technology

[Project page](https://sarim-mbzuai.github.io/advsim2real/) ·
[Paper (arXiv)](https://arxiv.org/abs/2610.08773) ·
[Checkpoints](https://huggingface.co/Sarim-Hash/advsim2real-stage1-curr1epoch-exec2epochs-iter3-nprop150)

## Abstract

Web agents complete user requests by reading and acting on pages that third parties write, so an
instruction planted on a page can redirect the agent away from the user's goal. The agent cannot
simply ignore the page, because the page also holds the values and controls the task requires.
Current defenses fine-tune the agent on injections fixed before training, and attackers that adapt to
the trained model bypass them. Adversarial training lets the attacker adapt but keeps the tasks fixed,
so a task stops teaching once the agent solves it. We introduce AdvSim2Real, which co-evolves a task
curriculum, an injection adversary, and the agent inside a frozen web world model. The curriculum is
rewarded for tasks the agent solves about half of the time, and the adversary only for a success flip,
an injection that turns a judged success into a failure. Training in the simulator makes a 4B agent
both more capable and more robust: its completion rises with and without attacks, holds against a
frontier-model adversary it never trained against, and its capability gain carries over to a real
browser. On 150 web tasks, AdvSim2Real raises completion under this unseen adversary by 33.6%
relative to the base agent. We release our code, the benchmark, and all checkpoint results.

## Overview

Web agents read and act on pages that third parties write, so an instruction planted on a page can
redirect the agent away from the user's goal. AdvSim2Real co-evolves a **task curriculum**, an
**injection adversary** and the **web agent** (the executor) inside a frozen web world model
(WebWorld-14B). The curriculum is rewarded for tasks the agent solves about half of the time, and the
adversary only for a **success flip**: an injection that turns a judged success into a failure. Every
trajectory is graded by a frozen LLM judge (Qwen3.8-27B).

![AdvSim2Real overview](assets/fig_overview.png)

## Highlights

On the 150-task benchmark, the trained Qwen3.5-4B executor becomes both more capable and more robust:

- **Clean completion** rises from 74.89% to 81.33%.
- **Completion under the three learned adversaries** rises from 48.07% to 57.48%, matching a hosted
  Qwen3.5-9B under attack (56.96%) and beating it clean (78.22%).
- **Against Kimi-K3**, a frontier-model adversary that took no part in training, completion rises
  from 23.00% to 30.72%, a 33.6% relative gain.
- **In a real Chromium browser**, without any world-model call, strict success on the submitted form
  rises from 25.56% to 44.44% after Stage 1.

## Method

Training runs in two stages, each for three rounds. In **Stage 1** the curriculum proposes tasks, the
executor runs each several times in the world model, the judge grades every run, and the curriculum
is rewarded for tasks solved about half the time; fresh tasks from the updated curriculum then train
the executor. In **Stage 2** the curriculum is frozen, the adversary proposes one injection and the
step to insert it, each saved clean run is replayed to that step with the injection, and the adversary
is rewarded only for success flips; the executor then trains on new attacks, earlier attacks and clean
tasks.

![Two-stage training](assets/fig_method.png)

The same injected notice diverts the base agent but not the trained one:

![Injected notice example](assets/fig_example.png)

## Results

**Judged task completion in WebWorld-14B (%)**, mean and sample standard deviation over three rollout
seeds on the 150 tasks; Mean weights Adv v1-v3 equally.

| Training | Executor | Clean | Adv v1 | Adv v2 | Adv v3 | Mean |
|---|---|---|---|---|---|---|
| Initial | Base | 74.89 ± 1.39 | 51.11 ± 2.34 | 48.00 ± 4.16 | 45.11 ± 1.54 | 48.07 ± 0.56 |
| Stage 1 | Capability iter 1 | 78.00 ± 1.76 | 57.11 ± 2.69 | 54.00 ± 2.91 | 53.11 ± 6.41 | 54.74 ± 3.11 |
| | Capability iter 2 | 77.11 ± 0.38 | 58.89 ± 1.39 | 53.11 ± 4.54 | 54.22 ± 1.92 | 55.41 ± 1.68 |
| | Capability iter 3 | 79.33 ± 1.76 | 57.11 ± 1.39 | 52.35 ± 4.12 | 53.56 ± 3.67 | 54.34 ± 2.93 |
| Stage 1 + 2 | Robust iter 1 | 77.33 ± 0.67 | 55.11 ± 2.14 | 52.67 ± 4.67 | 53.33 ± 2.40 | 53.70 ± 1.89 |
| | Robust iter 2 | 78.00 ± 1.15 | 62.00 ± 2.00 | 54.44 ± 2.69 | 52.89 ± 1.39 | 56.44 ± 1.15 |
| | **Robust iter 3** | **81.33 ± 2.31** | **62.89 ± 4.73** | **54.89 ± 3.91** | **54.67 ± 3.53** | **57.48 ± 0.56** |
| Hosted 9B | Qwen3.5-9B | 78.22 ± 2.14 | 58.89 ± 3.36 | 57.10 ± 2.68 | 54.89 ± 1.68 | 56.96 ± 0.13 |

**Completion under the unseen Kimi-K3 adversary (%)**, mean and sample standard deviation over two
rollout seeds; Clean repeats the no-adversary column above.

| Executor | Clean | Kimi-K3 |
|---|---|---|
| Base | 74.89 ± 1.39 | 23.00 ± 3.30 |
| Robust iter 1 | 77.33 ± 0.67 | 29.67 ± 4.24 |
| Robust iter 2 | 78.00 ± 1.15 | 30.33 ± 0.47 |
| **Robust iter 3** | **81.33 ± 2.31** | **30.72 ± 3.04** |

**Sim-to-real transfer (%)**: clean runs of the 150 tasks in a real Chromium browser. Strict success is
a deterministic check of the submitted form; Correct fields counts the 746 target values per seed.

| Executor | Strict success | Correct fields |
|---|---|---|
| Base | 25.56 ± 1.02 | 52.50 ± 3.14 |
| Capability iter 1 | 31.78 ± 8.34 | 61.89 ± 7.45 |
| Capability iter 2 | 43.56 ± 5.18 | 71.45 ± 4.97 |
| **Capability iter 3** | **44.44 ± 5.00** | **73.24 ± 2.24** |

Where the gains come from: (a) removing Stage 1, (b) Kimi-K3 against the learned adversaries,
(c) world-model verdicts against browser outcomes, (d) attacked completion by skill stratum.

![Analysis](assets/fig_analysis.png)

## Layout

```text
run.sh                          launcher: stage1 (Algorithm 1), stage2 (Algorithm 2), eval, all
data/tasks.json                 the 150 benchmark tasks (goal, page)
assets/                         figures from the paper
advsim2real/
  world/                        WebWorld world-model client and rollouts
    rollouts.py                 executor episodes in the world model, clean forks
    consistency.py              trajectory signatures and terminal outcome keys
    render.py                   render gate for injected pages
  judge/
    llm_judge.py                frozen LLM judge (DeepInfra), verdict cache
    judge_rollouts.py           K executor rollouts per task, judge verdicts, p_hat (Eq. 2)
  curriculum/
    prompts.py                  curriculum prompt and <PAGE>/<GOAL> parsing
    propose.py                  sample task proposals from the served curriculum
    train.py                    curriculum update (Eq. 3)
  executor/
    train.py                    executor update (Eq. 4)
  adversary/
    prompts.py                  adversary prompts and attack parsing
    rollouts.py                 clean controls, forked attacks, success-flip reward (Eq. 5)
    build_tasks.py              one attack per base task from the served adversary
    train.py                    adversary update
    stage2_data.py              Stage-2 attackable tasks and executor mixture
  evaluation/
    robustness.py               clean and reactive-attack evaluation (Tables 1, 3)
  training/
    grpo.py                     in-process GRPO + LoRA trainer (Eq. 6)
    rewards.py                  reward shaping (Eqs. 3-5)
    schedules.py                epoch and replay schedules
```

Every step is a module run from the repository root, e.g.
`python -m advsim2real.judge.judge_rollouts --help`.

## Requirements

Python 3.11, `torch`, `transformers>=5`, `peft`, `vllm`, `requests`:

```bash
pip install -e .
```

GPUs for WebWorld-14B, two 4B vLLM servers and the trainer (`EXEC_SERVE_GPU`, `SECOND_SERVE_GPU`, `TGPU`).
The judge is called through DeepInfra: set `DEEPINFRA_TOKEN` (environment or `.env`).

## Run

```bash
vllm serve Qwen/WebWorld-14B --served-model-name WebWorld-14B --port 8004 --max-model-len 16384
bash run.sh stage1     # runs_stage1/{curr_v1..3, exec_v1..3}   Capability iter 1-3
bash run.sh stage2     # runs_stage2/{adv_v1..3,  exec_v1..3}   Adv v1-3, Robust iter 1-3
EVAL_ADV=runs_stage2/adv_v1 bash run.sh eval      # base, exec_v3 of both stages; clean + attacked
```

Stage-1 defaults follow App. C.6 (150 proposal positions, K=6, one curriculum epoch, two
executor epochs, G=6, LoRA rank 16, lr 1e-5, KL 0.01, 4,096-token cap). All settings are
environment variables at the top of `run.sh`.

## Citation

```bibtex
@article{hashmi2026advsim2real,
  title   = {AdvSim2Real: Training Web Agents Against Adaptive Prompt Injection in a Web World Model},
  author  = {Hashmi, Sarim and Ranjan, Mukul and Mishra, Kshitij and Kuznetsov, Mikhail and
             Vepakomma, Praneeth and Lukas, Nils},
  journal = {arXiv preprint arXiv:2610.08773},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.08773}
}
```

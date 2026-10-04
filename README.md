# AdvSim2Real

Code for *AdvSim2Real: Training Web Agents Against Adaptive Prompt Injection in a Web World Model*.

A curriculum, an injection adversary and a Qwen3.5-4B web agent (the executor) are trained with
GRPO + LoRA inside the frozen WebWorld-14B world model; every trajectory is graded by a frozen
LLM judge (Qwen3.8-27B).

## Layout

```text
run.sh                          launcher: stage1 (Algorithm 1), stage2 (Algorithm 2), eval, all
data/tasks.json                 the 150 benchmark tasks (source_id, goal, page)
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

# AdvSim2Real

Code for *AdvSim2Real: Training Web Agents Against Adaptive Prompt Injection in a Web World Model*.

A curriculum, an injection adversary and a Qwen3.5-4B web agent (the executor) are trained with
GRPO + LoRA inside the frozen WebWorld-14B world model; every trajectory is graded by a frozen
LLM judge (Qwen3.8-27B).

| file | role |
| --- | --- |
| `run.sh` | launcher: `stage1` (Algorithm 1), `stage2` (Algorithm 2), `eval`, `all` |
| `build_tasks.py`, `web_curriculum.py` | curriculum proposals |
| `judge_rollouts.py`, `kimi_judge.py` | executor rollouts in the world model, judge verdicts, p_hat (Eq. 2) |
| `train_curriculum.py` | curriculum update (Eq. 3) |
| `train_executor.py` | executor update (Eq. 4) |
| `train_adversary.py`, `adv_rollouts.py`, `adv_prompts.py`, `adv_world.py`, `build_adv_tasks.py` | adversary, success-flip reward (Eq. 5) |
| `stage2_data.py` | Stage-2 attackable tasks and executor mixture |
| `eval_robustness.py` | clean and reactive-attack evaluation (Tables 1, 3) |
| `web_rollouts.py`, `web_consistency.py`, `_lib/` | world-model client, trajectory signatures, GRPO trainer (Eq. 6) |
| `data/tasks.json` | the 150 benchmark tasks (`source_id`, `goal`, `page`) |

## Requirements

Python 3.11, `torch`, `transformers>=5`, `peft`, `vllm`, `requests`. GPUs for WebWorld-14B,
two 4B vLLM servers and the trainer (`EXEC_SERVE_GPU`, `SECOND_SERVE_GPU`, `TGPU`).
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

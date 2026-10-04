"""Train the adversary (GRPO + LoRA) against the frozen served executor with the success-flip reward.

Each eligible base task (clean_p_hat >= --min_clean_success) gets K clean controls once.
Every generated attack is forked from those controls at its chosen turn and judged; the
reward (Eq. 5) counts rendered clean-success -> attacked-failure flips. Tasks whose
controls all fail are excluded. The diversity penalty is computed per task group.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

from advsim2real.adversary.prompts import ADVERSARY_SYSTEM_IMPERATIVE, parse_adv_completion
from advsim2real.adversary.rollouts import adversary_reward, k_rollouts_adv, prepare_clean_baselines
from advsim2real.judge.llm_judge import DEFAULT_MODEL as JUDGE_MODEL, DEFAULT_CACHE_PATH as JUDGE_CACHE, load_api_key, reference_for_task
from advsim2real.training.grpo import GRPOConfig, GRPOTrainer
from advsim2real.training.rewards import repetition_penalty

INVALID_REWARD = -1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--init_adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base_tasks", required=True, help="clean-judged tasks with clean_p_hat (advsim2real.adversary.stage2_data prepare)")
    ap.add_argument("--min_clean_success", type=float, default=0.5)
    ap.add_argument("--K", type=int, default=4, help="clean controls per task")
    ap.add_argument("--max_steps", type=int, default=8)
    ap.add_argument("--outcome_mode", default="final_state")
    ap.add_argument("--executor_temperature", type=float, default=0.7)
    ap.add_argument("--exec_api", default="http://127.0.0.1:8005/v1")
    ap.add_argument("--exec_model", default="exec")
    ap.add_argument("--world_api", default="http://127.0.0.1:8004/v1")
    ap.add_argument("--world_model", default="WebWorld-14B")
    ap.add_argument("--judge_model", default=JUDGE_MODEL)
    ap.add_argument("--judge_workers", type=int, default=16)
    ap.add_argument("--judge_retries", type=int, default=6)
    ap.add_argument("--judge_timeout", type=int, default=120)
    ap.add_argument("--judge_cache", default=JUDGE_CACHE)
    ap.add_argument("--rep_lambda", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=0, help=">0: complete passes over tasks (overrides --steps)")
    ap.add_argument("--group_size", type=int, default=6, help="attack proposals per task")
    ap.add_argument("--prompts_per_step", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--kl_beta", type=float, default=1e-2)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--micro_bs", type=int, default=2)
    ap.add_argument("--attn_impl", default=None)
    ap.add_argument("--grad_checkpoint", type=int, default=0)
    args = ap.parse_args()

    # one base per source task, screened on its measured clean success
    bases = {}
    for t in json.loads(Path(args.base_tasks).read_text()):
        key = t.get("source_id") or " ".join(t["goal"].lower().split())
        if key not in bases and t.get("clean_p_hat") is not None and float(t["clean_p_hat"]) >= args.min_clean_success:
            bases[key] = {"goal": t["goal"], "page": t["page"], "reference": reference_for_task(t)}
    api_key = load_api_key()
    ev = dict(K=args.K, max_steps=args.max_steps, executor_temperature=args.executor_temperature,
              exec_api=args.exec_api, world_api=args.world_api, exec_model=args.exec_model,
              world_model=args.world_model, outcome_mode=args.outcome_mode, judge_model=args.judge_model,
              judge_workers=args.judge_workers, judge_retries=args.judge_retries,
              judge_timeout=args.judge_timeout, judge_cache=args.judge_cache, judge_api_key=api_key)

    # K clean controls per base, captured once against the frozen executor
    controls = {}
    for key, b in bases.items():
        paired = prepare_clean_baselines(b["page"], b["goal"], reference=b["reference"], **ev)
        print(f"[adv] clean controls {key}: {paired.n_clean_success}/{args.K} good", flush=True)
        if paired.n_clean_success:
            controls[key] = paired
    if not controls:
        sys.exit("ERROR: no task has a successful clean control; adversary training not started")

    tr = GRPOTrainer(GRPOConfig(
        model_path=args.model, out_dir=args.out, lora_rank=args.lora_rank, lr=args.lr, kl_beta=args.kl_beta,
        steps=args.steps, epochs=args.epochs, group_size=args.group_size, prompts_per_step=args.prompts_per_step,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature, micro_bs=args.micro_bs,
        init_adapter=args.init_adapter, attn_impl=args.attn_impl, grad_checkpoint=bool(args.grad_checkpoint)))
    prompts, prompt_base = [], {}
    for key in controls:
        ptext = tr.tok.apply_chat_template(
            [{"role": "system", "content": "/no_think\n" + ADVERSARY_SYSTEM_IMPERATIVE},
             {"role": "user", "content": f"Goal: {bases[key]['goal']}\n\nInitial Page:\n{bases[key]['page']}\n\nPropose one attack."}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompts.append(ptext)
        prompt_base[ptext] = key

    Path(args.out).mkdir(parents=True, exist_ok=True)
    log_f, log_lock = open(Path(args.out) / "adversary_log.jsonl", "a"), threading.Lock()

    def reward_fn(prompts_b, completions):
        parsed = [parse_adv_completion(c) for c in completions]
        keys = [prompt_base[p] for p in prompts_b]
        # diversity penalty per task group, minus the singleton floor (a unique attack pays 0)
        rep = [0.0] * len(completions)
        groups = {}
        for i, p in enumerate(parsed):
            if p.is_valid:
                groups.setdefault(keys[i], []).append(i)
        for idxs in groups.values():
            pen = repetition_penalty([parsed[i].instruction for i in idxs], lam=args.rep_lambda)
            for j, i in enumerate(idxs):
                rep[i] = max(0.0, pen[j] - args.rep_lambda / len(idxs))
        rewards = []
        for i, p in enumerate(parsed):
            if not p.is_valid:
                rewards.append(INVALID_REWARD)
                continue
            b = bases[keys[i]]
            stats = k_rollouts_adv(b["page"], b["goal"], p.instruction, clean_baselines=controls[keys[i]],
                                   adv_turn=p.turn, reference=b["reference"], **ev)
            r = adversary_reward(stats, p.marker, rep_penalty=rep[i], invalid_reward=INVALID_REWARD)
            rewards.append(r)
            with log_lock:
                log_f.write(json.dumps({"ts": time.time(), "source_id": keys[i], "instruction": p.instruction,
                                        "marker": p.marker, "turn": p.turn, "target": p.target,
                                        "clean_verdicts": stats.clean_verdicts, "verdicts": stats.verdicts,
                                        "rep_penalty": rep[i], "R_A": r}, ensure_ascii=False) + "\n")
                log_f.flush()
        return rewards

    tr.train(prompts, reward_fn)
    log_f.close()


if __name__ == "__main__":
    main()

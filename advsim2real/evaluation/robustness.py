"""Judged completion of one served executor, clean or under a reactive attacker (Eq. 1; Tables 1, 3).

attacked: before every world transition the attacker sees the goal, the current page and the
pending action and either waits or injects (at most once). Episodes whose attacker or rollout
failed are excluded from the denominator, never counted as clean successes.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from advsim2real.adversary.rollouts import sample_reactive_decision
from advsim2real.judge.llm_judge import DEFAULT_CACHE_PATH, DEFAULT_MODEL, judge_trajectory, load_api_key, reference_for_task
from advsim2real.world.rollouts import rollout_episode


def rollout(task, idx, seed, args, inject):
    """Turns of one episode, or None when the rollout/attacker hit an infrastructure failure."""
    adv_decide, failed = None, [False]
    if inject:
        def adv_decide(page, action, tstep):
            for _ in range(max(1, args.adv_retries)):
                atk = sample_reactive_decision(args.adversary_api, args.adversary_model, task["goal"], page,
                                               action, seed=seed * 100003 + idx * 131 + tstep)
                if atk is not None:
                    return atk.instruction if atk.is_valid else None     # invalid / <WAIT/> = wait
            failed[0] = True
            return None
    r = rollout_episode(task["page"], task["goal"], max_steps=args.max_steps,
                        executor_temperature=args.temperature, exec_api=args.executor_api,
                        world_api=args.world_api, exec_model=args.executor_model,
                        world_model=args.world_model, outcome_mode=args.outcome_mode, seed=seed,
                        adv_decide=adv_decide)
    return None if (r.error or failed[0]) else r.turns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--mode", choices=["clean", "attacked"], default="clean")
    ap.add_argument("--adversary_api", default="")
    ap.add_argument("--adversary_model", default="adv")
    ap.add_argument("--adv_retries", type=int, default=3)
    ap.add_argument("--executor_api", default="http://127.0.0.1:8005/v1")
    ap.add_argument("--executor_model", default="exec")
    ap.add_argument("--world_api", default="http://127.0.0.1:8004/v1")
    ap.add_argument("--world_model", default="WebWorld-14B")
    ap.add_argument("--seeds", default="0 1 2")
    ap.add_argument("--max_steps", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--outcome_mode", default="final_state")
    ap.add_argument("--judge_model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max_retries", type=int, default=10)
    ap.add_argument("--judge_timeout", type=int, default=120)
    ap.add_argument("--cache_path", default=DEFAULT_CACHE_PATH)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.mode == "attacked" and not args.adversary_api:
        sys.exit("ERROR: --mode attacked needs --adversary_api")

    tasks = json.loads(Path(args.bench).read_text())
    api_key = load_api_key()
    per_seed = []
    for seed in [int(s) for s in args.seeds.split()]:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            turns = list(ex.map(lambda it: rollout(it[1], it[0], seed, args, args.mode == "attacked"),
                                list(enumerate(tasks))))
        jobs = [(t, tr) for t, tr in zip(tasks, turns) if tr is not None]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            verdicts = list(ex.map(lambda j: judge_trajectory(
                j[0]["goal"], j[0]["page"], j[1], gold_signature=reference_for_task(j[0]),
                model=args.judge_model, api_key=api_key, max_retries=args.max_retries,
                timeout=args.judge_timeout, cache_path=args.cache_path), jobs))
        graded = [v for v in verdicts if v in ("good", "bad")]
        acc = 100 * graded.count("good") / len(graded) if graded else 0.0
        per_seed.append({"seed": seed, "accuracy": acc, "n_good": graded.count("good"), "n_graded": len(graded),
                         "n_tasks": len(tasks)})
        print(f"[eval] seed={seed} mode={args.mode} completion={acc:.2f}% ({graded.count('good')}/{len(graded)})", flush=True)
    accs = [r["accuracy"] for r in per_seed]
    summary = {"mode": args.mode, "executor_model": args.executor_model, "per_seed": per_seed,
               "mean_accuracy": statistics.fmean(accs),
               "std_accuracy": statistics.stdev(accs) if len(accs) > 1 else 0.0}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"[eval] {args.mode}: {summary['mean_accuracy']:.2f} ± {summary['std_accuracy']:.2f}", flush=True)


if __name__ == "__main__":
    main()

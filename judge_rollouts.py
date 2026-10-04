"""Assess the served executor (Eq. 2): K rollouts per task in WebWorld, each graded by the judge.

Writes the judged-tasks file the trainers read:
  {goal, page, source_id, adv_*, rollouts: [{trajectory, signature, outcome, judge_verdict,
   injection_page}], p_hat (fraction judged good), pseudo_label (most common outcome among
   good rollouts), bucket}.
Attacked tasks are rolled out with their injection at adv_turn. Rollouts with >= max_repeat
identical consecutive actions are marked bad without a judge call (0 disables). Any
unresolved rollout or judgment aborts without writing output.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from web_rollouts import rollout_episode                                   # noqa: E402
from web_consistency import extract_outcome                                # noqa: E402
from kimi_judge import DEFAULT_CACHE_PATH, DEFAULT_MODEL, judge_trajectory, load_api_key, reference_for_task  # noqa: E402

ROLLOUT_SEED_BASE = 1000


def bucket_for(p_hat: float, lo: float, hi: float) -> str:
    return "easy" if p_hat > hi else "hard" if p_hat < lo else "frontier"


def is_degenerate(trajectory: list, max_repeat: int) -> bool:
    """>= max_repeat consecutive identical actions (exact strings)."""
    if max_repeat <= 0:
        return False
    run, prev = 0, None
    for turn in trajectory:
        action = (turn.get("action") or "").strip()
        run = run + 1 if action == prev else 1
        if run >= max_repeat:
            return True
        prev = action
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=8)
    ap.add_argument("--executor_api", default="http://127.0.0.1:8005/v1")
    ap.add_argument("--executor_model", default="exec")
    ap.add_argument("--world_api", default="http://127.0.0.1:8004/v1")
    ap.add_argument("--world_model", default="WebWorld-14B")
    ap.add_argument("--outcome_mode", default="final_state", choices=["final_click", "final_state"])
    ap.add_argument("--executor_temperature", type=float, default=0.7)
    ap.add_argument("--lo", type=float, default=0.3)
    ap.add_argument("--hi", type=float, default=0.8)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--judge_model", default=DEFAULT_MODEL)
    ap.add_argument("--max_retries", type=int, default=10)
    ap.add_argument("--judge_timeout", type=int, default=120)
    ap.add_argument("--max_repeat", type=int, default=3)
    ap.add_argument("--cache_path", default=DEFAULT_CACHE_PATH)
    args = ap.parse_args()

    tasks = json.loads(Path(args.tasks).read_text())
    api_key = load_api_key()

    def rollout_task(task):
        def one(k):
            return rollout_episode(task["page"], task["goal"], max_steps=args.max_steps,
                                   executor_temperature=args.executor_temperature,
                                   exec_api=args.executor_api, world_api=args.world_api,
                                   exec_model=args.executor_model, world_model=args.world_model,
                                   outcome_mode=args.outcome_mode, seed=ROLLOUT_SEED_BASE + k,
                                   adv_instruction=task.get("adv_instruction"),
                                   adv_turn=int(task.get("adv_turn", 1) or 1))
        with ThreadPoolExecutor(max_workers=args.K) as ex:
            return list(ex.map(one, range(args.K)))

    all_rollouts = [rollout_task(t) for t in tasks]
    if any(r.error for rolls in all_rollouts for r in rolls):
        sys.exit("ERROR: rollout infrastructure failure; no judged output written")

    jobs, index = [], []
    for ti, (task, rolls) in enumerate(zip(tasks, all_rollouts)):
        for ri, r in enumerate(rolls):
            if not is_degenerate(r.turns, args.max_repeat):
                jobs.append((task, r.turns)); index.append((ti, ri))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        verdicts = list(ex.map(lambda j: judge_trajectory(
            j[0]["goal"], j[0]["page"], j[1], gold_signature=reference_for_task(j[0]), model=args.judge_model,
            api_key=api_key, max_retries=args.max_retries, timeout=args.judge_timeout,
            cache_path=args.cache_path), jobs))
    if any(v not in ("good", "bad") for v in verdicts):
        sys.exit("ERROR: unresolved judge results; no judged output written")
    vmap = dict(zip(index, verdicts))

    out = []
    for ti, (task, rolls) in enumerate(zip(tasks, all_rollouts)):
        records = []
        for ri, r in enumerate(rolls):
            records.append({"trajectory": r.turns, "signature": r.signature,
                            "outcome": extract_outcome(r.turns, args.outcome_mode),
                            "judge_verdict": vmap.get((ti, ri), "bad"), "injection_page": r.injection_page})
        good = [rec["outcome"] for rec in records if rec["judge_verdict"] == "good"]
        p_hat = len(good) / len(records)
        labels = [o for o in good if o is not None]
        out.append({"goal": task["goal"], "page": task["page"], "gold_signature": task.get("gold_signature"),
                    "gold_reference": task.get("gold_reference"), "source_id": task.get("source_id"),
                    "adv_instruction": task.get("adv_instruction"), "adv_marker": task.get("adv_marker"),
                    "adv_target": task.get("adv_target", ""), "adv_turn": task.get("adv_turn", 1),
                    "rollouts": records, "p_hat": p_hat,
                    "pseudo_label": Counter(labels).most_common(1)[0][0] if labels else None,
                    "bucket": bucket_for(p_hat, args.lo, args.hi)})
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    n_good = sum(r["judge_verdict"] == "good" for t in out for r in t["rollouts"])
    print(f"[judge_rollouts] {len(out)} tasks, judged good {n_good}/{len(out) * args.K} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

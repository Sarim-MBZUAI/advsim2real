"""Stage-2 task pools. `prepare`: clean-judged tasks -> attackable tasks (clean_p_hat >= min).
`mix`: executor pool = all fresh attacks + historical attacks + clean tasks (default shares
0.50 / 0.25 / 0.25), sampled without replacement; prior rollouts/verdicts are discarded."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

ATTACK_KEYS = ("adv_instruction", "adv_marker", "adv_target", "adv_turn", "adv_source")


def raw_task(task: dict, *, clean: bool = False) -> dict:
    """Keep only task/attack inputs."""
    out = {k: task[k] for k in ("goal", "page", "gold_signature", "gold_reference", "clean_p_hat") if k in task}
    out["source_id"] = "stage2_" + hashlib.sha256(json.dumps(
        [task["goal"], task["page"]], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
    if not clean:
        out.update({k: task[k] for k in ATTACK_KEYS if k in task})
    return out


def prepare_clean(tasks: list[dict], minimum: float) -> tuple[list[dict], list[dict]]:
    clean, seen = [], set()
    for task in tasks:
        item = raw_task(task, clean=True)
        if item["source_id"] not in seen:
            seen.add(item["source_id"])
            item["clean_p_hat"] = float(task["p_hat"])
            clean.append(item)
    eligible = [dict(t) for t in clean if t["clean_p_hat"] >= minimum]
    if not eligible:
        raise SystemExit("ERROR: no clean-solvable tasks for the adversary")
    return eligible, clean


def mix_tasks(fresh, clean, history, *, replay_fraction=.25, clean_fraction=.25, seed=0):
    rng = random.Random(seed)
    key = lambda t: json.dumps([t["goal"], t["page"]] + [t.get(k) for k in ATTACK_KEYS[:4]])
    current, older, seen = [], [], set()
    for source, dest, label in ((fresh, current, "fresh"), (history, older, "replay")):
        for t in source:
            item = raw_task(t)
            if key(item) not in seen:
                seen.add(key(item))
                item["pool_source"] = label
                dest.append(item)
    clean_pool = list({t["source_id"]: {**raw_task(t, clean=True), "pool_source": "clean"} for t in clean}.values())
    share = 1 - replay_fraction - clean_fraction
    n_replay = min(len(older), round(len(current) * replay_fraction / share))
    n_clean = min(len(clean_pool), round(len(current) * clean_fraction / share))
    selected = current + rng.sample(older, n_replay) + rng.sample(clean_pool, n_clean)
    rng.shuffle(selected)
    return selected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--tasks", required=True); p.add_argument("--out", required=True)
    p.add_argument("--clean_out", required=True); p.add_argument("--min_clean_success", type=float, default=.5)
    m = sub.add_parser("mix")
    m.add_argument("--fresh", required=True); m.add_argument("--clean", required=True)
    m.add_argument("--history", nargs="*", default=[]); m.add_argument("--out", required=True)
    m.add_argument("--replay_fraction", type=float, default=.25); m.add_argument("--clean_fraction", type=float, default=.25)
    m.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    read = lambda path: json.loads(Path(path).read_text())
    if args.command == "prepare":
        eligible, clean = prepare_clean(read(args.tasks), args.min_clean_success)
        Path(args.out).write_text(json.dumps(eligible, indent=2, ensure_ascii=False))
        Path(args.clean_out).write_text(json.dumps(clean, indent=2, ensure_ascii=False))
        print(f"[stage2] {len(eligible)}/{len(clean)} tasks eligible for attacker training")
    else:
        mixed = mix_tasks(read(args.fresh), read(args.clean), [t for h in args.history for t in read(h)],
                          replay_fraction=args.replay_fraction, clean_fraction=args.clean_fraction, seed=args.seed)
        Path(args.out).write_text(json.dumps(mixed, indent=2, ensure_ascii=False))
        print(f"[stage2] executor mixture: {len(mixed)} tasks")


if __name__ == "__main__":
    main()

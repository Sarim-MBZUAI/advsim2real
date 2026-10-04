"""One attack per base task from the served adversary ("adv"); a task without a valid attack is an error."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adv_prompts import ADVERSARY_SYSTEM_IMPERATIVE                    # noqa: E402
from adv_rollouts import sample_attack                                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="base tasks to attack")
    ap.add_argument("--out", required=True)
    ap.add_argument("--adversary_api", default="http://127.0.0.1:8006/v1")
    ap.add_argument("--adversary_model", default="adv")
    ap.add_argument("--adv_max_tokens", type=int, default=512)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=-1, help=">=0: seeded, reproducible attack sampling")
    args = ap.parse_args()
    pool = json.loads(Path(args.pool).read_text())

    def one(idx_item):
        idx, item = idx_item
        a = sample_attack(args.adversary_api, args.adversary_model, item["goal"], item["page"],
                          system=ADVERSARY_SYSTEM_IMPERATIVE, max_tokens=args.adv_max_tokens, retries=args.retries,
                          seed=None if args.seed < 0 else args.seed * 100003 + idx)
        if a is None:
            raise ValueError(f"task {item.get('source_id', idx)}: no valid attack from the adversary")
        return {**item, "adv_instruction": a.instruction, "adv_marker": a.marker, "adv_source": "adversary",
                "adv_target": a.target, "adv_turn": a.turn}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        out = list(ex.map(one, enumerate(pool)))
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {len(out)} attacked tasks -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

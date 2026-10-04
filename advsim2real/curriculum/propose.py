"""Sample <PAGE>/<GOAL> proposals from the served curriculum ("curr"); drop near-duplicate goals."""
from __future__ import annotations

import argparse
import difflib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from advsim2real.curriculum.prompts import WEB_CURRICULUM_SYSTEM, WEB_DOMAINS, parse_web_completion


def propose_task(api_url, model, domain, *, max_tokens, temperature=1.0, timeout=180, retries=3):
    """One valid proposal (up to `retries` samples), else None."""
    for _ in range(retries):
        try:
            resp = requests.post(f"{api_url}/chat/completions", json={
                "model": model, "max_tokens": max_tokens, "temperature": temperature,
                "messages": [{"role": "system", "content": WEB_CURRICULUM_SYSTEM},
                             {"role": "user", "content": f"Generate one web task proposal for: {domain}."}],
                "chat_template_kwargs": {"enable_thinking": False}}, timeout=timeout)
            resp.raise_for_status()
            parsed = parse_web_completion(resp.json()["choices"][0]["message"]["content"])
        except requests.RequestException:
            return None
        if parsed.is_valid:
            return parsed
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--curriculum_api", default="http://127.0.0.1:8006/v1")
    ap.add_argument("--curriculum_model", default="curr")
    ap.add_argument("--n_proposals", type=int, default=40)
    ap.add_argument("--propose_max_tokens", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--dedup_threshold", type=float, default=0.85)
    args = ap.parse_args()

    def one(i):
        p = propose_task(args.curriculum_api, args.curriculum_model, WEB_DOMAINS[i % len(WEB_DOMAINS)],
                         max_tokens=args.propose_max_tokens, temperature=args.temperature)
        return {"goal": p.goal, "page": p.page, "gold_signature": "", "source_id": f"curr_{i}"} if p else None

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        cand = [c for c in ex.map(one, range(args.n_proposals)) if c]
    kept, norms = [], []
    for c in cand:
        n = " ".join(c["goal"].lower().split())
        if all(difflib.SequenceMatcher(None, n, m).ratio() < args.dedup_threshold for m in norms):
            kept.append(c); norms.append(n)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(kept, indent=2, ensure_ascii=False))
    print(f"wrote {len(kept)} proposals ({args.n_proposals} positions) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

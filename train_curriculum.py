"""Train the curriculum (GRPO + LoRA) by replaying its judged proposals (Eq. 3).

Each judged task is a completion the curriculum already emitted; it is teacher-forced under
the prompt it was proposed with and rewarded by
    R_C = gate * max(0, q(p_hat) - rep_penalty),  gate = (pseudo_label exists and p_hat > 0.1).
Groups mix difficulties (round-robin deal after sorting by reward); one pass per epoch.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG_DIR))

from _lib.rewards import curriculum_reward, repetition_penalty         # noqa: E402
from web_curriculum import WEB_CURRICULUM_SYSTEM, WEB_DOMAINS, parse_web_completion  # noqa: E402
from _lib.grpo import GRPOConfig, GRPOTrainer                          # noqa: E402
from _lib.schedules import replay_epoch_schedule                       # noqa: E402


def agent0_curriculum_reward(p_hat: float, pseudo_label: str | None, rep_penalty: float = 0.0) -> float:
    ok = pseudo_label is not None and p_hat > 0.1
    return max(0.0, float(curriculum_reward(p_hat)) - rep_penalty) if ok else 0.0


def reconstruct_completion(page: str, goal: str) -> str:
    return f"<PAGE>\n{page.strip()}\n</PAGE>\n<GOAL>\n{goal.strip()}\n</GOAL>"


def _domain_for(task: dict, fallback_i: int) -> str:
    """Proposal i (source_id curr_i) was prompted with WEB_DOMAINS[i % 12]."""
    tail = str(task.get("source_id") or "").rsplit("_", 1)[-1]
    return WEB_DOMAINS[(int(tail) if tail.isdigit() else fallback_i) % len(WEB_DOMAINS)]


def build_replay_items(tasks: list[dict]) -> list[dict]:
    items, seen = [], set()
    for i, t in enumerate(tasks):
        page, goal = (t.get("page") or "").strip(), (t.get("goal") or "").strip()
        if not (page and goal) or t.get("p_hat") is None:
            continue
        comp = reconstruct_completion(page, goal)
        parsed = parse_web_completion(comp)
        if not (parsed.is_valid and parsed.page == page and parsed.goal == goal) or comp in seen:
            continue
        seen.add(comp)
        items.append({"completion": comp, "p_hat": float(t["p_hat"]), "pseudo_label": t.get("pseudo_label"),
                      "domain": _domain_for(t, i),
                      "base_reward": agent0_curriculum_reward(float(t["p_hat"]), t.get("pseudo_label"))})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--init_adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks_file", required=True, help="judged proposals (judge_rollouts.py)")
    ap.add_argument("--rep_lambda", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--group_size", type=int, default=6)
    ap.add_argument("--prompts_per_step", type=int, default=2, help="replay groups per update")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--kl_beta", type=float, default=1e-2)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--micro_bs", type=int, default=2)
    ap.add_argument("--attn_impl", default=None)
    ap.add_argument("--grad_checkpoint", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    items = build_replay_items(json.loads(Path(args.tasks_file).read_text()))
    if len(items) < 2:
        sys.exit(f"ERROR: only {len(items)} replayable proposal(s)")
    plan = replay_epoch_schedule(items, args.group_size, args.prompts_per_step, args.epochs, random.Random(args.seed))
    tr = GRPOTrainer(GRPOConfig(
        model_path=args.model, out_dir=args.out, lora_rank=args.lora_rank, lr=args.lr, kl_beta=args.kl_beta,
        steps=len(plan), group_size=args.group_size, prompts_per_step=args.prompts_per_step,
        max_new_tokens=args.max_new_tokens, micro_bs=args.micro_bs, init_adapter=args.init_adapter,
        seed=args.seed, attn_impl=args.attn_impl, grad_checkpoint=bool(args.grad_checkpoint)))

    def render_prompt(domain: str) -> str:
        return tr.tok.apply_chat_template(
            [{"role": "system", "content": "/no_think\n" + WEB_CURRICULUM_SYSTEM},
             {"role": "user", "content": f"Generate one web task proposal for: {domain}."}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    by_completion = {it["completion"]: it for it in items}

    def reward_fn(_prompts, completions):
        parsed = [parse_web_completion(c) for c in completions]
        valid = [i for i, p in enumerate(parsed) if p.is_valid]
        rep = dict(zip(valid, repetition_penalty([parsed[i].goal for i in valid], lam=args.rep_lambda)))
        out = []
        for i, c in enumerate(completions):
            it = by_completion.get(c)
            out.append(agent0_curriculum_reward(it["p_hat"], it["pseudo_label"], rep[i])
                       if i in rep and it is not None else 0.0)
        return out

    def sampler(step):
        _, groups = plan[step - 1]
        prompts, comps, owner = [], [], []
        for gi, grp in enumerate(groups):
            for idx in grp:
                prompts.append(render_prompt(items[idx]["domain"]))
                comps.append(items[idx]["completion"])
                owner.append(gi)
        return prompts, comps, owner

    tr.train_on_completions(sampler, reward_fn)


if __name__ == "__main__":
    main()

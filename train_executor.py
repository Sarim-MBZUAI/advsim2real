"""Train the executor (GRPO + LoRA) multi-turn inside WebWorld; rewards from the live judge (Eq. 4).

Per task, G trajectories are rolled out with the in-process policy: the policy's action tokens
are trained, world-model pages and chat glue are masked. Every trajectory is judged fresh:
good = +1; bad = -1, or, on attacked tasks, clip(-wg + wr * (1 - 2*followed_target)).
Group-normalized advantages are scaled by f(p_hat) of the task's judged solve rate.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

PKG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG_DIR))

from _lib.rewards import advantage_scaler                             # noqa: E402
from _lib.grpo import GRPOConfig, GRPOTrainer                         # noqa: E402
from _lib.schedules import executor_task_schedule                     # noqa: E402
from web_consistency import trajectory_signature                      # noqa: E402
from web_rollouts import AGENT_SYSTEM, WORLD_API, WORLD_MODEL, _WebWorld, parse_action  # noqa: E402
from kimi_judge import (DEFAULT_MODEL as JUDGE_MODEL, DEFAULT_CACHE_PATH as JUDGE_CACHE,  # noqa: E402
                        judge_trajectory, load_api_key, reference_for_task)

EXEC_SYSTEM = "/no_think\n" + AGENT_SYSTEM
TERMINAL = ("send_msg_to_user",)


def rollout_group(tr: GRPOTrainer, task: dict, group_size, *, world_api, world_model, max_steps,
                  action_len, temperature, top_p, max_state_chars):
    """group_size trajectories, turn-synchronous: one batched generate per turn, concurrent
    WebWorld calls. Returns [(ids, action_mask, turns, error)]. An attack in the task is
    injected once, at transition adv_turn."""
    tok, model = tr.tok, tr.model
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    goal, page = task["goal"], task["page"]
    adv_instr, adv_turn = task.get("adv_instruction"), int(task.get("adv_turn", 1) or 1)
    ids0 = tok.apply_chat_template(
        [{"role": "system", "content": EXEC_SYSTEM},
         {"role": "user", "content": f"Goal: {goal}\n\nPage State:\n{page}\n\nNext action:"}],
        add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True)["input_ids"][0].cuda()
    trajs = [{"ids": ids0.clone(), "mask": torch.zeros(ids0.shape[0], dtype=torch.bool, device=ids0.device),
              "turns": [], "state": page, "world": _WebWorld(api_url=world_api, model=world_model),
              "alive": True, "error": None} for _ in range(group_size)]

    for turn in range(max_steps):
        live = [t for t in trajs if t["alive"]]
        if not live:
            break
        W = max(t["ids"].shape[0] for t in live)
        batch = torch.full((len(live), W), tr.pad_id, dtype=ids0.dtype, device=ids0.device)
        attn = torch.zeros((len(live), W), dtype=torch.long, device=ids0.device)
        for j, t in enumerate(live):
            batch[j, W - t["ids"].shape[0]:] = t["ids"]
            attn[j, W - t["ids"].shape[0]:] = 1
        model.config.use_cache = True
        with torch.no_grad():
            out = model.generate(batch, attention_mask=attn, do_sample=True, temperature=temperature, top_p=top_p,
                                 max_new_tokens=action_len, eos_token_id=im_end, pad_token_id=tr.pad_id)
        model.config.use_cache = False

        pending = []
        for j, t in enumerate(live):
            row = out[j, W:]
            eos = (row == im_end).nonzero(as_tuple=False)
            new = row[: eos[0, 0].item() + 1] if eos.numel() else row
            t["ids"] = torch.cat([t["ids"], new])
            t["mask"] = torch.cat([t["mask"], torch.ones(new.shape[0], dtype=torch.bool, device=new.device)])
            action = parse_action(re.sub(r"<think>.*?</think>", "", tok.decode(new, skip_special_tokens=True), flags=re.DOTALL))
            if action is None:
                t["alive"] = False
                continue
            t["turns"].append({"action": action, "state_shown": t["state"]})
            if action.startswith(TERMINAL) or turn == max_steps - 1:
                t["alive"] = False
                continue
            t["closed"] = new[-1].item() == im_end
            pending.append((t, action))

        def advance(item, _turn=turn):
            t, action = item
            inj = adv_instr if adv_instr and adv_turn == _turn + 1 else None
            nxt = (t["world"].reset(initial_state=page, first_action=action, adv_instruction=inj) if _turn == 0
                   else t["world"].step(action, adv_instruction=inj))
            return None if nxt is None else nxt[:max_state_chars]

        with ThreadPoolExecutor(max_workers=max(1, len(pending))) as ex:
            states = list(ex.map(advance, pending))
        for (t, _), state in zip(pending, states):
            if state is None:
                t["alive"], t["error"] = False, "world_error"
                continue
            t["state"] = state
            glue = ("" if t["closed"] else "<|im_end|>") + \
                f"\n<|im_start|>user\nPage State:\n{state}\n\nNext action:<|im_end|>\n" \
                f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
            g = tok(glue, add_special_tokens=False, return_tensors="pt").input_ids[0].cuda()
            t["ids"] = torch.cat([t["ids"], g])
            t["mask"] = torch.cat([t["mask"], torch.zeros(g.shape[0], dtype=torch.bool, device=g.device)])
    return [(t["ids"], t["mask"], t["turns"], t["error"]) for t in trajs]


def _norm_sig_token(tok: str, *, drop_value: bool = True) -> str:
    tok = re.sub(r"\s+", " ", (tok or "").strip().lower())
    match = re.fullmatch(r'(.*?)\s*=\s*"([^"]*)"', tok)
    if match:
        field, value = match.group(1).strip(), match.group(2).strip()
        return field if drop_value else f'{field}="{value}"'
    return tok


def _took_adv_target(signature: str, adv_target: str) -> bool:
    """Did the trajectory take the attacker's declared target? A quoted value must match too."""
    if not adv_target or not signature:
        return False
    has_value = re.search(r'=\s*"[^"]*"\s*$', adv_target) is not None
    want = _norm_sig_token(adv_target, drop_value=not has_value)
    tokens = re.split(r'\|(?=(?:[^"]*"[^"]*")*[^"]*$)', signature)
    return bool(want) and any(_norm_sig_token(t, drop_value=not has_value) == want for t in tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--init_adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks_file", required=True, help="judged tasks (judge_rollouts.py) with p_hat")
    ap.add_argument("--world_api", default=WORLD_API)
    ap.add_argument("--world_model", default=WORLD_MODEL)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=0, help=">0: complete passes over tasks (overrides --steps)")
    ap.add_argument("--group_size", type=int, default=4)
    ap.add_argument("--prompts_per_step", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=8)
    ap.add_argument("--max_seq_len", type=int, default=8192, help="keep the last N tokens for the update")
    ap.add_argument("--action_len", type=int, default=64)
    ap.add_argument("--max_state_chars", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl_beta", type=float, default=1e-2)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--judge_model", default=JUDGE_MODEL)
    ap.add_argument("--judge_workers", type=int, default=16)
    ap.add_argument("--judge_retries", type=int, default=6)
    ap.add_argument("--judge_timeout", type=int, default=120)
    ap.add_argument("--judge_cache", default=JUDGE_CACHE)
    ap.add_argument("--harden_goal_weight", type=float, default=0.5)
    ap.add_argument("--harden_resist_weight", type=float, default=0.5)
    ap.add_argument("--attn_impl", default=None)
    ap.add_argument("--grad_checkpoint", type=int, default=0)
    args = ap.parse_args()

    tasks = json.loads(Path(args.tasks_file).read_text())
    plan = executor_task_schedule(list(range(len(tasks))), args.prompts_per_step, args.epochs, args.steps,
                                  random.Random(0))
    api_key = load_api_key()
    tr = GRPOTrainer(GRPOConfig(
        model_path=args.model, out_dir=args.out, lora_rank=args.lora_rank, lr=args.lr, kl_beta=args.kl_beta,
        steps=len(plan), group_size=args.group_size, prompts_per_step=args.prompts_per_step, micro_bs=1,
        temperature=args.temperature, init_adapter=args.init_adapter, attn_impl=args.attn_impl,
        grad_checkpoint=bool(args.grad_checkpoint)))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    log = open(Path(args.out) / "executor_log.jsonl", "a")

    def judge(job):
        task, turns = job
        return judge_trajectory(task["goal"], task["page"], turns, gold_signature=reference_for_task(task),
                                model=args.judge_model, api_key=api_key, max_retries=args.judge_retries,
                                timeout=args.judge_timeout, cache_path=args.judge_cache)

    for s, (_, rows) in enumerate(plan, start=1):
        t0 = time.time()
        seqs, masks, owner, rewards, scales = [], [], [], [], []
        for gi, row in enumerate(rows):
            task = tasks[row]
            group = [g for g in rollout_group(tr, task, args.group_size, world_api=args.world_api,
                                              world_model=args.world_model, max_steps=args.max_steps,
                                              action_len=args.action_len, temperature=args.temperature,
                                              top_p=1.0, max_state_chars=args.max_state_chars)
                     if g[3] is None]                                   # drop world failures
            with ThreadPoolExecutor(max_workers=max(1, min(args.judge_workers, len(group)))) as ex:
                verdicts = list(ex.map(judge, [(task, g[2]) for g in group]))
            group = [(g, v) for g, v in zip(group, verdicts) if v in ("good", "bad")]   # drop judge failures
            if len(group) < 2:
                continue
            target = task.get("adv_target")
            for (ids, mask, turns, _), verdict in group:
                if verdict == "good":
                    r = 1.0
                elif target:
                    followed = _took_adv_target(trajectory_signature(turns), target)
                    r = max(-1.0, min(1.0, -args.harden_goal_weight
                                      + args.harden_resist_weight * (-1.0 if followed else 1.0)))
                else:
                    r = -1.0
                seqs.append(ids); masks.append(mask); owner.append(gi); rewards.append(r)
                scales.append(advantage_scaler(float(task.get("p_hat", 1.0))))
                log.write(json.dumps({"step": s, "task": row, "verdict": verdict, "reward": r,
                                      "actions": [t["action"] for t in turns]}) + "\n")
        log.flush()
        if not seqs:
            print(f"[exec] step {s}/{len(plan)}: no gradable groups", flush=True)
            continue
        adv = tr.group_advantages(torch.tensor(rewards), owner, scales)
        seqs = [x[-args.max_seq_len:] for x in seqs]
        masks = [m[-args.max_seq_len:] for m in masks]
        loss = tr._pg_update(seqs, masks, adv.tolist())
        print(f"[exec] step {s}/{len(plan)}  good_rate={sum(r > 0 for r in rewards) / len(rewards):.2f}"
              f"  loss={loss:+.4f}  n={len(seqs)}  {time.time()-t0:.1f}s", flush=True)
    log.close()
    tr.save_adapter(args.out)


if __name__ == "__main__":
    main()

"""In-process GRPO + LoRA trainer (transformers + peft); loss of Eq. 6."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from advsim2real.training.schedules import executor_task_schedule


@dataclass
class GRPOConfig:
    model_path: str
    out_dir: str
    lora_rank: int = 16
    lora_alpha: int = 32
    lr: float = 1e-5
    steps: int = 50
    group_size: int = 8          # samples per prompt (comparison group)
    prompts_per_step: int = 4
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    kl_beta: float = 0.0         # KL to the adapter-disabled base
    grad_clip: float = 1.0
    micro_bs: int = 4
    seed: int = 0
    init_adapter: str | None = None
    attn_impl: str | None = None
    grad_checkpoint: bool = False
    epochs: int = 0              # >0: complete passes over prompts, overrides steps


def _left_pad(seqs, pad_value):
    return torch.nn.utils.rnn.pad_sequence(
        [s.flip(0) for s in seqs], batch_first=True, padding_value=pad_value).flip(1)


def _eos_with_im_end(tok, eos):
    """Model eos id(s) plus ChatML <|im_end|> and <|im_start|>."""
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    for t in ("<|im_end|>", "<|im_start|>"):
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id and tid not in eos:
            eos.append(tid)
    return eos or None


class GRPOTrainer:
    def __init__(self, cfg: GRPOConfig):
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.tok = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.pad_id = self.tok.pad_token_id

        # Qwen3.5 is an image-text model: load it text-only and keep LoRA off the vision tower.
        archs = getattr(AutoConfig.from_pretrained(cfg.model_path, trust_remote_code=True), "architectures", None) or []
        multimodal = any(("ConditionalGeneration" in a) or ("ImageTextToText" in a) for a in archs)
        cls = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM
        model = cls.from_pretrained(cfg.model_path, dtype=torch.bfloat16, trust_remote_code=True,
                                    attn_implementation=cfg.attn_impl or "eager").to("cuda")
        model.config.use_cache = False
        self.eos_ids = _eos_with_im_end(self.tok, model.generation_config.eos_token_id)
        if cfg.init_adapter:
            self.model = PeftModel.from_pretrained(model, cfg.init_adapter, is_trainable=True)
        else:
            targets = (["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                       if multimodal else "all-linear")
            self.model = get_peft_model(model, LoraConfig(
                task_type="CAUSAL_LM", r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
                target_modules=targets, lora_dropout=0.0, bias="none"))
        if cfg.grad_checkpoint:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()
        self.model.print_trainable_parameters()
        self.opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=cfg.lr)

    @staticmethod
    def group_advantages(rewards: torch.Tensor, groups: list[int], scales: list[float] | None = None):
        adv = torch.zeros_like(rewards)
        g = torch.tensor(groups)
        for gi in g.unique():
            m = g == gi
            grp = rewards[m]
            adv[m] = (grp - grp.mean()) / (grp.std(unbiased=False) + 1e-6)
        if scales is not None:               # per-sample scale on the normalized advantage
            adv = adv * torch.as_tensor(scales, dtype=adv.dtype)
        return adv

    def _pg_update(self, seqs, masks, advs) -> float:
        """Token-mean policy gradient on masked (policy-generated) tokens + KL to the base."""
        cfg = self.cfg
        self.model.train()                   # transformers 5 gates checkpointing on .training
        advs = torch.as_tensor(advs, dtype=torch.float32, device="cuda")
        denom = max(int(sum(int(m[1:].sum()) for m in masks)), 1)
        self.opt.zero_grad(set_to_none=True)
        total = 0.0
        for i in range(0, len(seqs), cfg.micro_bs):
            sl = slice(i, i + cfg.micro_bs)
            ids = _left_pad(seqs[sl], self.pad_id)
            mk = _left_pad([m.long() for m in masks[sl]], 0).bool()
            attn = (ids != self.pad_id).long()
            logits = self.model(input_ids=ids, attention_mask=attn).logits[:, :-1, :]
            tgt = ids[:, 1:]
            V = logits.size(-1)
            tok = (-F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), reduction="none")).view(tgt.shape)
            tmask = mk[:, 1:]
            loss = -(advs[sl].unsqueeze(1) * tok * tmask).sum() / denom
            if cfg.kl_beta > 0:
                with torch.no_grad(), self.model.disable_adapter():
                    rl = self.model(input_ids=ids, attention_mask=attn).logits[:, :-1, :]
                    rlp = (-F.cross_entropy(rl.reshape(-1, V), tgt.reshape(-1), reduction="none")).view(tgt.shape)
                kl = (torch.exp(rlp - tok) - (rlp - tok) - 1.0) * tmask
                loss = loss + cfg.kl_beta * kl.sum() / denom
            loss.backward()
            total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], cfg.grad_clip)
        self.opt.step()
        self.model.eval()                    # keep the KV cache for the next generate()
        return total

    # ── sampled single-turn groups (adversary) ──────────────────────────────
    @torch.no_grad()
    def _generate(self, prompt_text: str):
        cfg = self.cfg
        enc = self.tok(prompt_text, return_tensors="pt", add_special_tokens=False).to("cuda")
        self.model.config.use_cache = True
        out = self.model.generate(**enc, do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p,
                                  max_new_tokens=cfg.max_new_tokens, num_return_sequences=cfg.group_size,
                                  pad_token_id=self.pad_id, eos_token_id=self.eos_ids)
        self.model.config.use_cache = False
        return out, enc.input_ids.shape[1]

    def step(self, prompts, reward_fn):
        seqs, masks, owner, texts, comp_prompts = [], [], [], [], []
        for pi, ptext in enumerate(prompts):
            full, Lp = self._generate(ptext)
            for s in full:
                pos = torch.arange(s.shape[0], device=s.device)
                seqs.append(s)
                masks.append((pos >= Lp) & (s != self.pad_id))
                owner.append(pi)
                texts.append(self.tok.decode(s[Lp:], skip_special_tokens=True))
                comp_prompts.append(ptext)
        rewards = torch.tensor(reward_fn(comp_prompts, texts), dtype=torch.float32)
        loss = self._pg_update(seqs, masks, self.group_advantages(rewards, owner).tolist())
        return {"loss": loss, "reward_mean": float(rewards.mean()), "reward_std": float(rewards.std(unbiased=False))}

    def train(self, prompts, reward_fn):
        cfg = self.cfg
        plan = executor_task_schedule(list(range(len(prompts))), cfg.prompts_per_step, cfg.epochs,
                                      cfg.steps, random.Random(cfg.seed))
        for s, (_, rows) in enumerate(plan, start=1):
            t0 = time.time()
            m = self.step([prompts[i] for i in rows], reward_fn)
            print(f"[grpo] step {s}/{len(plan)}  reward={m['reward_mean']:+.3f}±{m['reward_std']:.3f}"
                  f"  loss={m['loss']:+.4f}  {time.time()-t0:.1f}s", flush=True)
        self.save_adapter(cfg.out_dir)

    # ── teacher-forced groups on stored completions (curriculum replay) ─────
    @torch.no_grad()
    def _encode_forced(self, prompt_text: str, completion_text: str):
        p_ids = self.tok(prompt_text, add_special_tokens=False).input_ids
        c_ids = self.tok(completion_text, add_special_tokens=False).input_ids
        if len(c_ids) > self.cfg.max_new_tokens:
            c_ids = c_ids[: self.cfg.max_new_tokens]
        else:
            c_ids = c_ids + [self.tok.convert_tokens_to_ids("<|im_end|>")]
        s = torch.tensor(p_ids + c_ids, dtype=torch.long, device="cuda")
        pos = torch.arange(s.shape[0], device=s.device)
        return s, (pos >= len(p_ids)) & (s != self.pad_id)

    def train_on_completions(self, sampler, reward_fn):
        """sampler(step) -> (prompts, completions, groups) for each replay update."""
        for s in range(1, self.cfg.steps + 1):
            prompts, completions, groups = sampler(s)
            seqs, masks = zip(*[self._encode_forced(p, c) for p, c in zip(prompts, completions)])
            rewards = torch.tensor(reward_fn(list(prompts), list(completions)), dtype=torch.float32)
            loss = self._pg_update(list(seqs), list(masks), self.group_advantages(rewards, list(groups)).tolist())
            print(f"[grpo/replay] step {s}/{self.cfg.steps}  reward={float(rewards.mean()):+.3f}"
                  f"  loss={loss:+.4f}", flush=True)
        self.save_adapter(self.cfg.out_dir)

    def save_adapter(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        print(f"[grpo] saved LoRA adapter -> {path}", flush=True)

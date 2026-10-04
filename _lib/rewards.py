"""Reward shaping (paper Eqs. 3-5)."""
from __future__ import annotations

import difflib
from collections import Counter


def curriculum_reward(p_hat: float) -> float:
    """q(p) = 1 - 2|p - 1/2|: peaks when the executor solves the task half the time."""
    return 1.0 - 2.0 * abs(p_hat - 0.5)


def advantage_scaler(p_hat: float) -> float:
    """f(p) = max(0.1, q(p)), applied to the group-normalized executor advantage."""
    return max(0.1, min(1.0, 1.0 - 2.0 * abs(p_hat - 0.5)))


def repetition_penalty(texts: list[str], lam: float = 1.0, threshold: float = 0.8) -> list[float]:
    """lam * |C_i| / B, with C_i the greedy difflib cluster (ratio >= threshold) of text i."""
    B = len(texts)
    if B == 0:
        return []
    norm = [" ".join(t.lower().split()) for t in texts]
    cluster_of, reps = [-1] * B, []
    for i, ti in enumerate(norm):
        for c, rep in enumerate(reps):
            if difflib.SequenceMatcher(None, ti, norm[rep]).ratio() >= threshold:
                cluster_of[i] = c
                break
        else:
            cluster_of[i] = len(reps)
            reps.append(i)
    sizes = Counter(cluster_of)
    return [lam * sizes[cluster_of[i]] / B for i in range(B)]

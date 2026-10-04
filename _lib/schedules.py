"""Training schedules: complete epochs (or fixed steps) over tasks and replay groups."""
from __future__ import annotations

import random


def make_epoch_groups(items: list[dict], group_size: int, rng: random.Random) -> list[list[int]]:
    """Deal reward-sorted replay items round-robin so every group mixes difficulties."""
    n = len(items)
    if n < 2:
        return []
    order = list(range(n))
    rng.shuffle(order)
    order.sort(key=lambda i: items[i]["base_reward"])
    count = max(1, n // max(2, int(group_size)))
    groups: list[list[int]] = [[] for _ in range(count)]
    for j, idx in enumerate(order):
        groups[j % count].append(idx)
    kept = [g for g in groups if len(g) >= 2]
    rng.shuffle(kept)
    return kept


def replay_epoch_schedule(items: list[dict], group_size: int, groups_per_step: int,
                          epochs: int, rng: random.Random) -> list[tuple[int, list[list[int]]]]:
    """Exact replay passes with a fresh deal per epoch; the last update may be smaller."""
    schedule = []
    for epoch in range(1, epochs + 1):
        groups = make_epoch_groups(items, group_size, rng)
        for start in range(0, len(groups), groups_per_step):
            schedule.append((epoch, groups[start:start + groups_per_step]))
    return schedule


def executor_task_schedule(indices: list[int], tasks_per_step: int, epochs: int,
                           steps: int, rng: random.Random) -> list[tuple[int | None, list[int]]]:
    """Epochs visit every task once per shuffled pass; epochs=0 samples `steps` batches."""
    if not epochs:
        return [(None, [rng.choice(indices) for _ in range(tasks_per_step)]) for _ in range(steps)]
    schedule = []
    for epoch in range(1, epochs + 1):
        order = list(indices)
        rng.shuffle(order)
        for start in range(0, len(order), tasks_per_step):
            schedule.append((epoch, order[start:start + tasks_per_step]))
    return schedule

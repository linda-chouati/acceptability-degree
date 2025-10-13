from __future__ import annotations
from typing import Dict
from .framework import ArgumentationFramework

def weighted_h_categorizer(
    af: ArgumentationFramework,
    w: Dict[str, float],
    epsilon: float = 1e-6,
    max_iter: int = 1000,
) -> Dict[str, float]:
    A = af.A
    atk = af.attackers_map()
    x_prev = {a: float(w.get(a, 0.0)) for a in A}

    for _ in range(max_iter):
        x_next = {}
        for a in A:
            denom = 1.0 + sum(x_prev[b] for b in atk[a])
            x_next[a] = (w.get(a, 0.0) / denom) if denom != 0 else x_prev[a]
        if max(abs(x_next[a] - x_prev[a]) for a in A) < epsilon:
            return x_next
        x_prev = x_next
    # si ca n a pas convergé (normalement cas rare) -> on retourne donc la dernière itération
    return x_prev

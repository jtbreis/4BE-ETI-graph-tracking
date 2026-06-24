"""Global greedy assignment of disjoint multi-frame paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PathCandidate:
    """
    One valid path with cost (lower is better).

    ``start_layer`` is 0 for paths beginning at the first frame of the block,
    or 1 for paths beginning at the second frame.

    ``indices`` has length 3 or 4 (local particle index in each frame used).
    """

    start_layer: int
    indices: tuple[int, ...]
    cost: float

    @property
    def length(self) -> int:
        return len(self.indices)


def path_indices_key(candidate: PathCandidate) -> tuple[int, ...]:
    """Canonical key for matching the same physical path across passes."""
    return candidate.indices


def filter_bidirectional_four_frame(
    forward: list[PathCandidate],
    backward: list[PathCandidate],
) -> tuple[list[PathCandidate], set[tuple[int, ...]]]:
    """
    Keep forward 4-frame candidates whose index tuple also appears backward.

    Returns (filtered forward list, intersection key set).
    """
    fwd_4 = [c for c in forward if c.length == 4 and c.start_layer == 0]
    bwd_keys = {
        path_indices_key(c)
        for c in backward
        if c.length == 4 and c.start_layer == 0
    }
    both = {path_indices_key(c) for c in fwd_4} & bwd_keys
    kept = [c for c in fwd_4 if path_indices_key(c) in both]
    return kept, both


def greedy_assign_paths(candidates: list[PathCandidate]) -> list[PathCandidate]:
    """
    Pick disjoint paths: prefer longer paths, then lower cost.

    Each detection index may appear at most once per frame layer.
    """
    if not candidates:
        return []

    order = sorted(
        range(len(candidates)),
        key=lambda i: (-candidates[i].length, candidates[i].cost),
    )
    used: list[set[int]] = [set(), set(), set(), set()]
    chosen: list[PathCandidate] = []

    for idx in order:
        c = candidates[idx]
        layers = range(c.start_layer, c.start_layer + c.length)
        if any(c.indices[j] in used[layer] for j, layer in enumerate(layers)):
            continue
        for j, layer in enumerate(layers):
            used[layer].add(c.indices[j])
        chosen.append(c)

    return chosen

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

    ``hop_distances`` stores each track-box hop distance (mm) at graph build time.
    """

    start_layer: int
    indices: tuple[int, ...]
    cost: float
    hop_distances: tuple[float, ...] = ()

    @property
    def length(self) -> int:
        return len(self.indices)


def path_indices_key(candidate: PathCandidate) -> tuple[int, ...]:
    """Canonical key for matching the same physical path across passes."""
    return candidate.indices


def combine_bidirectional_four_frame(
    forward: list[PathCandidate],
    backward: list[PathCandidate],
) -> tuple[list[PathCandidate], dict[str, int]]:
    """Keep only 4-frame paths found in both forward and backward enumerations."""
    fwd_4 = {
        path_indices_key(c): c
        for c in forward
        if c.length == 4 and c.start_layer == 0
    }
    bwd_4 = {
        path_indices_key(c): c
        for c in backward
        if c.length == 4 and c.start_layer == 0
    }

    both = set(fwd_4) & set(bwd_4)
    fwd_only = set(fwd_4) - both
    bwd_only = set(bwd_4) - both

    combined: list[PathCandidate] = []
    for key in both:
        fc = fwd_4[key]
        bc = bwd_4[key]
        combined.append(
            PathCandidate(
                start_layer=0,
                indices=key,
                cost=fc.cost + bc.cost,
            )
        )

    stats = {
        "n_bidirectional_4frame": len(both),
        "n_forward_only_4frame": len(fwd_only),
        "n_backward_only_4frame": len(bwd_only),
        "n_combined_candidates": len(combined),
    }
    return combined, stats


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

#!/usr/bin/env python3
"""Compare Python vs Numba spatial-grid linking (correctness + timing)."""

from __future__ import annotations

import time

from graph_tracking.assign import greedy_assign_paths, path_indices_key
from graph_tracking.link_block import LinkParams, link_block
from graph_tracking.load_data import load_block
from graph_tracking.spatial_grid_numba import numba_available


def _candidate_keys(candidates):
    return sorted(
        (c.start_layer, c.indices, round(c.cost, 12))
        for c in candidates
    )


def _assigned_keys(assigned):
    return sorted(
        (c.start_layer, c.indices, round(c.cost, 12))
        for c in assigned
    )


def compare_block(h5path: str, start: int, params_kw: dict) -> dict:
    block = load_block(h5path, start)
    kw = dict(params_kw)

    p_py = LinkParams(**kw, use_numba=False)
    p_nb = LinkParams(**kw, use_numba=True)

    t0 = time.perf_counter()
    r_py = link_block(block, p_py)
    a_py = greedy_assign_paths(r_py.candidates)
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    r_nb = link_block(block, p_nb)
    a_nb = greedy_assign_paths(r_nb.candidates)
    t_nb = time.perf_counter() - t0

    same_candidates = _candidate_keys(r_py.candidates) == _candidate_keys(r_nb.candidates)
    same_assigned = _assigned_keys(a_py) == _assigned_keys(a_nb)

    return {
        "t_py": t_py,
        "t_nb": t_nb,
        "candidates_py": len(r_py.candidates),
        "candidates_nb": len(r_nb.candidates),
        "tracks_py": len(a_py),
        "tracks_nb": len(a_nb),
        "same_candidates": same_candidates,
        "same_assigned": same_assigned,
    }


def main() -> int:
    if not numba_available():
        print("SKIP: numba not installed")
        return 0

    synth = "graph_tracking/synthetic_data/two_particles/rays_out_cpp.h5"
    run1 = (
        "data/Windtunnel_PTV_TNTI/X300cm_PTV_above_TNTI/"
        "TNTI_opposing_gravity/Run1/rays_out_cpp.h5"
    )
    base = dict(
        box_size_track=0.2,
        initial_x_lo=0.0,
        initial_x_hi=3.5,
        initial_y_lo=-1.5,
        initial_y_hi=1.5,
        initial_z_lo=-1.5,
        initial_z_hi=1.5,
        max_branch=30,
        allow_3_frame=False,
        progress_interval=0,
    )

    print("=== Synthetic two-particle ===")
    s = compare_block(synth, 0, {**base, "max_seeds": 0})
    print(f"  python {s['t_py']:.3f}s | numba {s['t_nb']:.3f}s | speedup {s['t_py']/max(s['t_nb'],1e-9):.2f}x")
    print(f"  candidates {s['candidates_py']} vs {s['candidates_nb']} | match={s['same_candidates']}")
    print(f"  tracks {s['tracks_py']} vs {s['tracks_nb']} | match={s['same_assigned']}")
    assert s["same_candidates"] and s["same_assigned"]

    print("\n=== Opposing Run1 (all seeds, forward) ===")
    r = compare_block(run1, 0, {**base, "max_seeds": 0, "use_bidirectional_graph": False})
    print(f"  python {r['t_py']:.2f}s | numba {r['t_nb']:.2f}s | speedup {r['t_py']/max(r['t_nb'],1e-9):.2f}x")
    print(f"  candidates {r['candidates_py']} vs {r['candidates_nb']} | match={r['same_candidates']}")
    print(f"  tracks {r['tracks_py']} vs {r['tracks_nb']} | match={r['same_assigned']}")
    assert r["same_candidates"] and r["same_assigned"]

    print("\nOK: Numba matches Python results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

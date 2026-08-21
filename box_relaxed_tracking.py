"""
Progressive box_size relaxation with per-block velocity gates.

1. Build the candidate graph once (up to box_size_max).
2. At box_size_min, histogram seed-valid candidates and detect the velocity peak
   per 4-frame block; gate to peak ± σ (fixed σ across runs).
3. Seed pass starts at box_size_min; if too tight (0 tracks), advance to the next
   larger box_size without retrying failed sizes. Relaxation continues from there.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, replace

import numpy as np

from .assign import PathCandidate
from .link_block import LinkParams
from .load_data import FramePositions, block_starts_for_nframes, load_all_frames
from .pdf_relaxed_tracking import pass_output_slug, write_filtered_stm
from .tracking_state import (
    VELOCITY_VARIABLES,
    UsedDetections,
    velocity_component,
)


@dataclass
class BoxRelaxedTrackingConfig:
    max_stereo_error: float = 0.1
    box_size_min: float = 0.15
    box_size_max: float = 0.5
    box_size_step: float = 0.05
    velocity_variable: str = "vx"
    velocity_target: float | None = None  # optional override; None = detect peak from histogram
    velocity_sigma: float = 1.0  # fixed ±σ around detected peak (same for all runs)
    velocity_bin_count: int = 0  # 0 = auto (coarse histogram at tight box)
    velocity_band_margin: float = 2.0  # max multiplier on σ at final relax pass (seed = 1×σ)
    dt: float = 1e-3
    workers: int = 4
    max_blocks: int = 0
    min_track_length: int = 3
    rep_rate: float = 10.0
    position_unit: str = "mm"
    link_params: LinkParams = field(default_factory=LinkParams)
    filtered_stm_name: str = "rays_out_box_filtered.h5"
    tracks_name: str = "tracks_box_relaxed.h5part"
    summary_name: str = "box_relaxed_tracking_summary.json"
    write_tracks_each_pass: bool = True
    pass_filtered_stm_template: str = "rays_out_box_pass_{pass_slug}.h5"
    pass_tracks_template: str = "tracks_box_pass_{pass_slug}.h5part"


@dataclass
class BlockReference:
    """Per-4-frame-block velocity peak detected at box_size_min."""

    block_start: int
    velocity_peak: float | None = None
    velocity_bands: list[tuple[float, float]] = field(default_factory=list)
    dominant_velocity: float | None = None
    dominant_bin_lo: float | None = None
    dominant_bin_hi: float | None = None
    dominant_bin_count: int = 0
    candidate_count: int = 0

    def bands_at_margin(self, sigma: float, margin: float) -> list[tuple[float, float]]:
        if self.velocity_peak is None:
            return []
        lo, hi = _peak_velocity_band(self.velocity_peak, sigma, margin)
        return [(lo, hi)]

    def to_dict(self) -> dict:
        return {
            "block_start": self.block_start,
            "velocity_peak": self.velocity_peak,
            "velocity_bands": self.velocity_bands,
            "dominant_velocity": self.dominant_velocity,
            "dominant_bin_lo": self.dominant_bin_lo,
            "dominant_bin_hi": self.dominant_bin_hi,
            "dominant_bin_count": self.dominant_bin_count,
            "candidate_count": self.candidate_count,
        }


def box_size_thresholds(cfg: BoxRelaxedTrackingConfig) -> list[float]:
    if cfg.box_size_step <= 0:
        raise ValueError("box_size_step must be positive")
    if cfg.box_size_max < cfg.box_size_min:
        raise ValueError("box_size_max must be >= box_size_min")
    values: list[float] = []
    size = cfg.box_size_min
    while size <= cfg.box_size_max + 1e-12:
        values.append(round(size, 6))
        size += cfg.box_size_step
    return values


def _auto_velocity_bins(n_samples: int) -> int:
    """Coarse histogram — fewer bins make the dominant mode easier to identify."""
    return int(max(6, min(24, n_samples // 8)))


def _representative_velocity(ap, variable: str) -> float | None:
    if ap.segment_velocities.size == 0:
        return None
    comps = [velocity_component(v, variable) for v in ap.segment_velocities]
    return float(np.mean(comps))


def _dominant_bin_mask(values: np.ndarray, n_bins: int) -> tuple[float, float, np.ndarray]:
    if values.size == 0:
        raise ValueError("values must not be empty")
    if values.size == 1:
        v = float(values[0])
        return v, v, np.ones(1, dtype=bool)
    lo_edge = float(np.min(values))
    hi_edge = float(np.max(values))
    if abs(hi_edge - lo_edge) < 1e-15:
        return lo_edge, hi_edge, np.ones(values.size, dtype=bool)
    n_bins = min(n_bins, max(2, values.size // 2))
    counts, edges = np.histogram(values, bins=n_bins, range=(lo_edge, hi_edge))
    if counts.size == 0 or int(counts.max()) == 0:
        return lo_edge, hi_edge, np.ones(values.size, dtype=bool)
    i = int(np.argmax(counts))
    bin_lo, bin_hi = float(edges[i]), float(edges[i + 1])
    if i == len(counts) - 1:
        mask = (values >= bin_lo) & (values <= bin_hi)
    else:
        mask = (values >= bin_lo) & (values < bin_hi)
    if not np.any(mask):
        mask = np.ones(values.size, dtype=bool)
        bin_lo, bin_hi = lo_edge, hi_edge
    return bin_lo, bin_hi, mask


def _peak_velocity_band(
    peak: float,
    sigma: float,
    margin: float,
) -> tuple[float, float]:
    half = max(float(sigma), 1e-6) * float(margin)
    return float(peak - half), float(peak + half)


def velocity_margin_for_pass(
    pass_index: int,
    n_passes: int,
    max_margin: float,
) -> float:
    """Seed (index 0) uses 1×σ; relax passes ramp linearly to max_margin × σ."""
    if pass_index <= 0 or n_passes <= 1 or max_margin <= 1.0:
        return 1.0
    n_relax = n_passes - 1
    relax_step = pass_index
    return 1.0 + (max_margin - 1.0) * relax_step / n_relax


def detect_velocity_peak(
    paths: list,
    *,
    variable: str,
    velocity_bins: int = 0,
    velocity_sigma: float = 1.0,
    velocity_target: float | None = None,
) -> tuple[float | None, dict]:
    """
    Histogram seed-valid candidates at the tight box; return peak center.

    Peak is the busiest histogram-bin center unless ``velocity_target`` overrides.
    """
    paired: list[tuple[float, object]] = []
    for ap in paths:
        v = _representative_velocity(ap, variable)
        if v is not None and np.isfinite(v):
            paired.append((v, ap))
    meta: dict = {
        "velocity_peak": None,
        "dominant_velocity": None,
        "dominant_bin_lo": None,
        "dominant_bin_hi": None,
        "dominant_bin_count": 0,
        "candidate_count": len(paired),
        "velocity_bins_used": 0,
        "velocity_sigma": velocity_sigma,
        "velocity_target_override": velocity_target,
    }
    if not paired:
        return None, meta

    values = np.array([p[0] for p in paired], dtype=np.float64)
    if velocity_target is not None:
        peak = float(velocity_target)
        meta["velocity_peak"] = peak
        meta["dominant_velocity"] = peak
        return peak, meta

    n_bins = velocity_bins if velocity_bins > 0 else _auto_velocity_bins(values.size)
    meta["velocity_bins_used"] = n_bins
    bin_lo, bin_hi, mask = _dominant_bin_mask(values, n_bins)
    peak = float(0.5 * (bin_lo + bin_hi))
    meta.update(
        {
            "velocity_peak": peak,
            "dominant_velocity": peak,
            "dominant_bin_lo": bin_lo,
            "dominant_bin_hi": bin_hi,
            "dominant_bin_count": int(np.sum(mask)),
        }
    )
    return peak, meta


def _path_seed_error(
    frames: list[FramePositions],
    block_graph,
    path: PathCandidate,
) -> float:
    if not path.indices:
        return float("inf")
    layer = int(path.start_layer)
    local_idx = int(path.indices[0])
    if layer < 0 or layer >= len(block_graph.global_maps):
        return float("inf")
    gmap = block_graph.global_maps[layer]
    if local_idx < 0 or local_idx >= len(gmap):
        return float("inf")
    global_idx = int(gmap[local_idx])
    fi = int(block_graph.block.frame_indices[layer])
    fr = frames[fi]
    if fr.error is None:
        return 0.0
    if global_idx < 0 or global_idx >= fr.n:
        return float("inf")
    return float(fr.error[global_idx])


def paths_at_tight_box(
    frames: list[FramePositions],
    block_graph,
    paths: list,
    *,
    max_stereo_error: float,
    max_hop: float,
) -> list:
    """Candidates at the initial (tight) box_size whose seed passes the STM error filter."""
    kept: list = []
    for ap in paths:
        if ap.max_hop_distance > max_hop + 1e-12:
            continue
        if _path_seed_error(frames, block_graph, ap.path) > max_stereo_error + 1e-12:
            continue
        kept.append(ap)
    return kept


def build_dominant_mode_references(
    block_graphs: dict,
    frames: list[FramePositions],
    cfg: BoxRelaxedTrackingConfig,
) -> dict[int, BlockReference]:
    """Per-block velocity peak from candidates at box_size_min."""
    refs: dict[int, BlockReference] = {}
    for start, block_graph in block_graphs.items():
        tight_paths = paths_at_tight_box(
            frames,
            block_graph,
            block_graph.paths,
            max_stereo_error=cfg.max_stereo_error,
            max_hop=cfg.box_size_min,
        )
        peak, meta = detect_velocity_peak(
            tight_paths,
            variable=cfg.velocity_variable,
            velocity_bins=cfg.velocity_bin_count,
            velocity_sigma=cfg.velocity_sigma,
            velocity_target=cfg.velocity_target,
        )
        seed_band = (
            list(BlockReference(block_start=0, velocity_peak=peak).bands_at_margin(
                cfg.velocity_sigma, 1.0
            ))
            if peak is not None
            else []
        )
        refs[int(start)] = BlockReference(
            block_start=int(start),
            velocity_peak=peak,
            velocity_bands=seed_band,
            dominant_velocity=meta.get("dominant_velocity"),
            dominant_bin_lo=meta.get("dominant_bin_lo"),
            dominant_bin_hi=meta.get("dominant_bin_hi"),
            dominant_bin_count=int(meta.get("dominant_bin_count", 0)),
            candidate_count=int(meta.get("candidate_count", 0)),
        )
    return refs


def _block_refs_at_margin(
    mode_refs: dict[int, BlockReference],
    sigma: float,
    margin: float,
) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for start, ref in mode_refs.items():
        d = ref.to_dict()
        d["velocity_bands"] = ref.bands_at_margin(sigma, margin)
        out[int(start)] = d
    return out


def process_box_tracking_pass(
    frames: list[FramePositions],
    block_graphs: dict,
    used: UsedDetections,
    cfg: BoxRelaxedTrackingConfig,
    box_size: float,
    *,
    apply_velocity_gate: bool,
    mode_refs: dict[int, BlockReference] | None = None,
    velocity_margin: float = 1.0,
    workers: int = 1,
    verbose: bool = True,
    pass_label: str = "pass",
    accepted_store=None,
) -> dict:
    from .relaxation_engine import PassCriteria, run_relaxation_pass

    starts = block_starts_for_nframes(len(frames), max_blocks=cfg.max_blocks)
    block_ref_dict = (
        _block_refs_at_margin(mode_refs, cfg.velocity_sigma, velocity_margin)
        if apply_velocity_gate and mode_refs
        else {}
    )
    criteria = PassCriteria(
        max_error=cfg.max_stereo_error,
        max_box_size=box_size,
        velocity_variable=cfg.velocity_variable,
        apply_velocity_gate=apply_velocity_gate,
        block_refs=block_ref_dict or None,
    )
    _, stats = run_relaxation_pass(
        block_graphs,
        frames,
        used,
        criteria,
        cfg.dt,
        workers=workers,
        verbose=verbose,
        pass_label=pass_label,
        log_prefix="box",
        max_blocks=cfg.max_blocks,
        accepted_store=accepted_store,
    )
    stats["box_size"] = box_size
    stats["velocity_margin"] = velocity_margin
    stats["velocity_rejected"] = stats.get("filtered_rejected", 0)
    stats["apply_velocity_gate"] = apply_velocity_gate
    if apply_velocity_gate and mode_refs:
        stats["block_references"] = [
            {
                **mode_refs[start].to_dict(),
                "velocity_bands": mode_refs[start].bands_at_margin(
                    cfg.velocity_sigma, velocity_margin
                ),
            }
            for start in starts
            if start in mode_refs and mode_refs[start].velocity_peak is not None
        ]
    else:
        stats["block_references"] = []
    return stats


def write_box_pass_tracks(
    stm_path: str,
    output_dir: str,
    frames: list[FramePositions],
    used: UsedDetections,
    cfg: BoxRelaxedTrackingConfig,
    block_graphs: dict,
    accepted_store,
    *,
    pass_label: str,
    verbose: bool = True,
) -> dict:
    from .relaxation_engine import write_accepted_paths_h5part

    slug = pass_output_slug(pass_label)
    filtered_stm = os.path.join(
        output_dir, cfg.pass_filtered_stm_template.format(pass_slug=slug)
    )
    tracks_out = os.path.join(output_dir, cfg.pass_tracks_template.format(pass_slug=slug))
    matched_masks = used.matched_masks(frames)
    n_written = write_filtered_stm(stm_path, filtered_stm, matched_masks, frames=frames)
    if verbose:
        print(
            f"[box] pass {pass_label}: matched STM → {filtered_stm} ({n_written} detections)",
            flush=True,
        )
        print(f"[box] pass {pass_label}: writing tracks → {tracks_out}", flush=True)
    track_result = write_accepted_paths_h5part(
        block_graphs,
        accepted_store,
        output_dir,
        tracks_out,
        min_track_length=cfg.min_track_length,
        dt=cfg.dt,
        rep_rate=cfg.rep_rate,
        position_unit=cfg.position_unit,
        verbose=verbose,
    )
    return {
        "pass_label": pass_label,
        "pass_slug": slug,
        "filtered_stm": filtered_stm,
        "tracks_h5part": tracks_out,
        "matched_detections": n_written,
        "tracking": track_result,
    }


def run_box_relaxed_tracking(
    stm_path: str,
    output_dir: str,
    cfg: BoxRelaxedTrackingConfig | None = None,
    *,
    verbose: bool = True,
) -> dict:
    cfg = cfg or BoxRelaxedTrackingConfig()
    stm_path = os.path.abspath(stm_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if cfg.velocity_variable not in VELOCITY_VARIABLES:
        raise ValueError(f"velocity_variable must be one of {VELOCITY_VARIABLES}")

    box_sizes = box_size_thresholds(cfg)
    if not box_sizes:
        raise ValueError("No box_size values to process")

    if verbose:
        print(f"[box] Input STM: {stm_path}", flush=True)
        print(
            f"[box] initial hop {box_sizes[0]:.4f} mm, relax to {box_sizes[-1]:.4f} mm "
            f"(step {cfg.box_size_step}); stereo error <= {cfg.max_stereo_error}; "
            f"velocity gate on {cfg.velocity_variable}",
            flush=True,
        )
        if cfg.link_params.use_bidirectional_graph:
            print("[box] bidirectional graph: ON (consensus 4-frame paths only)", flush=True)
        print(
            f"[box] velocity gate: detect peak at hop={cfg.box_size_min:.4f} mm, "
            f"±{cfg.velocity_sigma} {cfg.velocity_variable}, "
            f"widen to ±{cfg.velocity_band_margin}×σ on last relax pass",
            flush=True,
        )
        print(f"[box] workers: {cfg.workers}", flush=True)

    frames = load_all_frames(stm_path, with_error=True)
    if not frames:
        raise ValueError("STM file has no frames")
    if frames[0].error is None:
        raise ValueError("STM xyze dataset must include error row (4 x N)")

    pool_available = sum(
        int(np.sum(frames[fi].error <= cfg.max_stereo_error))
        for fi in range(len(frames))
    )
    if pool_available == 0:
        raise ValueError(
            f"No particles with error <= {cfg.max_stereo_error}. "
            "Raise max_stereo_error or use a higher-threshold STM file."
        )
    if verbose:
        print(f"[box] particle pool: {pool_available} detections", flush=True)

    from .relaxation_engine import AcceptedPathStore, build_all_block_graphs, write_accepted_paths_h5part

    graph_params = replace(cfg.link_params, box_size_track=cfg.box_size_max)
    starts = block_starts_for_nframes(len(frames), max_blocks=cfg.max_blocks)
    block_graphs = build_all_block_graphs(
        frames,
        starts,
        graph_params,
        cfg.max_stereo_error,
        cfg.dt,
        workers=cfg.workers,
        verbose=verbose,
        log_prefix="box",
    )
    mode_refs = build_dominant_mode_references(block_graphs, frames, cfg)
    if verbose:
        n_with_peak = sum(1 for r in mode_refs.values() if r.velocity_peak is not None)
        print(
            f"[box] detected velocity peaks at hop={cfg.box_size_min:.4f} mm: "
            f"{n_with_peak}/{len(mode_refs)} blocks",
            flush=True,
        )
        peaks = [r.velocity_peak for r in mode_refs.values() if r.velocity_peak is not None]
        if peaks:
            print(
                f"[box]   {cfg.velocity_variable} peak range "
                f"[{min(peaks):.3f}, {max(peaks):.3f}], "
                f"median={float(np.median(peaks)):.3f}",
                flush=True,
            )
            sample = next(r for r in mode_refs.values() if r.velocity_peak is not None)
            lo, hi = sample.bands_at_margin(cfg.velocity_sigma, 1.0)[0]
            print(
                f"[box]   seed window example: [{lo:.3f}, {hi:.3f}] "
                f"(peak={sample.velocity_peak:.3f}, "
                f"{sample.candidate_count} tight-box candidates)",
                flush=True,
            )

    used = UsedDetections()
    accepted_store = AcceptedPathStore()
    pass_stats: list[dict] = []
    use_velocity_gate = cfg.velocity_sigma > 0

    seed_start_idx: int | None = None
    seed_stats: dict | None = None

    for idx, box in enumerate(box_sizes):
        margin = velocity_margin_for_pass(0, len(box_sizes), cfg.velocity_band_margin)
        pass_label = "seed" if idx == 0 else f"hop<={box:.4f}"
        stats = process_box_tracking_pass(
            frames,
            block_graphs,
            used,
            cfg,
            box,
            apply_velocity_gate=use_velocity_gate,
            mode_refs=mode_refs if use_velocity_gate else None,
            velocity_margin=margin,
            workers=cfg.workers,
            verbose=verbose,
            pass_label=pass_label,
            accepted_store=accepted_store,
        )
        if stats["tracks_assigned"] > 0:
            seed_start_idx = idx
            seed_stats = stats
            seed_stats["pass_label"] = "seed"
            seed_stats["pass_role"] = "seed"
            if verbose and idx > 0:
                print(
                    f"[box] seed at hop={box:.4f} mm: {stats['tracks_assigned']} tracks "
                    f"(skipped {idx} tighter box_size step(s))",
                    flush=True,
                )
            if cfg.write_tracks_each_pass:
                pass_outputs = write_box_pass_tracks(
                    stm_path,
                    output_dir,
                    frames,
                    used,
                    cfg,
                    block_graphs,
                    accepted_store,
                    pass_label="seed",
                    verbose=verbose,
                )
                seed_stats.update(
                    {
                        "filtered_stm": pass_outputs["filtered_stm"],
                        "tracks_h5part": pass_outputs["tracks_h5part"],
                        "matched_detections": pass_outputs["matched_detections"],
                        "tracking": pass_outputs["tracking"],
                    }
                )
            pass_stats.append(seed_stats)
            break

        if verbose:
            print(
                f"[box] hop={box:.4f} mm: 0 tracks (too tight), "
                f"advancing to next box_size",
                flush=True,
            )
        stats["skipped"] = True
        stats["pass_role"] = "seed_probe"
        pass_stats.append(stats)

    if seed_start_idx is None or seed_stats is None:
        raise ValueError(
            f"No tracks assigned at any box_size in "
            f"[{box_sizes[0]:.4f}, {box_sizes[-1]:.4f}] mm. "
            "Raise box_size_max, loosen max_stereo_error, or widen velocity gate."
        )

    n_remaining = len(box_sizes) - seed_start_idx
    threshold = box_sizes[seed_start_idx]
    for relax_idx, next_box in enumerate(box_sizes[seed_start_idx + 1 :], start=1):
        pass_label = f"relax<={next_box:.4f}"
        relax_margin = velocity_margin_for_pass(
            relax_idx, n_remaining, cfg.velocity_band_margin
        )
        stats = process_box_tracking_pass(
            frames,
            block_graphs,
            used,
            cfg,
            next_box,
            apply_velocity_gate=use_velocity_gate,
            mode_refs=mode_refs if use_velocity_gate else None,
            velocity_margin=relax_margin,
            workers=cfg.workers,
            verbose=verbose,
            pass_label=pass_label,
            accepted_store=accepted_store,
        )
        stats["box_size_lo"] = threshold
        stats["box_size_hi"] = next_box
        if cfg.write_tracks_each_pass:
            pass_outputs = write_box_pass_tracks(
                stm_path,
                output_dir,
                frames,
                used,
                cfg,
                block_graphs,
                accepted_store,
                pass_label=pass_label,
                verbose=verbose,
            )
            stats.update(
                {
                    "filtered_stm": pass_outputs["filtered_stm"],
                    "tracks_h5part": pass_outputs["tracks_h5part"],
                    "matched_detections": pass_outputs["matched_detections"],
                    "tracking": pass_outputs["tracking"],
                }
            )
        pass_stats.append(stats)
        if verbose:
            print(
                f"[box] relax ({threshold:.4f}, {next_box:.4f}]: "
                f"+{stats['tracks_assigned']} tracks, "
                f"{stats['velocity_rejected']} velocity-rejected, "
                f"velocity window {relax_margin:.2f}×σ, "
                f"matched total={stats['matched_total']}",
                flush=True,
            )
        threshold = next_box

    matched_masks = used.matched_masks(frames)
    filtered_stm = os.path.join(output_dir, cfg.filtered_stm_name)
    n_written = write_filtered_stm(stm_path, filtered_stm, matched_masks, frames=frames)
    tracks_out = os.path.join(output_dir, cfg.tracks_name)

    if verbose:
        print(f"[box] matched STM: {filtered_stm} ({n_written} detections)", flush=True)
        print(f"[box] writing tracks from accepted paths → {tracks_out}", flush=True)

    track_result = write_accepted_paths_h5part(
        block_graphs,
        accepted_store,
        output_dir,
        tracks_out,
        min_track_length=cfg.min_track_length,
        dt=cfg.dt,
        rep_rate=cfg.rep_rate,
        position_unit=cfg.position_unit,
        verbose=verbose,
    )

    summary = {
        "input_stm": stm_path,
        "output_dir": output_dir,
        "filtered_stm": filtered_stm,
        "tracks_h5part": tracks_out,
        "config": {
            **{f.name: getattr(cfg, f.name) for f in fields(cfg) if f.name != "link_params"},
            "link_params": asdict(cfg.link_params),
        },
        "particle_pool_detections": pool_available,
        "box_sizes": box_sizes,
        "seed_box_size": box_sizes[seed_start_idx],
        "seed_box_skipped": seed_start_idx,
        "dominant_velocity_bins": [r.to_dict() for r in mode_refs.values()],
        "passes": pass_stats,
        "matched_detections_total": n_written,
        "tracking": track_result,
    }
    summary_path = os.path.join(output_dir, cfg.summary_name)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    summary["summary_json"] = summary_path

    if verbose:
        print(f"[box] summary: {summary_path}", flush=True)
        print(
            f"[box] done: {n_written} matched detections, "
            f"{track_result.get('total_tracks', 0)} tracks",
            flush=True,
        )
    return summary

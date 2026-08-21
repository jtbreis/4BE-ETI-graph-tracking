"""
Iterative PDF-gated particle acceptance from a high-threshold STM file.

1. Build candidate path graphs once at the maximum error ceiling.
2. Build a reference PDF of a kinematic variable (e.g. v_x) from seed-pass tracks.
3. Relax the error ceiling in steps; at each step drop graph edges that fail the
   tighter error / PDF criteria, then re-assign on still-unmatched detections.
4. Write the union of all matched detections and final tracks.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields

import h5py
import numpy as np

from .link_block import LinkParams
from .load_data import (
    FramePositions,
    block_starts_for_nframes,
    get_nframes,
    load_all_frames,
)
from .tracking_state import (
    VELOCITY_VARIABLES,
    UsedDetections,
    VelocityPdf,
    segment_velocities_from_paths,
    velocity_component,
)


@dataclass
class PdfTrackingConfig:
    initial_max_error: float = 0.05
    max_error: float = 0.2
    error_step: float = 0.01
    pdf_variable: str = "vx"
    pdf_percentile_lo: float = 1.0
    pdf_percentile_hi: float = 99.0
    dt: float = 1e-3
    box_size_track: float = 0.2
    workers: int = 4
    max_blocks: int = 0
    min_track_length: int = 3
    rep_rate: float = 10.0
    position_unit: str = "mm"
    link_params: LinkParams = field(default_factory=LinkParams)
    filtered_stm_name: str = "rays_out_pdf_filtered.h5"
    tracks_name: str = "tracks_pdf_relaxed.h5part"
    summary_name: str = "pdf_tracking_summary.json"
    write_tracks_each_pass: bool = True
    pass_filtered_stm_template: str = "rays_out_pdf_pass_{pass_slug}.h5"
    pass_tracks_template: str = "tracks_pdf_pass_{pass_slug}.h5part"


def process_tracking_pass(
    frames: list[FramePositions],
    block_graphs: dict,
    used: UsedDetections,
    error_threshold: float,
    dt: float,
    *,
    pdf: VelocityPdf | None = None,
    max_blocks: int = 0,
    workers: int = 1,
    verbose: bool = True,
    pass_label: str = "pass",
    accepted_store=None,
) -> tuple[np.ndarray, dict]:
    """Filter pre-built block graphs and assign tracks for one error threshold."""
    from .relaxation_engine import AcceptedPathStore, PassCriteria, run_relaxation_pass

    criteria = PassCriteria(
        max_error=error_threshold,
        global_pdf=pdf,
        apply_velocity_gate=pdf is not None,
    )
    velocities, stats = run_relaxation_pass(
        block_graphs,
        frames,
        used,
        criteria,
        dt,
        workers=workers,
        verbose=verbose,
        pass_label=pass_label,
        log_prefix="pdf",
        max_blocks=max_blocks,
        accepted_store=accepted_store,
    )
    stats["error_threshold"] = error_threshold
    stats["pdf_rejected"] = stats.get("filtered_rejected", 0)
    return velocities, stats


def write_filtered_stm(
    src_path: str,
    dst_path: str,
    masks: list[np.ndarray],
    *,
    frames: list[FramePositions] | None = None,
) -> int:
    """Write a filtered copy of an STM HDF5 file. Returns total particles written."""
    src_path = os.path.abspath(src_path)
    dst_path = os.path.abspath(dst_path)
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)

    if frames is None:
        frames = load_all_frames(src_path, with_error=True)

    total = 0
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for ak, av in src.attrs.items():
            dst.attrs[ak] = av
        for fi, fr in enumerate(frames):
            mask = np.asarray(masks[fi], dtype=bool)
            if mask.size != fr.n:
                raise ValueError(f"Mask length mismatch at frame {fi}")
            key = f"frame{fi:05d}"
            grp = dst.create_group(key)
            idx = np.flatnonzero(mask)
            n = int(idx.size)
            total += n
            xyze = np.zeros((4, n), dtype=np.float64)
            if n:
                xyze[0] = fr.x[idx]
                xyze[1] = fr.y[idx]
                xyze[2] = fr.z[idx]
                if fr.error is not None:
                    xyze[3] = fr.error[idx]
            grp.create_dataset("xyze", data=xyze)
            if key in src and "camrayids" in src[key]:
                cam = np.array(src[key]["camrayids"])
                if cam.ndim == 2 and cam.shape[1] == fr.n:
                    grp.create_dataset("camrayids", data=cam[:, idx])
            for extra in ("diameter", "intensity", "mass"):
                if key in src and extra in src[key]:
                    data = np.array(src[key][extra])
                    if data.ndim == 2 and data.shape[0] == fr.n:
                        grp.create_dataset(extra, data=data[idx])
                    elif data.ndim == 2 and data.shape[1] == fr.n:
                        grp.create_dataset(extra, data=data[:, idx])
    return total


def pass_output_slug(pass_label: str) -> str:
    """Filesystem-safe slug from a pass label (seed, relax<=0.0600, …)."""
    label = pass_label.strip().lower()
    if label == "seed":
        return "seed"
    if label.startswith("relax<="):
        value = label.split("<=", 1)[1].strip()
        try:
            return f"relax_{float(value):.4f}"
        except ValueError:
            pass
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label)


def write_pass_tracks(
    stm_path: str,
    output_dir: str,
    frames: list[FramePositions],
    used: UsedDetections,
    cfg: PdfTrackingConfig,
    block_graphs: dict,
    accepted_store,
    *,
    pass_label: str,
    verbose: bool = True,
) -> dict:
    """Write filtered STM + h5part from accumulated accepted paths (no re-linking)."""
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
            f"[pdf] pass {pass_label}: matched STM → {filtered_stm} ({n_written} detections)",
            flush=True,
        )
        print(f"[pdf] pass {pass_label}: writing tracks → {tracks_out}", flush=True)
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


def stm_max_error_attr(h5path: str, default: float) -> float:
    try:
        with h5py.File(h5path, "r") as f:
            if "maxdistance" in f.attrs:
                return float(f.attrs["maxdistance"])
    except OSError:
        pass
    return default


def run_pdf_relaxed_tracking(
    stm_path: str,
    output_dir: str,
    cfg: PdfTrackingConfig | None = None,
    *,
    verbose: bool = True,
) -> dict:
    cfg = cfg or PdfTrackingConfig()
    stm_path = os.path.abspath(stm_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if cfg.pdf_variable not in VELOCITY_VARIABLES:
        raise ValueError(f"pdf_variable must be one of {VELOCITY_VARIABLES}")

    if cfg.max_error <= 0:
        cfg.max_error = stm_max_error_attr(stm_path, cfg.max_error)

    cfg.link_params.box_size_track = cfg.box_size_track

    if verbose:
        print(f"[pdf] Input STM: {stm_path}", flush=True)
        print(
            f"[pdf] seed error<={cfg.initial_max_error}, relax to {cfg.max_error} "
            f"step {cfg.error_step}; PDF variable={cfg.pdf_variable}",
            flush=True,
        )
        if cfg.link_params.use_bidirectional_graph:
            print("[pdf] bidirectional graph: ON (consensus 4-frame paths only)", flush=True)
        if not cfg.link_params.allow_3_frame:
            print("[pdf] 4-frame tracks only (3-frame paths disabled)", flush=True)
        print(f"[pdf] workers: {cfg.workers}", flush=True)

    frames = load_all_frames(stm_path, with_error=True)
    if not frames:
        raise ValueError("STM file has no frames")
    if frames[0].error is None:
        raise ValueError("STM xyze dataset must include error row (4 x N)")

    seed_available = sum(
        int(np.sum(frames[fi].error <= cfg.initial_max_error))
        for fi in range(len(frames))
    )
    if seed_available == 0:
        raise ValueError(
            f"No particles with error <= {cfg.initial_max_error}. "
            "Lower initial_max_error or use an STM run with a higher maxdistance."
        )

    if verbose:
        print(f"[pdf] seed pool (unmatched): {seed_available} detections", flush=True)

    from dataclasses import replace as dc_replace

    from .relaxation_engine import AcceptedPathStore, build_all_block_graphs, write_accepted_paths_h5part

    graph_params = dc_replace(cfg.link_params, box_size_track=cfg.box_size_track)
    starts = block_starts_for_nframes(len(frames), max_blocks=cfg.max_blocks)
    block_graphs = build_all_block_graphs(
        frames,
        starts,
        graph_params,
        cfg.max_error,
        cfg.dt,
        workers=cfg.workers,
        verbose=verbose,
        log_prefix="pdf",
    )

    used = UsedDetections()
    accepted_store = AcceptedPathStore()
    seed_velocities, seed_stats = process_tracking_pass(
        frames,
        block_graphs,
        used,
        cfg.initial_max_error,
        cfg.dt,
        pdf=None,
        max_blocks=cfg.max_blocks,
        workers=cfg.workers,
        verbose=verbose,
        pass_label="seed",
        accepted_store=accepted_store,
    )
    if seed_velocities.size == 0:
        raise ValueError("Seed tracking produced no velocity segments for PDF reference")
    if cfg.write_tracks_each_pass:
        pass_outputs = write_pass_tracks(
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

    comps = np.array([velocity_component(v, cfg.pdf_variable) for v in seed_velocities])
    pdf = VelocityPdf.from_values(
        comps,
        cfg.pdf_variable,
        cfg.pdf_percentile_lo,
        cfg.pdf_percentile_hi,
    )
    if verbose:
        print(
            f"[pdf] reference PDF ({cfg.pdf_variable}): "
            f"n={pdf.values.size}, range=[{pdf.lo:.4f}, {pdf.hi:.4f}], "
            f"mean={pdf.mean:.4f}, std={pdf.std:.4f}",
            flush=True,
        )

    relax_stats: list[dict] = [seed_stats]
    threshold = cfg.initial_max_error
    while threshold + 1e-12 < cfg.max_error:
        next_threshold = min(cfg.max_error, threshold + cfg.error_step)
        pass_label = f"relax<={next_threshold:.4f}"
        _, stats = process_tracking_pass(
            frames,
            block_graphs,
            used,
            next_threshold,
            cfg.dt,
            pdf=pdf,
            max_blocks=cfg.max_blocks,
            workers=cfg.workers,
            verbose=verbose,
            pass_label=pass_label,
            accepted_store=accepted_store,
        )
        stats["threshold_lo"] = threshold
        stats["threshold_hi"] = next_threshold
        if cfg.write_tracks_each_pass:
            pass_outputs = write_pass_tracks(
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
        relax_stats.append(stats)
        if verbose:
            print(
                f"[pdf] relax ({threshold:.4f}, {next_threshold:.4f}]: "
                f"+{stats['tracks_assigned']} tracks, "
                f"{stats['pdf_rejected']} PDF-rejected, "
                f"matched total={stats['matched_total']}",
                flush=True,
            )
        threshold = next_threshold

    matched_masks = used.matched_masks(frames)
    filtered_stm = os.path.join(output_dir, cfg.filtered_stm_name)
    n_written = write_filtered_stm(stm_path, filtered_stm, matched_masks, frames=frames)
    tracks_out = os.path.join(output_dir, cfg.tracks_name)

    if verbose:
        print(f"[pdf] matched STM: {filtered_stm} ({n_written} detections)", flush=True)
        print(f"[pdf] writing tracks from accepted paths → {tracks_out}", flush=True)

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
        "seed_pool_detections": seed_available,
        "pdf": pdf.to_dict(),
        "passes": relax_stats,
        "matched_detections_total": n_written,
        "tracking": track_result,
    }
    summary_path = os.path.join(output_dir, cfg.summary_name)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    summary["summary_json"] = summary_path

    if verbose:
        print(f"[pdf] summary: {summary_path}", flush=True)
        print(
            f"[pdf] done: {n_written} matched detections, "
            f"{track_result.get('total_tracks', 0)} tracks",
            flush=True,
        )
    return summary

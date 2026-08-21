"""
Sensitivity of graph tracking to stereomatching error threshold.

For each error ceiling in a range, filter particles in-memory (error <= threshold)
from the input STM, run graph-based 4-frame tracking, and write only the per-threshold
tracks files plus a summary JSON. The input rays_out_cpp.h5 is never rewritten.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields

import numpy as np

from .link_block import LinkParams
from .load_data import get_nframes, load_all_errors
from .pdf_relaxed_tracking import stm_max_error_attr


@dataclass
class ErrorSensitivityConfig:
    error_min: float = 0.05
    error_max: float = 0.2
    error_step: float = 0.01
    dt: float = 1e-3
    box_size_track: float = 0.2
    workers: int = 4
    max_blocks: int = 0
    min_track_length: int = 3
    rep_rate: float = 10.0
    position_unit: str = "mm"
    link_params: LinkParams = field(default_factory=LinkParams)
    tracks_template: str = "tracks_error_{error:.4f}.h5part"
    summary_name: str = "error_sensitivity_summary.json"


def error_thresholds(cfg: ErrorSensitivityConfig) -> list[float]:
    if cfg.error_step <= 0:
        raise ValueError("error_step must be positive")
    if cfg.error_max < cfg.error_min:
        raise ValueError("error_max must be >= error_min")
    values: list[float] = []
    threshold = cfg.error_min
    while threshold <= cfg.error_max + 1e-12:
        values.append(round(threshold, 6))
        threshold += cfg.error_step
    return values


def error_masks_for_threshold(
    errors: list[np.ndarray],
    error_threshold: float,
) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for err in errors:
        masks.append(np.asarray(err <= error_threshold, dtype=bool))
    return masks


def run_error_sensitivity_tracking(
    stm_path: str,
    output_dir: str,
    cfg: ErrorSensitivityConfig | None = None,
    *,
    verbose: bool = True,
) -> dict:
    from .run_tracking import run_graph_tracking

    cfg = cfg or ErrorSensitivityConfig()
    stm_path = os.path.abspath(stm_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if cfg.error_max <= 0:
        cfg.error_max = stm_max_error_attr(stm_path, cfg.error_max)
    if cfg.error_max < cfg.error_min:
        raise ValueError(
            f"error_max ({cfg.error_max}) must be >= error_min ({cfg.error_min})"
        )

    cfg.link_params.box_size_track = cfg.box_size_track
    thresholds = error_thresholds(cfg)
    if not thresholds:
        raise ValueError("No error thresholds to process")

    nframes = get_nframes(stm_path)
    if nframes == 0:
        raise ValueError("STM file has no frames")

    if verbose:
        print(f"[error-sensitivity] Input STM: {stm_path}", flush=True)
        print(
            f"[error-sensitivity] thresholds {thresholds[0]:.4f}..{thresholds[-1]:.4f} "
            f"step {cfg.error_step} ({len(thresholds)} value(s))",
            flush=True,
        )
        print(
            f"[error-sensitivity] workers: {cfg.workers} "
            f"(filter in-memory from input STM; no filtered STM files written)",
            flush=True,
        )

    errors = load_all_errors(stm_path)

    runs: list[dict] = []
    for idx, threshold in enumerate(thresholds, start=1):
        if verbose:
            print(
                f"[error-sensitivity] [{idx}/{len(thresholds)}] error <= {threshold:.4f}",
                flush=True,
            )
        masks = error_masks_for_threshold(errors, threshold)
        n_available = int(sum(int(np.sum(mask)) for mask in masks))
        if n_available == 0:
            if verbose:
                print(
                    f"[error-sensitivity]   skip: no particles with error <= {threshold:.4f}",
                    flush=True,
                )
            runs.append(
                {
                    "error_threshold": threshold,
                    "available_detections": 0,
                    "skipped": True,
                }
            )
            continue

        tracks_out = os.path.join(output_dir, cfg.tracks_template.format(error=threshold))
        if verbose:
            print(
                f"[error-sensitivity]   {n_available} detections after filter "
                f"(from {stm_path})",
                flush=True,
            )
            print(f"[error-sensitivity]   graph tracking → {tracks_out}", flush=True)

        track_result = run_graph_tracking(
            stm_path,
            output_dir,
            output=tracks_out,
            max_blocks=cfg.max_blocks,
            workers=cfg.workers,
            link_params=cfg.link_params,
            min_track_length=cfg.min_track_length,
            dt=cfg.dt,
            rep_rate=cfg.rep_rate,
            position_unit=cfg.position_unit,
            show_write_progress=False,
            verbose=verbose,
            masks=masks,
        )
        runs.append(
            {
                "error_threshold": threshold,
                "available_detections": n_available,
                "tracks_h5part": tracks_out,
                "tracking": track_result,
                "skipped": False,
            }
        )
        if verbose:
            print(
                f"[error-sensitivity]   done: {track_result.get('total_tracks', 0)} tracks",
                flush=True,
            )

    summary = {
        "input_stm": stm_path,
        "output_dir": output_dir,
        "config": {
            **{f.name: getattr(cfg, f.name) for f in fields(cfg) if f.name != "link_params"},
            "link_params": asdict(cfg.link_params),
        },
        "thresholds": thresholds,
        "runs": runs,
    }
    summary_path = os.path.join(output_dir, cfg.summary_name)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    summary["summary_json"] = summary_path

    if verbose:
        completed = sum(1 for run in runs if not run.get("skipped"))
        print(
            f"[error-sensitivity] summary: {summary_path} "
            f"({completed}/{len(thresholds)} threshold(s) tracked)",
            flush=True,
        )
    return summary

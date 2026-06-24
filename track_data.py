"""Build a 4BE-ETI-compatible data object from graph tracking assignments."""

from __future__ import annotations

import numpy as np
from python_4be_eti.utils.basic_utils import a

from .assign import PathCandidate
from .load_data import BlockFrames


def block_to_tracking_data(
    block: BlockFrames,
    assigned: list[PathCandidate],
) -> a:
    """
    Fill ``Count`` with track IDs for assigned 4-frame paths.

    Unassigned particles keep Count == 0.
    """
    data = a()
    xs, ys, zs, slices = [], [], [], []
    for fr in block.frames:
        xs.append(fr.x.reshape(-1, 1))
        ys.append(fr.y.reshape(-1, 1))
        zs.append(fr.z.reshape(-1, 1))
        slices.append(np.full((fr.n, 1), fr.frame_index, dtype=int))

    data.x = np.concatenate(xs).ravel()
    data.y = np.concatenate(ys).ravel()
    data.z = np.concatenate(zs).ravel()
    data.Slice = np.concatenate(slices).ravel()
    data.Count = np.zeros(data.x.shape, dtype=np.int64)
    data.CountTemp = np.zeros_like(data.Count)
    data.Cost = -np.ones_like(data.Count, dtype=np.float64)
    data.Area = -np.ones_like(data.Count, dtype=np.float64)

    # Global flat index offsets per frame layer in block
    offsets = [0]
    for fr in block.frames[:-1]:
        offsets.append(offsets[-1] + fr.n)

    for track_id, path in enumerate(assigned, start=1):
        for j, local_idx in enumerate(path.indices):
            layer = path.start_layer + j
            flat = offsets[layer] + local_idx
            data.Count[flat] = track_id

    return data


def merge_block_data(accumulator: a | None, block_data: a) -> a:
    """Append one block's particles into a combined data object."""
    if accumulator is None:
        return block_data
    accumulator.x = np.concatenate([accumulator.x, block_data.x])
    accumulator.y = np.concatenate([accumulator.y, block_data.y])
    accumulator.z = np.concatenate([accumulator.z, block_data.z])
    accumulator.Slice = np.concatenate([accumulator.Slice, block_data.Slice])
    accumulator.Count = np.concatenate([accumulator.Count, block_data.Count])
    accumulator.CountTemp = np.concatenate([accumulator.CountTemp, block_data.CountTemp])
    accumulator.Cost = np.concatenate([accumulator.Cost, block_data.Cost])
    accumulator.Area = np.concatenate([accumulator.Area, block_data.Area])
    return accumulator

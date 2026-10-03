"""Stability fusion for independently visible physical slots."""

from collections import deque

import numpy as np


class SlotStabilizer:
    """Fuse each slot independently; an invisible third slot cannot erase two visible slots."""

    def __init__(self, config, slot_count=3):
        self.history_size = int(config.get("history_size", 7))
        self.required_frames = int(config.get("required_frames", 3))
        self.max_center_spread_px = float(config.get("max_center_spread_px", 32.0))
        self.max_area_relative_spread = float(config.get("max_area_relative_spread", 0.18))
        self.slot_count = int(slot_count)
        self.history = deque(maxlen=self.history_size)

    def reset(self):
        self.history.clear()

    def update(self, observations):
        """Update with ``{slot_id: detection_or_none}`` and return fused slots.

        A slot is stable only when it has enough valid observations of the same
        color. Other slots are evaluated independently.
        """
        normalized = {slot_id: observations.get(slot_id) for slot_id in range(1, self.slot_count + 1)}
        self.history.append(normalized)
        if len(self.history) < self.required_frames:
            return False, {}

        recent = list(self.history)[-self.required_frames:]
        fused = {}
        for slot_id in range(1, self.slot_count + 1):
            items = [frame[slot_id] for frame in recent if frame[slot_id] is not None]
            if len(items) < self.required_frames:
                continue
            colors = {item.color for item in items}
            if len(colors) != 1:
                continue
            values = np.asarray(
                [[float(item.u), float(item.v), float(item.area)] for item in items],
                dtype=np.float64,
            )
            center_spread = float(np.max(np.ptp(values[:, :2], axis=0)))
            area_median = float(np.median(values[:, 2]))
            area_spread = float(np.ptp(values[:, 2]) / max(area_median, 1.0))
            if center_spread > self.max_center_spread_px:
                continue
            if area_spread > self.max_area_relative_spread:
                continue
            item = items[-1]
            item.u = float(np.median(values[:, 0]))
            item.v = float(np.median(values[:, 1]))
            item.area = area_median
            fused[slot_id] = item
        return bool(fused), fused


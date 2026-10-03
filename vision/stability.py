from collections import deque

import numpy as np


class DetectionStabilizer:
    def __init__(self, config):
        self.config = config
        self.history = deque(maxlen=int(config["history_size"]))
        self.last_fused = []

    def reset(self):
        self.history.clear()
        self.last_fused = []

    def update(self, detections):
        current = {
            item.color: (float(item.u), float(item.v), float(item.area), item)
            for item in detections
            if item.valid_top
        }
        self.history.append(current)

        required = int(self.config["required_frames"])
        if len(self.history) < required:
            return False, list(self.last_fused)

        recent = list(self.history)[-required:]
        common_colors = set(recent[0])
        for frame in recent[1:]:
            common_colors.intersection_update(frame)
        if not common_colors:
            # Preserve the last geometrically valid result for a very short
            # detection gap.  The live preview must not blink empty when a
            # moving object briefly loses a threshold pixel.
            return False, list(self.last_fused)

        fused = []
        for color in sorted(common_colors, key=lambda name: recent[-1][name][3].color_id):
            values = np.asarray([frame[color][:3] for frame in recent], dtype=np.float64)
            center_spread = np.max(np.ptp(values[:, :2], axis=0))
            area_median = np.median(values[:, 2])
            area_spread = np.ptp(values[:, 2]) / max(area_median, 1.0)
            if center_spread > float(self.config["max_center_spread_px"]):
                continue
            if area_spread > float(self.config["max_area_relative_spread"]):
                continue

            item = recent[-1][color][3]
            item.u = float(np.median(values[:, 0]))
            item.v = float(np.median(values[:, 1]))
            item.area = float(area_median)
            fused.append(item)

        stable = bool(fused)
        if stable:
            self.last_fused = list(fused)
        return stable, fused if fused else list(self.last_fused)

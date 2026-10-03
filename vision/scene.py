"""Scene-specific spatial constraints for material detection.

The color detector is shared by both competition areas, but the valid image
region is different.  Keeping that constraint here prevents background objects
from becoming color candidates before classification starts.
"""

import cv2
import numpy as np


class SceneConstraint:
    def __init__(self, config, source_size):
        self.config = config
        self.source_width, self.source_height = map(int, source_size)
        detection = config.get("detection", {})
        self.mode = str(detection.get("scene_mode", "raw_material"))
        self.scenes = config.get("scenes", {})

    def set_mode(self, mode):
        mode = str(mode).strip().lower()
        if mode not in self.scenes:
            raise ValueError("unknown scene mode: " + mode)
        self.mode = mode

    def max_targets(self):
        scene = self.scenes.get(self.mode, {})
        return max(1, int(scene.get("max_targets", 6)))

    def is_processing_area(self):
        return self.mode == "processing_area"

    def is_raw_material(self):
        return self.mode == "raw_material"

    def mask(self, frame_shape):
        """Return a uint8 mask in the current processing-image coordinates."""
        height, width = frame_shape[:2]
        scene = self.scenes.get(self.mode, {})
        mask = np.zeros((height, width), dtype=np.uint8)
        # The raw-material scene has one authoritative spatial gate: the
        # detected turntable mask in MaterialDetector.  The old polygon was
        # measured for a different camera framing and could erase the real
        # disc before color detection.  Returning a full mask here keeps the
        # disc gate independent and leaves polygon constraints for the board
        # scene where they are still intentional.
        if self.is_raw_material() and bool(scene.get("use_disc_roi", False)):
            mask.fill(255)
            return mask
        shape = str(scene.get("roi_shape", "full"))

        if shape == "polygon":
            points = scene.get("polygon_source", [])
            if len(points) >= 3:
                scaled = np.asarray(
                    [
                        (
                            round(float(x) * width / self.source_width),
                            round(float(y) * height / self.source_height),
                        )
                        for x, y in points
                    ],
                    dtype=np.int32,
                )
                cv2.fillPoly(mask, [scaled], 255)
                return mask

        if shape == "ellipse":
            ellipse = scene.get("ellipse_source", {})
            center = ellipse.get("center", [])
            axes = ellipse.get("axes", [])
            if len(center) == 2 and len(axes) == 2:
                cx = round(float(center[0]) * width / self.source_width)
                cy = round(float(center[1]) * height / self.source_height)
                ax = max(1, round(float(axes[0]) * width / self.source_width))
                ay = max(1, round(float(axes[1]) * height / self.source_height))
                cv2.ellipse(
                    mask,
                    (cx, cy),
                    (ax, ay),
                    float(ellipse.get("angle_deg", 0.0)),
                    0,
                    360,
                    255,
                    -1,
                )
                return mask

        mask.fill(255)
        return mask

    def summary(self):
        scene = self.scenes.get(self.mode, {})
        return {
            "mode": self.mode,
            "max_targets": self.max_targets(),
            "roi_shape": scene.get("roi_shape", "full"),
            "partial_visibility_allowed": bool(
                scene.get("partial_visibility_allowed", False)
            ),
        }

"""Material detection constrained to the three predicted turntable slots."""

import copy

import cv2
import numpy as np

from .detector import MaterialDetector
from .slot_tracker import ThreeSlotTracker


class SlotMaterialDetector:
    """Run the existing color/top-center detector independently in each slot ROI."""

    def __init__(self, config):
        self.config = config
        local_config = copy.deepcopy(config)
        # A slot ROI is intentionally tight around one material, so the
        # material occupies a much larger fraction of the patch than it does
        # in a full frame. Keep the same color/top logic but relax only the
        # frame-relative area gates for this local detector.
        local_config["material"]["min_area_frame_ratio"] = 0.004
        local_config["material"]["max_area_frame_ratio"] = 0.72
        local_config["material"]["black_max_area_frame_ratio"] = 0.72
        self.base = MaterialDetector(local_config)
        turntable = config["turntable"]
        self.roi_half_size_source = float(turntable.get("slot_roi_half_size_px", 260.0))
        self.max_center_distance_source = float(turntable.get("slot_max_center_distance_px", 150.0))

    def detect_slots(
        self,
        frame_bgr,
        tracker: ThreeSlotTracker,
        source_size=(2560, 1440),
        refine_top=False,
    ):
        source_width, source_height = source_size
        scale_x = frame_bgr.shape[1] / float(source_width)
        scale_y = frame_bgr.shape[0] / float(source_height)
        centers = tracker.predicted_centers()
        observations = {}
        debug = {}
        for slot_id, (source_u, source_v) in centers.items():
            center_x = int(round(source_u * scale_x))
            center_y = int(round(source_v * scale_y))
            half_w = max(24, int(round(self.roi_half_size_source * scale_x)))
            half_h = max(24, int(round(self.roi_half_size_source * scale_y)))
            x0 = max(0, center_x - half_w)
            y0 = max(0, center_y - half_h)
            x1 = min(frame_bgr.shape[1], center_x + half_w)
            y1 = min(frame_bgr.shape[0], center_y + half_h)
            if x1 - x0 < 24 or y1 - y0 < 24:
                debug[slot_id] = {"reason": "slot_roi_outside_frame", "roi": [x0, y0, x1, y1]}
                continue
            patch = frame_bgr[y0:y1, x0:x1].copy()
            # Keep the local detector away from the rectangular ROI boundary;
            # this blocks outside text/cables without assuming a white disc.
            mask = np.zeros(patch.shape[:2], np.uint8)
            local_center = (center_x - x0, center_y - y0)
            radius = int(round(min(patch.shape[:2]) * 0.44))
            cv2.circle(mask, local_center, radius, 255, -1)
            white = np.median(patch.reshape(-1, 3), axis=0).astype(np.uint8)
            patch[mask == 0] = white

            _, candidates, local_debug = self.base.detect(patch, refine_top=refine_top)
            best = None
            best_distance = None
            for candidate in candidates:
                candidate.u += x0
                candidate.v += y0
                distance_source = np.hypot(
                    (candidate.u - center_x) / max(scale_x, 1e-6),
                    (candidate.v - center_y) / max(scale_y, 1e-6),
                )
                if distance_source > self.max_center_distance_source:
                    continue
                if best is None or distance_source < best_distance:
                    best = candidate
                    best_distance = distance_source
            if best is not None:
                # MaterialDetector returns area in patch pixels.  Convert all
                # published geometry back to the camera source coordinate
                # system, including area, so stability thresholds remain
                # comparable with full-frame detections.
                best.u /= max(scale_x, 1e-6)
                best.v /= max(scale_y, 1e-6)
                best.area /= max(scale_x * scale_y, 1e-12)
                observations[slot_id] = best
            debug[slot_id] = {
                "roi": [x0, y0, x1, y1],
                "predicted_center": [center_x, center_y],
                "candidate_count": len(candidates),
                "selected_distance_source_px": best_distance,
                "local": local_debug,
            }
        return observations, debug

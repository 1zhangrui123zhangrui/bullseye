from dataclasses import dataclass
from math import pi

import cv2
import numpy as np

from .scene import SceneConstraint


@dataclass
class Disc:
    cx: float
    cy: float
    radius: float
    confidence: float


@dataclass
class Detection:
    color: str
    color_id: int
    u: float
    v: float
    area: float
    confidence: float
    top_confidence: float
    valid_top: bool
    center_source: str

    def as_dict(self):
        return {
            "color": self.color,
            "color_id": self.color_id,
            "u": round(self.u, 3),
            "v": round(self.v, 3),
            "area": round(self.area, 1),
            "confidence": round(self.confidence, 3),
            "top_confidence": round(self.top_confidence, 3),
            "valid_top": self.valid_top,
            "center_source": self.center_source,
        }


class MaterialDetector:
    """Detect material colors and top centers without requiring a background."""

    def __init__(self, config):
        self.config = config
        self.detection_cfg = config.get("detection", {})
        self.disc_cfg = config["disc"]
        self.material_cfg = config["material"]
        self.top_cfg = config["top"]
        self.colors = config["colors"]
        self.scene = SceneConstraint(
            config,
            (config["camera"]["width"], config["camera"]["height"]),
        )
        self.uncalibrated_preview = any(
            not spec.get("calibrated", False) for spec in self.colors.values()
        )
        # The turntable is stationary relative to the camera in the raw-material
        # view.  Reusing a validated circle prevents HoughCircles from jumping
        # to cables, printed text, or small bright regions on individual frames.
        self._last_disc = None

    def detect(self, frame_bgr, refine_top=True):
        # Maix camera can return an empty frame during startup or buffer
        # handoff.  A bad frame must be skipped, never passed to OpenCV.
        if frame_bgr is None or not hasattr(frame_bgr, "shape") or frame_bgr.size == 0:
            return None, [], {"reason": "empty_frame", "candidates": []}
        if len(frame_bgr.shape) != 3 or frame_bgr.shape[0] < 2 or frame_bgr.shape[1] < 2:
            return None, [], {"reason": "invalid_frame_shape", "candidates": []}
        disc = None
        x0, y0 = 0, 0
        roi = frame_bgr
        local_disc = None
        roi_box = (0, 0, frame_bgr.shape[1], frame_bgr.shape[0])
        scene_mask = self.scene.mask(frame_bgr.shape)

        scene_cfg = self.scene.scenes.get(self.scene.mode, {})
        # Processing/rough-processing views do not guarantee that all three
        # target rings are visible.  They also have no white turntable to use
        # as a prerequisite.  Keep detection open over the board band; the
        # optional ring-local gate below can be enabled after ring geometry is
        # calibrated, but never reject a frame merely because fewer than three
        # rings are visible.
        use_disc_roi = bool(
            self.detection_cfg.get("use_disc_roi", False)
            or scene_cfg.get("use_disc_roi", False)
        )
        if use_disc_roi:
            disc = self._get_disc(frame_bgr)
            if disc is None:
                # Do not expose the full frame when the disc detector misses a
                # transient frame.  Reuse the last valid geometry if present;
                # otherwise report a miss and wait for the next frame.
                return None, [], {"reason": "disc_not_found", "candidates": []}
            roi_box = self._disc_roi(frame_bgr.shape, disc)
            x0, y0, x1, y1 = roi_box
            height, width = frame_bgr.shape[:2]
            x0, x1 = max(0, min(int(x0), width)), max(0, min(int(x1), width))
            y0, y1 = max(0, min(int(y0), height)), max(0, min(int(y1), height))
            if x1 <= x0 or y1 <= y0:
                return None, [], {"reason": "empty_disc_roi", "candidates": []}
            roi = frame_bgr[y0:y1, x0:x1]
            if roi is None or roi.size == 0:
                return None, [], {"reason": "empty_disc_roi", "candidates": []}
            local_disc = Disc(disc.cx - x0, disc.cy - y0, disc.radius, disc.confidence)
            scene_mask = scene_mask[y0:y1, x0:x1]

        if roi is None or roi.size == 0:
            return None, [], {"reason": "empty_roi", "candidates": []}
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        chromatic_mask, black_mask = self._material_masks(
            roi,
            local_disc,
            hsv=hsv,
        )
        chromatic_mask = cv2.bitwise_and(chromatic_mask, scene_mask)
        black_mask = cv2.bitwise_and(black_mask, scene_mask)
        # Keep the two candidate sources separate.  A single OR mask makes
        # shadows, dark borders and black materials merge into one contour.
        material_mask = cv2.bitwise_or(chromatic_mask, black_mask)
        candidate_specs = [
            (contour, "chromatic")
            for contour in self._candidate_contours(
                chromatic_mask, local_disc, kind="chromatic"
            )
        ]
        candidate_specs.extend(
            (contour, "black")
            for contour in self._candidate_contours(
                black_mask, local_disc, kind="black"
            )
        )
        detections = []
        debug_candidates = []
        use_full_classifier = refine_top and bool(candidate_specs)
        color_masks = self._color_masks(hsv) if use_full_classifier else None
        lab = None
        if use_full_classifier and roi.size:
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

        for contour, candidate_kind in candidate_specs:
            clipped = self._touches_frame_border(contour, roi.shape)
            # In the raw-material view every published candidate must be fully
            # inside the inner turntable circle.  Clipped candidates are only
            # useful for moving-board tracking, not for this fixed turntable.
            if use_disc_roi and self.scene.is_raw_material() and clipped:
                debug_candidates.append({
                    "bbox": tuple(int(v) for v in cv2.boundingRect(contour)),
                    "reason": "raw_material_border_rejected",
                    "candidate_kind": candidate_kind,
                })
                continue
            if candidate_kind == "black" and not self._validate_black_candidate(
                roi, contour, hsv, clipped
            ):
                x, y, w, h = cv2.boundingRect(contour)
                debug_candidates.append(
                    {
                        "bbox": (x + x0, y + y0, w, h),
                        "color": "black",
                        "reason": "black_candidate_rejected",
                        "candidate_kind": candidate_kind,
                    }
                )
                continue
            # Black candidates use the same lightweight center path while the
            # disc is moving. Expensive top-face analysis is reserved for a
            # stopped/grab refinement call; doing it for every black blob was
            # the dominant source of the 300 ms detection spikes.
            # In the moving processing-area view, keep a clipped chromatic
            # object available for tracking.  It is explicitly marked invalid
            # for grabbing; dropping it here causes the displayed target to
            # jump/disappear while the robot or turntable is moving.
            processing_partial = self.scene.is_processing_area() and clipped
            # Color classification must use the calibrated LAB/HSV path for
            # raw-material objects.  The fast path is only appropriate for
            # processing-area tracking, where latency is more important than
            # separating yellow/red and blue/light-blue under changing light.
            # The real-time path must not run CLAHE/Canny/ellipse fitting on
            # every object in every frame.  It uses the top-band center; the
            # caller can request the geometric refinement only while stopped.
            candidate_refine = use_full_classifier
            if candidate_refine:
                if clipped and not processing_partial:
                    detection, debug = self._analyze_candidate_fast(
                        contour,
                        x0,
                        y0,
                        hsv,
                        clipped=True,
                        candidate_kind=candidate_kind,
                    )
                else:
                    detection, debug = self._analyze_candidate(
                        roi,
                        contour,
                        x0,
                        y0,
                        local_disc,
                        hsv,
                        lab,
                        color_masks,
                        True,
                        candidate_kind=candidate_kind,
                    )
            else:
                detection, debug = self._analyze_candidate_fast(
                    contour,
                    x0,
                    y0,
                    hsv,
                    clipped=clipped,
                    candidate_kind=candidate_kind,
                )
            if detection is not None and processing_partial and candidate_kind != "black":
                detection.valid_top = False
                detection.center_source = "partial_tracking"
                detection.confidence *= 0.8
                debug["partial"] = True
                debug["reason"] = "partial_tracking_only"
            debug["candidate_kind"] = candidate_kind
            debug_candidates.append(debug)
            if detection is not None:
                detections.append(detection)

        detections = self._deduplicate_colors(detections)
        max_targets = self.scene.max_targets()
        if len(detections) > max_targets:
            detections = sorted(
                detections,
                key=lambda item: (item.valid_top, item.confidence),
                reverse=True,
            )[:max_targets]
        detections.sort(key=lambda item: item.color_id)
        return disc, detections, {
            "mode": self.scene.mode,
            "scene": self.scene.summary(),
            "roi": roi_box,
            "material_mask": material_mask,
            "mask_pixels": {
                "chromatic": int(cv2.countNonZero(chromatic_mask)),
                "black": int(cv2.countNonZero(black_mask)),
            },
            "candidates": debug_candidates,
        }

    def _get_disc(self, frame_bgr):
        """Resolve the fixed raw-material turntable geometry once and reuse it."""
        manual = self.disc_cfg.get("manual_circle")
        if manual:
            return Disc(float(manual[0]), float(manual[1]), float(manual[2]), 1.0)
        if self._last_disc is not None:
            return self._last_disc
        disc = self.detect_disc(frame_bgr)
        if disc is not None:
            self._last_disc = disc
        return disc

    def set_scene_mode(self, mode):
        """Switch scene constraints when the robot reports a new work area."""
        self.scene.set_mode(mode)

    def detect_disc(self, frame_bgr):
        if frame_bgr is None or not hasattr(frame_bgr, "size") or frame_bgr.size == 0:
            return None
        manual = self.disc_cfg.get("manual_circle")
        if manual:
            return Disc(float(manual[0]), float(manual[1]), float(manual[2]), 1.0)

        scale = float(self.disc_cfg["downscale"])
        small = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if small is None or small.size == 0:
            return None
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            (0, 0, int(self.disc_cfg["white_v_min"])),
            (179, int(self.disc_cfg["white_s_max"]), 255),
        )
        h, w = small.shape[:2]
        min_r = min(w, h) * float(self.disc_cfg["min_radius_ratio"])
        max_r = min(w, h) * float(self.disc_cfg["max_radius_ratio"])

        # Use the largest coherent white region.  HoughCircles is intentionally
        # not part of the runtime path: it can select text, cables, or a bright
        # object instead of the actual turntable and caused the ROI to jump.
        size = max(5, int(round(min(small.shape[:2]) * 0.025)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        image_center = np.array([w * 0.5, h * 0.5], dtype=np.float32)
        best = None
        image_area = float(w * h)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < image_area * float(self.disc_cfg.get("min_component_area_ratio", 0.12)):
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if not min_r <= radius <= max_r * 1.35:
                continue
            circularity = min(1.0, area / max(pi * radius * radius, 1.0))
            center_distance = np.linalg.norm(np.array([cx, cy]) - image_center)
            center_score = max(0.0, 1.0 - center_distance / max(np.hypot(w, h) * 0.7, 1.0))
            area_score = min(1.0, area / (image_area * 0.40))
            score = 0.58 * area_score + 0.27 * circularity + 0.15 * center_score
            if best is None or score > best[0]:
                best = (score, cx, cy, radius)

        if best is None:
            return None
        score, cx, cy, radius = best
        result = Disc(cx / scale, cy / scale, radius / scale, float(score))
        return result

    def _disc_roi(self, shape, disc):
        height, width = shape[:2]
        padding = disc.radius * float(self.disc_cfg["roi_padding_ratio"])
        radius = disc.radius + padding
        x0 = max(0, int(np.floor(disc.cx - radius)))
        y0 = max(0, int(np.floor(disc.cy - radius)))
        x1 = min(width, int(np.ceil(disc.cx + radius)))
        y1 = min(height, int(np.ceil(disc.cy + radius)))
        return x0, y0, x1, y1

    def _disc_inner_mask(self, shape, disc):
        mask = np.zeros(shape[:2], dtype=np.uint8)
        margin = disc.radius * max(
            float(self.disc_cfg["inner_margin_ratio"]),
            0.075 if self.scene.is_raw_material() else 0.035,
        )
        cv2.circle(
            mask,
            (int(round(disc.cx)), int(round(disc.cy))),
            max(1, int(round(disc.radius - margin))),
            255,
            -1,
        )
        return mask

    def _color_masks(self, hsv):
        color_masks = {}
        for name, spec in self.colors.items():
            color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for h_min, h_max, s_min, s_max, v_min, v_max in spec["hsv_ranges"]:
                color_mask = cv2.bitwise_or(
                    color_mask,
                    cv2.inRange(
                        hsv,
                        (int(h_min), int(s_min), int(v_min)),
                        (int(h_max), int(s_max), int(v_max)),
                    ),
                )
            color_masks[name] = color_mask
        return color_masks

    def _material_masks(self, roi_bgr, disc=None, hsv=None):
        if hsv is None:
            hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        chromatic = cv2.inRange(
            hsv,
            (
                0,
                int(self.material_cfg["chromatic_s_min"]),
                int(self.material_cfg["chromatic_v_min"]),
            ),
            (179, 255, 255),
        )
        black = cv2.inRange(
            hsv,
            (0, 0, 0),
            (179, 255, int(self.material_cfg["black_v_max"])),
        )
        # Preserve dark objects under the intense lower-disc light.  The
        # geometric/contrast gate below rejects the white-disc shadows and
        # black printed background that this wider mask may include.
        if disc is not None:
            reflective_black = cv2.inRange(
                hsv,
                (0, 0, int(self.material_cfg["black_v_max"])),
                (179, 120, min(125, int(self.material_cfg["black_v_max"]) + 35)),
            )
            black = cv2.bitwise_or(black, reflective_black)
        if disc is not None:
            # Apply the turntable mask once, here.  The caller applies the
            # scene mask separately; keeping one disc gate prevents the old
            # crop-plus-mask double filtering that dropped edge materials.
            disc_mask = self._disc_inner_mask(roi_bgr.shape, disc)
            chromatic = cv2.bitwise_and(chromatic, disc_mask)
            black = cv2.bitwise_and(black, disc_mask)
            reference_size = disc.radius
        else:
            reference_size = min(roi_bgr.shape[:2])

        kernel_size = max(
            3,
            int(round(reference_size * float(self.material_cfg["morph_kernel_ratio"]))) | 1,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        if disc is not None and self.scene.is_raw_material():
            # A full-size kernel erases the pale-blue/yellow top face after
            # resizing.  Use one smaller opening and one close to bridge the
            # shaded highlight without deleting the object.
            small_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (max(3, kernel_size // 2 * 2 + 1),) * 2
            )
            chromatic = cv2.morphologyEx(chromatic, cv2.MORPH_OPEN, small_kernel, iterations=1)
            chromatic = cv2.morphologyEx(chromatic, cv2.MORPH_CLOSE, small_kernel, iterations=1)
        else:
            chromatic = cv2.morphologyEx(chromatic, cv2.MORPH_OPEN, kernel, iterations=1)
            chromatic = cv2.morphologyEx(chromatic, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Black regions are especially prone to joining through shadows.  A
        # lighter close keeps the material body connected without growing thin
        # dark background structures into large candidates.
        black_kernel = kernel
        if disc is not None and self.scene.is_raw_material():
            black_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (max(3, kernel_size // 2 * 2 + 1),) * 2
            )
        black = cv2.morphologyEx(black, cv2.MORPH_OPEN, black_kernel, iterations=1)
        black = cv2.morphologyEx(black, cv2.MORPH_CLOSE, black_kernel, iterations=1)
        return chromatic, black

    def _material_mask(self, roi_bgr, disc=None, hsv=None):
        """Compatibility helper for tools that only need a display mask."""
        chromatic, black = self._material_masks(roi_bgr, disc, hsv=hsv)
        return cv2.bitwise_or(chromatic, black)

    def _candidate_contours(self, mask, disc=None, kind="chromatic"):
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if disc is not None:
            reference_area = pi * disc.radius * disc.radius
            min_area = reference_area * float(self.material_cfg["min_area_disc_ratio"])
            max_area = reference_area * float(self.material_cfg["max_area_disc_ratio"])
        else:
            reference_area = float(mask.shape[0] * mask.shape[1])
            min_area = reference_area * float(self.material_cfg["min_area_frame_ratio"])
            max_area = reference_area * float(self.material_cfg["max_area_frame_ratio"])
        border_margin = int(
            round(min(mask.shape[:2]) * float(self.material_cfg["border_margin_ratio"]))
        )
        accepted = []

        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            clipped = (
                x <= border_margin
                or y <= border_margin
                or x + w >= mask.shape[1] - border_margin
                or y + h >= mask.shape[0] - border_margin
            )
            candidate_min_area = (
                min_area * float(self.material_cfg["clipped_min_area_scale"])
                if clipped
                else min_area
            )
            if kind == "black":
                candidate_min_area = max(
                    candidate_min_area,
                    float(self.material_cfg.get("black_min_area", 0.0)),
                )
                black_max_ratio = float(
                    self.material_cfg.get("black_max_area_frame_ratio", 0.08)
                )
                if area > mask.shape[0] * mask.shape[1] * black_max_ratio:
                    continue
            # At the 320x180 processing size, a material can be compressed by
            # the resize and by the bright underside glare.  Keep a modest
            # lower bound, but retain the upper bound that protects against a
            # merged shadow/background region.
            if kind == "chromatic" and self.scene.is_raw_material():
                candidate_min_area = min(candidate_min_area, reference_area * 0.0015)
            if kind == "black" and self.scene.is_raw_material():
                candidate_min_area = min(candidate_min_area, reference_area * 0.0018)
            if not candidate_min_area <= area <= max_area:
                continue
            fill = area / max(float(w * h), 1.0)
            aspect = w / max(float(h), 1.0)
            minimum_fill = (
                float(self.material_cfg["clipped_min_fill_ratio"])
                if clipped
                else float(self.material_cfg["min_fill_ratio"])
            )
            if fill < minimum_fill:
                continue
            if clipped:
                if not 0.12 <= aspect <= 8.0:
                    continue
                if kind == "black" and (
                    w >= mask.shape[1] * 0.42 or h >= mask.shape[0] * 0.72
                ):
                    continue
            elif not 0.35 <= aspect <= 2.85:
                continue
            if kind == "black":
                perimeter = cv2.arcLength(contour, True)
                compactness = 4.0 * pi * area / max(perimeter * perimeter, 1.0)
                if compactness < float(
                    self.material_cfg.get("black_min_compactness", 0.12)
                ):
                    continue
            accepted.append(contour)
        return accepted

    def _validate_black_candidate(self, roi_bgr, contour, hsv, clipped):
        """Reject dark shadows and frame structures before black classification."""
        x, y, w, h = cv2.boundingRect(contour)
        if self.scene.is_processing_area():
            # In the board view a true black material is a compact top object.
            # Text, cables and the turntable border are elongated or touch the
            # scene boundary.  Keep this gate independent of HSV so widening
            # the black color range cannot re-introduce those false positives.
            image_h, image_w = roi_bgr.shape[:2]
            side = max(w, h)
            short = max(1, min(w, h))
            aspect = side / float(short)
            area = float(cv2.contourArea(contour))
            frame_area = float(image_h * image_w)
            if aspect > 2.2 or area > frame_area * 0.06:
                return False
            edge = max(2, int(round(min(image_h, image_w) * 0.018)))
            if x <= edge or y <= edge or x + w >= image_w - edge or y + h >= image_h - edge:
                return False
        local = np.zeros((h, w), dtype=np.uint8)
        shifted = contour.copy()
        shifted[:, :, 0] -= x
        shifted[:, :, 1] -= y
        cv2.drawContours(local, [shifted], -1, 255, -1)
        erode = max(1, int(round(min(w, h) * 0.12)))
        inner = cv2.erode(
            local,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1)
            ),
        )
        if cv2.countNonZero(inner) < 12:
            return False
        values = hsv[y:y + h, x:x + w, 2]
        inner_mean = float(cv2.mean(values, mask=inner)[0])

        # A border-connected black blob spanning a large part of the image is
        # almost always the camera frame, not a material.  Small clipped
        # materials remain available for color preview and are marked invalid
        # for grabbing by the caller.
        if clipped and (w >= roi_bgr.shape[1] * 0.42 or h >= roi_bgr.shape[0] * 0.72):
            return False

        dilated = cv2.dilate(
            local,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * max(2, erode) + 1, 2 * max(2, erode) + 1)
            ),
        )
        ring = cv2.subtract(dilated, local)
        ring_count = cv2.countNonZero(ring)
        if ring_count == 0:
            return False
        ring_mean = float(cv2.mean(values, mask=ring)[0])
        contrast_min = float(self.material_cfg.get("black_min_local_contrast", 8.0))
        # For clipped objects the surrounding ring may be incomplete, so use a
        # lower threshold; the border/compactness checks still protect it.
        required = contrast_min * (0.45 if clipped else 1.0)
        return inner_mean <= float(self.material_cfg["black_v_max"]) + 18.0 and (
            ring_mean - inner_mean >= required
        )

    def _touches_frame_border(self, contour, shape):
        x, y, w, h = cv2.boundingRect(contour)
        margin = int(
            round(min(shape[:2]) * float(self.material_cfg["border_margin_ratio"]))
        )
        return (
            x <= margin
            or y <= margin
            or x + w >= shape[1] - margin
            or y + h >= shape[0] - margin
        )

    def _analyze_candidate(
        self,
        roi_bgr,
        contour,
        roi_x,
        roi_y,
        disc=None,
        hsv=None,
        lab=None,
        color_masks=None,
        refine_top=True,
        candidate_kind="chromatic",
    ):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        component_mask = np.zeros(roi_bgr.shape[:2], dtype=np.uint8)
        cv2.drawContours(component_mask, [contour], -1, 255, -1)

        erode_size = max(3, int(round(min(w, h) * float(self.material_cfg["sample_erode_ratio"]))) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
        sample_mask = cv2.erode(component_mask, kernel, iterations=1)
        if candidate_kind == "black":
            color = "black"
            color_id = int(self.colors["black"]["id"])
            color_confidence = self._black_color_confidence(sample_mask, hsv)
            color_debug = {"candidate_kind": "black"}
        elif refine_top:
            color, color_id, color_confidence, color_debug = self._classify_color(
                roi_bgr,
                sample_mask,
                hsv=hsv,
                lab=lab,
                color_masks=color_masks,
            )
        else:
            color, color_id, color_confidence, color_debug = self._classify_color_fast(
                sample_mask,
                color_masks,
            )
        if color == "unknown":
            return None, {
                "bbox": (x + roi_x, y + roi_y, w, h),
                "color": color,
                "reason": "color_uncertain",
                "color_debug": color_debug,
            }

        if not refine_top:
            axis_center = self._contour_center(contour)
            detection = Detection(
                color,
                color_id,
                float(axis_center[0] + roi_x),
                float(axis_center[1] + roi_y),
                area,
                color_confidence,
                0.0,
                True,
                "coarse_contour",
            )
            return detection, {
                "bbox": (x + roi_x, y + roi_y, w, h),
                "ellipse": None,
                "color": color,
                "center_source": "coarse_contour",
                "color_debug": color_debug,
            }

        # The center of the complete trapezoid is below the top face.  Use the
        # upper silhouette band as the geometric fallback/quality anchor; this
        # is also usable when a black top edge has too little Canny contrast.
        local_component = component_mask[y:y + h, x:x + w]
        local_center = self._fast_top_center(local_component)
        axis_center = (float(x + local_center[0]), float(y + local_center[1]))
        top_center, top_confidence, ellipse = self._find_top_center(
            roi_bgr, component_mask, contour, disc
        )
        center_source = "top_ellipse"
        if top_center is not None:
            max_distance = max(w, h) * float(self.top_cfg["max_top_axis_distance_bbox_ratio"])
            distance = np.linalg.norm(np.asarray(top_center) - np.asarray(axis_center))
            if distance > max_distance:
                top_center = None

        allow_fallback = bool(self.top_cfg.get("allow_axis_fallback", False))
        if candidate_kind == "black":
            allow_fallback = False
        if top_center is None and allow_fallback:
            top_center = axis_center
            top_confidence = float(self.top_cfg["fallback_confidence"])
            center_source = "material_axis_fallback"

        valid_top = top_center is not None
        if not valid_top:
            return None, {
                "bbox": (x + roi_x, y + roi_y, w, h),
                "color": color,
                "reason": "top_not_found",
                "color_debug": color_debug,
            }

        u = float(top_center[0] + roi_x)
        v = float(top_center[1] + roi_y)
        confidence = float(np.clip(0.55 * color_confidence + 0.45 * top_confidence, 0.0, 1.0))
        detection = Detection(
            color, color_id, u, v, area, confidence, top_confidence, True, center_source
        )
        return detection, {
            "bbox": (x + roi_x, y + roi_y, w, h),
            "ellipse": ellipse,
            "color": color,
            "center_source": center_source,
            "color_debug": color_debug,
        }

    def _analyze_candidate_fast(
        self,
        contour,
        roi_x,
        roi_y,
        hsv,
        clipped=False,
        candidate_kind="chromatic",
    ):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        local_contour = contour.copy()
        local_contour[:, :, 0] -= x
        local_contour[:, :, 1] -= y

        component_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(component_mask, [local_contour], -1, 255, -1)
        erode_size = max(
            3,
            int(round(min(w, h) * float(self.material_cfg["sample_erode_ratio"]))) | 1,
        )
        sample_mask = cv2.erode(
            component_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size)),
        )
        if candidate_kind == "black":
            color = "black"
            color_id = int(self.colors["black"]["id"])
            color_confidence = self._black_color_confidence(
                sample_mask, hsv[y:y + h, x:x + w]
            )
            color_debug = {"candidate_kind": "black"}
        else:
            local_color_masks = self._color_masks(hsv[y:y + h, x:x + w])
            color, color_id, color_confidence, color_debug = self._classify_color_fast(
                sample_mask,
                local_color_masks,
                allow_black=False,
            )
        if color == "unknown":
            return None, {
                "bbox": (x + roi_x, y + roi_y, w, h),
                "color": color,
                "reason": "color_uncertain",
                "color_debug": color_debug,
            }

        center_x, center_y = self._fast_top_center(component_mask)
        detection = Detection(
            color,
            color_id,
            float(x + center_x + roi_x),
            float(y + center_y + roi_y),
            area,
            color_confidence,
            0.0,
            not clipped,
            "clipped_no_center" if clipped else "coarse_top_band",
        )
        return detection, {
            "bbox": (x + roi_x, y + roi_y, w, h),
            "ellipse": None,
            "color": color,
            "center_source": "clipped_no_center" if clipped else "coarse_top_band",
            "clipped": clipped,
            "color_debug": color_debug,
        }

    def _black_color_confidence(self, sample_mask, hsv_patch):
        values = hsv_patch[:, :, 2][sample_mask > 0]
        if len(values) == 0:
            return 0.0
        limit = float(self.material_cfg["black_v_max"])
        return float(np.clip(np.mean(values <= limit), 0.0, 1.0))

    def _fast_top_center(self, component_mask):
        height, width = component_mask.shape
        band_height = max(
            1,
            min(
                height,
                int(round(height * float(self.top_cfg["fast_center_band_ratio"]))),
            ),
        )
        top_band = component_mask[:band_height]
        moments = cv2.moments(top_band, binaryImage=True)
        if moments["m00"] > 0:
            return (
                float(moments["m10"] / moments["m00"]),
                float(moments["m01"] / moments["m00"]),
            )
        return float(width * 0.5), float(band_height * 0.5)

    @staticmethod
    def _contour_center(contour):
        moments = cv2.moments(contour)
        return (
            float(moments["m10"] / max(moments["m00"], 1e-6)),
            float(moments["m01"] / max(moments["m00"], 1e-6)),
        )

    @staticmethod
    def _axis_center(contour):
        # With a vertical optical axis the full silhouette is concentric with the
        # 30 mm top face. This fallback is explicitly reported to the caller.
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipseAMS(contour)
                return float(ellipse[0][0]), float(ellipse[0][1])
            except cv2.error:
                pass
        return MaterialDetector._contour_center(contour)

    def _classify_color_fast(self, sample_mask, color_masks, allow_black=True):
        sample_count = cv2.countNonZero(sample_mask)
        if sample_count < 20:
            return "unknown", 0, 0.0, {}

        coverages = {}
        for name, color_mask in color_masks.items():
            overlap = cv2.bitwise_and(color_mask, sample_mask)
            coverages[name] = cv2.countNonZero(overlap) / float(sample_count)
        # Blue and light-blue overlap under exposure changes.  Prefer the
        # calibrated hue/brightness profile instead of raw mask coverage.
        chromatic = sorted(
            ((coverage, name) for name, coverage in coverages.items() if name != "black"),
            reverse=True,
        )
        best_chromatic_coverage, best_chromatic = chromatic[0]
        # Never allow the broad red hue mask to win on a yellow candidate just
        # because its coverage is marginally larger.  Yellow has a distinct
        # hue band and high brightness in the calibrated videos.
        if coverages.get("yellow", 0.0) >= 0.08:
            best_chromatic = "yellow"
            best_chromatic_coverage = coverages["yellow"]
        elif coverages.get("light_blue", 0.0) >= 0.08 and coverages.get("blue", 0.0) < coverages["light_blue"] * 1.15:
            best_chromatic = "light_blue"
            best_chromatic_coverage = coverages["light_blue"]
        minimum = float(self.material_cfg["min_chromatic_coverage"])
        if best_chromatic_coverage >= minimum:
            color = best_chromatic
            confidence = best_chromatic_coverage
        elif not allow_black:
            # The contour already came from the chromatic mask.  Do not let a
            # dark patch inside it turn into a black material; keep the best
            # chromatic label for tracking and let calibration set the final
            # confidence threshold later.
            color = best_chromatic
            confidence = best_chromatic_coverage
        else:
            color = "black"
            confidence = coverages.get("black", 0.0)

        if confidence < minimum and not self.uncalibrated_preview:
            return "unknown", 0, confidence, {"hsv_coverage": coverages}
        return (
            color,
            int(self.colors[color]["id"]),
            float(np.clip(confidence, 0.0, 1.0)),
            {"hsv_coverage": coverages},
        )

    @staticmethod
    def _yellow_hsv_score(hsv_pixels):
        """Score yellow by hue distance, while tolerating strong lighting."""
        if len(hsv_pixels) == 0:
            return 0.0
        h = hsv_pixels[:, 0].astype(np.float32)
        s = hsv_pixels[:, 1].astype(np.float32)
        v = hsv_pixels[:, 2].astype(np.float32)
        hue_score = np.clip(1.0 - np.abs(h - 27.0) / 13.0, 0.0, 1.0)
        sat_score = np.clip((s - 70.0) / 100.0, 0.0, 1.0)
        value_score = np.clip((v - 70.0) / 150.0, 0.0, 1.0)
        return float(np.mean(hue_score * (0.55 + 0.25 * sat_score + 0.20 * value_score)))

    def _classify_color(
        self,
        roi_bgr,
        sample_mask,
        hsv=None,
        lab=None,
        color_masks=None,
    ):
        if lab is None:
            lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
        if hsv is None:
            hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        if color_masks is None:
            color_masks = self._color_masks(hsv)
        pixels = lab[sample_mask > 0]
        hsv_pixels = hsv[sample_mask > 0]
        if len(pixels) < 20:
            return "unknown", 0, 0.0, {}

        lab_median = np.median(pixels, axis=0).astype(np.float32)
        hsv_median = np.median(hsv_pixels, axis=0).astype(np.float32)
        color_coverages = {}
        for name, color_mask in color_masks.items():
            color_coverages[name] = float(
                np.mean(color_mask[sample_mask > 0] > 0)
            )

        # Yellow is the most frequent red confusion in the supplied preview.
        # Use the pixel hue distribution as an independent veto: a real yellow
        # top face has a compact hue cluster around 20--34, whereas red wraps
        # around 0/179.  This is deliberately applied before LAB ranking.
        yellow_score = self._yellow_hsv_score(hsv_pixels)
        yellow_coverage = color_coverages.get("yellow", 0.0)
        yellow_is_real = (
            yellow_coverage >= 0.045
            and yellow_score >= 0.36
            and 16.0 <= float(hsv_median[0]) <= 45.0
        )
        if yellow_is_real:
            color_coverages["yellow"] = max(color_coverages["yellow"], yellow_score)
            color_coverages["red"] *= 0.08

        # Explicit red/yellow separation.  Red's calibrated hue is near the
        # wrap-around (0/179), while yellow is centered near 27.  A yellow
        # sample must not be relabelled red merely because the LAB model is
        # broad after illumination changes.
        if color_coverages.get("yellow", 0.0) >= 0.045 and 16.0 <= float(hsv_median[0]) <= 45.0:
            color_coverages["red"] *= 0.25

        chromatic_coverage = max(
            coverage
            for name, coverage in color_coverages.items()
            if name != "black"
        )
        allow_black = chromatic_coverage < float(
            self.material_cfg["min_chromatic_coverage"]
        )
        distances = []
        for name, spec in self.colors.items():
            if name == "black" and not allow_black:
                continue
            mean = np.asarray(spec["lab_mean"], dtype=np.float32)
            std = np.maximum(np.asarray(spec["lab_std"], dtype=np.float32), 8.0)
            distance = float(np.sqrt(np.mean(np.square((lab_median - mean) / std))))
            hsv_coverage = color_coverages[name]
            if hsv_coverage < float(self.material_cfg["min_chromatic_coverage"]):
                distance *= 1.35
            else:
                distance *= max(0.65, 1.0 - 0.25 * hsv_coverage)
            distances.append((distance, name, int(spec["id"])))

        # Resolve the blue/light-blue boundary using LAB chroma.  Light blue
        # is substantially brighter and less saturated in the supplied videos;
        # this prevents a blue object at the bottom of the disc from drifting
        # into the light-blue label after shading lowers its hue confidence.
        if {item[1] for item in distances} >= {"blue", "light_blue"}:
            blue = next(item for item in distances if item[1] == "blue")
            light = next(item for item in distances if item[1] == "light_blue")
            if float(hsv_median[2]) < 125.0 and float(hsv_median[1]) > 150.0:
                distances = [item for item in distances if item[1] != "light_blue"] + [blue]
            elif float(hsv_median[2]) > 145.0 and float(hsv_median[1]) < 190.0:
                distances = [item for item in distances if item[1] != "blue"] + [light]
            distances.sort(key=lambda item: item[0])
        else:
            distances.sort(key=lambda item: item[0])
        first = distances[0]
        second = distances[1]
        margin = (second[0] - first[0]) / max(second[0], 1e-6)
        confidence = np.clip(0.65 * np.exp(-0.45 * first[0]) + 0.35 * max(margin, 0.0), 0.0, 1.0)
        if margin < float(self.material_cfg["min_color_margin"]):
            confidence *= 0.6

        debug = {
            "lab_median": lab_median.tolist(),
            "hsv_median": hsv_median.tolist(),
            "hsv_coverage": color_coverages,
            "distance": first[0],
            "margin": float(margin),
        }
        if (
            confidence < float(self.material_cfg["min_color_confidence"])
            and not self.uncalibrated_preview
        ):
            return "unknown", 0, float(confidence), debug

        return first[1], first[2], float(confidence), debug

    @staticmethod
    def _hsv_matches(hsv_value, ranges):
        h, s, v = hsv_value
        return any(
            h_min <= h <= h_max and s_min <= s <= s_max and v_min <= v <= v_max
            for h_min, h_max, s_min, s_max, v_min, v_max in ranges
        )

    def _find_top_center(self, roi_bgr, component_mask, contour, disc=None):
        x, y, w, h = cv2.boundingRect(contour)
        padding = int(round(max(w, h) * float(self.top_cfg["crop_padding_ratio"])))
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(roi_bgr.shape[1], x + w + padding)
        y1 = min(roi_bgr.shape[0], y + h + padding)
        crop = roi_bgr[y0:y1, x0:x1]
        crop_mask = component_mask[y0:y1, x0:x1]
        if crop is None or crop.size == 0 or crop_mask is None or crop_mask.size == 0:
            return None, 0.0, None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        edges = cv2.Canny(
            blurred,
            int(self.top_cfg["canny_low"]),
            int(self.top_cfg["canny_high"]),
        )
        edges = cv2.bitwise_and(edges, cv2.dilate(crop_mask, np.ones((5, 5), np.uint8)))
        edge_contours = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2]

        bbox_scale = float(max(w, h))
        min_d = bbox_scale * float(self.top_cfg["min_diameter_bbox_ratio"])
        max_d = bbox_scale * float(self.top_cfg["max_diameter_bbox_ratio"])
        component_moments = cv2.moments(contour)
        body_cx = component_moments["m10"] / max(component_moments["m00"], 1e-6)
        body_cy = component_moments["m01"] / max(component_moments["m00"], 1e-6)
        radial = None
        radial_norm = 0.0
        if disc is not None:
            radial = np.array([body_cx - disc.cx, body_cy - disc.cy], dtype=np.float32)
            radial_norm = np.linalg.norm(radial)
            if radial_norm > 1e-6:
                radial /= radial_norm

        best = None
        for edge_contour in edge_contours:
            if len(edge_contour) < 18:
                continue
            try:
                ellipse = cv2.fitEllipseAMS(edge_contour)
            except cv2.error:
                continue
            (ecx, ecy), (axis_a, axis_b), angle = ellipse
            major = max(axis_a, axis_b)
            minor = min(axis_a, axis_b)
            if not min_d <= major <= max_d:
                continue
            axis_ratio = minor / max(major, 1e-6)
            if axis_ratio < float(self.top_cfg["min_axis_ratio"]):
                continue

            local_x = int(round(ecx))
            local_y = int(round(ecy))
            if not (0 <= local_x < crop_mask.shape[1] and 0 <= local_y < crop_mask.shape[0]):
                continue
            if crop_mask[local_y, local_x] == 0:
                continue

            support = self._ellipse_edge_support(edges, ellipse)
            if support < float(self.top_cfg["min_edge_support"]):
                continue

            global_center = np.array([ecx + x0, ecy + y0], dtype=np.float32)
            offset = global_center - np.array([body_cx, body_cy], dtype=np.float32)
            radial_projection = 0.0
            radial_alignment = 0.5
            if radial_norm > 1e-6 and np.linalg.norm(offset) > 1e-6:
                radial_projection = float(np.dot(offset, radial) / max(bbox_scale, 1.0))
                # Under perspective the top face moves away from the disc center.
                # Near zero is accepted for a true top-down installation, while a
                # clear inward displacement is a side-wall false positive.
                if radial_projection < -0.035:
                    continue
                radial_alignment = float(np.clip(0.5 + radial_projection * 1.8, 0.0, 1.0))
            diameter_score = max(0.0, 1.0 - abs(major / bbox_scale - 0.60) / 0.32)
            score = 0.42 * support + 0.25 * axis_ratio + 0.18 * diameter_score + 0.15 * radial_alignment
            if best is None or score > best[0]:
                best = (score, global_center, ((ecx + x0, ecy + y0), (axis_a, axis_b), angle))

        if best is None:
            return None, 0.0, None
        score, center, ellipse = best
        return (float(center[0]), float(center[1])), float(np.clip(score, 0.0, 1.0)), ellipse

    @staticmethod
    def _ellipse_edge_support(edges, ellipse, samples=96):
        (cx, cy), (axis_a, axis_b), angle = ellipse
        theta = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
        x = 0.5 * axis_a * np.cos(theta)
        y = 0.5 * axis_b * np.sin(theta)
        radians = np.deg2rad(angle)
        xs = np.rint(cx + x * np.cos(radians) - y * np.sin(radians)).astype(np.int32)
        ys = np.rint(cy + x * np.sin(radians) + y * np.cos(radians)).astype(np.int32)
        valid = (xs >= 0) & (xs < edges.shape[1]) & (ys >= 0) & (ys < edges.shape[0])
        if not np.any(valid):
            return 0.0
        dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        return float(np.mean(dilated[ys[valid], xs[valid]] > 0))

    @staticmethod
    def _deduplicate_colors(detections):
        best = {}
        for detection in detections:
            previous = best.get(detection.color)
            if previous is None or detection.confidence > previous.confidence:
                best[detection.color] = detection
        return list(best.values())


def draw_debug(
    frame_bgr,
    disc,
    detections,
    stable=False,
    output_size=None,
    coordinate_size=None,
):
    if output_size is None:
        output = frame_bgr.copy()
        scale_x = scale_y = 1.0
    else:
        if output_size == (frame_bgr.shape[1], frame_bgr.shape[0]):
            output = frame_bgr.copy()
        else:
            output = cv2.resize(frame_bgr, output_size, interpolation=cv2.INTER_AREA)
        coordinate_width, coordinate_height = coordinate_size or (
            frame_bgr.shape[1],
            frame_bgr.shape[0],
        )
        scale_x = output_size[0] / coordinate_width
        scale_y = output_size[1] / coordinate_height
    if disc is not None:
        cv2.circle(
            output,
            (int(round(disc.cx * scale_x)), int(round(disc.cy * scale_y))),
            int(round(disc.radius * min(scale_x, scale_y))),
            (255, 200, 0),
            max(1, int(round(4 * min(scale_x, scale_y)))),
        )
    for item in detections:
        center = (int(round(item.u * scale_x)), int(round(item.v * scale_y)))
        color = (0, 255, 0) if item.valid_top else (0, 0, 255)
        cv2.drawMarker(output, center, color, cv2.MARKER_CROSS, max(10, int(36 * min(scale_x, scale_y))), 2)
        if item.valid_top:
            label = f"{item.color} ({item.u:.1f},{item.v:.1f}) {item.confidence:.2f}"
        else:
            label = f"{item.color} CLIPPED - MOVE INTO VIEW"
        cv2.putText(output, label, (center[0] + 8, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
    cv2.putText(
        output,
        "STABLE" if stable else "TRACKING",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0) if stable else (0, 190, 255),
        2,
    )
    return output

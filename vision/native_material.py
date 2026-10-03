from dataclasses import dataclass
from math import cos, hypot, radians, sin


@dataclass
class Material:
    color: str
    color_id: int
    x: int
    y: int
    w: int
    h: int
    pixels: int
    u: float
    v: float
    confidence: float
    predicted: bool = False
    hits: int = 0
    missed: int = 0
    still_frames: int = 0
    refined: bool = False
    center_source: str = "blob"


DEFAULT_THRESHOLDS = {
    # MaixPy find_blobs uses L in [0, 100] and a/b in [-128, 127].
    "red": [12, 58, 24, 82, 0, 64],
    "yellow": [42, 100, -34, 24, 38, 108],
    "blue": [4, 52, 12, 72, -104, -24],
    "green": [18, 88, -86, -12, 0, 72],
    "black": [0, 32, -32, 32, -38, 38],
    "light_blue": [24, 100, -42, 30, -68, 18],
}


def box_overlap(a, b):
    """Intersection relative to the smaller box, including nested masks."""
    width = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    height = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    return width * height / float(max(1, min(a.w * a.h, b.w * b.h)))


def circle_value(circle, name, index):
    value = getattr(circle, name, None)
    if callable(value):
        value = value()
    if value is not None:
        return value
    try:
        return circle[index]
    except (IndexError, KeyError, TypeError):
        return None


class NativeMaterialDetector:
    """Fast material detector based on MaixPy's native LAB blob finder."""

    def __init__(self, config):
        self.config = config
        self.camera_cfg = config["camera"]
        self.runtime_cfg = config.get("runtime", {})
        self.scene_cfg = config["scenes"]
        self.colors = config.get("colors", {})
        self.color_names = list(self.colors.keys())
        native = config.get("native_material", {})
        self.thresholds = native.get("lab_thresholds", DEFAULT_THRESHOLDS)
        self.min_pixels = int(native.get("min_pixels", 80))
        self.min_box_area = int(native.get("min_box_area", 280))
        self.max_box_area = int(native.get("max_box_area", 18000))
        self.min_fill = float(native.get("min_fill", 0.18))
        self.min_aspect = float(native.get("min_aspect", 0.30))
        self.max_aspect = float(native.get("max_aspect", 3.20))
        self.duplicate_distance = float(native.get("duplicate_distance", 24))
        top = config.get("top_refine", {})
        self.top_refine_enabled = bool(top.get("enabled", True))
        self.top_refine_threshold = int(top.get("circle_threshold", 2200))
        self.top_refine_min_radius = float(top.get("min_radius_ratio", 0.16))
        self.top_refine_max_radius = float(top.get("max_radius_ratio", 0.72))
        self.top_refine_center_tolerance = float(top.get("center_tolerance_ratio", 0.55))
        self.top_refine_margin = int(top.get("roi_margin", 8))
        self.top_refine_min_magnitude = float(top.get("min_magnitude", 0.0))
        self.top_color_min_fill = float(top.get("color_min_fill", 0.08))
        self.top_color_min_margin = float(top.get("color_min_margin", 0.08))
        self.mode = config.get("detection", {}).get("scene_mode", "raw_material")

    def set_scene_mode(self, mode):
        if mode not in self.scene_cfg:
            raise ValueError("unknown scene mode: " + mode)
        self.mode = mode

    def _scene(self):
        return self.scene_cfg.get(self.mode, self.scene_cfg["raw_material"])

    def _roi(self, image):
        width, height = image.width(), image.height()
        raw = self._scene().get("roi_capture")
        if not raw:
            return [0, 0, width, height]
        x, y, w, h = [int(v) for v in raw]
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        return [x, y, w, h]

    @staticmethod
    def _value(blob, name, index):
        candidate = getattr(blob, name, None)
        if callable(candidate):
            try:
                return candidate()
            except TypeError:
                pass
        if candidate is not None and not callable(candidate):
            return candidate
        try:
            return blob[index]
        except (IndexError, KeyError, TypeError):
            if isinstance(blob, dict):
                keys = (name, {0: "x", 1: "y", 2: "w", 3: "h"}.get(index, name))
                for key in keys:
                    if key in blob:
                        return blob[key]
        raise ValueError("unsupported Maix blob object")

    def _blob_values(self, blob):
        x = int(self._value(blob, "x", 0))
        y = int(self._value(blob, "y", 1))
        w = int(self._value(blob, "w", 2))
        h = int(self._value(blob, "h", 3))
        try:
            pixels = int(self._value(blob, "pixels", 4))
        except ValueError:
            pixels = w * h
        try:
            cx = float(self._value(blob, "cx", 5))
            cy = float(self._value(blob, "cy", 6))
        except ValueError:
            cx, cy = x + w * 0.5, y + h * 0.5
        return x, y, w, h, max(0, pixels), cx, cy

    def _ellipse_value(self, px, py, ellipse):
        if len(ellipse) < 4:
            return 0.0
        ecx, ecy, rx, ry = [float(v) for v in ellipse[:4]]
        angle = float(ellipse[4]) if len(ellipse) >= 5 else 0.0
        if rx <= 0 or ry <= 0:
            return 0.0
        theta = radians(angle)
        dx, dy = px - ecx, py - ecy
        local_x = dx * cos(theta) + dy * sin(theta)
        local_y = -dx * sin(theta) + dy * cos(theta)
        return (local_x / rx) ** 2 + (local_y / ry) ** 2

    def _inside_plate(self, x, y, w, h):
        ellipse = self._scene().get("plate_ellipse_capture")
        if not ellipse:
            return True
        margin = float(self._scene().get("plate_margin_capture", 6.0))
        values = [float(v) for v in ellipse]
        values[2] = max(1.0, values[2] - margin)
        values[3] = max(1.0, values[3] - margin)
        # Require the complete bounding box to remain on the plate. This is
        # stricter than testing only the center and rejects edge/background
        # blobs without touching the color pipeline.
        corners = (
            (x, y),
            (x + w - 1, y),
            (x, y + h - 1),
            (x + w - 1, y + h - 1),
        )
        return all(self._ellipse_value(x, y, values) <= 1.0 for x, y in corners)

    def _candidate(self, color, color_id, blob):
        x, y, w, h, pixels, cx, cy = self._blob_values(blob)
        if w <= 0 or h <= 0:
            return None
        box_area = w * h
        aspect = w / float(h)
        fill = pixels / float(box_area)
        if pixels < self.min_pixels or not self.min_box_area <= box_area <= self.max_box_area:
            return None
        if not self.min_aspect <= aspect <= self.max_aspect or fill < self.min_fill:
            return None
        if not self._inside_plate(x, y, w, h):
            return None
        confidence = min(1.0, 0.45 * min(1.0, fill) + 0.55 * min(1.0, pixels / 900.0))
        return Material(color, color_id, x, y, w, h, pixels, cx, cy, confidence)

    def _find(self, image, threshold, roi):
        # Keep the keyword form used by current MaixPy. The fallback handles
        # older firmware builds that do not expose every optional argument.
        try:
            return image.find_blobs(
                [threshold],
                roi=roi,
                x_stride=2,
                y_stride=2,
                area_threshold=self.min_box_area,
                pixels_threshold=self.min_pixels,
                merge=False,
            )
        except TypeError:
            return image.find_blobs(
                [threshold],
                roi=roi,
                x_stride=2,
                y_stride=2,
                area_threshold=self.min_box_area,
                pixels_threshold=self.min_pixels,
            )

    def _find_all(self, image, roi):
        """Run one native scan for all six thresholds when code() is available."""
        thresholds = [self.thresholds[name] for name in self.color_names if name in self.thresholds]
        names = [name for name in self.color_names if name in self.thresholds]
        if not thresholds:
            return []
        try:
            blobs = image.find_blobs(
                thresholds,
                roi=roi,
                x_stride=2,
                y_stride=2,
                area_threshold=self.min_box_area,
                pixels_threshold=self.min_pixels,
                merge=False,
            )
        except TypeError:
            blobs = image.find_blobs(
                thresholds,
                roi=roi,
                x_stride=2,
                y_stride=2,
                area_threshold=self.min_box_area,
                pixels_threshold=self.min_pixels,
            )

        results = []
        for blob in blobs:
            code_value = getattr(blob, "code", None)
            if callable(code_value):
                code_value = code_value()
            if code_value is None:
                # Older runtimes may omit code(); use the compatible six-pass
                # path rather than guessing a color label.
                return None
            code_value = int(code_value)
            # Do not turn an absent or ambiguous label into a guessed color.
            if code_value <= 0 or code_value & (code_value - 1):
                continue
            for index, name in enumerate(names):
                if code_value & (1 << index):
                    spec = self.colors[name]
                    item = self._candidate(name, int(spec.get("id", index + 1)), blob)
                    if item is not None:
                        results.append(item)
        return results

    def _deduplicate(self, materials):
        materials.sort(key=lambda item: item.confidence, reverse=True)
        accepted = []
        for item in materials:
            duplicate = False
            for index, previous in enumerate(accepted):
                overlap = box_overlap(item, previous)
                if overlap >= 0.55 or hypot(item.u - previous.u, item.v - previous.v) < self.duplicate_distance:
                    # A substantial colored region inside a dark mask is
                    # stronger color evidence than its shaded side wall.
                    if overlap >= 0.55:
                        if previous.color == "black" and item.color != "black" and item.pixels >= previous.pixels * 0.5:
                            accepted[index] = item
                        elif item.color == "black" and previous.color != "black" and previous.pixels < item.pixels * 0.5:
                            accepted[index] = item
                    duplicate = True
                    break
            if not duplicate:
                accepted.append(item)
        return accepted

    def detect(self, image):
        roi = self._roi(image)
        materials = self._find_all(image, roi)
        if materials is None:
            materials = []
            for name, spec in self.colors.items():
                threshold = self.thresholds.get(name)
                if threshold is None:
                    continue
                try:
                    blobs = self._find(image, threshold, roi)
                except Exception:
                    blobs = []
                color_id = int(spec.get("id", len(materials) + 1))
                for blob in blobs:
                    item = self._candidate(name, color_id, blob)
                    if item is not None:
                        materials.append(item)

        materials = self._deduplicate(materials)
        materials.sort(key=lambda item: item.pixels, reverse=True)
        max_targets = int(self._scene().get("max_targets", 2))
        return sorted(materials[:max_targets], key=lambda item: item.color_id)

    def _find_top_circle(self, image, item):
        """Find one top-face circle in a small candidate ROI, only on still frames."""
        finder = getattr(image, "find_circles", None)
        if not callable(finder):
            return None
        margin = self.top_refine_margin
        x = max(0, item.x - margin)
        y = max(0, item.y - margin)
        right = min(image.width(), item.x + item.w + margin)
        bottom = min(image.height(), item.y + item.h + margin)
        roi = [x, y, max(1, right - x), max(1, bottom - y)]
        radius = max(2.0, min(item.w, item.h) * 0.5)
        r_min = max(2, int(radius * self.top_refine_min_radius))
        r_max = max(r_min + 1, int(radius * self.top_refine_max_radius))
        try:
            circles = finder(
                threshold=self.top_refine_threshold,
                x_stride=2,
                y_stride=2,
                roi=roi,
                r_min=r_min,
                r_max=r_max,
                r_step=2,
            )
        except TypeError:
            try:
                circles = finder(self.top_refine_threshold, 2, 2, roi, r_min, r_max, 2)
            except (TypeError, ValueError):
                return None
        except (RuntimeError, ValueError):
            return None
        best = None
        for circle in circles or []:
            cx = circle_value(circle, "x", 0)
            cy = circle_value(circle, "y", 1)
            cr = circle_value(circle, "r", 2)
            magnitude = circle_value(circle, "magnitude", 3)
            if None in (cx, cy, cr):
                continue
            cx, cy, cr = float(cx), float(cy), float(cr)
            if magnitude is not None and float(magnitude) < self.top_refine_min_magnitude:
                continue
            distance = hypot(cx - item.u, cy - item.v)
            if distance > max(item.w, item.h) * self.top_refine_center_tolerance:
                continue
            score = (float(magnitude or 1.0), -distance, cr)
            if best is None or score > best[0]:
                best = (score, cx, cy, cr)
        return best

    def refine_top(self, image, materials):
        """Refine stable candidates; the caller must decide when the scene is still."""
        if not self.top_refine_enabled:
            return materials
        for item in materials:
            result = self._find_top_circle(image, item)
            if result is None:
                continue
            _, cx, cy, radius = result
            item.u, item.v = cx, cy
            item.x = int(round(cx - radius))
            item.y = int(round(cy - radius))
            item.w = max(1, int(round(radius * 2)))
            item.h = item.w
            item.refined = True
            item.center_source = "top_circle"
            color = self._classify_top(image, item, radius)
            if color is not None:
                item.color = color
                item.color_id = int(self.colors[color].get("id", item.color_id))
        return materials

    def _classify_top(self, image, item, radius):
        """Score LAB thresholds only inside the refined top-face patch."""
        half = max(2, int(round(radius * 0.42)))
        x = max(0, int(round(item.u - half)))
        y = max(0, int(round(item.v - half)))
        right = min(image.width(), x + half * 2 + 1)
        bottom = min(image.height(), y + half * 2 + 1)
        roi = [x, y, max(1, right - x), max(1, bottom - y)]
        scores = []
        for name in self.color_names:
            threshold = self.thresholds.get(name)
            if threshold is None:
                continue
            try:
                blobs = image.find_blobs(
                    [threshold], roi=roi, x_stride=1, y_stride=1,
                    area_threshold=3, pixels_threshold=3, merge=False,
                )
            except TypeError:
                try:
                    blobs = image.find_blobs(
                        [threshold], roi=roi, x_stride=1, y_stride=1,
                        area_threshold=3, pixels_threshold=3,
                    )
                except (RuntimeError, ValueError):
                    blobs = []
            except (RuntimeError, ValueError):
                blobs = []
            pixels = 0
            for blob in blobs or []:
                try:
                    pixels += self._blob_values(blob)[4]
                except (ValueError, TypeError):
                    continue
            scores.append((pixels / float(max(1, roi[2] * roi[3])), name))
        if not scores:
            return None
        scores.sort(reverse=True)
        best, name = scores[0]
        second = scores[1][0] if len(scores) > 1 else 0.0
        if best < self.top_color_min_fill or best - second < self.top_color_min_margin:
            return None
        return name


class MaterialTracker:
    """Keep short association memory; publish only this frame's observations."""

    def __init__(self, alpha=0.55, max_missed=3, stable_frames=3, still_center_px=1.5):
        self.alpha = float(alpha)
        self.max_missed = int(max_missed)
        self.stable_frames = int(stable_frames)
        self.still_center_px = float(still_center_px)
        self.items = {}

    def update(self, detections):
        seen = set()
        output = []
        for item in detections:
            previous = self.items.get(item.color)
            if previous is not None and box_overlap(item, previous) >= 0.25:
                movement = hypot(item.u - previous.u, item.v - previous.v)
                old_u, old_v = item.u, item.v
                item.u = previous.u + self.alpha * (item.u - previous.u)
                item.v = previous.v + self.alpha * (item.v - previous.v)
                item.x += int(round(item.u - old_u))
                item.y += int(round(item.v - old_v))
                hits = previous.hits + 1
                still_frames = previous.still_frames + 1 if movement <= self.still_center_px else 0
            else:
                hits = 1
                still_frames = 0
            item.hits = hits
            item.still_frames = still_frames
            item.missed = 0
            item.predicted = False
            self.items[item.color] = item
            seen.add(item.color)
            output.append(item)

        for color, previous in list(self.items.items()):
            if color in seen:
                continue
            previous.missed += 1
            if previous.missed <= self.max_missed:
                previous.predicted = True
            else:
                del self.items[color]
        return output

    def stable(self, item):
        return (
            getattr(item, "hits", 0) >= self.stable_frames
            and getattr(item, "still_frames", 0) >= self.stable_frames
            and not item.predicted
        )

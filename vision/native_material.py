from dataclasses import dataclass
from math import hypot


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


DEFAULT_THRESHOLDS = {
    # MaixPy find_blobs uses L in [0, 100] and a/b in [-128, 127].
    "red": [12, 58, 24, 82, 0, 64],
    "yellow": [42, 100, -34, 24, 38, 108],
    "blue": [4, 52, 12, 72, -104, -24],
    "green": [18, 88, -86, -12, 0, 72],
    "black": [0, 32, -32, 32, -38, 38],
    "light_blue": [24, 100, -42, 30, -68, 18],
}


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

    def _inside_plate(self, cx, cy, roi):
        ellipse = self._scene().get("plate_ellipse_capture")
        if not ellipse:
            return True
        ecx, ecy, rx, ry = [float(v) for v in ellipse]
        if rx <= 0 or ry <= 0:
            return True
        return ((cx - ecx) / rx) ** 2 + ((cy - ecy) / ry) ** 2 <= 1.0

    def _candidate(self, color, color_id, blob, roi):
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
        if not self._inside_plate(cx, cy, roi):
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
            if code_value == 0:
                code_value = 1
            for index, name in enumerate(names):
                if code_value & (1 << index):
                    spec = self.colors[name]
                    item = self._candidate(name, int(spec.get("id", index + 1)), blob, roi)
                    if item is not None:
                        results.append(item)
        return results

    def _deduplicate(self, materials):
        materials.sort(key=lambda item: item.confidence, reverse=True)
        accepted = []
        for item in materials:
            duplicate = False
            for previous in accepted:
                if hypot(item.u - previous.u, item.v - previous.v) < self.duplicate_distance:
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
                    item = self._candidate(name, color_id, blob, roi)
                    if item is not None:
                        materials.append(item)

        materials = self._deduplicate(materials)
        materials.sort(key=lambda item: item.pixels, reverse=True)
        max_targets = int(self._scene().get("max_targets", 2))
        return sorted(materials[:max_targets], key=lambda item: item.color_id)


class MaterialTracker:
    """Small EMA tracker with a short hold window for transient missed blobs."""

    def __init__(self, alpha=0.55, max_missed=3, stable_frames=3):
        self.alpha = float(alpha)
        self.max_missed = int(max_missed)
        self.stable_frames = int(stable_frames)
        self.items = {}

    def update(self, detections):
        seen = set()
        output = []
        for item in detections:
            previous = self.items.get(item.color)
            if previous is not None:
                item.u = previous.u + self.alpha * (item.u - previous.u)
                item.v = previous.v + self.alpha * (item.v - previous.v)
                item.x = int(round(item.u - item.w * 0.5))
                item.y = int(round(item.v - item.h * 0.5))
                hits = previous.hits + 1
            else:
                hits = 1
            item.hits = hits
            item.missed = 0
            self.items[item.color] = item
            seen.add(item.color)
            output.append(item)

        for color, previous in list(self.items.items()):
            if color in seen:
                continue
            previous.missed += 1
            if previous.missed <= self.max_missed:
                previous.predicted = True
                previous.confidence *= 0.85
                output.append(previous)
            else:
                del self.items[color]
        return output

    def stable(self, item):
        return getattr(item, "hits", 0) >= self.stable_frames and not item.predicted

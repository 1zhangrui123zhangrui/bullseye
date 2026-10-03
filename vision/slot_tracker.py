"""Three-slot turntable geometry and visibility state.

This module is intentionally independent of color segmentation.  It gives the
material detector a predicted local search region for each physical ring and
keeps a slot alive when that ring is temporarily outside the camera view.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class TurntableOrbit:
    """Projected ellipse of the three ring centers in image pixels."""

    cx: float
    cy: float
    axis_x: float
    axis_y: float
    rotation_deg: float = 0.0

    def point(self, phase_deg: float):
        phase = math.radians(float(phase_deg))
        rotation = math.radians(float(self.rotation_deg))
        x = 0.5 * self.axis_x * math.cos(phase)
        y = 0.5 * self.axis_y * math.sin(phase)
        return (
            self.cx + x * math.cos(rotation) - y * math.sin(rotation),
            self.cy + x * math.sin(rotation) + y * math.cos(rotation),
        )

    def residual(self, u: float, v: float):
        """Approximate pixel distance from a point to the projected orbit."""
        phase = math.radians(float(self.rotation_deg))
        cos_a, sin_a = math.cos(phase), math.sin(phase)
        dx, dy = float(u) - self.cx, float(v) - self.cy
        x = dx * cos_a + dy * sin_a
        y = -dx * sin_a + dy * cos_a
        a = max(self.axis_x * 0.5, 1.0)
        b = max(self.axis_y * 0.5, 1.0)
        return abs(math.hypot(x / a, y / b) - 1.0) * min(a, b)


@dataclass
class SlotState:
    slot_id: int
    phase_deg: float
    visible: bool = False
    color: str | None = None
    color_id: int | None = None
    u: float | None = None
    v: float | None = None
    confidence: float = 0.0
    missed_frames: int = 0


class ThreeSlotTracker:
    """Predict and preserve the three physical slots during rotation."""

    def __init__(
        self,
        orbit: TurntableOrbit,
        phase0_deg: float = 0.0,
        angular_spacing_deg: float = 120.0,
        max_missed_frames: int = 15,
        max_step_deg: float = 8.0,
    ):
        self.orbit = orbit
        self.phase0_deg = float(phase0_deg)
        self.angular_spacing_deg = float(angular_spacing_deg)
        self.max_missed_frames = int(max_missed_frames)
        self.max_step_deg = float(max_step_deg)
        self.slots = [
            SlotState(slot_id=index + 1, phase_deg=self.phase0_deg + index * self.angular_spacing_deg)
            for index in range(3)
        ]
        self.angle_deg = 0.0

    def reset_visibility(self):
        for slot in self.slots:
            slot.visible = False
            slot.missed_frames = 0

    def set_angle(self, angle_deg: float):
        self.angle_deg = float(angle_deg)

    def estimate_angle(self, detections, max_distance_px=None, global_search=False):
        """Estimate turntable angle from one or more detected top centers.

        The search is deliberately coarse and bounded by the previous angle.
        It is a slot-assignment aid during rotation, not a replacement for the
        final stopped-frame center refinement. A single visible object is used
        only as a coarse phase cue; publication still requires multi-frame
        stability.
        """
        if not detections:
            return self.angle_deg, False
        limit = float(max_distance_px or 180.0)
        if global_search:
            candidates = np.arange(-180.0, 180.0, 2.0)
        else:
            center = self.angle_deg
            candidates = np.arange(
                center - self.max_step_deg,
                center + self.max_step_deg + 0.01,
                1.0,
            )
        best = None
        for angle in candidates:
            predicted = [
                self.orbit.point(slot.phase_deg + float(angle))
                for slot in self.slots
            ]
            distance_matrix = np.asarray(
                [
                    [math.hypot(float(item.u) - px, float(item.v) - py) for px, py in predicted]
                    for item in detections
                ],
                dtype=np.float64,
            )
            # Brute-force assignment is tiny here (three physical slots) and
            # prevents both detections from being matched to one slot.
            import itertools

            pair_count = min(2, len(detections))
            score = min(
                sum(distance_matrix[item_index, slot_index] for item_index, slot_index in zip(item_indices, slot_indices))
                for item_indices in itertools.combinations(range(len(detections)), pair_count)
                for slot_indices in itertools.permutations(range(3), pair_count)
            )
            if best is None or score < best[0]:
                best = (score, float(angle))
        if best is None or best[0] > limit * min(2, len(detections)):
            return self.angle_deg, False
        if not global_search:
            delta = ((best[1] - self.angle_deg + 180.0) % 360.0) - 180.0
            if abs(delta) > self.max_step_deg + 0.1:
                return self.angle_deg, False
        self.angle_deg = best[1]
        return self.angle_deg, True

    def predicted_centers(self):
        return {
            slot.slot_id: self.orbit.point(slot.phase_deg + self.angle_deg)
            for slot in self.slots
        }

    def mark_unseen(self):
        for slot in self.slots:
            slot.missed_frames += 1
            slot.visible = False

    def update(self, detections, max_distance_px: float, slot_observations=None):
        """Assign detections to predicted slots and preserve slot identity.

        Each detection must expose ``u``, ``v``, ``color``, ``color_id`` and
        ``confidence``.  Unmatched slots remain in memory but are marked
        invisible; their color is not erased until ``max_missed_frames``.
        """
        # SlotMaterialDetector has already made a constrained decision inside
        # each physical ROI.  Preserve that identity instead of globally
        # re-matching nearby observations, which can swap colors at crossings.
        if slot_observations is not None:
            used_slots = set()
            for slot_id, detection in slot_observations.items():
                if detection is None or not 1 <= int(slot_id) <= len(self.slots):
                    continue
                slot = self.slots[int(slot_id) - 1]
                px, py = self.predicted_centers()[int(slot_id)]
                distance = math.hypot(float(detection.u) - px, float(detection.v) - py)
                if distance > float(max_distance_px):
                    continue
                slot.visible = True
                slot.missed_frames = 0
                slot.color = detection.color
                slot.color_id = int(detection.color_id)
                slot.u = float(detection.u)
                slot.v = float(detection.v)
                slot.confidence = float(detection.confidence)
                used_slots.add(int(slot_id))
            for slot in self.slots:
                if slot.slot_id in used_slots:
                    continue
                slot.visible = False
                slot.missed_frames += 1
                if slot.missed_frames > self.max_missed_frames:
                    slot.color = None
                    slot.color_id = None
                    slot.u = None
                    slot.v = None
                    slot.confidence = 0.0
            return self.snapshot()

        predictions = self.predicted_centers()
        pairs = []
        for slot in self.slots:
            px, py = predictions[slot.slot_id]
            for index, detection in enumerate(detections):
                distance = math.hypot(float(detection.u) - px, float(detection.v) - py)
                pairs.append((distance, slot.slot_id, index))
        used_slots = set()
        used_detections = set()
        for distance, slot_id, index in sorted(pairs):
            if distance > float(max_distance_px):
                break
            if slot_id in used_slots or index in used_detections:
                continue
            slot = self.slots[slot_id - 1]
            detection = detections[index]
            slot.visible = True
            slot.missed_frames = 0
            slot.color = detection.color
            slot.color_id = int(detection.color_id)
            slot.u = float(detection.u)
            slot.v = float(detection.v)
            slot.confidence = float(detection.confidence)
            used_slots.add(slot_id)
            used_detections.add(index)

        for slot in self.slots:
            if slot.slot_id in used_slots:
                continue
            slot.visible = False
            slot.missed_frames += 1
            if slot.missed_frames > self.max_missed_frames:
                slot.color = None
                slot.color_id = None
                slot.u = None
                slot.v = None
                slot.confidence = 0.0
        return self.snapshot()

    def snapshot(self):
        return [
            {
                "slot_id": slot.slot_id,
                "visible": slot.visible,
                "color": slot.color,
                "color_id": slot.color_id,
                "u": slot.u,
                "v": slot.v,
                "confidence": slot.confidence,
                "missed_frames": slot.missed_frames,
            }
            for slot in self.slots
        ]

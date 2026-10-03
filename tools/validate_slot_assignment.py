"""Validate slot assignment on recorded multi-material turntable videos."""

import argparse
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from vision.config import load_config
from vision.slot_detector import SlotMaterialDetector
from vision.slot_tracker import ThreeSlotTracker, TurntableOrbit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--step", type=int, default=15)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_config()
    orbit_cfg = config["turntable"]["orbit"]
    if not orbit_cfg:
        raise SystemExit("turntable.orbit is not configured")
    orbit = TurntableOrbit(
        float(orbit_cfg["cx"]),
        float(orbit_cfg["cy"]),
        float(orbit_cfg["axis_x"]),
        float(orbit_cfg["axis_y"]),
        float(orbit_cfg.get("rotation_deg", 0.0)),
    )
    tracker = ThreeSlotTracker(
        orbit,
        phase0_deg=float(config["turntable"]["phase0_deg"]),
        angular_spacing_deg=float(config["turntable"]["angular_spacing_deg"]),
        max_missed_frames=int(config["turntable"]["max_missed_frames"]),
        # The validator samples every Nth frame; scale the per-frame phase
        # bound accordingly while the board runtime uses the base value.
        max_step_deg=float(config["turntable"].get("max_step_deg", 8.0)) * max(1, int(args.step)),
    )
    detector = SlotMaterialDetector(config)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    records = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % max(1, args.step):
            frame_index += 1
            continue
        # Use local slot ROIs.  This intentionally excludes external objects,
        # printed characters and cables before color detection.
        working = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        # Coarse full-frame detections are used only to estimate the current
        # turntable angle. Their coordinates are never published as results.
        coarse = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        coarse_detector = detector.base
        _, coarse_detections, _ = coarse_detector.detect(coarse, refine_top=False)
        for item in coarse_detections:
            item.u *= 4.0
            item.v *= 4.0
        coarse_detections = [
            item
            for item in coarse_detections
            if tracker.orbit.residual(item.u, item.v) <= 180.0
        ]
        angle, angle_ok = tracker.estimate_angle(
            coarse_detections,
            max_distance_px=float(config["turntable"]["max_slot_distance_px"]),
            global_search=not records,
        )
        observations, slot_debug = detector.detect_slots(
            working,
            tracker,
            source_size=(2560, 1440),
            refine_top=False,
        )
        detections = list(observations.values())
        # Re-run at the updated angle when the initial estimate was usable.
        if angle_ok:
            observations, slot_debug = detector.detect_slots(
                working, tracker, source_size=(2560, 1440), refine_top=False
            )
        slots = tracker.update(
            list(observations.values()),
            max_distance_px=float(config["turntable"]["max_slot_distance_px"]),
            slot_observations=observations,
        )
        slot_debug = {
            str(slot_id): {
                key: float(value) if hasattr(value, "item") else value
                for key, value in details.items()
                if key != "local"
            }
            for slot_id, details in slot_debug.items()
        }
        records.append(
            {
                "frame": frame_index,
                "time_s": frame_index / fps if fps else None,
                "detection_count": len(detections),
                "detections": [item.as_dict() for item in observations.values()],
                "angle_deg": round(float(angle), 3),
                "angle_ok": angle_ok,
                "slots": slots,
                "slot_debug": slot_debug,
            }
        )
        frame_index += 1
    capture.release()

    output = args.output or args.video.with_name(args.video.stem + "_slot_validation.json")
    result = {
        "video": str(args.video),
        "frame_count": frame_count,
        "fps": fps,
        "sample_step": int(args.step),
        "records": records,
        "frames_with_two_or_more_detections": sum(item["detection_count"] >= 2 for item in records),
        "frames_with_angle_estimate": sum(item["angle_ok"] for item in records),
    }
    # Quality counters are intentionally diagnostic, not a competition score.
    # They make geometry mismatch and temporal instability visible in reports.
    color_switches = 0
    coordinate_jumps = 0
    previous = {}
    for record in records:
        current = {item["slot_id"]: item for item in record["slots"]}
        for slot_id, item in current.items():
            old = previous.get(slot_id)
            if old is not None:
                if old.get("color") and item.get("color") and old["color"] != item["color"]:
                    color_switches += 1
                if old.get("u") is not None and item.get("u") is not None:
                    jump = ((float(item["u"]) - float(old["u"])) ** 2 +
                            (float(item["v"]) - float(old["v"])) ** 2) ** 0.5
                    if jump > 120.0:
                        coordinate_jumps += 1
        previous = current
    result["diagnostics"] = {
        "slot_color_switches": color_switches,
        "coordinate_jumps_over_120_source_px": coordinate_jumps,
        "note": "These counters diagnose the supplied geometry/video pair; they are not accuracy labels without ground truth.",
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "records": len(records),
        "frames_with_two_or_more_detections": result["frames_with_two_or_more_detections"],
        "frames_with_angle_estimate": result["frames_with_angle_estimate"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

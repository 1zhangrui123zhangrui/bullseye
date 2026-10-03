"""First-stage bullseye detector for MaixCAM Pro.

This stage validates the converted YOLO11 model only.  It reports the
detector-box center; geometric ring-center refinement is added after the
model is confirmed on the real camera.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from maix import app, camera, display, image, nn, time

TARGET_CONFIG_PATH = PROJECT_ROOT / "config" / "target.json"


def load_target_config():
    with TARGET_CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def model_path(config):
    configured = str(config["model"])
    # MaixVision uploads project files under /root.  Keep the configured
    # device path, while allowing local execution to resolve the project copy.
    if configured.startswith("/root/"):
        local_copy = PROJECT_ROOT / configured.removeprefix("/root/")
        if local_copy.exists():
            return str(local_copy)
    return configured


def main():
    target_config = load_target_config()
    detector = nn.YOLO11(model=model_path(target_config), dual_buff=True)
    cam = camera.Camera(
        detector.input_width(),
        detector.input_height(),
        detector.input_format(),
    )
    disp = display.Display()

    conf_th = float(target_config.get("confidence", 0.50))
    iou_th = float(target_config.get("iou", 0.45))
    print_interval = max(1, int(target_config.get("print_interval_frames", 10)))
    frame_index = 0
    last_tick = time.ticks_ms()
    fps = 0.0

    print("bullseye model:", model_path(target_config))
    print("input:", detector.input_width(), "x", detector.input_height())
    print("labels:", detector.labels)

    while not app.need_exit():
        frame_index += 1
        frame = cam.read()
        objects = detector.detect(frame, conf_th=conf_th, iou_th=iou_th)

        results = []
        for obj in objects:
            cx = float(obj.x) + float(obj.w) * 0.5
            cy = float(obj.y) + float(obj.h) * 0.5
            label = detector.labels[obj.class_id]
            results.append({
                "label": label,
                "class_id": int(obj.class_id),
                "score": round(float(obj.score), 4),
                "x": int(obj.x),
                "y": int(obj.y),
                "w": int(obj.w),
                "h": int(obj.h),
                "cx": round(cx, 2),
                "cy": round(cy, 2),
            })
            frame.draw_rect(obj.x, obj.y, obj.w, obj.h, color=image.COLOR_RED)
            frame.draw_circle(int(cx), int(cy), 4, color=image.COLOR_GREEN)
            frame.draw_string(
                int(obj.x),
                max(0, int(obj.y) - 22),
                f"{label} {float(obj.score):.2f}",
                color=image.COLOR_RED,
            )

        now = time.ticks_ms()
        elapsed = max(1, now - last_tick)
        instant_fps = 1000.0 / elapsed
        fps = instant_fps if fps <= 0 else fps * 0.85 + instant_fps * 0.15
        last_tick = now
        frame.draw_string(8, 8, f"BULLSEYE  {len(results)}  FPS {fps:.1f}", color=image.COLOR_GREEN)
        disp.show(frame)

        if frame_index % print_interval == 0:
            print(json.dumps({
                "frame": frame_index,
                "fps": round(fps, 2),
                "detections": results,
            }, ensure_ascii=False))


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from maix import app, camera, display, image, time

from vision.native_material import MaterialTracker, NativeMaterialDetector


CONFIG_PATH = PROJECT_ROOT / "config" / "material_native.json"
BUILD_ID = "MATERIAL_NATIVE_V3_TOP_STABLE"


def load_config():
    import json

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def configure_camera(cam, cfg):
    if cfg.get("exposure_us") is not None:
        cam.exposure(int(cfg["exposure_us"]))
    if cfg.get("gain") is not None:
        cam.gain(int(cfg["gain"]))
    gains = cfg.get("white_balance_gain")
    if gains is not None:
        cam.awb_mode(camera.AwbMode.Manual)
        cam.set_wb_gain(gains)


def draw_results(img, detections, detector, source_size, fps):
    sx = source_size[0] / float(img.width())
    sy = source_size[1] / float(img.height())
    for item in detections:
        x = int(item.x)
        y = int(item.y)
        color = image.COLOR_ORANGE if item.predicted else image.COLOR_GREEN
        img.draw_rect(x, y, int(item.w), int(item.h), color, thickness=2)
        img.draw_cross(int(item.u), int(item.v), image.COLOR_BLUE, size=10, thickness=2)
        if item.refined:
            img.draw_circle(int(item.u), int(item.v), max(2, int(item.w * 0.5)), color, thickness=2)
        label = "%s %d,%d" % (
            item.color,
            int(round(item.u * sx)),
            int(round(item.v * sy)),
        )
        img.draw_string(x, max(0, y - 18), label, color=color, scale=1.0)

    mode = detector.mode.upper()
    img.draw_string(4, 4, "%s %s" % (BUILD_ID, mode), color=image.COLOR_YELLOW, scale=1.2)
    img.draw_string(4, 24, "FPS %.1f  N%d" % (fps, len(detections)), color=image.COLOR_YELLOW, scale=1.1)


def main():
    config = load_config()
    camera_cfg = config["camera"]
    cam = camera.Camera(
        int(camera_cfg["capture_width"]),
        int(camera_cfg["capture_height"]),
        fps=int(camera_cfg["fps"]),
        buff_num=int(camera_cfg["buffer_count"]),
    )
    cam.skip_frames(int(camera_cfg.get("warmup_frames", 10)))
    configure_camera(cam, camera_cfg)

    detector = NativeMaterialDetector(config)
    print("VISION_BUILD=" + BUILD_ID)
    print("scene ROI:", config["scenes"][detector.mode])
    tracking = config.get("tracking", {})
    tracker = MaterialTracker(
        alpha=tracking.get("alpha", 0.55),
        max_missed=tracking.get("max_missed", 3),
        stable_frames=tracking.get("stable_frames", 3),
        still_center_px=tracking.get("still_center_px", 1.5),
    )
    disp = display.Display()
    source_size = (int(camera_cfg["width"]), int(camera_cfg["height"]))
    last_ms = time.ticks_ms()
    fps = 0.0
    frame_count = 0

    while not app.need_exit():
        frame_count += 1
        frame = cam.read()
        if frame is None:
            continue
        detections = detector.detect(frame)
        detections = tracker.update(detections)
        if detections and all(tracker.stable(item) for item in detections):
            refine_every = max(1, int(config.get("top_refine", {}).get("every_n_still_frames", 4)))
            if frame_count % refine_every == 0:
                detections = detector.refine_top(frame, detections)
                # Refinement may correct a color label; keep the tracker key
                # aligned with the published material identity.
                tracker.items = {item.color: item for item in detections}

        now = time.ticks_ms()
        dt = max(1, now - last_ms)
        instant = 1000.0 / dt
        fps = instant if fps == 0 else fps * 0.85 + instant * 0.15
        last_ms = now
        draw_results(frame, detections, detector, source_size, fps)
        disp.show(frame)


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from maix import app, camera, display, image

from vision.config import load_config, save_config
from vision.detector import MaterialDetector


# Change this value and run once for every material color.
TARGET_COLOR = "red"
COLLECT_FRAMES = 30


def apply_camera_config(cam, config):
    if config.get("exposure_us") is not None:
        cam.exposure(int(config["exposure_us"]))
    if config.get("gain") is not None:
        cam.gain(int(config["gain"]))
    wb_gain = config.get("white_balance_gain")
    if wb_gain is not None:
        cam.awb_mode(camera.AwbMode.Manual)
        cam.set_wb_gain(wb_gain)


def main():
    config = load_config()
    if TARGET_COLOR not in config["colors"]:
        raise ValueError("Unknown TARGET_COLOR: " + TARGET_COLOR)

    camera_cfg = config["camera"]
    runtime_cfg = config["runtime"]
    capture_size = (
        int(runtime_cfg["capture_width"]),
        int(runtime_cfg["capture_height"]),
    )
    cam = camera.Camera(
        capture_size[0], capture_size[1], image.Format.FMT_BGR888,
        fps=camera_cfg["fps"],
        buff_num=int(runtime_cfg["capture_buffer_count"]),
    )
    cam.skip_frames(camera_cfg["warmup_frames"])
    apply_camera_config(cam, camera_cfg)
    detector = MaterialDetector(config)
    disp = display.Display()
    samples = []
    processing_size = (
        int(runtime_cfg["processing_width"]),
        int(runtime_cfg["processing_height"]),
    )

    while not app.need_exit() and len(samples) < COLLECT_FRAMES:
        maix_frame = cam.read()
        if capture_size == processing_size:
            processing_maix = maix_frame
        else:
            processing_maix = maix_frame.resize(processing_size[0], processing_size[1])
        frame = image.image2cv(processing_maix, ensure_bgr=False, copy=False)
        mask = detector._material_mask(frame)
        contours = detector._candidate_contours(mask)
        if len(contours) != 1:
            preview = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            cv2.putText(preview, "PLACE EXACTLY ONE MATERIAL", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            disp.show(image.cv2image(preview, bgr=True, copy=False))
            continue

        contour = contours[0]
        component = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(component, [contour], -1, 255, -1)
        x, y, w, h = cv2.boundingRect(contour)
        erode_size = max(3, int(round(min(w, h) * 0.10)) | 1)
        component = cv2.erode(component, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size)))
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[component > 0]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[component > 0]
        if len(lab) > 50:
            samples.append((np.median(lab, axis=0), np.median(hsv, axis=0)))

        preview = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.putText(preview, f"{TARGET_COLOR}: {len(samples)}/{COLLECT_FRAMES}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        disp.show(image.cv2image(preview, bgr=True, copy=False))

    if len(samples) < COLLECT_FRAMES:
        return
    lab_values = np.asarray([item[0] for item in samples], dtype=np.float32)
    hsv_values = np.asarray([item[1] for item in samples], dtype=np.float32)
    spec = config["colors"][TARGET_COLOR]
    spec["lab_mean"] = np.median(lab_values, axis=0).round(3).tolist()
    spec["lab_std"] = np.maximum(np.std(lab_values, axis=0) * 2.5, 8.0).round(3).tolist()
    spec["calibrated"] = True

    h_low, s_low, v_low = np.percentile(hsv_values, 2, axis=0)
    h_high, s_high, v_high = np.percentile(hsv_values, 98, axis=0)
    if TARGET_COLOR != "red":
        spec["hsv_ranges"] = [[
            max(0, int(h_low) - 4), min(179, int(h_high) + 4),
            max(0, int(s_low) - 20), 255,
            max(0, int(v_low) - 20), 255,
        ]]
    save_config(config)
    print("calibrated", TARGET_COLOR, spec)


if __name__ == "__main__":
    main()

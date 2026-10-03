"""Build six-color HSV/LAB profiles from the supplied single-material videos.

The tool is conservative: it uses only the largest plausible component in each
frame, rejects border-connected components, and writes a backup before updating
vision.json. Review the printed sample counts before using the result on board.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

COLOR_FILES = {"red": "2.mp4", "blue": "3.mp4", "light_blue": "4.mp4",
               "yellow": "6.mp4", "green": "7.mp4", "black": "8.mp4"}


def collect(video, color, max_frames=240):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    lab_values, hsv_values = [], []
    index = 0
    while len(lab_values) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        index += 1
        # Sampling every second frame reduces near-duplicate samples while
        # retaining lighting and rotation variation.
        if index % 2:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if color == "black":
            mask = cv2.inRange(hsv, (0, 0, 0), (179, 255, 105))
        else:
            # Chromatic candidates only.  Using the whole hue range lets a
            # dark cable or shadow become the largest contour in yellow clips.
            hue_ranges = {
                "yellow": [(15, 45)], "blue": [(95, 145)],
                "light_blue": [(80, 125)], "green": [(35, 90)],
                "red": [(0, 15), (165, 179)],
            }
            mask = np.zeros(hsv.shape[:2], np.uint8)
            for h0, h1 in hue_ranges[color]:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, (h0, 45, 35), (h1, 255, 255)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if not contours:
            continue
        height, width = mask.shape
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            if area < width * height * 0.001 or area > width * height * 0.35:
                continue
            if x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2:
                continue
            candidates.append((area, contour))
        if not candidates:
            continue
        _, contour = max(candidates, key=lambda item: item[0])
        component = np.zeros_like(mask)
        cv2.drawContours(component, [contour], -1, 255, -1)
        x, y, w, h = cv2.boundingRect(contour)
        erode = max(3, int(round(min(w, h) * 0.12)) | 1)
        component = cv2.erode(component, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode)))
        pixels = component > 0
        if int(pixels.sum()) < 100:
            continue
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab_values.append(np.median(lab[pixels], axis=0))
        hsv_values.append(np.median(hsv[pixels], axis=0))
    cap.release()
    if len(lab_values) < 10:
        raise RuntimeError(f"too few valid samples for {color}: {len(lab_values)}")
    return np.asarray(lab_values, np.float32), np.asarray(hsv_values, np.float32)


def profile(lab, hsv, color):
    spec = {
        "lab_mean": np.median(lab, axis=0).round(3).tolist(),
        "lab_std": np.maximum(np.std(lab, axis=0) * 2.5, 8.0).round(3).tolist(),
        "calibrated": True,
    }
    low = np.percentile(hsv, 2, axis=0)
    high = np.percentile(hsv, 98, axis=0)
    s0, v0 = max(0, int(low[1]) - 18), max(0, int(low[2]) - 18)
    if color == "red" or low[0] < 8 and high[0] > 170:
        spec["hsv_ranges"] = [[0, min(20, int(high[0]) + 4), s0, 255, v0, 255],
                               [max(155, int(low[0]) - 4), 179, s0, 255, v0, 255]]
    else:
        spec["hsv_ranges"] = [[max(0, int(low[0]) - 4), min(179, int(high[0]) + 4),
                                s0, 255, v0, 255]]
    return spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--config", type=Path, default=Path("MaixCAMPro_Vision/config/vision.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    backup = args.config.with_suffix(".before_video_calibration.json")
    backup.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {}
    for color, filename in COLOR_FILES.items():
        lab, hsv = collect(args.data_dir / filename, color)
        config["colors"][color].update(profile(lab, hsv, color))
        report[color] = {"samples": len(lab), "lab_median": np.median(lab, axis=0).round(2).tolist(),
                         "hsv_median": np.median(hsv, axis=0).round(2).tolist()}
    args.config.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"config": str(args.config), "backup": str(backup), "profiles": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Measure color distributions from the labelled single-material videos.

This is an offline evidence step.  It does not write ``vision.json`` and does
not pretend that video measurements replace final MaixCAM calibration.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


VIDEO_BY_COLOR = {
    "red": "2.mp4",
    "blue": "3.mp4",
    "light_blue": "4.mp4",
    "yellow": "6.mp4",
    "green": "7.mp4",
    "black": "8.mp4",
}

ROUGH_RANGES = {
    "red": [(0, 12), (168, 179)],
    "blue": [(100, 145)],
    "light_blue": [(82, 118)],
    "yellow": [(15, 45)],
    "green": [(35, 95)],
    "black": [(0, 179)],
}


def disk_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 145), (179, 105, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    result = np.zeros(mask.shape, np.uint8)
    if contours:
        cv2.drawContours(result, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    return result


def material_mask(hsv, color):
    if color == "black":
        return cv2.inRange(hsv, (0, 0, 0), (179, 255, 78))
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for low, high in ROUGH_RANGES[color]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, (low, 45, 30), (high, 255, 255)))
    return mask


def choose_component(mask, color, disk):
    mask = cv2.bitwise_and(mask, disk)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    options = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if not 2500 <= area <= 260000 or min(w, h) < 35:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        if disk[int(round(cy)), int(round(cx))] == 0:
            continue
        fill = area / max(float(w * h), 1.0)
        perimeter = cv2.arcLength(contour, True)
        compactness = 4.0 * np.pi * area / max(perimeter * perimeter, 1.0)
        if color == "black" and compactness < 0.12:
            continue
        options.append((area * (0.6 + 0.4 * min(fill, 1.0)), contour))
    if not options:
        return None
    return max(options, key=lambda item: item[0])[1]


def robust_sample(frame, contour):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    component = np.zeros(hsv.shape[:2], np.uint8)
    cv2.drawContours(component, [contour], -1, 255, -1)
    x, y, w, h = cv2.boundingRect(contour)
    erosion = max(3, int(round(min(w, h) * 0.13)) | 1)
    component = cv2.erode(component, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion, erosion)))
    return lab[component > 0], hsv[component > 0], component


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--preview-dir", type=Path, default=None)
    parser.add_argument("--step", type=int, default=12)
    args = parser.parse_args()
    profiles = {}
    for color, filename in VIDEO_BY_COLOR.items():
        path = args.data_dir / filename
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise SystemExit(f"cannot open {path}")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = 0
        lab_values = []
        hsv_values = []
        areas = []
        accepted = 0
        preview = None
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % max(1, args.step):
                frame_index += 1
                continue
            # Profile statistics are invariant to this scale and the reduced
            # frame keeps the offline pass practical on H.264 source videos.
            frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            disk = disk_mask(frame)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            contour = choose_component(material_mask(hsv, color), color, disk)
            if contour is not None:
                lab, hsv_pixels, component = robust_sample(frame, contour)
                if len(lab) >= 50:
                    lab_values.append(np.median(lab, axis=0))
                    hsv_values.append(np.median(hsv_pixels, axis=0))
                    areas.append(float(cv2.contourArea(contour)))
                    accepted += 1
                    if preview is None:
                        preview = frame.copy()
                        cv2.drawContours(preview, [contour], -1, (0, 255, 0), 2)
                        cv2.putText(preview, f"{color} frame {frame_index}", (10, 28), 0, .75, (0, 255, 255), 2)
            frame_index += 1
        capture.release()
        if not lab_values:
            profiles[color] = {"accepted": 0, "video": filename}
            continue
        lab_array = np.asarray(lab_values, np.float32)
        hsv_array = np.asarray(hsv_values, np.float32)
        profiles[color] = {
            "video": filename,
            "total_frames": total,
            "accepted_frames": accepted,
            "lab_median_of_frame_medians": np.median(lab_array, axis=0).round(3).tolist(),
            "lab_frame_median_min": np.min(lab_array, axis=0).round(3).tolist(),
            "lab_frame_median_max": np.max(lab_array, axis=0).round(3).tolist(),
            "lab_frame_median_std": np.std(lab_array, axis=0).round(3).tolist(),
            "hsv_median_of_frame_medians": np.median(hsv_array, axis=0).round(3).tolist(),
            "hsv_frame_median_min": np.min(hsv_array, axis=0).round(3).tolist(),
            "hsv_frame_median_max": np.max(hsv_array, axis=0).round(3).tolist(),
            "hsv_frame_median_std": np.std(hsv_array, axis=0).round(3).tolist(),
            "area_median": round(float(np.median(areas)), 3),
        }
        if preview is not None and args.preview_dir is not None:
            args.preview_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.preview_dir / f"{color}_profile_preview.jpg"), preview)
    output = args.output or args.data_dir / "single_color_profiles.json"
    output.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "profiles": profiles}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

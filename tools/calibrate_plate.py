"""Offline plate fit from reviewed outer-edge points on a video frame.

White background can connect to a white plate. Never infer the plate from the
largest white region alone. Points are source-image [x, y] pairs, spread over
the visible left, bottom and right arcs; omit the image border and black rings.
Review the JSON report and preview before updating the runtime config.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def fit_boundary(points, source_size, capture_size):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 8:
        raise ValueError("select at least eight outer-edge [x, y] points")
    if not np.all(np.isfinite(points)):
        raise ValueError("boundary points must be finite")
    if np.any(points < 0) or np.any(points >= np.asarray(source_size)):
        raise ValueError("boundary points must be inside the source image")
    # Fit after scaling, also handling nonuniform image scaling correctly.
    points = points * (np.asarray(capture_size) / np.asarray(source_size))
    (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(points.astype(np.float32))
    rx, ry = axis_a / 2.0, axis_b / 2.0
    if min(rx, ry) <= 0 or not np.all(np.isfinite([cx, cy, rx, ry, angle])):
        raise ValueError("invalid fitted ellipse")
    theta = np.deg2rad(angle)
    delta = points - [cx, cy]
    local_x = delta[:, 0] * np.cos(theta) + delta[:, 1] * np.sin(theta)
    local_y = -delta[:, 0] * np.sin(theta) + delta[:, 1] * np.cos(theta)
    error = np.abs(np.sqrt((local_x / rx) ** 2 + (local_y / ry) ** 2) - 1) * min(rx, ry)
    if np.max(error) > 5:
        raise ValueError("points do not fit one ellipse within five capture pixels")
    return [cx, cy, rx, ry, angle], float(np.max(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--points", type=Path, required=True,
                        help="JSON array of reviewed source-image boundary points")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--capture-width", type=int, default=640)
    parser.add_argument("--capture-height", type=int, default=360)
    parser.add_argument("--margin-capture", type=float, default=8.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.frame < 0 or min(args.capture_width, args.capture_height) <= 0 or args.margin_capture < 0:
        parser.error("invalid frame, capture size or margin")

    capture = cv2.VideoCapture(str(args.video))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        parser.error("cannot read selected video frame")
    height, width = frame.shape[:2]
    points = json.loads(args.points.read_text(encoding="utf-8-sig"))
    capture_size = [args.capture_width, args.capture_height]
    ellipse, error = fit_boundary(points, [width, height], capture_size)
    cx, cy, rx, ry, angle = ellipse
    if args.margin_capture >= min(rx, ry):
        parser.error("margin removes the entire ellipse")
    preview = cv2.resize(frame, tuple(capture_size))
    cv2.ellipse(preview, ((cx, cy), (rx * 2, ry * 2), angle), (255, 255, 0), 1)
    cv2.ellipse(preview, ((cx, cy), ((rx - args.margin_capture) * 2,
                                  (ry - args.margin_capture) * 2), angle), (0, 255, 255), 1)
    for px, py in np.asarray(points) * (np.asarray(capture_size) / [width, height]):
        cv2.circle(preview, (round(px), round(py)), 2, (0, 0, 255), -1)
    output = args.output or args.video.with_name(args.video.stem + "_plate_ellipse.json")
    preview_path = output.with_suffix(".png")
    encoded, data = cv2.imencode(".png", preview)
    if not encoded:
        raise RuntimeError("cannot encode preview")
    preview_path.write_bytes(data.tobytes())
    result = {
        "video": str(args.video), "frame": args.frame,
        "points": str(args.points), "source_size": [width, height],
        "capture_size": capture_size,
        "plate_ellipse_capture": [round(v, 3) for v in ellipse],
        "plate_margin_capture": args.margin_capture,
        "max_fit_error_capture": round(error, 3),
        "preview": str(preview_path),
        "review_required": True,
        "format": "[cx, cy, rx, ry, angle_deg]; margin applied only at runtime",
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

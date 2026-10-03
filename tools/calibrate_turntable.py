"""Offline calibration of the three black material slots on the turntable.

The tool deliberately does not modify the runtime configuration.  It creates a
measurement report from an empty-turntable video so that the slot geometry can
be reviewed before it is used on the MaixCAM.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def largest_white_region(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 105), (179, 105, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if not contours:
        return mask
    largest = max(contours, key=cv2.contourArea)
    region = np.zeros_like(mask)
    cv2.drawContours(region, [largest], -1, 255, -1)
    return region


def ellipse_radial_residual(points, ellipse):
    (cx, cy), (axis_a, axis_b), angle = ellipse
    a = max(float(axis_a) * 0.5, 1.0)
    b = max(float(axis_b) * 0.5, 1.0)
    radians = math.radians(float(angle))
    cos_a, sin_a = math.cos(radians), math.sin(radians)
    points = np.asarray(points, dtype=np.float64)
    dx = points[:, 0] - float(cx)
    dy = points[:, 1] - float(cy)
    x = dx * cos_a + dy * sin_a
    y = -dx * sin_a + dy * cos_a
    radial = np.sqrt((x / a) ** 2 + (y / b) ** 2)
    return np.abs(radial - 1.0) * min(a, b)


def ring_candidates(frame, white_region, max_x_ratio=0.80):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.2)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=110,
        param1=70,
        param2=28,
        minRadius=120,
        maxRadius=190,
    )
    if circles is None:
        return []

    height, width = gray.shape
    candidates = []
    for cx, cy, radius in np.round(circles[0]).astype(np.int32):
        if not (0 <= cx < width and 0 <= cy < height):
            continue
        # In this camera view the right edge contains printed black text whose
        # arcs pass Hough's circle test.  The empty-disc recording shows that
        # real slot centers stay left of this boundary; keep this calibration
        # guard explicit rather than silently using a global detector rule.
        if cx > width * float(max_x_ratio):
            continue
        if white_region[cy, cx] == 0:
            continue
        inner_radius = max(12, int(round(radius * 0.68)))
        inner = np.zeros_like(white_region)
        cv2.circle(inner, (int(cx), int(cy)), inner_radius, 255, -1)
        inside = white_region[inner > 0]
        if len(inside) == 0 or float(np.mean(inside > 0)) < 0.70:
            continue

        # A real thin ring has dark pixels near its circumference, while its
        # center remains mostly white.  This rejects most external text arcs.
        annulus = np.zeros_like(white_region)
        cv2.circle(annulus, (int(cx), int(cy)), int(round(radius * 1.00)), 255, 3)
        annulus_inside = white_region[annulus > 0]
        inside_ratio = float(np.mean(annulus_inside > 0)) if len(annulus_inside) else 0.0
        if inside_ratio < 0.78:
            continue
        annulus_values = gray[(annulus > 0) & (white_region > 0)]
        dark_ratio = float(np.mean(annulus_values < 150)) if len(annulus_values) else 0.0
        if dark_ratio < 0.10:
            continue
        candidates.append(
            {
                "x": int(cx),
                "y": int(cy),
                "radius": int(radius),
                "score": round(dark_ratio, 4),
            }
        )

    # Hough can return nearby versions of the same ring.
    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = []
    for candidate in candidates:
        if all(
            math.hypot(candidate["x"] - other["x"], candidate["y"] - other["y"]) > 75
            for other in selected
        ):
            selected.append(candidate)
    return selected


def fit_common_orbit(points, seed=7):
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 5:
        return None, np.zeros(len(points), dtype=bool)

    rng = np.random.default_rng(seed)
    best_ellipse = None
    best_inliers = np.zeros(len(points), dtype=bool)
    iterations = min(500, max(80, len(points) * 3))
    for _ in range(iterations):
        sample = points[rng.choice(len(points), size=5, replace=False)]
        try:
            ellipse = cv2.fitEllipse(sample)
        except cv2.error:
            continue
        (cx, cy), (axis_a, axis_b), _ = ellipse
        if min(axis_a, axis_b) < 250 or max(axis_a, axis_b) > 2600:
            continue
        residual = ellipse_radial_residual(points, ellipse)
        inliers = residual < 85.0
        if int(inliers.sum()) > int(best_inliers.sum()):
            best_ellipse, best_inliers = ellipse, inliers

    if best_ellipse is None or int(best_inliers.sum()) < 5:
        return None, best_inliers
    try:
        refined = cv2.fitEllipse(points[best_inliers])
    except cv2.error:
        refined = best_ellipse
    refined_residual = ellipse_radial_residual(points, refined)
    return refined, refined_residual < 85.0


def ellipse_parameter(points, ellipse):
    (cx, cy), (axis_a, axis_b), angle = ellipse
    radians = math.radians(float(angle))
    cos_a, sin_a = math.cos(radians), math.sin(radians)
    points = np.asarray(points, dtype=np.float64)
    dx = points[:, 0] - float(cx)
    dy = points[:, 1] - float(cy)
    x = dx * cos_a + dy * sin_a
    y = -dx * sin_a + dy * cos_a
    return np.arctan2(y / max(float(axis_b), 1.0), x / max(float(axis_a), 1.0))


def phase_for_point(point, ellipse):
    phase = ellipse_parameter(np.asarray([point], dtype=np.float32), ellipse)[0]
    return float(np.degrees(phase) % 360.0)


def track_candidates(records, max_jump=230.0, max_missed=2):
    """Link ring candidates in time; static text arcs fail the motion test."""
    active = []
    finished = []
    for record in records:
        candidates = record["candidates"]
        for track in active:
            track["missed"] += 1
        used = set()
        updated = []
        for track in sorted(active, key=lambda item: item["missed"]):
            if not candidates:
                continue
            distances = []
            for index, candidate in enumerate(candidates):
                if index in used:
                    distances.append(float("inf"))
                    continue
                last = track["points"][-1]
                distance = math.hypot(candidate["x"] - last[0], candidate["y"] - last[1])
                radius_delta = abs(candidate["radius"] - track["radius"])
                distances.append(distance + 2.5 * radius_delta)
            best_index = int(np.argmin(distances)) if distances else -1
            if best_index >= 0 and distances[best_index] <= max_jump:
                candidate = candidates[best_index]
                used.add(best_index)
                track["points"].append((candidate["x"], candidate["y"]))
                track["radius_values"].append(candidate["radius"])
                track["last_frame"] = record["frame"]
                track["missed"] = 0
                updated.append(track)
            elif track["missed"] > max_missed:
                finished.append(track)
            else:
                updated.append(track)
        for index, candidate in enumerate(candidates):
            if index in used:
                continue
            updated.append(
                {
                    "start_frame": record["frame"],
                    "last_frame": record["frame"],
                    "points": [(candidate["x"], candidate["y"])],
                    "radius": candidate["radius"],
                    "radius_values": [candidate["radius"]],
                    "missed": 0,
                }
            )
        active = updated
    finished.extend(active)
    useful = []
    for track in finished:
        points = np.asarray(track["points"], dtype=np.float32)
        if len(points) < 8:
            continue
        span = float(np.max(np.ptp(points, axis=0)))
        path = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        if span < 180.0 or path < 260.0:
            continue
        track["point_count"] = len(points)
        track["motion_span_px"] = round(span, 3)
        track["path_length_px"] = round(path, 3)
        track["radius_median_px"] = round(float(np.median(track["radius_values"])), 3)
        track.pop("missed", None)
        useful.append(track)
    useful.sort(key=lambda item: item["point_count"], reverse=True)
    return useful


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--preview", type=Path, default=None)
    parser.add_argument("--max-x-ratio", type=float, default=0.80)
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    step = max(1, int(args.step))

    records = []
    all_points = []
    preview_frames = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step != 0:
            index += 1
            continue
        white_region = largest_white_region(frame)
        candidates = ring_candidates(frame, white_region, args.max_x_ratio)
        records.append({"frame": index, "time_s": index / fps if fps else None, "candidates": candidates})
        for candidate in candidates:
            all_points.append((candidate["x"], candidate["y"]))
        if len(preview_frames) < 24 and index % max(step, frame_count // 24 or 1) == 0:
            preview = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            ratio_x, ratio_y = 640 / frame.shape[1], 360 / frame.shape[0]
            for candidate in candidates:
                center = (int(candidate["x"] * ratio_x), int(candidate["y"] * ratio_y))
                cv2.circle(preview, center, max(3, int(candidate["radius"] * ratio_x)), (0, 0, 255), 2)
            cv2.putText(preview, f"frame {index}", (8, 22), 0, 0.65, (0, 255, 255), 2)
            preview_frames.append(preview)
        index += 1
    capture.release()

    ellipse, inliers = fit_common_orbit(all_points)
    tracks = track_candidates(records)
    tracked_points = [point for track in tracks for point in track["points"]]
    tracked_ellipse, tracked_inliers = fit_common_orbit(tracked_points)
    result = {
        "video": str(args.video),
        "frame_count": frame_count,
        "fps": fps,
        "sample_step": step,
        "candidate_filter": {"max_x_ratio": args.max_x_ratio},
        "candidate_count": len(all_points),
        "tracked_candidate_count": len(tracked_points),
        "tracks": tracks,
        "common_orbit": None,
        "records": records,
    }
    if ellipse is not None:
        (cx, cy), (axis_a, axis_b), angle = ellipse
        result["common_orbit"] = {
            "center": [round(float(cx), 3), round(float(cy), 3)],
            "axes": [round(float(axis_a), 3), round(float(axis_b), 3)],
            "angle_deg": round(float(angle), 3),
            "inlier_count": int(inliers.sum()),
            "inlier_ratio": round(float(inliers.mean()), 4),
        }
        params = ellipse_parameter(np.asarray(all_points)[inliers], ellipse)
        result["inlier_angle_range_deg"] = [
            round(float(np.degrees(np.min(params))), 3),
            round(float(np.degrees(np.max(params))), 3),
        ]
        initial_phases = [
            phase_for_point((item["x"], item["y"]), ellipse)
            for item in records[0]["candidates"]
        ] if records else []
        result["slot_geometry"] = {
            "slot_count": 3,
            "angular_spacing_deg": 120.0,
            "ring_radius_median_px": round(float(np.median([
                item["radius"] for record in records for item in record["candidates"]
            ])), 3) if records else None,
            "initial_visible_phases_deg": [round(value, 3) for value in initial_phases],
            "phase_reference": "The first phase is an arbitrary image/video reference; map it to physical ring 1/2/3 after the competition ring labels are visible.",
        }
    if tracked_ellipse is not None:
        (cx, cy), (axis_a, axis_b), angle = tracked_ellipse
        result["tracked_orbit"] = {
            "center": [round(float(cx), 3), round(float(cy), 3)],
            "axes": [round(float(axis_a), 3), round(float(axis_b), 3)],
            "angle_deg": round(float(angle), 3),
            "inlier_count": int(tracked_inliers.sum()),
            "inlier_ratio": round(float(tracked_inliers.mean()), 4),
        }
    else:
        result["tracked_orbit"] = None

    output = args.output or args.video.with_name(args.video.stem + "_turntable_calibration.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.preview is not None and preview_frames:
        rows = []
        for start in range(0, len(preview_frames), 6):
            row = preview_frames[start:start + 6]
            while len(row) < 6:
                row.append(np.zeros_like(preview_frames[0]))
            rows.append(cv2.hconcat(row))
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.preview), cv2.vconcat(rows))

    print(json.dumps({
        "output": str(output),
        "candidate_count": len(all_points),
        "tracked_candidate_count": len(tracked_points),
        "common_orbit": result["common_orbit"],
        "tracked_orbit": result["tracked_orbit"],
        "track_count": len(tracks),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

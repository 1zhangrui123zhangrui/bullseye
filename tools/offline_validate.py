import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from vision.config import load_config
from vision.detector import MaterialDetector, draw_debug


def iter_frames(path):
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        # cv2.imread on Windows may fail for Chinese paths.
        data = path.read_bytes()
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            yield 0, frame
        return

    capture = cv2.VideoCapture(str(path))
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % 15 == 0:
            yield index, frame
        index += 1
    capture.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument(
        "--scene",
        choices=("raw_material", "processing_area"),
        default=None,
        help="override the scene mode from vision.json",
    )
    parser.add_argument("--no-images", action="store_true", help="do not write debug JPEGs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    detector = MaterialDetector(load_config())
    if args.scene is not None:
        detector.set_scene_mode(args.scene)
    paths = sorted(args.input.glob("*")) if args.input.is_dir() else [args.input]
    records = []
    for path in paths:
        for frame_index, frame in iter_frames(path):
            disc, detections, _ = detector.detect(frame)
            record = {
                "file": path.name,
                "frame": frame_index,
                "disc": None if disc is None else disc.__dict__,
                "materials": [item.as_dict() for item in detections],
            }
            records.append(record)
            if not args.no_images:
                debug = draw_debug(frame, disc, detections)
                output_path = args.output / f"{path.stem}_{frame_index:06d}.jpg"
                encoded, buffer = cv2.imencode(".jpg", debug, [cv2.IMWRITE_JPEG_QUALITY, 94])
                if encoded:
                    output_path.write_bytes(buffer.tobytes())
            print(json.dumps(record, ensure_ascii=False))

    with (args.output / "results.json").open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from maix import app, camera, display, image

from vision.config import load_config


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
    camera_cfg = config["camera"]
    cam = camera.Camera(
        camera_cfg["width"],
        camera_cfg["height"],
        image.Format.FMT_BGR888,
        fps=camera_cfg["fps"],
        buff_num=camera_cfg["buffer_count"],
    )
    cam.skip_frames(camera_cfg["warmup_frames"])
    apply_camera_config(cam, camera_cfg)
    disp = display.Display()
    saved_focus_frame = False

    while not app.need_exit():
        frame = cam.read()
        preview = frame.resize(640, 360)
        preview.draw_string(8, 8, "GC4653 2560x1440", image.COLOR_GREEN)
        disp.show(preview)
        # Save one native-resolution frame for checking optical focus. Repeated
        # file writes would add I/O load and create unnecessary files.
        if not saved_focus_frame:
            path = "/root/gc4653_focus_test.jpg"
            frame.save(path)
            print("saved focus test:", path)
            saved_focus_frame = True


if __name__ == "__main__":
    main()

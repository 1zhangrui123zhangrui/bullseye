# MaixCAM Pro Material Vision

This project now uses a native MaixPy material pipeline. The runtime entry is
the project root `main.py`; run the complete folder from MaixVision.

## Runtime pipeline

`apps/material_detect.py` performs the following work per frame:

1. Capture one `640x360` Maix image at the configured camera FPS.
2. Run one multi-threshold LAB `Image.find_blobs()` scan.
3. Filter blobs by scene ROI, area, aspect ratio, fill ratio, and optional plate ellipse.
4. Use the blob `cx/cy` as the material center and map it to the `2560x1440` logical image.
5. Apply a short per-color EMA tracker for transient misses.
6. Draw on the same image and call `disp.show(frame)` once.

The runtime deliberately does not use OpenCV, Numpy, full-frame LAB conversion,
Hough circles, CLAHE, or per-frame ellipse fitting. Those operations were the
source of the previous frame-rate failures. Geometric center refinement should
be a separate stopped/grab operation, not part of the moving loop.

## Files

```text
main.py
apps/material_detect.py
vision/native_material.py
config/material_native.json
apps/target_detect.py          optional target-ring test
```

`config/material_native.json` contains provisional LAB thresholds converted
from the supplied videos. They must be checked on the final MaixCAM Pro
lighting and exposure. The six threshold ranges are native MaixPy LAB values:
`L[0,100]`, `a[-128,127]`, `b[-128,127]`.

The raw-material ROI is currently `[52, 0, 546, 290]` in the `640x360`
capture image. It is a conservative starting gate, not a substitute for a
final installation calibration. When the camera is fixed, measure the actual
turntable ellipse and set `plate_ellipse_capture` to
`[center_x, center_y, radius_x, radius_y]`.

## Why this design

The official MaixPy examples and documentation use native `find_blobs` for
color segmentation and return blob rectangle/pixel/center information. The
official camera documentation recommends setting the capture resolution/FPS
at camera construction and using `buff_num=1` to reduce latency. MaixVision
preview is supplied by `disp.show(img)`, so the loop does not create a second
OpenCV display buffer.

References:

- https://github.com/sipeed/MaixPy
- https://github.com/sipeed/MaixPy/blob/main/docs/doc/zh/vision/find_blobs.md
- https://github.com/sipeed/MaixPy/blob/main/docs/doc/zh/vision/camera.md
- https://github.com/sipeed/MaixPy/blob/main/docs/doc/zh/basic/maixvision.md
- https://github.com/lc007-xl/MaixCamPro-Laser-Tracking-System

The local XAUT reference was also reviewed. Its HSV mask plus largest contour
and `minAreaRect` center are useful principles, but its fixed single-target
thresholds and repeated OpenCV passes are not suitable as the new board runtime.

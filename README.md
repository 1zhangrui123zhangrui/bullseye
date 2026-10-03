# MaixCAM Pro Material Vision

This project now uses a native MaixPy material pipeline. The material runtime
entry is the project root `main.py`; run the complete folder from MaixVision.
`target_main.py` is the separate target-ring entry for bullseye testing.

## Runtime pipeline

`apps/material_detect.py` performs the following work per frame:

1. Capture one `640x360` Maix image at the configured camera FPS.
2. Run one multi-threshold LAB `Image.find_blobs()` scan.
3. Filter blobs by scene ROI, area, aspect ratio, fill ratio, and optional plate ellipse.
4. Use the blob `cx/cy` as the material center and map it to the `2560x1440` logical image.
5. Suppress overlapping color boxes, then apply EMA to current observations.
   Missed observations remain association memory only and are not displayed.
6. When a tracked candidate has stayed still for several frames, search only
   its small ROI for a top-face circle. The refined circle center replaces the
   color-blob centroid, and color is rescored only in the circle interior.
7. Draw on the same image and call `disp.show(frame)` once.

The runtime deliberately does not use OpenCV, Numpy, full-frame LAB conversion,
Hough circles, CLAHE, or per-frame ellipse fitting. Geometric center refinement should
be a separate stopped/grab operation, not part of the moving loop.

## Files

```text
main.py
target_main.py
apps/material_detect.py
apps/target_detect.py
vision/native_material.py
config/material_native.json
config/target.json
models/model_9512.mud
models/model_9512.cvimodel
tools/calibrate_plate.py       offline outer-plate ellipse calibration
```

`config/material_native.json` contains provisional LAB thresholds converted
from the supplied videos. They must be checked on the final MaixCAM Pro
lighting and exposure. The six threshold ranges are native MaixPy LAB values:
`L[0,100]`, `a[-128,127]`, `b[-128,127]`.

The raw-material ROI is currently `[52, 0, 546, 290]` in the `640x360`
capture image. A fitted outer-plate ellipse is now also enabled:
`plate_ellipse_capture = [cx, cy, rx, ry, angle_deg]`. The runtime checks the
actual candidate bounding box inside this ellipse, not a box centered on its
pixel centroid. Current geometry is a provisional fit to the visible outer
arc in screenshot `caf81045-2a48-471f-8883-2ff024d12588`, with capture-image
ellipse `[323.53, 5.01, 261.62, 283.92, 168.95]`. Camera pose changes require
new calibration. The old automatic white-region fit from `0.mp4` was invalid:
it included pale background connected to the plate. Do not reuse that report.

For a new video, select at least eight source-image points spread along the
actual left, bottom and right plate arcs. Store them as a JSON array of
`[x, y]` pairs; exclude the cut-off image border and the three black rings.
Run on the computer:

```text
python tools/calibrate_plate.py <empty_turntable.mp4> --points <edge_points.json> --output <report.json>
```

Inspect the generated PNG overlay first: cyan is the fitted edge and yellow
is the inset acceptance boundary. A low fit error alone is not verification.
After confirming alignment, copy the reported `plate_ellipse_capture` value into
`config/material_native.json`. The report contains the fitted, unshrunk
ellipse; keep `plate_margin_capture` as the single safety margin applied by
the runtime. The tool is offline only and never runs on the MaixCAM.

Color boxes overlapping at least 55% of the smaller box compete as one object.
A chromatic region needs at least half the black candidate's matched pixels
to supersede it; small colored reflections alone do not override black.
This is a geometry heuristic, not a guarantee of correct color classification.
Pixel centers remain blob centroids, not refined top-circle centers. Scaling
their coordinates to 2560x1440 does not create full-resolution precision.

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

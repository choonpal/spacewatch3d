# Alternate real RGB-D source for the map-change demo

Checked 2026-10-15. The selected source is **Bonn RGB-D Dynamic Dataset / removing_nonobstructing_box**, from the University of Bonn. This is a real object removal within one continuous recording. Two stable temporal windows can supply before/after maps; it is **not a pair of independently revisited room scans**. [Official dataset and sequence download](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html)

## Direct access and actual contents

- Official anonymous [ZIP download](https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/rgbd_bonn_removing_nonobstructing_box.zip), tested with HTTP 206 byte ranges; total **271,656,752 bytes**. No login, form, or mirror was used.
- The ZIP directory was inspected directly: 994 entries, **494 RGB PNGs, 494 depth PNGs, 496 ground-truth poses**, plus `rgb.txt`, `depth.txt`, directories. First/last RGB timestamps are `1548265286.56559` / `1548265303.08894` (16.52335 seconds).
- RGB originals are 640×480. The dataset page states depth is registered to RGB. Camera intrinsics: fx=542.822841, fy=542.576870, cx=315.593520, cy=237.756098; distortion [0.039903, -0.099343, -0.000730, -0.000144, 0]. [Official calibration](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html)
- Depth PNG values divide by **5000** to obtain the nominal depth in meters; zero is invalid. This matches the original authors' `sensor.depth_factor=5000` in their example and the TUM format to which Bonn refers. Do not use a millimeter divisor of 1000. [Original ReFusion example, lines 36–44](https://github.com/PRBonn/refusion/blob/master/src/example/example.cpp#L36-L44), [TUM official formats](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)

## Actual visual inspection and proposed epochs

Original RGBs retrieved individually from the ZIP and viewed, with no object labels supplied to a detector:

| Timestamp | Seconds from first RGB | Visible state |
|---|---:|---|
|1548265288.20838|1.643|Large cardboard box on the floor, person absent; fairly sharp|
|1548265289.54890|2.983|Same box stationary, person absent|
|1548265290.55454|3.989|Same box stationary, person absent; sharp|
|1548265292.06219|5.497|Person enters beside the box; omit from static epoch fusion|
|1548265296.55318|9.988|Box and person gone; moderate motion blur|
|1548265298.56382|11.998|Box and person gone; clear empty floor|
|1548265301.44615|14.881|Box and person gone; sharp, viewpoint farther left|
|1548265303.08894|16.523|Box and person gone; sharp|

Use approximately **+1.0..4.5 seconds** and **+11.5..16.3 seconds**, or narrower sharp windows within them. Do not fuse the person manipulating the box into either static map. The time windows are selected for stable reconstruction, not change-label inference. The actual detector must infer the region from geometry/depth. This example contains visible removal; do not fabricate all three event types from this single manipulation.

Small originals and the exact ZIP inventory are in `outputs/bonn-source-inspection-20261015/` (`rgb/`, `depth/`, `groundtruth.txt`, `rgb.txt`, `depth.txt`, `inventory.json`). Full source ZIP/extraction is managed separately by the implementing task in `data/bonn/removing_nonobstructing_box/`.

## Bonn pose convention: special calibration is essential

`groundtruth.txt` has `timestamp tx ty tz qx qy qz qw`. TUM describes that format as camera orientation/position relative to the world, but Bonn additionally documents a ROS coordinate bug and a fixed sensor-to-marker calibration. **Do not use the raw GT matrix directly as an RGB optical pose.** [Bonn evaluation/calibration section](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html), [TUM pose convention](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)

The official Bonn Python helper computes `T_g = S @ T_0 @ S @ M`; `S^-1=S`. [Original helper](https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/compute_global_transformation.py). The maintained RTAB-Map implementation explicitly uses Bonn pose-format 12, `S @ pose @ S @ M`, then removes its own internal optical-axis transform. [Bonn format selection and intrinsics](https://github.com/introlab/rtabmap/blob/master/tools/RgbdDataset/main.cpp#L209-L224), [Bonn format-12 implementation](https://github.com/introlab/rtabmap/blob/master/corelib/src/Graph.cpp)

```
S = [[-1,0,0,0],
     [0,0,1,0],
     [0,1,0,0],
     [0,0,0,1]]
M = [[1.0157,0.1828,-0.2389,0.0113],
     [0.0009,-0.8431,-0.6413,-0.0098],
     [-0.3009,0.6147,-0.8085,0.0111],
     [0,0,0,1]]
```

The published `M[:3,:3]` is a near-uniform **similarity**, not a pure rotation: measured singular values [1.05934755, 1.05928623, 1.05923757], determinant 1.18862586. A rigid-camera TSDF interface can use its polar rotation `R=U@Vt`, mean scale **1.05929045**, camera pose `S @ GT(t) @ S @ [R,t_M]`, and depth multiplied by that mean scale after dividing by 5000. This factorization is an implementation inference from the published calibration; the tiny anisotropy remains an approximation. Do not pass the scaled 3×3 block to an interface that assumes orthonormal extrinsics.

Read-only probe `outputs/bonn-source-inspection-20261015/pose_probe.py` transformed three actual depth pairs (RGB observations at +1.64s vs +5.50s/+11.03s/+14.88s) and measured nearest-neighbor distances without ICP or labels. Naive raw-GT median distances were **33.2/24.6/18.5cm**; official calibrated similarity gave **2.15/3.62/4.92cm**; polar-rotation factor without depth-scale compensation gave **2.35/3.99/4.95cm**. These are coordinate sanity checks across partially overlapping views, including changed/occluded surfaces, **not detector accuracy metrics**. The detector must still tolerate depth noise, camera calibration error, blur, and partial overlap.

## Why the initially considered MultiChange3D was not selected

MultiChange3D is genuinely different from Rescan and provides aligned RGB-D-reconstructed XYZRGB point-cloud pairs, with change labels that can be ignored. Its currently listed distribution is PLY point clouds and does not provide the raw RGB-D frames or poses required by the current photo/depth workflow. TU Wien's object-change data similarly offers reconstructed PCD publicly but says raw rosbags require contacting the author. ETH ASL offers aligned meshes and rosbags, but the old direct listing timed out during this bounded check. [MultiChange3D official distribution](https://github.com/3DOM-FBK/multichange3d), [TU Wien official data contents](https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/object-change-detection-dataset-of-indoor-environments/), [ETH ASL contents](https://github.com/ethz-asl/change_detection_ds)

Credit the Bonn dataset authors and cite Palazzolo et al., ReFusion (IROS 2019), as the official page requests. Do not silently apply the ReFusion **code** license to the dataset: the dataset page does not visibly state a separate data license in the inspected text.

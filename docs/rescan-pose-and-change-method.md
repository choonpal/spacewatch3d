# Rescan camera recovery and geometry attention regions

Checked 2026-10-14 (Asia/Seoul). Primary-source research for the real-pair viewer. This note describes formats and proposed validation; it is not a report of completed camera recovery or detection accuracy.

## What the source data establishes

Rescan contains real repeated indoor RGB-D captures. Its released `gt_segmentation/*.ply` is a **surfel point cloud**, with XYZ, normals, RGB, radius, semantic class and temporally persistent instance ID. The documented header has vertices and no triangle faces. Stable IDs express object associations across captures; they are annotations, not predicted change labels. The archive audit in [real-instance-pair-access.md](real-instance-pair-access.md) found no calibration or camera-trajectory sidecars. [Authors' dataset page](https://rescan.cs.princeton.edu/)

The supplemental paper identifies the acquisition system as the Structure Occipital Sensor and **ScanNet Capture App**, and the reconstruction as Fine-to-Coarse Global Registration. It also explains ambiguous associations among similar objects and partial-scan failure modes. This establishes the raw-stream lineage, but does not supply missing camera calibration. [Rescan supplement, sections 2 and 3.4](https://rescan.cs.princeton.edu/assets/Halber19Rescan_supp.pdf)

## Concrete raw depth decoder

Official sources:

- [ScanNet depth2pgm](https://github.com/ScanNet/ScanNet/blob/master/ScannerApp/depth2pgm/depth2pgm.cpp)
- [Occipital codec header](https://github.com/ScanNet/ScanNet/blob/master/ScannerApp/depth2pgm/uplinksimple_image-codecs.h), [raw header](https://raw.githubusercontent.com/ScanNet/ScanNet/master/ScannerApp/depth2pgm/uplinksimple_image-codecs.h)
- [Shift-to-depth lookup table](https://github.com/ScanNet/ScanNet/blob/master/ScannerApp/depth2pgm/uplinksimple_shift2depth.h), [raw header](https://raw.githubusercontent.com/ScanNet/ScanNet/master/ScannerApp/depth2pgm/uplinksimple_shift2depth.h)

The raw `.depth` consists of records: little-endian `uint32` compressed payload size, then that many codec bytes. Frame dimensions are external; the helper defaults to 640 × 480. The payload is an MSB-first differential/run-length stream initialized with last value zero:

| Prefix | Decoded shift value |
| --- | --- |
| `00` | Same as previous |
| `11` | Previous + 1 |
| `10` | Previous − 1 |
| `010` + 5 bits | Repeat previous **N + 5** times |
| `01100` | Previous − 2 |
| `01101` | Previous + 2 |
| `0111` + 11 bits | Absolute reset |

Values are sensor **shifts**, not millimeters. Apply the official lookup table; the decoder marks lookup values at/above the invalid maximum 9729 as zero. Valid lookup outputs convert millimeters to meters by dividing by 1000. Preserve source byte offsets and decoded frame ordinal, and reject truncated payloads, impossible runs or incorrect decoded pixel counts. Do not treat the official header's abbreviated run-length comment as the implementation: it omits the `+5` written in the decoder.

The capture implementation writes **two** `double` timestamp arrays after all compressed records: depth timestamps, followed by RGB timestamps, each with the captured-frame count. The standalone helper reads only the first array. Without the metadata frame count, a candidate record count N can be validated when remaining file bytes equal 16N, both timestamp arrays are finite and monotonic, and decoded RGB frame count agrees. Check the actual Rescan stream before asserting this layout. [ScanNet capture implementation, `writeDepthFrame` and `closeFile`](https://github.com/ScanNet/ScanNet/blob/master/ScannerApp/Scanner/ViewController%2BSensor.mm)

## Intrinsics and RGB alignment

The ScanNet app's default VGA parameters provide **initial estimates**, not the missing Rescan calibration: depth `fx = fy = 570.5`, center `(320,240)`; RGB `fx = fy = 578`. Hardware-registered depth defaults to false. [Official Options structure](https://github.com/ScanNet/ScanNet/blob/master/ScannerApp/Scanner/ViewController.h)

At runtime the app replaces these values from the sensor projection matrices and writes calibration to `.txt`. The helper uses image centers `(width−1)/2`, `(height−1)/2`. The app's high-resolution RGB mode captures 2592 × 1936 and encodes half resolution, 1296 × 968; a commented RGB focal initialization suggests approximately 1170.45 pixels after halving. This is a search seed only. The metadata format has independent RGB/depth focal lengths and centers plus `colorToDepthExtrinsics`. [Capture implementation](https://github.com/ScanNet/ScanNet/blob/master/ScannerApp/Scanner/ViewController%2BSensor.mm), [RGB resolution settings](https://github.com/ScanNet/ScanNet/blob/master/ScannerApp/Scanner/ViewController.mm), [metadata loader](https://github.com/ScanNet/ScanNet/blob/master/Converter/src/metaData.h)

Resize an image's own intrinsic parameters with its image resize; crop offsets also change the principal point. **Scaling depth intrinsics to RGB dimensions does not calibrate a separate RGB camera.** The original calibration is unavailable in the inspected release, so estimated poses/intrinsics must remain explicitly marked as estimated. [OpenCV camera model and calibration](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

### Proposed route to original-frame linkage

1. Decode actual depth and original RGB. Retain ordinal, raw-stream timestamps, source scan ID and original image dimensions. H.264 decoding order must be display order; validate count and synchronization instead of assigning an assumed constant FPS timestamp.
2. Back-project depth using an estimated depth K; make a coarse point cloud with valid depths and normals. Start near the documented VGA parameters and evaluate a bounded focal-length search.
3. Register a depth-frame cloud to **its own capture's released surfel map**, using several FPFH/RANSAC proposals and robust point-to-plane ICP refinement. A global proposal is needed because ICP expects an adequate initial alignment. Retain multiple plausible hypotheses in repeated furniture or planar-only views. [Open3D global registration](https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html)
4. Accept a depth pose only with broad spatial support, normal agreement and plausible camera position, plus reprojection of held-out depth pixels to the surfel map. Report overlap/inlier ratio, residual quantiles and the gap from the second-best hypothesis. Registration RMSE alone can favor a wrong chair or a single wall.
5. Recover RGB mapping separately: initialize the RGB-to-depth rigid offset and independent RGB K, then align static RGB edges/features to map-rendered RGB/geometry across several frames. Rendering must retain a pixel-to-3D map. Feature correspondences can support `solvePnPRansac`; nonlinear refinement needs validation on features/frames withheld from fitting. **OpenCV `SOLVEPNP_UPNP` is documented as broken and falls back to EPnP**, so it must not be used as a focal-length estimator. [OpenCV PnP and refinement](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)
6. Select before/after keyframes only after camera validation and depth visibility checks. Compare projected region depth to observed depth, measure visible area and framing, and prefer sharper, closer views with independent viewpoints. For a removal/addition counterpart, the frame must observe the region's empty location; an occluded or unscanned counterpart is unknown.

These are proposed recovery steps, not guaranteed automatic recovery. If a depth pose is good but RGB extrinsics are unverified, show the **original RGB candidate frame** with an unverified-link label; do not draw a precise region overlay or claim a camera-ground-truth connection. Generated map renderings are useful diagnostics and must be labeled separately from original RGB.

The authors' original reconstruction software requires per-frame configuration and camera intrinsics too; it does not infer the omitted calibration by simply opening the released PLY. [Fine-to-Coarse `fetregister` instructions](https://github.com/mhalber/Fine-To-Coarse-Registration/blob/master/fetregister/Readme.md)

## What a restored mesh actually contains

The source points measure samples of observed surfaces. A mesh adds connectivity: `vertices[N,3]` are positions; `triangles[M,3]` index three vertices per face. Optional vertex normals and RGB support shading, while face/vertex labels and provenance preserve the closest source surfel/instance. A texture map is optional and is not implied by vertex colors. Keep raw surfels, restored mesh and annotation tables distinct.

Rescan's official pipeline restores a mesh with Poisson reconstruction, using width 0.01, scale 1.25, point weight 0.1, samples per node 5, color and density output, followed by density trimming. [Official restoration script](https://github.com/mhalber/Rescan/blob/master/scripts/run_poisson_reconstruction.py)

Different sampling, reconstruction depth, smoothing and trimming can produce different vertex counts and triangles for the **same physical surface**. Comparing vertex indices, triangle identities or triangle counts across independently restored scans is consequently an invalid physical-change detector. Poisson can extrapolate triangles into sparse or unobserved regions; low-support faces require trimming/masking. [Open3D surface reconstruction documentation](https://www.open3d.org/docs/release/tutorial/geometry/surface_reconstruction.html)

## Objective assessment of geometry-only attention regions

**The user's proposal is sensible as a review-priority mechanism:** a region with persistent, supported surface differences deserves closer observation even when instance segmentation is missing or wrong. The following is a proposed design; its scores are evidence measures, not calibrated probabilities that a change occurred.

### Compare observed surfaces, not mesh topology

- Align scans first on large stable geometry, with robust rejection of moving objects. Use a withheld stable-region residual to measure alignment quality. Full-scene ICP can absorb object motion into the alignment or shift toward a large changed object.
- Sample mesh surfaces uniformly by area, or compare original surfels with local normals. Compute both directed distances and point-to-surface distance; nearest **vertex** distance is affected by triangulation density. Point-to-plane distance suppresses tangential sampling noise but needs a maximum Euclidean correspondence distance and normal agreement so unrelated surfaces do not become matches.
- Retain local raw-point support, density, roughness, reconstruction-to-source error and boundary proximity. A smooth restored patch without measured support is not change evidence.
- Group supported discrepancies in fixed world-space voxels, with minimum area/volume and persistence across scales. Object labels may explain or group a candidate afterward; candidate discovery should run with all instance labels removed as a separate experiment.
- Separate **difference**, **observed free space**, and **unknown coverage**. A bounding-box overlap or missing surface alone does not establish object removal. Depth rays from trustworthy cameras are needed to distinguish observed empty space from occlusion/unscanned areas.

For a local normal-direction displacement estimate, M3C2 compares neighborhood means rather than raw nearest vertices and calculates an uncertainty bound from roughness, point count and registration error. The official implementation uses `LoD95 = 1.96 × (sqrt(sigma_before²/n_before + sigma_after²/n_after) + registration_rms)` with sufficient neighborhood support. This is a useful model for an uncertainty-aware attention score; correlated RGB-D reconstruction samples mean nominal statistical confidence still needs empirical checking. Do not call a simplified nearest-neighbor score “M3C2” or display a 95% confidence claim without implementing and validating its assumptions. [CloudCompare M3C2 documentation](https://www.cloudcompare.org/doc/wiki/index.php/M3C2_(plugin)), [official implementation](https://github.com/CloudCompare/CloudCompare/blob/master/plugins/core/Standard/qM3C2/src/qM3C2Process.cpp)

### Experiments that make the judgment measurable

| Experiment | Question resolved | Report |
| --- | --- | --- |
| Same surfels, independently reconstructed with two resolutions | Does mesh reconstruction itself trigger attention? | False candidate fraction/area and residual quantiles |
| Deterministic odd/even or seeded point split of one scan, reconstructed independently | Is a region stable to sampling density? | Candidate area, retained support and repeatability |
| Small pose perturbations consistent with measured stable-region residual | Is the apparent change within alignment uncertainty? | Candidate survival by perturbation magnitude |
| Same real pair, instance IDs available versus removed | Does discovery depend on perfect segmentation? | Candidate locations/ranks and overlap with annotated objects |
| Real pair plus original before/after keyframe inspection | Are candidates physically supported and interpretable? | Reviewed changed / unchanged / uncertain counts, visibility reasons |
| All real-pair candidates and all failed links, without selecting only attractive examples | Does the result generalize within this pair? | Full inventory and thresholds, along with representative failures |

The first two controls estimate algorithm/sampling sensitivity, **not independent repeated-capture sensor noise**; a same-scan split shares the original reconstruction errors. One real pair cannot establish general accuracy across rooms. Stable instance associations in Rescan can help evaluation, but centroid displacement of a partial object is not an independent ground-truth motion measurement. Similar objects have documented identity ambiguities. Preserve those cases rather than count them as definite errors or successes.

### Viewer labels and evidence

Use `유심 관찰 구역` for supported geometric differences and show distance magnitude, area, support, alignment uncertainty and persistence. Use `관측 부족` when coverage is insufficient; `복원 민감` when the same-scan controls produce similar discrepancies; and `사진으로 확인` only when an actual frame visibly supports the conclusion. Store estimated camera status separately from geometric candidate confidence. Changes found with instance annotations and geometry-only candidates should be inspectable separately.

The useful conclusion is not “mesh difference equals physical change.” It is that an attention region can **prioritize human review while preserving uncertainty**, and the controls above show when the priority score follows reconstruction artifacts instead of actual scene changes.

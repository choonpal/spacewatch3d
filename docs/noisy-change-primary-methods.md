# Noisy surface change and attention-region scoring

Research date: 2026-10-15. Scope updated by the user: **investigate existing techniques/models that identify possibly changed regions in noisy reconstructed 3D maps; do not implement anything.** No viewer, algorithm code, dataset or detector output is changed by this research. Project-specific scoring design is deferred; this note records existing published methods only.

## Research conclusion

Existing work does address this problem without relying on researcher-supplied object-instance labels. The most relevant families are statistically tested point-cloud surface differences (M3C2 and patch-based M3C2), TSDF/free-space differencing, and plane-supported local SDF comparison. They use different inputs and do not offer interchangeable guarantees. Bare mesh pairs omit some observation information retained by TSDF/depth pipelines.

Researcher-supplied instance labels identify object membership; they do not by themselves make XYZ measurements clean. A mesh can look cleaner because reconstruction smooths and extrapolates. Open3D explicitly documents that Poisson reconstruction can create triangles in poorly supported areas; its returned density identifies weakly supported vertices. Hence a mesh-only discrepancy can reflect reconstruction settings or missing observations. [Open3D surface reconstruction](https://www.open3d.org/docs/release/tutorial/geometry/surface_reconstruction.html)

## Existing indoor geometric methods, checked in depth

### Fehr et al., ICRA 2017 — TSDF-based change detection

**Input:** registered per-session TSDF reconstructions, generated from RGB-D observations and poses. A bare triangle mesh alone does not preserve its TSDF measurement weights or observed-free/unknown-space information. GT object segmentation is not required.

**Existing evidence/output:** voxel-wise absolute TSDF difference is thresholded; labels are transferred to mesh vertices, refined by mesh region growing and split into changed object candidates. This is geometric change evidence, not a calibrated change-probability output.

**Noise/coverage:** low measurement-weight voxels are filtered; 3D erosion suppresses scattered noise and dilation recovers geometry. TSDF weights distinguish unobserved space from observed empty space, while a static-map update handles partial occlusions. Alignment must already be reliable; the authors acknowledge remaining local/nonrigid reconstruction misalignment and over/undersegmentation, especially with only two observations. Planar changes are rejected by the object-discovery stage, limiting structural-change coverage.

**Training:** no learned detector or pretrained checkpoint is required. Evaluation includes real Google Tango RGB-D recordings of an office, living room and uncontrolled lounge. [Author-hosted paper](https://tisl.cs.utoronto.ca/publication/201705-icra-tsdf_change_detection/icra17-tsdf_change_detection.pdf)

**Code status:** the public [ETH dataset repository](https://github.com/ethz-asl/change_detection_ds) contains README and previews; it is not the detector implementation. Its release describes aligned meshes, RGB, point clouds and camera transforms. The [official dataset page](https://projects.asl.ethz.ch/datasets/change-detection/) and publication were checked; runnable author detector code has not been verified. Do not present this method as an immediately installable package.

### Fu et al., RA-L/IROS 2022 — PlaneSDF

**Input:** point clouds from two sessions, either complete reconstructions or streams, already in a common world frame. No GT semantic/instance segmentation is needed: planes and object blobs are extracted geometrically.

**Noise/alignment/coverage:** scene geometry is partitioned into plane-associated SDF volumes. Relative plane-pose alignment mitigates small local drift. Height-map differences nominate regions; connected-component filtering/dilation removes speckles; 3D SDF shape comparison rejects false candidates. Height maps explicitly encode unobserved regions. This is local reconstruction-error handling, not tolerance to arbitrary registration failure.

**Published score:** each key voxel has `f = [shape histogram, weighted SDF]`; the paper defines `sim(f,f') = 1 / (1 + alpha * ||f-f'||_2)`. A blob's mean histogram similarity `Havg` below `delta_blob` produces a change label. Outputs include 2D location masks and changed 3D voxels. Similarity is not a calibrated probability.

**Training/limits:** non-learning method, with real noisy indoor RGB-D reconstruction evaluation as well as synthetic scenes. Objects hidden in the height direction and noisy/deforming sofa boundaries are reported failures; horizontal/vertical supporting planes constrain applicability. [Original paper, sections III–IV](https://arxiv.org/pdf/2207.08323)

**Code status:** the [author's publication page](https://jiahui-fu.github.io/) links Paper and Video for PlaneSDF, without an algorithm-code link. The official paper and repository searches did not establish publicly runnable author code/checkpoints. This is a relevant existing method with a reproducibility/access limitation, not a confirmed ready-to-use model.

### Yang and Schwieger, 2023 — patch-based M3C2

**Input/output:** two registered point clouds; no mesh, GT instance matching or learned checkpoint required. Supervoxel patches support local planar projection and prism-shaped comparison regions. Outputs are surface distances plus a detection threshold/significance decision.

The method addresses cylindrical-neighborhood edge mixing, smoothing and the confusion of actual surface geometry with measurement noise. It incorporates within-patch correlation into an effective sample count; imperfect patches and inappropriate stochastic assumptions still matter. It measures separation along specified directions, not automatic semantic moved/added/removed identity. [Primary paper](https://doi.org/10.1016/j.jag.2023.103535)

**Naming caution:** this 2023 **patch-based M3C2 (PB-M3C2)** is distinct from Zahs et al.'s 2022 **correspondence-driven plane-based M3C2**, which appears as `PBM3C2` in py4dgeo. Do not claim that installing that class reproduces the 2023 method. Public author implementation/checkpoints have not been verified. [2023 paper comparison of the two approaches](https://www.sciencedirect.com/science/article/pii/S156984322300359X), [py4dgeo API](https://py4dgeo.readthedocs.io/en/latest/pythonapi.html)

### Existing-method suitability for the current noisy-map question

| Existing method | Direct input | Main noise/error handling | Existing numerical evidence | Training / runnable author artifact |
| --- | --- | --- | --- | --- |
| M3C2 / M3C2-EP | Aligned point clouds; EP needs additional uncertainty metadata | Neighborhood roughness/count and alignment uncertainty; EP propagates covariance | Signed separation and LoD; significant/non-significant/unknown | No detector training; public CloudCompare/py4dgeo implementations |
| Patch-based M3C2 (2023) | Aligned point clouds | Local planar patches, projection, spatial correlation | Distance and LoD/significance | No training; author implementation not verified |
| Fehr TSDF (2017) | Aligned TSDF volumes derived from RGB-D/poses | Measurement weights, erosion/dilation, free/unknown distinction | TSDF difference and changed-region/object masks | No training; public data verified, detector code not verified |
| PlaneSDF (2022) | Aligned complete/streamed point clouds | Plane-wise drift handling, 2D filtering, 3D SDF shape evidence | Key-voxel similarity and blob similarity threshold | No training; author algorithm code not verified |
| NDT spatial change (2019) | Precise reference point map plus live range/RGB-D points and localization | Voxel point-distribution classes, overlapping grid, repeated-observation voting | Voxel discrepancy/classification and accumulated votes | No learned detector; author implementation not verified |
| NDF change detection (2022) | Separated partial object point clouds | Category shape completion and relative neighboring-object layouts | Shape similarity and graph-edge agreement thresholds | Category-level training required; change-detector code/checkpoints not verified |

The best research match depends on which evidence remains available: rough/noisy geometry alone favors M3C2 variants; indoor plane-supported object changes make PlaneSDF particularly relevant; preserving depth-derived observation/free-space information makes TSDF methods useful for partial coverage. This is a comparison of published capabilities, not a decision to implement any of them.

### Katsura et al., ROBOMECH 2019 — NDT spatial change detection

This existing method compares a high-precision reference point-cloud map with newly acquired RGB-D/range points localized to the map. Normal-distribution voxels are classified by the point distribution; overlapping voxels and votes over sequential observations suppress unstable detections. It produces detected change voxels instead of a calibrated probability map, without learning a neural detector. [Original open-access paper](https://link.springer.com/article/10.1186/s40648-019-0148-8)

The scale and reference-quality assumptions matter: reported experiments use comparatively large voxel/noise settings, so indoor small-object performance cannot be inferred. The authors report a false change where single-view occlusion makes an observed sheet look like a line. Therefore repeated voting is useful but not a complete visibility solution. Runnable author algorithm code/pretrained models were not verified; no training is needed for the geometric method.

## Primary methods worth using

| Method | Relevant mechanism | Use in this project |
| --- | --- | --- |
| M3C2, Lague et al. 2013 | Local normal-aligned surface means and a roughness/count/alignment-dependent detection threshold | Strong no-training baseline for geometric evidence; derive normals and neighborhoods without object segmentation |
| M3C2-EP, Winiwarter et al. 2021 | Propagate sensor and alignment uncertainties, including shared covariance | Later upgrade when trustworthy sensor and registration covariance inputs exist |
| Explicit ray-based change, Underwood et al. 2013 | Distinguish occupied surface, observed free space, and unobserved space | Prevent a missing scan or occlusion from becoming a confident added/removed result |
| Siamese KPConv, de Gélis et al. 2023 | Learn change segmentation directly from two point clouds | Future comparison; published urban benchmarks do not establish indoor RGB-D performance |
| SRGCD, Wu and Ma, CVPR 2026 | Geometric consistency selects stable seeds; controlled region growth improves boundaries | Research alternative for spatial coherence, requiring training/domain evaluation |

Sources: [M3C2 author manuscript](https://arxiv.org/pdf/1302.1183), [M3C2-EP research group](https://www.geog.uni-heidelberg.de/en/3dgeo/projects-of-the-3dgeo-research-group/m3c2-ep), [Underwood author-hosted paper](https://www-personal.acfr.usyd.edu.au/tbailey/papers/icra13underwood.pdf), [Siamese KPConv official code](https://github.com/IdeGelis/torch-points3d-SiameseKPConv), [SRGCD primary paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Wu_SRGCD_Stability-Driven_Region_Growth_Framework_for_3D_Change_Detection_CVPR_2026_paper.pdf).

### M3C2: the exact reference calculation

At a core point `x`, estimate a unit surface normal `n` at a suitable normal radius. Select points from each capture in a cylinder whose axis is `n` through `x`; the cylinder radius is the projection scale. Project the points onto that axis. Let `mu_A`, `mu_B` be their means, `s_A`, `s_B` their sample standard deviations, and `N_A`, `N_B` their counts. Then:

```text
delta = mu_B - mu_A
LoD95 = 1.96 * (sqrt(s_A^2/N_A + s_B^2/N_B) + registration_rms)
```

`delta` is signed surface separation, not the full movement vector of an identified object. `registration_rms` uses the same length unit as coordinates. The official CloudCompare implementation applies the formula when both neighborhoods meet the minimum sample count, and declares significant separation when `abs(delta) > LoD95`. [Official implementation, qM3C2Process.cpp](https://github.com/CloudCompare/CloudCompare/blob/master/plugins/core/Standard/qM3C2/src/qM3C2Process.cpp)

Normal scale controls whether roughness becomes an unstable normal or is averaged away. Corners, thin structures and overlapping surfaces can mix unrelated geometry in one cylinder. Use multiscale normals and record planarity and normal consistency. The original paper allows a robust median alternative, but its Gaussian mean confidence formula no longer directly applies; bootstrap intervals are discussed instead. A normal-direction measurement can miss a tangential displacement. Neither its sign nor a large distance establishes moved/addition/removal semantics. [M3C2 author manuscript, sections 3 and 6](https://arxiv.org/pdf/1302.1183)

**RGB-D caveat:** fused surfels/mesh vertices are correlated observations. Thousands of nearby samples are not necessarily thousands of independent measurements. Taking their raw count in a standard error can make thresholds unrealistically small. A voxel/block support count is a useful conservative engineering substitute, but does not create a proven effective sample size or a calibrated 95% confidence statement. A single ICP fit residual is also not automatically an independent registration uncertainty estimate.

### M3C2-EP: what additional inputs buy

The official `py4dgeo.M3C2EP` workflow requires an alignment covariance matrix, an affine transformation and its reduction point. The implementation additionally uses scan origins and sensor uncertainty information; it returns distance, uncertainty and local 3D covariance. Its purpose is to propagate measurement and alignment uncertainty instead of treating observed surface spread as measurement noise. [Official tutorial](https://py4dgeo.readthedocs.io/en/latest/m3c2ep.html), [official implementation](https://github.com/3dgeo-heidelberg/py4dgeo/blob/main/src/py4dgeo/m3c2ep.py)

A useful conceptual first-order propagation is:

```text
C_X = J C_input J^T
Var(delta | fixed n) = n^T (C_meanA + C_meanB - C_AB - C_AB^T) n
```

`J` is the Jacobian mapping measurement/alignment parameters to XYZ; `C_meanA/B` are covariance matrices of estimated local surface means; `C_AB` represents shared cross-capture errors when present. Shared alignment uncertainty does not vanish simply by increasing the number of points. This is a conceptual explanation, not a complete reimplementation of M3C2-EP. The authors' later primary paper explains Jacobian propagation and correlated alignment errors. [Winiwarter et al. 2023, section 3.2](https://esurf.copernicus.org/articles/11/593/2023/)

Current Rescan camera poses are estimates and no independent sensor/registration covariance has been established. Do not manufacture zero covariance, reuse an arbitrary constant as measured covariance, or call a nearest-neighbor/noise-weighted method full M3C2-EP. Match matrix ordering and transformation conventions to the installed implementation before any future integration.

### Visibility: missing geometry is not observed emptiness

For a measured depth `D(u)` and transformed candidate point depth `z` in a valid calibrated camera:

```text
abs(z - D(u)) <= tolerance: surface is supported
z < D(u) - tolerance:       observed free space at the candidate location
z > D(u) + tolerance:       candidate lies behind a measured surface (occluded)
invalid/outside image:     unknown
```

These rules depend on valid poses, intrinsics, depth units, synchronization and uncertainty margins. Test multiple views when possible. An object at a location in B and observed free space at that location in A supports addition; reverse the observations for removal. Movement needs correspondence between disappearing and appearing geometry. This occupied/free/unknown distinction is the central useful contribution of explicit ray-based change reasoning. [Underwood et al. 2013](https://www-personal.acfr.usyd.edu.au/tbailey/papers/icra13underwood.pdf)

CloudCompare's official documentation likewise leaves points with no counterpart inside the cylinder as NaN: a hole, hidden part or insufficient search length can cause this. Do not replace NaN by a maximum change score. RGB photo availability alone does not establish observed free space or calibrated visibility. [CloudCompare M3C2 documentation](https://www.cloudcompare.org/doc/wiki/index.php/M3C2_(plugin))

## Objective evaluation when a reference becomes available

Current real Rescan pairs have instance labels and existing change estimates, but no independent complete change-region reference has been established for this attention channel. Do not treat the current detector output, its manually selected display list, or transferred instance labels as independent truth. Do not report precision, recall or accuracy as measured before reference labels exist.

Use fixed-world voxel or surface-area weighting rather than raw vertices, which vary with reconstruction density. Predefine one-to-one region matching (for example a declared voxel IoU or distance/coverage tolerance) so splitting one true region into ten candidates does not earn ten true positives. Publish unobserved coverage as unknown and show the fraction of map included in evaluation.

| Measurement | Definition and interpretation |
| --- | --- |
| Region precision | `TP/(TP+FP)`: how many regions a person opens contain a reference change |
| Region recall | `TP/(TP+FN)`: how many reference changes are surfaced |
| PR curve and average precision | Sweep score threshold; useful when changed regions are rare. State whether reporting sklearn AP or interpolated/trapezoidal PR area |
| FROC | Reference-region recall versus false regions per scan pair; directly exposes the review burden |
| Precision@k / Recall@k | Useful findings and captured changes within a fixed review budget, e.g. first 5 or 10 regions |
| Unchanged-area false alarms | False candidate area fraction and candidates per room/pair on genuinely repeated unchanged captures |
| Stability | Region/score agreement under reasonable density, meshing-resolution, normal-scale and alignment perturbations |

Precision, recall and AP definitions are provided by [scikit-learn evaluation documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics). FROC is an established detection-localization protocol; adapting false positives per scan to false regions per scan pair is a project design choice. [Primary ANODE09 evaluation protocol](https://anode09.grand-challenge.org/Details/)

For a useful initial reference, review raw images/depth plus meshes while hiding algorithm scores, annotate changed/unchanged/unknown regions, and include random non-candidate areas to avoid measuring precision only. Reviewers should not use segmentation identity alone to decide actual movement. Separate tuning rooms from evaluation rooms; use room/pair bootstrap intervals rather than pretending adjacent vertices are independent trials.

Controlled perturbations of the same cloud can test noise, density, holes, reconstruction choices, residual rigid offsets and known synthetic object transforms. Those tests have known injected changes but remain **synthetic sensitivity tests**. Two reconstructions of the same surfels share the acquisition errors; random splits of the same capture are also not independent repeat scans. Neither establishes real-world false-positive rates. Preserve these distinctions in the report, without adding a control dashboard to the everyday viewer.

## Learned alternatives: keep the comparison honest

### Fu et al., IROS 2022 — Robust Change Detection Based on Neural Descriptor Fields

This is a more directly relevant indoor learned candidate than urban change networks. **Input is already separated partial object point clouds**, not two unsegmented whole-room meshes. A category-trained NDF predicts full-shape codes/centers; shape similarity associates objects, and relative neighboring-object layouts help distinguish movement from localization drift. The paper uses cosine shape similarity and thresholded edge-vector agreement, with changed-object outputs; these are not calibrated whole-map change probabilities. [Original paper](https://arxiv.org/html/2208.01014v1)

It specifically addresses partial observation overlap and noisy localization, and evaluates synthetic and real sequences containing mug instances that are added, removed and rearranged. Category training and object extraction remain prerequisites; the paper does not establish arbitrary-category, poorly segmented noisy-room-mesh performance. The [official project](https://yilundu.github.io/ndf_change/) links Paper and Video; change-detection algorithm code/checkpoints have not been verified. Do not confuse separately available foundational NDF manipulation code with a released implementation of this change detector.

Siamese KPConv supplies official training/evaluation code and multiclass raw-cloud change detection. Its urban LiDAR context and learned weights require indoor RGB-D adaptation and held-out evaluation; quoted urban scores are not this application's scores. [Official paper](https://doi.org/10.1016/j.isprsjprs.2023.02.001), [official code](https://github.com/IdeGelis/torch-points3d-SiameseKPConv)

DC3DCD is an unsupervised learned alternative, still requiring optimization and domain choices rather than eliminating evaluation. [Authors' preprint](https://arxiv.org/abs/2305.05421)

SRGCD's stable-seed/controlled-region-growth formulation is relevant to suppressing scattered and leaking boundaries. The CVPR paper evaluates urban change datasets; it is evidence for a method worth comparing, not a guarantee for Rescan furniture or noisy indoor reconstructions. A geometry-and-uncertainty baseline is cheaper and more interpretable to establish first. [Primary CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Wu_SRGCD_Stability-Driven_Region_Growth_Framework_for_3D_Change_Detection_CVPR_2026_paper.pdf)

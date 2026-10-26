# Segmentation-independent geometric change evidence

Research date: 2026-10-15. Literature and public source inspection only. No package installation, detector inference, dataset processing, or viewer edit.

## Scope

The target is a whole-scene change-suspect field even when instance segmentation fails. Inputs may include RGB-D observations and camera poses. Rescan is temporary sample data, not a detector or an architectural constraint. Geometric neighborhoods, tiles and clustering are permitted; successful object masks must not gate candidate generation.

The methods below supply geometric evidence or a learned discrepancy score. None by itself establishes the identity of a moved object, all three event types, calibrated probability, and visibility handling under arbitrary indoor reconstruction noise.

## 1. M3C2: accessible numerical baseline, bounded indoor evidence

M3C2 compares local point neighborhoods along an estimated surface normal without object labels or a mesh. Its published outputs include signed distance, distance uncertainty, significance, and optional neighborhood spread/count. Larger support reduces noise sensitivity while averaging more structure. The official tool explicitly warns that missing counterparts produce NaN: holes, occlusion, or insufficient search length can all cause this. Therefore an undefined distance is not a disappearance detection. Normals, support scale and mixed surfaces remain consequential. This is a useful *surface-change measurement*, not a full object-event detector. [Official CloudCompare documentation](https://www.cloudcompare.org/doc/wiki/index.php/M3C2_(plugin))

The original uncertainty model uses local roughness, sample counts and a global registration term. Its synthetic noisy-surface and outdoor TLS experiments do not establish RGB-D furniture-change accuracy. Correlated observations fused from repeated depth frames need particular caution: more stored points are not automatically more independent evidence. [Original paper](https://arxiv.org/abs/1302.1183)

There is genuine indoor RGB-D evidence: MultiChange3D reports M3C2 F1 52.2%, precision 51.8%, recall 55.4% on its Open space RGB-D summary. The dataset uses RealSense D435i, BAD-SLAM surfels, manually verified stable-region registration and removal of gross artifacts. Thus it is not a raw-noise stress test. [Published benchmark, Tables 4–5](https://isprs-annals.copernicus.org/articles/XI-2-2026/639/2026/isprs-annals-XI-2-2026-639-2026.pdf#page=6)

The released metric code thresholds absolute M3C2 distance at the configured fixed threshold; it does not evaluate a local-LoD decision. Do not treat that number as a ranking of every uncertainty-aware M3C2 configuration. [Official evaluation implementation](https://github.com/3DOM-FBK/multichange3d/blob/main/evaluation/compute_metrics.py)

**Public detector status:** actual CloudCompare plugin and py4dgeo implementation exist. The Python class computes cloud-to-cloud distance/uncertainty from geometric inputs; no mask input was found. [Current py4dgeo source](https://github.com/3dgeo-heidelberg/py4dgeo/blob/main/src/py4dgeo/m3c2.py)

**Score meaning:** distance and detection limit are existing outputs. A proposed UI ratio such as absolute distance divided by LoD would be an application-derived evidence index, not an existing calibrated “probability of change.” Region aggregation would also be additional application logic.

## 2. M3C2-EP: stronger uncertainty model, additional acquisition information required

M3C2-EP propagates measurement and alignment uncertainty to local 3D covariance and a detection limit. The official public implementation is real, not a code-coming-soon placeholder. Its tutorial returns distance, uncertainty fields and covariance. Required metadata include acquisition positions and point-source IDs, sensor configuration, a 3 × 4 transformation and 12 × 12 alignment covariance. [Official tutorial](https://py4dgeo.readthedocs.io/en/stable/m3c2ep.html)

Source inspection confirms use of per-acquisition `origin`, `sigma_range`, `sigma_scan`, `sigma_yaw`, neighborhood means/covariances and a chi-square-derived 95% limit. Insufficient supporting points leave measurements undefined. No semantic or instance labels are required. The current implementation models laser range/angles; RGB-D camera poses alone are not its complete uncertainty input. Translating depth-camera measurement/fusion uncertainty into a suitable model requires additional work. [Current implementation](https://github.com/3dgeo-heidelberg/py4dgeo/blob/main/src/py4dgeo/m3c2ep.py)

The paper evaluates simulated scenarios and an outdoor rock-glacier TLS series. No verified actual indoor RGB-D robustness result for M3C2-EP was established in this search. Better uncertainty modeling is not evidence of superior indoor addition/removal/movement classification. [Original DOI](https://doi.org/10.1016/j.isprsjprs.2021.06.011)

**Assessment:** relevant when defensible uncertainty information exists. It is not a drop-in guarantee obtained merely by replacing M3C2 with EP or supplying an ICP RMSE.

## 3. M3C2-LOG: segmentation-free geometry, current release is lightweight

The author repository describes graph aggregation of position and orientation, comparing both changes and associated uncertainty. This extends the evidence beyond normal-distance magnitude alone. Its stated domain is topographic change. [Author README](https://github.com/jiashoujun/M3C2-LOG/blob/main/README.md)

The actual current repository was inspected through the GitHub API. Root entries were README and `M3C2-LOG (Lightweight version)`. Its main MATLAB script reads two XYZ clouds plus `Correspondence.txt`, computes normals and two-stage graph aggregation, then exports `GraphD.txt` and `GraphF.txt`. Correspondence is an external prerequisite; its comment says it can come from M3C2. These are numerical geometric differences, not calibrated probabilities. The inspected script contains no uncertainty/significance calculation. Therefore do not describe this current release as a verified full LoD detector. Stale third-party documentation refers to `change_estimation.m`, absent from the inspected current tree. [Current main script](https://github.com/jiashoujun/M3C2-LOG/blob/main/M3C2-LOG%20(Lightweight%20version)/M3C2_LOG_main.m)

**Assessment:** real code and no instance-mask prerequisite, but it does not solve missing observations or establish object identity. Current public implementation and full paper capabilities must be distinguished.

## 4. ROI Similarity: directly relevant region score, limited evidence and unverified public implementation

This 2026 method starts with ICP/SOR, adaptive nearest-neighbor change seeds and Euclidean RBNN clustering, rather than semantic instances. Keypoint triangles support a regional similarity combining normal agreement and structural network features: lower similarity indicates change. This is not a calibrated posterior. Real underground-garage LiDAR evaluation reports 8 detected added targets and 5 false-alarm regions; tested M3C2 and LOG each detect 3 with 8 false alarms. Targets are added people/bags/boxes, not a validated three-event benchmark. Simulation separately includes 2 cm observation noise. Strongly undulating surfaces can fragment ROIs. No runnable author detector repository was verified. It also locally reconstructs keypoint triangle surfaces; it does not require an already supplied instance mesh. [Original paper, §§2–4 and Table 4](https://www.mdpi.com/2079-9292/15/20/4629)

**Assessment:** closer to the requested region-level judgment than a pure point-distance threshold, but not a verified ready-to-run solution or a demonstrated general RGB-D result. Its own region construction can fail, even though semantic segmentation is unnecessary.

## 5. SPICD-Net: useful precedent, not a noise-robust recommendation

This indoor method directly compares aligned cloud tiles without instance segmentation and outputs Class-1 softmax confidence plus a spatial heatmap. Classes are no-change, changed and different-tile, not moved/added/removed. Main evaluation is synthetic alterations of real scans; a limited unseen-room experiment reports P 0.583, R 1.000, F1 0.737. The author reports rapid precision collapse under Gaussian coordinate jitter and sharp degradation under translation/yaw perturbations. Synthetic threshold 0.17 changes to 0.50 after real-environment optimization. Thus softmax is not shown to be a transferable calibrated change probability. Data/code are available from the corresponding author upon reasonable request; no public runnable repository was verified. [Paper, §§3.6/3.9, 5.8/5.11, 6.3 and availability statement](https://www.mdpi.com/2673-2688/7/4/141)

**Assessment:** it meets the “no instance prerequisite” criterion and has native tile scoring. It fails the stronger claim that existing results establish robust detection under our noisy RGB-D conditions. Do not promote synthetic F1 0.84 as that evidence.

## Decision for the wider investigation

1. Keep M3C2 as an accessible, explainable geometric evidence baseline. Its strengths are numeric distance/uncertainty and verified code, not demonstrated best indoor accuracy.
2. Consider EP only after checking whether the capture/fusion pipeline supplies defensible uncertainty metadata; poses alone are insufficient.
3. ROI Similarity is a relevant existing region-score design, but public reproducibility and generalization remain unresolved. LOG's current code is a lightweight difference calculator.
4. Do not recommend SPICD-Net as the answer to coordinate-noise robustness merely because it is segmentation-independent.
5. RGB-D with poses offers free-space/occlusion evidence beyond two final clouds. Combine this review with the separate investigation of observation-aware methods; do not solve the problem by adding new mask prerequisites.

These are research conclusions and proposed selection criteria. No detector was run and no claim about performance on our future captures or current Rescan data has been measured.

# Statistical comparison of two noisy meshes: M3C2 family

Research date: 2026-10-15. Research only: no detector installation, inference, reconstruction, mesh processing experiment, or viewer change. This note concerns only two imperfect t0/t1 geometry meshes, including uneven reconstructed surfaces and bridges between adjacent objects. It does not propose improving instances, recognizing objects, or identifying moved-object identity.

## Answer for mesh-only inputs

**M3C2 is technically usable after sampling points on both meshes, but its normal 95% detection limit cannot automatically be interpreted as calibrated confidence in physical scene change.** M3C2-PM and M3C2-EP require uncertainty information not encoded by mesh geometry alone. Correspondence-driven plane-based M3C2 requires plane segmentation/association and is a specialized supplement, not a general solution for merged noisy surfaces.

| Method | Native input | Mesh-only route | Existing numerical output | Missing information / relevant restriction | Public implementation |
| --- | --- | --- | --- | --- | --- |
| M3C2 | Two aligned XYZ clouds; local normals; chosen physical neighborhood scales | Area-based samples from both mesh surfaces | Signed normal distance, roughness, support counts, local detection limit/significance | Sample count is synthetic after resampling; unknown reconstruction bias and registration error remain | CloudCompare and py4dgeo verified |
| M3C2-PM | Clouds plus externally determined per-point X/Y/Z precision maps | Sampled surfaces only if genuine uncertainty fields can also be transferred | Distance and precision-propagated detection limit | Geometry alone does not supply precision maps; local roughness is not equivalent to independent positional precision | CloudCompare verified |
| M3C2-EP | Clouds plus acquisition provenance/sensor error model/alignment covariance | Not a complete drop-in input from two arbitrary meshes | Distance, local covariance and detection limit | Mesh interpolation loses direct correspondence to individual acquisitions; additional sensor/fusion uncertainty is needed | py4dgeo verified |

Inputs/output sources: [CloudCompare M3C2 documentation](https://www.cloudcompare.org/doc/wiki/index.php/M3C2_(plugin)), [actual comparison source](https://github.com/CloudCompare/CloudCompare/blob/master/plugins/core/Standard/qM3C2/src/qM3C2Process.cpp), [M3C2-EP tutorial](https://py4dgeo.readthedocs.io/en/stable/m3c2ep.html).

## What mesh sampling does and does not establish

CloudCompare explicitly supports generating new points on a mesh with an adjustable point count or area density. Its library generates random barycentric points on each triangle, allocating expected count according to triangle area. Thus sampled points describe the supplied polygon surface; they are not additional sensor acquisitions. Surface sampling also avoids treating a nonuniform triangulation's vertex density as physical observation density. [Official mesh sampling API](https://cloudcompare.org/doc/CCLib/html/class_c_c_core_lib_1_1_mesh_sampling_tools.html)

**Statistical inference, not a published mesh experiment:** increasing samples on a fixed reconstructed triangle does not reduce the triangle's original position error or reveal whether a bridge between objects is real. Random sampling error in approximating the fixed mesh can decrease, while uncertainty about the real scene remains. These must not be conflated.

The original M3C2 model explicitly estimates local mean position uncertainty from roughness and counts. It assumes independent Gaussian normal-distance distributions for the two epochs. With fixed roughness and zero registration term, increasing both neighborhood counts fourfold halves the sampling term. Consequently an arbitrary mesh sampling density can change standard M3C2's nominal significance without new observations. This follows from the formula; it is not a measured robustness result for our meshes. [Original paper, section 3.3, PDF pp. 9-10](https://arxiv.org/pdf/1302.1183#page=9)

The current implementation makes the distinction particularly clear:

- Standard: variance term `s1²/n1 + s2²/n2`.
- PM: variance term `s1² + s2²`, using supplied precision propagated along the normal; no division by resampled point count in that term.
- Both then add the supplied registration contribution when forming the detection limit.

The PM implementation selects the supplied precision near each neighborhood center. This does not manufacture uncertainty from geometry or guarantee a complete correlated reconstruction model. [Actual source, ComputePMUncertainty and confidence calculation](https://github.com/CloudCompare/CloudCompare/blob/master/plugins/core/Standard/qM3C2/src/qM3C2Process.cpp)

## What uncertainty can be obtained from two meshes alone

One can calculate geometric distance, local surface roughness, normal stability, boundary/degeneracy indicators, and sensitivity to the chosen physical comparison scale. These characterize the geometry supplied to the algorithm. One cannot, from that fact alone, uniquely recover sensor accuracy, correlated reconstruction/fusion errors, filled-hole provenance, or full registration covariance. A smooth but biased surface can have low roughness; a highly curved correct surface can have high roughness. **Roughness is therefore an available geometric diagnostic, not a universal estimator of truth error.** This distinction follows from the M3C2 uncertainty sources and the PM/EP extra inputs. [M3C2, sections 2.3-2.4 and 6.2](https://arxiv.org/pdf/1302.1183#page=5), [M3C2-PM description](https://www.cloudcompare.org/doc/wiki/index.php/M3C2_(plugin)#Precision_maps)

M3C2-EP's real code requires scan-source IDs and per-acquisition origin/range/angular uncertainty, plus the alignment covariance. Its documented 12 x 12 covariance is additional input, not inferred from the two triangle surfaces. Merely assigning an ICP fit RMSE or default zero covariance does not establish calibrated reconstruction uncertainty. The implementation's native sensor model is laser range/angles. [Actual EP code](https://github.com/3dgeo-heidelberg/py4dgeo/blob/main/src/py4dgeo/m3c2ep.py)

## Significant, nonsignificant, and unknown are different

M3C2 significance means that the measured separation exceeds the detection limit **under its model and inputs**. It does not mean that the source is necessarily a physical change rather than a reconstruction artifact.

The original application explicitly reports that nonsignificant regions include genuinely stable geometry, low-density areas, high roughness, and surfaces with no net change. Hence **nonsignificant is not an existing suspicious-region label**. Similarly NaN can result from no corresponding surface, a hole/occlusion, or an insufficient search cylinder. [Original paper, section 5.3, PDF p. 20](https://arxiv.org/pdf/1302.1183#page=20), [official NaN explanation](https://www.cloudcompare.org/doc/wiki/index.php/M3C2_(plugin)#Computing_distance)

An application distinction between “sufficient geometric evidence” and “uncertain candidate” would need an explicit mapping from nonzero discrepancy plus support/normal/boundary/model diagnostics. It cannot label all nonsignificant or all missing results as suspected change. That mapping is additional design, not a built-in four-state M3C2 algorithm. A distance/LoD ratio would be an evidence index, not automatically a calibrated change probability.

## Published noise evidence and actual indoor results

The original M3C2 paper shows averaging benefits on synthetic noisy surfaces and outdoor TLS. It also reports failure of nominal uncertainty on unchanged rough surfaces: at excessive projection scales, detection limits become too small and false significant changes increase; bootstrap does not resolve the illustrated effect. Its scale recommendations are for the tested terrain, not indoor defaults. [Sections 5.2.1 and 7, PDF pp. 18/24](https://arxiv.org/pdf/1302.1183#page=18)

MultiChange3D provides a genuine indoor RGB-D point-cloud result, **not mesh-to-mesh validation**. Its Open space RGB-D summary reports M3C2 P 51.8%, R 55.4%, F1 52.2%; C2C on the same setting reports F1 75.4%. Data preparation manually verifies stable registration regions and removes gross spurious points. The tested M3C2 decision is a fixed 0.10 m distance threshold, not local-LoD classification. Spatially overlapping moved-chair surfaces can be labeled unchanged. These results should neither be sold as strong raw-mesh robustness nor used to condemn every LoD configuration. [Published paper, sections 3.2/4.4/6.1 and Tables 3-5](https://isprs-annals.copernicus.org/articles/XI-2-2026/639/2026/isprs-annals-XI-2-2026-639-2026.pdf), [actual evaluation threshold code](https://github.com/3DOM-FBK/multichange3d/blob/main/evaluation/compute_metrics.py)

No direct noisy indoor mesh-pair study establishing calibrated PM/EP uncertainty was verified. The accessible PM research concerns photogrammetric precision propagation; EP and the correspondence-plane study concern topographic/TLS examples. Avoid transferring their numerical accuracy to merged indoor meshes.

## Plane-based variant: useful restriction, not general coverage

Correspondence-driven PB-M3C2 extracts geometric planes, establishes corresponding planes, and measures normal separation/uncertainty. No semantic instances are required, but successful geometric plane extraction is a requirement. The paper's improvement was on a rock glacier. It does not establish coverage for bumpy merged object surfaces. [Primary publication](https://doi.org/10.1016/j.isprsjprs.2021.11.018)

The current py4dgeo implementation requires plane IDs/normals provided by external segmentation. The default mode learns plane correspondence from labeled pairs; a nearest-centroid alternative avoids training labels but retains plane association. Its LoD still uses plane roughness divided by point count, so arbitrary synthetic mesh sampling remains relevant. This plane-association requirement makes it a narrower geometric supplement, not the primary recommendation for the current scope. [Official tutorial](https://py4dgeo.readthedocs.io/en/stable/pbm3c2.html), [current source](https://github.com/3dgeo-heidelberg/py4dgeo/blob/main/src/py4dgeo/pbm3c2.py)

## Selection judgment

For two geometry-only meshes, standard M3C2 is a reproducible way to expose local surface difference and roughness, conditional on normal/neighborhood quality. Treat its synthetic-sample LoD as a model diagnostic rather than established 95% physical-change confidence. PM becomes a more defensible candidate when actual external per-location precision exists; EP requires richer acquisition/error provenance. Neither extension is justified solely by having a noisy mesh. Plane-based comparison is secondary where stable geometric planes are recoverable.

None of these methods can decide from mesh geometry alone whether a changed bridge/hole is physical change or reconstruction error when both causes produce the same two surfaces. The current evidence supports a transparent geometric candidate field with explicitly conditional confidence, not a claim that the uncertainty problem has already been solved for our inputs.

## Reading and verification record

- Freshly inspected CloudCompare mesh sampling documentation and actual M3C2/PM C++ confidence computation.
- Freshly inspected current py4dgeo EP metadata access and PB-M3C2 LoD implementation/tutorial.
- Read full relevant M3C2 pages; rendered and visually inspected PDF p. 10 (formula) and p. 18 (unchanged-surface uncertainty failure) in a temporary directory outside the repository.
- Re-read the previously downloaded published MultiChange3D source and visually checked the full table page (PDF p. 6), including the Open space RGB-D row and table captions. No new detector outputs were generated.

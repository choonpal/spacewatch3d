# M3C2 vs PlaneSDF: published evidence and the current Rescan pair

Scope correction: Rescan is temporary sample data, not a detector or a restriction on future method selection. The recommendation below is limited to those sample inputs. For the intended system and mesh-optional alternatives, see [change-detection-without-mesh.md](change-detection-without-mesh.md).

Research date: 2026-10-15. Research only: no algorithm, package, viewer or dataset changes, and no new experiment. Published papers, public evaluation code and existing local metadata were inspected. Source PDFs were downloaded for reading only.

## Decision for the current Rescan pair

**Project-specific inference, not a measured result:** M3C2 is the better first reproducible baseline for segmentation-independent surface-change evidence on our existing `scene_e_0 -> scene_e_2` point clouds. PlaneSDF is the more specialized research candidate for grouping plane-supported indoor object changes. Neither has demonstrated superior performance on this Rescan pair; neither alone establishes moved/added/removed object identity.

This preference is about input fit, explainable geometric output and implementation availability, **not a claim that M3C2 has the better published indoor accuracy**. A relevant indoor benchmark actually reports modest M3C2 results. PlaneSDF's stronger real-data score is from a different task that excludes furniture movement, so the two scores cannot be ranked against each other.

| Question | M3C2 | PlaneSDF |
| --- | --- | --- |
| Comparison unit | Local surface neighborhoods along a normal | Plane-associated SDF volumes, height-map locations and object blobs |
| Segmentation prerequisite | No GT instances or object classes | No GT instances; extracts supporting planes and connected blobs |
| Existing score | Signed separation plus location-dependent detection limit | SDF descriptor similarity aggregated per blob |
| Noise mechanism | Local averaging and roughness/sample-count/registration-dependent uncertainty | Plane-relative drift handling, component filtering and 3D blob shape comparison |
| Missing geometry | No counterpart can produce an undefined distance; not automatic disappearance | Unobserved height-map state is explicit; height-direction occlusion can still hide changes |
| Main indoor risks | Mixed surfaces at edges, thin chair parts, tangential movement, overlapping geometry, correlated fused observations | Missing/incorrect support planes, touching objects merged as blobs, sofa deformation/noisy boundaries, projection occlusion |
| Reproducibility now | Public CloudCompare and py4dgeo implementations verified | Runnable author detector implementation has not been verified |

Method sources: [M3C2 original paper](https://arxiv.org/pdf/1302.1183), [CloudCompare documentation](https://www.cloudcompare.org/doc/wiki/index.php/M3C2_(plugin)), [PlaneSDF paper](https://arxiv.org/pdf/2207.08323), [py4dgeo](https://github.com/3dgeo-heidelberg/py4dgeo).

## M3C2: actual indoor RGB-D evidence

### MultiChange3D, ISPRS Congress 2026

This is a directly relevant published indoor RGB-D benchmark, rather than extrapolation from the original M3C2 canyon experiment. Open space (RGB-D) was captured with Intel RealSense D435i, reconstructed with BAD-SLAM into surfels, then exported as points. The protocol uses manually verified stable regions for ICP and manually removes spurious sensor/occlusion points. Consequently this is real measured indoor geometry with preprocessing, not an unrestricted raw-noise stress test. [Published paper, sections 3.1-3.2](https://isprs-annals.copernicus.org/articles/XI-2-2026/639/2026/isprs-annals-XI-2-2026-639-2026.pdf)

| Method, Open space RGB-D | Precision | Recall | F1 | Overall accuracy | mIoU |
| --- | --- | --- | --- | --- | --- |
| M3C2 | 51.8% | 55.4% | 52.2% | 71.7% | 51.1% |
| C2C, same benchmark | 95.5% | 62.7% | 75.4% | 88.5% | 73.2% |

These are the authors' Table 4/5 summaries for testing pairs 0-1, 0-2 and 0-3, not our runs, individual-object scores or calibrated probabilities. Do not recompute the published aggregate F1 from aggregate precision/recall or claim the aggregation is a particular micro/macro scheme without more information. The final published PDF and the author's pinned repository PDF were cross-read; their reported values agree. [Table 4, published PDF p. 6 / printed p. 644](https://isprs-annals.copernicus.org/articles/XI-2-2026/639/2026/isprs-annals-XI-2-2026-639-2026.pdf#page=6)

Important implementation conditions:

- The indoor point clouds are spatially downsampled to 1 cm; Table 3 uses a fixed distance threshold of 0.10 m for C2C/M3C2. The released evaluation code thresholds the absolute `scalar_M3C2_distance`; it does not use the local M3C2 LoD field in that decision. This benchmark therefore evaluates a distance-threshold configuration, not every uncertainty-aware M3C2 configuration. [Sections 4.2/4.4 and Table 3](https://isprs-annals.copernicus.org/articles/XI-2-2026/639/2026/isprs-annals-XI-2-2026-639-2026.pdf#page=4), [official evaluation code](https://github.com/3DOM-FBK/multichange3d/blob/main/evaluation/compute_metrics.py)
- The text does not provide a full indoor normal/projection-scale parameter list or a controlled noise-sigma sweep. It is not possible to attribute all errors specifically to measurement noise from these results.
- The authors report that moved chairs with spatially overlapping geometry can be labeled unchanged, because distance methods do not track object motion. Fig. 2 also shows more scattered floor labels for the tested M3C2 result than the reference. These are limitations of that reported configuration. [Section 6.1 and Fig. 2](https://isprs-annals.copernicus.org/articles/XI-2-2026/639/2026/isprs-annals-XI-2-2026-639-2026.pdf#page=5)

### What the original M3C2 evidence does and does not add

The 2013 paper tests synthetic noisy surfaces and real outdoor TLS geometry. In a synthetic Gaussian-noise test (1 mm standard deviation), it reports surface-distance variation of 0.15 mm for M3C2 versus 1.00 mm for cloud-to-mesh. This demonstrates the averaging principle in that controlled setting, not furniture-change detection performance. The well-known 6 mm detection limit concerns high-precision TLS and millimetric registration outdoors; it is not an indoor RGB-D accuracy guarantee. [Original paper, section 3.4 and abstract](https://arxiv.org/pdf/1302.1183)

M3C2's parametric uncertainty calculation assumes independent distributions and an isotropic, spatially uniform registration term. Rescan surfels fuse repeated RGB-D measurements and are correlated; raw point count must not be treated as proof of a calibrated 95% guarantee. M3C2-EP is an error-propagation extension with additional sensor/alignment metadata, not an automatic remedy using only XYZ. [M3C2 sections 3.3/6](https://arxiv.org/pdf/1302.1183), [M3C2-EP documentation](https://py4dgeo.readthedocs.io/en/latest/m3c2ep.html)

## Current Rescan input fit

Read-only checks of `outputs/rescan-three-changes/review.json` and the released PLY header confirmed:

- Current real pair: `scene_e_0 -> scene_e_2`, in metres; original 365,039 and 205,704 measured surfels are retained.
- Original point attributes include XYZ, normals, RGB, radius, semantic ID and instance ID. A geometry-only comparison can ignore the semantic/instance fields completely.
- The current browser meshes are derived reconstructions (70,000 and 67,120 triangles). Changes in these triangles can include smoothing, meshing and support effects; measured point neighborhoods are the preferable comparison input for this question.
- A common frame and a saved robust ICP correction exist. Its reported inlier RMSE is 0.02167 m, which is a fit residual, not calibrated sensor noise, independent registration covariance or whole-scene accuracy.
- Raw RGB/depth and estimated frame poses exist, but no independently established sensor/alignment covariance is available for full M3C2-EP.

Local source: [existing review metadata](../outputs/rescan-three-changes/review.json), [current reconstruction source](../tools/prepare_rescan_pair.py). These checks did not run either candidate detector.

**Application judgment:** the retained raw point clouds and existing coordinate frame fit M3C2 directly, and its numerical output is useful for asking whether a region's surface difference exceeds its uncertainty. It still needs adequate neighborhoods and will not recover all chair movement, additions or disappearances by itself. PlaneSDF can also accept complete point clouds, so a missing ground-truth camera trajectory is not by itself a reason to reject it. Its extra plane/SDF/blob reconstruction and unverified author-code access make it a larger reproduction effort.

Rescan's current review contains large furniture changes, thin chair geometry and noisy/partial surfaces. PlaneSDF's real evaluation exclusion of furniture movement and its sofa/projection failures matter specifically here. Its published 0.3 m plane neighborhood and 2 cm height gate are experimental settings, not appropriate Rescan defaults established by this research.

The existing before/after object associations and original-photo links can remain a separate channel in a future design. Neither local distance nor SDF similarity alone supplies an instance movement vector or resolves missing observation versus true removal. No such integration was attempted.

## No head-to-head winner established

M3C2's 52.2% F1 and PlaneSDF's 0.76 average F1 use different datasets, annotations, preprocessing and targets. M3C2's cited RGB-D configuration uses a fixed 10 cm distance threshold; PlaneSDF's real evaluation excludes moved furniture/decoration and tunes dataset-dependent similarity parameters. Their percentages are not a controlled comparison. The existing evidence supports **M3C2 as an accessible baseline and PlaneSDF as a relevant specialized comparison**, not a measured accuracy ranking for Rescan.

## PlaneSDF: what the actual experiments establish

Primary publication: Fu et al., *PlaneSDF-based Change Detection for Long-term Dense Mapping*, RA-L/IROS 2022, arXiv v2 (2022-10-05), 8-page manuscript. [Original PDF](https://arxiv.org/pdf/2207.08323), [same-version HTML](https://arxiv.org/html/2207.08323v2)

### Synthetic and real evaluation are different

| Item | Synthetic tabletop experiment | Real indoor experiment |
| --- | --- | --- |
| Data | LiDAR-scanned furnished room with synthetic table/objects; simulated Quest 2 camera trajectories | Object Change Detection Dataset by Langer et al.; ASUS Xtion PRO Live on an HSR robot, Voxblox reconstructions |
| Quantity in the PlaneSDF paper | 20 selected source-target pairs, evaluated in both directions | Five scene types; paper states one reference plus 5–6 other reconstructions per scene |
| Changed targets | Tabletop objects added, removed or moved; examples include place swaps | 3–18 newly introduced YCB objects per annotated scene |
| Noise/overlap | Relatively complete surrounding views; reconstruction artifacts illustrated | Complete/partial maps, structural misalignment and noisy boundaries; no numerical overlap fraction or controlled noise-sigma sweep reported |
| What was measured | Changed-object precision/recall | Point-level precision/recall/F1, plus missed-object and false-cluster counts |

Locations: IV-A, PDF pp. 5–6; IV-D, p. 6; IV-B, pp. 5–6. [PDF](https://arxiv.org/pdf/2207.08323#page=5)

The synthetic result is 100% recall and 80% precision before 3D geometry checking, and 100% precision afterward. It is not the reported real-room result. The paper gives no exact controlled sensor-noise distribution for this tabletop experiment. [IV-D, PDF p. 6](https://arxiv.org/pdf/2207.08323#page=6)

### Real indoor numbers: Table I

The following values are the authors' reported Table I values, not measurements on Rescan. Each cell preserves the paper's notation. `M` counts changed objects with no changed points detected; `W` counts detected clusters overlapping no changed object. Precision/recall/F1 are point-level metrics, not object-classification accuracy or region-probability calibration.

| Real scene | PlaneSDF precision | PlaneSDF recall | PlaneSDF F1 | PlaneSDF M / W | “Best of [10]” F1 | “Best of [10]” M / W |
| --- | --- | --- | --- | --- | --- | --- |
| Small room | 0.50 ± 0.24 | 0.83 ± 0.14 | 0.59 ± 0.21 | 10 / 18 | 0.57 ± 0.22 | 6 / 28 |
| Big room | 0.78 ± 0.03 | 0.85 ± 0.15 | 0.81 ± 0.09 | 8 / 15 | 0.69 ± 0.10 | 2 / 50 |
| Living room, partial | 0.80 ± 0.05 | 0.87 ± 0.10 | 0.83 ± 0.05 | 4 / 13 | 0.72 ± 0.17 | 4 / 13 |
| Office, partial | 0.72 ± 0.10 | 0.94 ± 0.08 | 0.79 ± 0.06 | 0 / 4 | 0.54 ± 0.20 | 0 / 16 |
| Kitchen, partial | 0.77 ± 0.015 | 0.85 ± 0.05 | 0.81 ± 0.03 | 2 / 3 | 0.55 ± 0.11 | 0 / 55 |
| Paper's Average column | 0.72 ± 0.16 | 0.86 ± 0.12 | 0.76 ± 0.13 | 4.8 / 10.6 | 0.61 ± 0.16 | 2.8 / 34.2 |

Source: [Table I, PDF p. 7](https://arxiv.org/pdf/2207.08323#page=7), cross-read with [arXiv same-version HTML Table I](https://arxiv.org/html/2207.08323v2).

The Average column for “Best of [10]” also reports precision 0.64 ± 0.27 and recall 0.74 ± 0.14. PlaneSDF's reported F1 is higher and false clusters fewer, but missed objects are more numerous on average. Thus it is not a clean win on every detection metric.

**Aggregation caution:** the paper defines point-level precision/recall/F1, but its Table I caption/text does not explicitly identify whether the `±` entries are standard deviations, standard errors or confidence intervals, nor give a fully specified pooling/weighting recipe for the Average column. Preserve these as *reported table summaries*. Do not call them 95% intervals, recompute F1 from the aggregate precision/recall, or interpret the decimal `M/W` average as counts from a single scan. The kitchen precision spread is printed as `0.015`; it is preserved rather than silently rounded to `0.15`.

### Crucial evaluation scope restriction

For the real experiment, the authors explicitly remove detected points due to moved furniture and decorations from evaluation to follow the dataset's definition. Ground truth concerns newly introduced YCB objects. [IV-E, immediately below Table I, PDF p. 7](https://arxiv.org/pdf/2207.08323#page=7)

Therefore these real-room F1 numbers **do not demonstrate accuracy for all furniture movement, additions and disappearances**. The synthetic experiment includes swaps/removals, but cannot fill that real-world evidence gap. Rescan `scene_e_0 → scene_e_2` is a different dataset and change task; no score in this paper is a score for that pair.

The dataset owner's page additionally explains that the downloadable dataset was cleaned to remove duplicated Voxblox reconstruction/annotation points and provides updated baseline performance. This makes release/version matching necessary before treating baseline tables from different publications as identical protocols. [Official dataset page](https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/object-change-detection-dataset-of-indoor-environments/)

### Compared baselines, and what cannot be concluded

Table I compares Octomap, Meta-room, the best reported result from Langer et al. 2020, and an FPFH variant. The FPFH variant replaces PlaneSDF's 3D SDF validation with pointwise FPFH matching; it is not M3C2. `W` is not reported for FPFH because it does not produce cluster-level outputs. **M3C2/M3C2-EP is not a directly evaluated baseline.** [Table I and IV-E, PDF p. 7](https://arxiv.org/pdf/2207.08323#page=7)

There is consequently no published same-data, same-target head-to-head result here establishing that PlaneSDF outperforms M3C2. Relative suitability must be inferred from inputs, objectives, demonstrated conditions and implementation access, clearly labeled as an inference.

### Why noise handling can help, and the reported failures

PlaneSDF compares local plane-supported volumes after small plane-pose drift correction. Height differences nominate candidates; component filtering removes speckles; object-blob SDF geometry checks reject false candidates. Fig. 6 illustrates an unchanged pair of bottles near a wall with a partial/warped reconstruction, where pointwise FPFH produces false changes but PlaneSDF recognizes the scene as unchanged. [III-B–D and Fig. 6/IV-E](https://arxiv.org/pdf/2207.08323#page=7)

The authors also report failures: a baseball under a table is hidden along the height-projection direction; noisy/deforming sofa boundaries cause missed objects and reduced precision; attached planar parts such as a skillet bottom can be omitted. Supporting planes are limited to horizontal/vertical cases. [III-A and end of IV-E, PDF pp. 2, 8](https://arxiv.org/pdf/2207.08323#page=8)

These observations support a bounded claim of robustness to some real indoor reconstruction artifacts. They do not establish immunity to arbitrary noise, missing coverage or large pose errors.

### Parameters, processing cost and public implementation

| Reported parameter | Value / caveat |
| --- | --- |
| Plane-neighborhood fusion distance | 0.3 m |
| SDF voxel resolution | 7 mm |
| Plane normal cosine / offset gates | 0.95 / 0.2 m |
| Initial height-difference gate | 0.02 m |
| Descriptor bins | 5 × 5 × 6 per principal component |
| Synthetic descriptor coefficient / blob similarity threshold | 2 / 0.9 |
| Real-data tuning | Coefficient and blob threshold were additionally adjusted for dataset-specific cases; full per-scene values are not supplied in the text |

Source: [IV-C, PDF p. 6](https://arxiv.org/pdf/2207.08323#page=6). These are the paper's settings, not suggested defaults or measured tolerances for our data. A 0.2 m plane association gate is not a proven allowable registration error for reliable change detection.

The inspected 8-page publication does not report a numerical runtime, processor specification, peak RAM figure or detailed cost benchmark. The introduction's memory/reload benefit is an architectural argument, not a measured memory comparison. Do not claim verified real-time operation or a quantified speed advantage from this publication.

The [author's publication page](https://jiahui-fu.github.io/) lists Paper and Video for PlaneSDF; the paper links a supplementary video. Searches and those primary pages have not established publicly runnable author algorithm code or pretrained checkpoints. PlaneSDF is a non-learning method, so checkpoint absence is not itself a training requirement. Implementation availability remains **unverified**, not proof that no implementation exists anywhere. No repository was installed or executed for this investigation.

### Reading checks and source locations

- Main source: arXiv v2 PDF, 8 pages, 2022-10-05. Dataset/metric definitions: IV-A/B, PDF pp. 5–6. Parameters/synthetic result: IV-C/D, p. 6. Real results: Table I/IV-E, p. 7. Failure cases: p. 8.
- Same-version arXiv HTML was cross-read for the table's row/column relationships and the movement-exclusion statement.
- The initial delegated web interface exposed screenshot references without pixels. The parent researcher subsequently downloaded the source PDFs, rendered complete relevant pages with Poppler and visually inspected PlaneSDF Table I (PDF p. 7), MultiChange3D Fig. 2/Table 3 (p. 5), and the published MultiChange3D Table 4/5 (p. 6). Table transcription, column placement and the moved-furniture exclusion paragraph were checked against these pages.
- Local source PDFs for read-only review are `outputs/research-sources/m3c2-planesdf/PlaneSDF-2022.pdf` and `MultiChange3D-2026-published.pdf`; an author-repository manuscript was also cross-read. These are research source materials, not new detector outputs.
- No new Rescan evaluation or inference has been run. Any eventual preference for the current pair must distinguish published evidence from a project-specific, untested judgment.

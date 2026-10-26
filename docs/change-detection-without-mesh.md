# Change detection without requiring a finished mesh or clean instances

Research date: 2026-10-15. Corrected scope after user feedback: our own instance-segmented maps are not ready. Rescan is temporary sample data, not the detector or a constraint on method selection. The goal is detecting moved, added and removed objects/regions from our future captures under imperfect segmentation, partial visibility and reconstruction noise. Mesh output is optional. Research only; no installation, inference, detector implementation or viewer changes.

This note supersedes project-wide interpretations of the earlier pair-specific M3C2/PlaneSDF recommendation. That comparison was about the current sample files, not the best representation or detector for the intended system.

**Confirmed input update:** the user has now confirmed future RGB-D video and camera location information. Therefore, the RGB-only SceneDiff recommendation below is conditional, not the first choice for this confirmed input. Prioritize methods that consume measured depth and poses directly. Full camera rotation, intrinsics, RGB/depth alignment and a shared coordinate frame between captures remain necessary details; position XYZ alone is insufficient for the cited projection-based pipelines.

**Selection correction after further user feedback:** the central condition is detecting spatial change candidates even where noise makes instance detection/segmentation fail. Being compatible with RGB-D and not needing GT instances is insufficient. The earlier priority recommendation for Where Did I Leave My Glasses? is withdrawn: its priority map operates on already-created SceneObjects, and the detector returns empty detections when SAM produces no masks. This is not evidence of an independent change detector over unsegmented noisy regions. [Detector source](https://github.com/learnsyslab/perceive_semantix_release/blob/main/perceive_semantix_lib/src/perceive_semantix_lib/core/object_detector.py), [priority-map source](https://github.com/learnsyslab/perceive_semantix_release/blob/main/perceive_semantix_lib/src/perceive_semantix_lib/grid_maps/exploration_priority_map.py)

## Input-dependent shortlist

| Available future input | Most relevant candidates | Representation used for inference |
| --- | --- | --- |
| Before/after RGB videos | SceneDiff; SCaR-3D / 3DGS-CD when a dense baseline capture exists | Predicted point/depth maps and image masks; or Gaussian primitives |
| Before/after RGB images, potentially one per time | SceneDiff two-view; GOLDILOCS research reference | Predicted dense geometry, camera estimates and automatic masks |
| RGB-D or RGB-LiDAR plus camera poses in a shared metric frame | OASIS-Map; Where Did I Leave My Glasses? as a released probabilistic mapping candidate; Panoptic Multi-TSDF as a mapping baseline | Object point maps / volumetric TSDF, image correspondences |
| Only noisy aligned point clouds | M3C2 / ROI Similarity geometric baselines | Point neighborhoods / local geometric regions |

Existing automatic segmentation can become supporting evidence; the main candidates do not require a clean, researcher-labelled instance mesh as input. They still use predicted masks and can inherit mask/reconstruction errors. Their observation requirements differ; avoiding triangles does not eliminate geometry or visibility reasoning.

## SceneDiff: first RGB-video candidate

The method takes two captures as RGB image sequences, jointly estimates camera intrinsics/poses/depth with Pi3, chooses co-visible views, and compares geometry plus DINO appearance and SAM region features. It associates detections across views and then across time to output Added, Removed and Moved. Masks are predicted, not ground-truth masks supplied to the detector. [Author paper, sections 4.1-4.3](https://arxiv.org/html/2512.16908v1)

The public detector repository was verified to contain actual modules, configuration and `scripts/demo.py`, separate from the annotation tool. The demo accepts `--video1` and `--video2`, and lists object masks, RGB point clouds, instance clouds and per-point cost maps as outputs. Source inspection confirms object-mask saving and optional point-cloud visualization export (`vis_pc`); the YAML default disables visualization, but the demo overrides `vis_pc=True`, so its effective setting enables point-cloud export. Ground-truth visualization returns early if labels are absent and is not a detector-input requirement. Published scores can support candidate ranking; they are not calibrated probabilities. No prebuilt mesh or measured depth input is required. [Author code](https://github.com/yuqunw/scene_diff), [demo](https://github.com/yuqunw/scene_diff/blob/main/scripts/demo.py), [configuration](https://github.com/yuqunw/scene_diff/blob/main/configs/scenediff_config.yml), [inference implementation](https://github.com/yuqunw/scene_diff/blob/main/modules/scenediff.py)

Code availability is not proof that it has been run here: Pi3/SAM/DINO checkpoints and GPU dependencies remain prerequisites. The checked configuration has `dinov3.weight_url: TO_BE_FILLED`; README instructs obtaining DINOv3 checkpoint access. No weights were requested/downloaded and no package was installed.

The author benchmark uses 350 real video pairs, including varied scenes and cooking sequences, with 1,009 annotated changed objects. These annotations are evaluation targets, not inference inputs. V1 Table 2 reports per-scene AP 46.3/20.9 and type-aware AP 25.5/12.2 on SD-V/SD-K. The low type-aware values on cluttered scenes are a material limit; high relative-improvement slogans should not substitute for absolute scores. The evaluation focuses on locating changed objects, not mesh-boundary accuracy. These are author results, not our captures' results. [Paper sections 3.2/5.1 and Table 2](https://arxiv.org/html/2512.16908v1)

Reported failures include repetitive identical objects, insufficient co-visibility, reconstruction failure and strong lighting changes. SAM boundaries can disagree with ground truth. The paper's ablation finds appearance evidence stronger than geometric differences alone, supporting investigation beyond mesh differencing. This is not a guarantee of robustness to arbitrary segmentation corruption. [Sections 5.3, 6 and 7.4](https://arxiv.org/html/2512.16908v1)

## OASIS-Map: strongest RGB-D/pose-based match to deferred decisions

OASIS-Map uses automatic SAM2 masks, patch-level DINOv3 correspondences and accumulated evidence across observations; it builds object-level geometry from RGB-D or RGB-LiDAR. Camera poses must be provided in a shared metric frame. Pre-labelled instance meshes are not required. Its outputs include appeared, disappeared, static, moved, and Unknown when observation coverage is insufficient. This makes it directly relevant to imperfect segmentation and distinguishing missing observation from disappearance. [Author project](https://ori-drs.github.io/projects/oasis-map/), [2026 author preprint](https://arxiv.org/html/2607.14899v1)

It is an under-review research candidate: the project says Code Soon. A real preprint exists despite a Paper Soon graphic; do not say the paper is unavailable. The abstract's moved-association F1 and point-level change F1 are different metrics, and must not be conflated. No released detector implementation was verified.

## SCaR-3D and 3DGS-CD: image-to-3D alternatives

These build a baseline Gaussian representation from RGB images, render comparable viewpoints, infer image changes and aggregate them in 3D. SCaR-3D adds multi-view voting/pruning to reduce fragmented and inconsistent detections; 3DGS-CD also estimates object transformations. Both have actual author code, and provided baseline scene weights are linked. Custom captures require scene reconstruction/training, camera alignment and foundation-model masks. They do not need an instance-segmented triangle mesh, but their quality depends on baseline rendering and view coverage. [SCaR-3D code](https://github.com/zr-zhou0o0/SCaR-3D), [3DGS-CD code](https://github.com/520xyxyzq/3DGS-CD)

Do not call these universally more noise-robust: the SceneDiff authors report failures for 3DGS-CD under sparse, outward-facing captures. The underlying 3DGS-CD implementation was adapted in that comparison, so it is evidence about that protocol, not a universal ranking. These are useful candidates when dense reference images are available and 3D reconstruction is desired.

## GOLDILOCS: relevant paper, code not yet usable from the verified release

GOLDILOCS uses dense stereo geometry and SAM2 tracking from before/after RGB image pairs or image sets, addressing added/removed/moved and non-rigid changes without requiring prior meshes or measured camera poses. However, the author repository was checked: it contained README and documentation assets, and README explicitly says code will be released soon. It should be listed as a research reference, not as an executable released baseline. [Author project](https://goldilocs.github.io/), [author repository](https://github.com/almogfri/GOLDILOCS)

## Corrected recommendation

The methods above remain research references, not established solutions to the user's central segmentation-failure condition. OASIS-Map still has unreleased code and uses object maps and automatic masks. Where Did I Leave My Glasses? provides a real RGB-D/intrinsics/pose interface, object stationarity updates, re-identification and exploration-priority code, but its priority map is conditional on created objects. Its score represents modelled stationarity and can decay under non-observation; it is not a calibrated observation-only change probability. Its paper's priority-map evaluation is 2D robot exploration, not a ready-made 3D change-region benchmark. Its earlier first-choice recommendation is withdrawn. [Author paper](https://arxiv.org/html/2509.19851v2), [official implementation](https://github.com/learnsyslab/perceive_semantix_release), [priority-map source](https://github.com/learnsyslab/perceive_semantix_release/blob/main/perceive_semantix_lib/src/perceive_semantix_lib/grid_maps/exploration_priority_map.py)

The checked SceneDiff default inference estimates geometry and poses using Pi3; direct use of measured depth/poses was not verified in that path. Keep it as a public RGB event-detection comparison rather than assuming it makes best use of the confirmed input. Gaussian methods are a separate reconstruction-oriented option. Keep geometric cloud comparisons as baselines rather than making mesh availability the architectural gate.

This is a source-based shortlist, not a measured winner on our data. Before any later implementation, determine the actual available raw images/depth/poses and their projection model. In particular, a 360-degree panorama is not automatically interchangeable with the perspective camera images assumed by these pipelines; view conversion and camera consistency would need explicit treatment.

Research on the RGB-D/volumetric candidates is recorded separately in [change-detection-without-mesh-rgbd.md](change-detection-without-mesh-rgbd.md). No automatic scan or scheduled task was created.

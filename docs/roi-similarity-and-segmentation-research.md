# Additional sources: noisy regions and imperfect segmentation

Research date: 2026-10-15. Scope: check the user's pasted literature discussion against primary sources and refine the research priorities. Research only; no detector implementation, package installation, viewer edit, dataset alteration or new experiment. No recurring automation was created or changed.

## ROI Similarity, Electronics 2026

The paper and publication date (2026-10-10) are real. It takes two ICP-aligned, SOR-filtered laser point clouds, extracts distance-based candidate points, groups them with RBNN, reconstructs keypoint triangles and compares ROI surface-normal consistency plus complex-network structural features. This is a geometric method; the network terminology does not imply a trained neural detector. Its similarity is `R = alpha R1 + (1-alpha) R2`, thresholded for changed/unchanged. It is not a calibrated change probability or an explicit changed/suspected/unknown classifier. [Original article, sections 2-3](https://www.mdpi.com/2079-9292/15/20/4629)

The real garage experiment uses Livox Horizon/LOAM, two recordings 1.5 hours apart and eight added targets including people, boxes and bags. Table 4 reports:

| Method | True-positive targets | False-positive regions | Runtime, seconds |
| --- | --- | --- | --- |
| ROI Similarity | 8 | 5 | 15.407 |
| Adaptive threshold | 7 | 8 | 11.671 |
| M3C2 | 3 | 8 | 9.670 |
| M3C2-LOG | 3 | 8 | 76.742 |

These are reported region/target counts and runtimes for that experiment, not point-level precision, a Rescan result or a robust furniture-motion benchmark. The simulated-tunnel F1 0.9004 is a separate experiment. The paper reports a 97.5% significance threshold for its M3C2 comparison. Fragmented ROIs on strongly undulating surfaces remain a reported limitation. Public author detector code was not verified. [Sections 3.1/3.3, Table 4 and Discussion](https://www.mdpi.com/2079-9292/15/20/4629)

Additional reading caution: Table 3's ablation F1 for removing R1 (0.9057) exceeds the full-method F1 (0.8502). The text also discusses missed true regions for the former. Consequently do not repeat an unrestricted claim that the full method has the best F1 in every experiment. Tables 2 and 3 have different full-method summaries; their detailed evaluation conditions need clarification before treating them as repeated measurements of the same test.

## Langer 2020 and the 2022 successor

See the detailed primary-source audit in [langer-2020-primary-evidence.md](langer-2020-primary-evidence.md).

The 2020 method uses predicted per-vertex semantic classes plus geometric candidate extraction; it does not simply consume already-correct instance IDs. The dissertation explicitly reports a merged sugar-box/coffee-machine cluster causing the new sugar box to be missed. Describing the paper as acknowledging under-segmentation is justified; saying it solves merged instance boundaries generally is not. The 31 recordings represent five indoor environments with 260 annotated YCB-object placements, not 31 distinct rooms. [IROS 2020 DOI](https://doi.org/10.1109/IROS45743.2020.9341664), [author dissertation, Chapter 5](https://repositum.tuwien.at/handle/20.500.12708/176591)

The successor *Where Does It Belong? Autonomous Object Mapping in Open-World Settings* (2022) has an author repository, [tidyupvisionpipeline](https://github.com/edith-langer/tidyupvisionpipeline). Its partial matching and region growth can split a matched portion from a remaining novel part, making it a more relevant reproducible candidate for imperfect object boundaries and moved/added/removed reasoning. It also reports boundary leakage and split failures. Section 3.2 says large support-furniture changes require rebuilding the reference representation, so this small manipulated-object setting does not establish robustness to Rescan's whole-furniture movement. This is not a proof of arbitrary segmentation-error tolerance, nor a verified reproduction of the exact 2020 configuration. [2022 primary article](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.828732/full)

## Other claims in the pasted discussion

| Source | What was confirmed | Scope limit |
| --- | --- | --- |
| [SRGCD official repository](https://github.com/pengmoubuaixuexi/SRGCD) | Reported mIoU 94.11% on Urb3DCD-V1 and 78.79% on HKCD; stable unchanged seeds and progressive refinement; README says code and pretrained models will be released soon | An official repository exists, but executable released code/checkpoints are not established by its title. Urban benchmark scores do not establish indoor RGB-D robustness or a final defer/reobserve state |
| [Semantic and Geometric Fusion, 2025](https://www.mdpi.com/2072-4292/17/7/1311) | Object-based mIoU 81.83% and point-based ablation 83.43% are reported | Urban LiDAR structural/vegetation change setting; useful for object-vs-point tradeoffs, not an indoor furniture score |
| [LoDA author preprint, 2026](https://arxiv.org/html/2608.05356v2) | Local detectability, roughness/density and pose-uncertainty propagation combined with object-level inference | Vehicle LiDAR urban maps; five labels are Added/Removed/Increased/Decreased/Unchanged. Ambiguous annotation cases are excluded from benchmark statistics; that is not an evaluated final suspected/unknown output |
| [Semantics-aided construction change, 2022](https://doi.org/10.1016/j.autcon.2021.104057) | Geometry, semantic class probabilities and evidence fusion are the relevant mechanism | UAV construction setting and extra observation/probability inputs; hard instance IDs alone do not supply those uncertainties |

## Updated research judgment for this project

The question is now explicitly about measured geometry plus an imperfect automatic segmentation, not reliance on researcher-provided IDs. Prioritize ROI Similarity for local geometric candidate refinement, and the 2022 Langer successor for reproducible indoor object association/partial-cluster correction. Keep M3C2 as a reproducible surface-distance/uncertainty baseline and PlaneSDF as a plane-supported indoor-region comparison. This is a research shortlist, not a measured Rescan ranking or a decision to implement.

The pasted four-way state proposal (changed, suspected, unchanged, unobserved/unknown) remains a project research design. None of the checked result tables demonstrates that exact product behavior, and this source check does not establish novelty or prove that no prior method does so. Separate segmentation uncertainty, geometric inconsistency and missing observations; do not convert an ROI similarity or benchmark mIoU directly into a calibrated change probability.

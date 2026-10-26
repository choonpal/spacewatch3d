# Langer 2020 and the 2022 follow-up: primary evidence

Research date: 2026-10-15. Scope: verify published methods and available source code only. No experiment, package installation, viewer/data change or automation update was performed. Statements in the user's pasted research text are claims to check, not instructions to execute.

The 2020 publication is *Robust and Efficient Object Change Detection by Combining Global Semantic Information and Local Geometric Verification*, DOI [10.1109/IROS45743.2020.9341664](https://doi.org/10.1109/IROS45743.2020.9341664), IROS pp. 8453–8460. [Official publication record](https://repositum.tuwien.at/handle/20.500.12708/77137)

Method details and numerical results below were checked in the author's 2023 dissertation **Chapter 5**, which identifies that chapter with the 2020 work. The standalone IROS full text was not independently accessible through the tools; consequently Table 5.2 is cited as the dissertation's account, not an independently verified transcription of the IROS proceedings table. The dissertation's printed page numbers differ from PDF page numbers: printed p. 42 is PDF page 53. [Official thesis record and full text](https://repositum.tuwien.at/handle/20.500.12708/176591)

## 1. The 2020 method uses predicted semantic classes, not supplied instance identities

SparseConvNet trained on ScanNet supplies a semantic class per reconstruction vertex. Classes that may support objects restrict horizontal-plane fitting; points protruding above planes are Euclidean-clustered. Clusters predicted as `otherprop` are additional candidates. Thus the method does not assume reliable GT instance IDs, but is also not an arbitrary-instance-label comparison algorithm. The author explicitly discusses small ScanNet objects being undersegmented or merged into supporting structures. [Thesis §§5.2.1, 5.3.2.2, printed pp. 42–44, 47](https://repositum.tuwien.at/handle/20.500.12708/176591)

The target `novel object` includes a single object **or a heap** newly located in the vicinity. Slight movement of permanent furniture is intentionally tolerated. A separate final `suspected change / needs re-observation / unknown` category is not established by this method description. Open-world/unknown *object category* is different from unknown *change evidence*. [Thesis §5.2 and §5.1, printed pp. 40–42](https://repositum.tuwien.at/handle/20.500.12708/176591)

## 2. Local geometric verification: what it does, and the precise boundary failure

Verification first ICP-aligns surrounding support planes, then aligns cropped object geometry to the reference using restricted horizontal translation/yaw. Plane points are removed for the second alignment. A low reference overlap marks a candidate as newly introduced; tiny residual clusters are filtered. This absorbs some local reconstruction drift and small placement changes rather than classifying every raw point discrepancy as change. [Thesis §5.2.2, printed pp. 44–45](https://repositum.tuwien.at/handle/20.500.12708/176591)

Selected experimental settings: reference/current crop margins 0.20/0.05 m; plane/object ICP correspondence gates 0.05/0.15 m; point-difference threshold 0.014 m; novel-object overlap gate 0.7. These are configured gates, not measured uncertainty bounds or calibrated probabilities. [Thesis Table 5.1, printed p. 47](https://repositum.tuwien.at/handle/20.500.12708/176591)

Crucially, the author reports that a newly added sugar box touching a permanent coffee machine becomes one cluster. The coffee machine aligns well; the whole cluster is rejected, missing the box. Small objects may also falsely align to background. Therefore **the 2020 method identifies imperfect boundaries as a problem but does not solve arbitrary merged-instance separation**. [Thesis §5.3.5 and Fig. 5.5, printed pp. 49–51](https://repositum.tuwien.at/handle/20.500.12708/176591)

## 3. Experimental scope and exact metric interpretation

The dataset consists of five environments, **31 recordings including five clean references**, and 260 YCB object placements. It is not 31 independent rooms or 260 object categories. Full maps were reconstructed with Voxblox from robot RGB-D recordings. The task focuses on detecting introduced small objects while tolerating slight furniture rearrangement. [Thesis §§3.2, 5.3.1, printed pp. 20–22, 46](https://repositum.tuwien.at/handle/20.500.12708/176591)

Point precision is `TP/(TP+FP)`, recall `TP/(TP+FN)` and F1 their harmonic mean. Scores are averaged across setups per room and `±` denotes standard deviation. Missed objects (`M`, no GT point detected) and wrong clusters (`W`, no overlap with a GT object) are summed within each room; the table additionally reports an Average row. These measures are not equivalent to semantic change-type accuracy. [Thesis §§5.3.4–5.3.5, printed pp. 48–49](https://repositum.tuwien.at/handle/20.500.12708/176591)

| Table 5.2 Average row | Point precision | Point recall | Point F1 | M | W |
| --- | --- | --- | --- | --- | --- |
| Semantic labels alone | 0.23 ± 0.15 | 0.64 ± 0.21 | 0.32 ± 0.14 | 15.0 | 115.0 |
| Candidate detection without support-plane candidates | 0.64 ± 0.27 | 0.58 ± 0.19 | 0.57 ± 0.14 | 18.2 | 34.2 |
| Candidate detection without local verification | 0.20 ± 0.09 | 0.74 ± 0.14 | 0.31 ± 0.11 | 2.8 | 225.6 |
| Full 2020 approach, as reported in the thesis | 0.60 ± 0.23 | 0.69 ± 0.15 | 0.61 ± 0.16 | 5.6 | 71.0 |

Source: [Thesis Table 5.2, printed p. 50 / PDF page 61](https://repositum.tuwien.at/handle/20.500.12708/176591).

Two caveats matter when reusing numbers:

- The text also states 97.8% novel-object detection and 14 false positives per recording, but those summaries cannot be reconciled directly with the table's per-room missed totals and 260 placements without additional aggregation clarification. Prefer the explicitly defined table metrics; do not promote the sentence into an overall accuracy claim.
- PlaneSDF's later `Best of [10]` row combines best metric entries from this work's variants rather than denoting the full pipeline as one fixed configuration. Its average precision/recall/missed/wrong values differ from `Ours (full)`. Do not label that row the standalone Langer full-method result. This follows by comparing the two primary tables. [PlaneSDF Table I](https://arxiv.org/pdf/2207.08323#page=7), [Langer thesis Table 5.2](https://repositum.tuwien.at/handle/20.500.12708/176591)

The dataset owner's current download was cleaned to remove duplicate Voxblox/annotation points and has an updated performance table. Dataset versions therefore also matter. [Official dataset page](https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/object-change-detection-dataset-of-indoor-environments/)

## 4. The 2022 follow-up really does add partial matching and cluster splitting

*Where Does It Belong? Autonomous Object Mapping in Open-World Settings* is a distinct 2022 paper, corresponding to dissertation Chapter 6. Its outputs are **static, moved, removed and novel**. These are four exhaustive change/object states, not a confirmed/suspected/unknown triage. [Primary publication, §§3.1–3.5](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.828732/full)

Its relevant capabilities and bounds are:

- Local ICP and semilocal/global PPF matching associate object geometry. Matching points seed color/normal region growing; matched parts are separated and remaining parts become new candidates. Repeated matches can disentangle a heap across visits.
- This is conditional on identifiable matched parts and geometry/color boundaries. The paper reports region growing crossing the Pringles/tea-box boundary and failed separation under partial reconstruction. If changes never provide separating evidence, a heap may remain one object.
- Supporting surfaces and local reconstructions are central. Its task is manipulable objects on approximately stable supports; it states that significantly moved major furniture/support surfaces require rebuilding the reference. It is not demonstrated as arbitrary whole-room furniture-mesh change handling.
- Similarity thresholds and matching confidence are used internally, but the paper does not leave an explicit needs-more-observation final class.

Sources: [§§3.2, 3.4, 3.5.1–3.5.3 and §4.2 failures](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.828732/full). Its ObChange local-surface evaluation is different from the 2020 full-map evaluation; do not mix their counts or performance.

## 5. Public source code: verified follow-up, unverified standalone 2020 reproduction

Author repository [edith-langer/tidyupvisionpipeline](https://github.com/edith-langer/tidyupvisionpipeline) explicitly identifies itself as code for **the 2022 Frontiers paper**. The inspected default branch is `Frontiers_submission`, commit `365812063d1753c4bcb344512c759edd6e818f13`. It includes C++ matching, PPF and evaluation sources. No code was built or run.

Verified source locations:

- [README](https://github.com/edith-langer/tidyupvisionpipeline/blob/365812063d1753c4bcb344512c759edd6e818f13/README.md): paper identity and four output categories.
- [object_matching.cpp](https://github.com/edith-langer/tidyupvisionpipeline/blob/365812063d1753c4bcb344512c759edd6e818f13/change_detection/src/object_matching.cpp): weighted graph matching, partial matching and repeated cluster splitting.
- [local_object_verification.cpp](https://github.com/edith-langer/tidyupvisionpipeline/blob/365812063d1753c4bcb344512c759edd6e818f13/change_detection/src/local_object_verification.cpp): the current alignment implementation even contains a TODO about an IROS2020-style two-stage alignment, corroborating that this should not be labeled an unchanged 2020 reproduction.

A complete public author implementation specifically reproducing the 2020 semantic-candidate/two-stage-verification pipeline has not been verified. The 2022 code is useful existing implementation evidence, while its current buildability, performance on Rescan and direct compatibility with existing instance labels remain untested.

# Selecting one real Rescan pair with movement, addition and removal

Checked 2026-10-15 (Asia/Seoul). Bounded primary-source research supporting the real-pair viewer. This note does not establish a completed pair selection or completed frame registration.

## What the authors establish

The dataset has real repeated captures of 13 indoor spaces, with three to five captures per space. The supplemental paper says scene contents were moved between captures and additions/removals occurred in most cases. It provides temporally stable, unique instance IDs for object associations, and hand-authored alternative ID assignments for indistinguishable objects. It also explicitly documents incomplete scanning and partial geometry, which can make an object appear absent despite uncertain physical presence. [Rescan supplement, sections 2 and 3.4](https://rescan.cs.princeton.edu/assets/Halber19Rescan_supp.pdf)

No author-provided named archive pair was found that explicitly certifies **all three change types in the same two captures**. Supplemental Table 4 visualizes all scenes but does not give the archive member names in its extractable labels; figure order must not be silently mapped to `scene_a` through `scene_m`. Figure 6 discusses newly appearing trash bins, but supplies no archive ID or epoch pair. The additional 15-capture temporal-stability example is not evidence that the inspected 45-capture archive includes that sequence. [Same supplement, sections 3.2, 3.4 and Table 4](https://rescan.cs.princeton.edu/assets/Halber19Rescan_supp.pdf)

The official project documents `instance_idx` in each labeled surfel PLY and original `color`/`depth` streams. It describes stable IDs across time within a scene. It does **not** promise a unique ID namespace across different rooms, independent motion-transform ground truth, camera poses in the raw release, or a per-pair change-event catalog. [Authors' dataset page](https://rescan.cs.princeton.edu/)

## Exact ID and ambiguity semantics in the official code

The PLY-to-evaluation converter treats PLY instance ID **0 as valid**. It maps `instance_idx == 1024` to unlabelled evaluation ID 0, and otherwise maps to `instance_idx + 1`. Semantic/instance evaluation labels are packed as `1000 × class_idx + evaluation_instance_id`. Therefore a search that discards all PLY IDs less than or equal to zero would lose a valid object; do not confuse PLY IDs with converted evaluation IDs. [Official `create_eval_files` converter, `write_instance_transfer_file`](https://github.com/mhalber/Rescan/blob/master/apps/create_eval_files/main.cpp)

The temporal evaluator loads an optional same-basename `.txt` under the scene's `gt_segmentation` directory. Each row's first number is the base **zero-based PLY ID** and subsequent numbers are allowed alternatives; the `|` character is a delimiter. For example, `4 | 4 7` allows base PLY ID 4 to be scored against predictions 4 or 7. The evaluator adds 1 to all of these IDs, chooses the allowed prediction with the largest overlap for that GT row, and swaps that row's diagonal entry with the selected alternative before IoU computation. [Official `evaluate_instance_transfer.py`, `reorder_confusion`](https://github.com/mhalber/Rescan/blob/master/scripts/evaluate_instance_transfer.py)

Consequences for this viewer:

- Preserve original PLY IDs and the supplied ambiguity rows in provenance.
- These rows are association allowances, **not motion transformations or lists of added/removed objects**.
- The evaluator's row-wise score adjustment is not a globally consistent bijective object matching algorithm. Do not automatically mutate all scene IDs from these rows.
- A disappeared ID plus a new ID within a permitted ambiguity set should not be presented as definite physical removal/addition without geometry and original-image support.
- Correspondence for indistinguishable chairs may remain ambiguous even when movement of the chair arrangement is visible. A generic visible chair-change description can be more defensible than an exact identity assertion.

## Recommendation for selecting the same two real captures

Recommend **no named room/epoch from the paper alone**. Select from the archive's actual full PLY inventories, then keep both meshes and every displayed RGB frame from those same two source captures. Do not combine a moved object from one room with addition/removal examples from another room to satisfy the one-pair request.

### Concrete current shortlist: `scene_e_0 → scene_e_2`

The implementation owner's bounded 768 KiB compressed-prefix inspection shortlisted this pair: sampled afterward-only IDs 28 and 29 are chairs; sampled before-only IDs include table 10, chairs 14–17 and 22, other furniture 23, desk 24 and sofa 26. These were **prefix findings only**, not verified absence/motion or a completed selection.

Subsequent **full-cloud inventory reported by the implementation owner** confirms IDs 19, 20 and 21 occur in both captures, while IDs 28 and 29 occur only afterward, with 7,620 and 7,552 surfels respectively. Before has chair IDs `{14,15,16,17,19,20,21,22}` (eight), afterward `{19,20,21,28,29}` (five). This is substantial annotation/geometry evidence rather than an orphan prefix segment. Original before/after photos still need to establish the physical events and visible locations; this research task did not perform that inspection.

The exact ambiguity member `scene_e/gt_segmentation/scene_e_2.txt` was retrieved in memory using bounded HTTP ranges from the [official dataset ZIP](https://rescan.cs.princeton.edu/assets/rescan_dataset.zip). It is 24 uncompressed bytes, and ZIP reading verifies the member CRC. Its nonempty rows are:

```text
28 | 19 28
29 | 21 29
```

Thus exact identities of afterward chairs 28 and 29 can be exchanged with IDs 19 and 21 in the official evaluator. Both apparent prefix additions require special care; raw ID difference alone is weak evidence of a new physical chair. However, the allowance **does not prove there was no addition**: if actual before/after geometry and original frames establish an extra physical chair in an ambiguous group, the addition event may be valid while that chair's identity remains uncertain. Preserve event evidence separately from identity confidence.

In this particular full inventory, group `{19,28}` grows from one present member to two, and group `{21,29}` also grows from one to two. The fact that all chairs together decrease from eight to five does not contradict two additions and five removals. Multiplicity supports addition events modulo exact identity; it does not identify which indistinguishable chair moved, and it does not replace original-photo confirmation. Keep `scene_e_0 → scene_e_2` as the current preferred candidate rather than rejecting it merely because the TXT lists IDs 28 and 29.

The same archive contains `scene_e_1.txt` with `27 | 27 29` and has no `scene_e_0.txt` member. Do not apply an epoch-1 allowance indiscriminately to epoch 2. The full compressed PLY member sizes are 10,148,092 bytes for epoch 0 and 5,693,568 for epoch 2; these make complete annotation inspection practical. No large files were downloaded by this research task.

### Bounded ambiguity checks for fallback rooms

Exact small TXT members were also read from the same official ZIP in memory, with ZIP CRC verification:

| Member | Uncompressed bytes | Nonempty ambiguity rows |
| --- | ---: | --- |
| `scene_a_0.txt` | — | No such archive member |
| `scene_a_2.txt` | 78 | `4 \| 4 7`; `5 \| 4 6 5`; `6 \| 4 5 6`; `7 \| 4 7 11`; `10 \| 10 12`; `11 \| 7 11`; `12 \| 10 12` |
| `scene_a_3.txt` | 56 | `4 \| 4 7`; `6 \| 10 6`; `7 \| 4 7 11`; `11 \| 7 11`; `10 \| 10 6 12` |
| `scene_b_2.txt` | 117 | Bases 4, 5, 12, 13, 24 each allow `{4,5,12,13,24}`; bases 17 and 19 each allow `{17,19}` |
| `scene_b_3.txt` | 73 | Bases 17/19 allow `{17,19}`; bases 30/33 allow `{30,33}`; bases 32/35 allow `{32,35}` |

These files reside under the respective room's `gt_segmentation/` folder. At epoch `scene_a_3`, IDs 32–35 have no supplied allowance in the inspected epoch-2/3 files. At `scene_b_3`, IDs 32, 33 and 35 do have allowances, while ID 34 does not. This can guide an easier-to-explain fallback candidate if full inventories contain those added IDs. **No supplied allowance** means exactly that, not a guarantee that an object's appearance is unique or that addition occurred. The full clouds and original frames remain the evidence for a fallback event.

The local prefix sampler in `tools/find_rescan_change_pair.py` can reduce downloads and shortlist rooms, but its sampled ID sets and centroids are not evidence of full-capture absence or motion. An ID not seen in a prefix may occur later in the same PLY. Its centroid may be biased by the sampled subset. This is a limitation of that selection method, not an author claim.

For each shortlisted pair with before set B and after set A, using full labeled surfel files:

| Initial candidate | Required source evidence before calling it a physical event |
| --- | --- |
| `A − B`: added ID | Meaningful supported object geometry afterward; previously observed empty location or a clearly inspected original before frame; no ambiguity allowance explaining the identity difference |
| `B − A`: removed ID | Meaningful supported object geometry before; subsequently observed empty location or a clearly inspected original after frame; no ambiguity allowance explaining the identity difference |
| `A ∩ B`: shared ID with location difference | Stable background registration, object-surface fit/rigid displacement rather than centroid difference alone, and consistent original image evidence in both captures |

Structural IDs such as wall, floor and ceiling should not be counted as introduced or removed movable objects merely because scan extent or segmentation differs. A very small orphan segment is a weak selection candidate. Use a substantial, visually distinct object for each requested type when possible; chairs around a table are specifically documented as identity-ambiguous.

A chronological suffix comparison determines before/after order but does not make a skipped interval invalid: `scene_X_0 → scene_X_2` can be one real pair if it has clearer evidence than adjacent captures. Every displayed frame should retain its actual capture ID and frame ordinal. Pair selection should prefer shared room coverage and clear original-frame views over the largest number of differing IDs.

## Mesh and original-frame linkage boundaries

Rescan releases surfels rather than triangle meshes; the official pipeline separately reconstructs them using Poisson reconstruction. Reconstruct each selected capture from its own real surfels and preserve support/provenance rather than treating generated triangles as measurements. [Official reconstruction script](https://github.com/mhalber/Rescan/blob/master/scripts/run_poisson_reconstruction.py)

Camera recovery and RGB alignment constraints are detailed in [rescan-pose-and-change-method.md](rescan-pose-and-change-method.md). The raw-stream lineage supports depth decoding, but missing original calibration means derived camera mappings remain estimates until their correspondence is checked. An original RGB screenshot connected only by capture membership does not establish that the screenshot shows a particular changed object. Generated mesh renders must never be substituted for original before/after keyframes.

The user requested a quick viewer without diagnostic panels or extra testing. The interface can stay focused on the two meshes and original before/after frames. Retain source/evidence status in data and concise captions as needed; do not add a diagnostics page or broaden testing from this research. Clear frame support is still necessary before asserting the three physical change types.

## Current decision

The primary sources support Rescan as a suitable **real-source candidate**, but cannot name a certified three-change pair. `scene_e_0 → scene_e_2` is the current preferred candidate: full chair inventories support two group-level additions with exact identity ambiguity, and epoch-2 allowances are directly verified. The implementation owner's geometry and original-frame inspection must establish physical event locations and movement. Record that result as local data evidence, separately from the authors' dataset-wide statements. Do not announce a completed moved + added + removed demonstration until the same pair's actual files and original frames support all three.

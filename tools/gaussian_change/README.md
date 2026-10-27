# Mesh + Gaussian change detection

Fresh inputs run through mesh global registration → regional rigid correction
and residual translation calibration → same-pose Gaussian RGB/expected-depth/alpha
rendering → bidirectional pretrained SimSaC → person/common-observation filtering
→ multi-view 3D fusion and saved results. No saved detections are consumed.
Video reconstruction, Gaussian training and keypoint selection are upstream work.
Saved cameras are inputs; there is no planner or keypoint-version dependency here.

For native 4K display with local perspective inference, set:

```json
{"settings": {"width": 4096, "height": 2048, "cube_size": 2048,
  "inference_mode": "perspective_tiles", "perspective_tile_size": 1040,
  "perspective_tile_overlap": 32}}
```

At every supplied camera both maps are freshly rendered into six 100-degree
perspective faces. Each 2048-square face supplies four overlapping 1040-square
tiles, giving 24 paired local views. SimSaC still uses its pretrained 520/256
branches, but receives local perspective tiles rather than the resized complete
panorama. Both directional probability maps are averaged in tile overlaps and
projected back into the common 4K panorama for person/alpha/support filtering and
3D lifting. Per-view `directional-inference.json` records source image hashes,
tile rectangles and zero complete-panorama model calls. The existing `panorama`
mode remains available for reproducing the baseline.

From the repository root on the existing CUDA host:

```sh
python -m pip install -e .
python -m spacewatch3d.tasks.task_5a_change_detection --manifest input.local.json --runtime runtime.local.json --output outputs/change-run
```

The same command works through an already authenticated SSH file-queue relay.
Add `remote_session` (local relay directory), `remote_root` (new server jobs root),
`python` (server interpreter), `environment` (CUDA/PATH/library variables), and
`input_location` (`server` or `local`) to the local runtime JSON. Server input
paths must be absolute; local mode transfers the six mesh/map/camera input files.
The adapter transfers this source, launches the same worker, mirrors progress,
and downloads the results. Passwords are not accepted or stored by this code.
Deployment paths and capture-specific settings belong in ignored `.local/` or
`*.local.json` files, not in Git. Use a new output directory for every run.

Optional runtime `download_profile: viewer` downloads tour images, 3D outputs and
audit JSON while keeping dense per-view arrays and native perspective source images
on the GPU host. `download-receipt.json` records the full remote evidence directory;
the default `full` profile downloads everything. No server evidence is deleted.

`python -m tools.gaussian_change.pipeline` is an equivalent repository entry point.
`--validate-only` checks file references, hashes, coordinate bridges and camera
contracts without running geometry fitting or a GPU. That is not an inference test.

## Input contract

See `input.example.json` and `runtime.example.json`.

- Each capture supplies a triangle mesh, standard binary little-endian 3DGS PLY
  (xyz, opacity logits, log-scales, wxyz rotations, SH coefficients), saved cameras,
  and **explicit** within-capture coordinate bridges into a +Y-up frame with
  shared declared units. Global Sim3 also estimates relative uniform scale;
  it does not recover an unknown absolute metric scale from images.
- Mesh NPZ accepts `p` or `vertices`, plus `faces`. Optional `face_semantic` or
  `face_class_id` uses the existing MTPano wall=0/floor=3 convention. Without labels,
  normals/heights provide a structural proxy, recorded as such in the result.
  PLY/OBJ triangle meshes supported by Open3D are also accepted.
- Cameras JSON accepts `requests[].reference_from_camera` (saved render requests),
  `cameras[].camera_to_world`, or `poses[].camera_to_world`. Declare `opencv`
  (right/down/forward) or `opengl` (right/up/back). Camera rotations must be rigid.
- `world_from_mesh`, `world_from_gaussian`, `world_from_camera` are within one
  capture. They do **not** supply/replay the before/after registration result.
- All saved camera poses from **both paths** are used by default. At each pose,
  both corrected maps are rendered from the exact same common camera. Optional
  `max_views_per_capture` explicitly makes a smaller smoke run and is recorded.

## Methods and results

The global adapter extends the mesh SimSaC branch's FPFH/RANSAC/point-to-plane ICP
strategy with a relative scale fit, symmetric overlap and competing-solution gates.
It is a fresh Sim3 fit.
It does not replay the older experiment's saved GeoTransformer matrix. Regions
are derived from reference mesh XZ occupancy and frozen after global fitting;
the old experiment's hardcoded 13-room partition is not reused for new inputs.
Regional fitting reuses the original robust structural `rigid_icp` and residual
estimator. Disjoint samples fit local SE3, calibrate residual translation, and
validate support. Unsupported regions remain recorded and are excluded from
change candidates. Geometric overlap does not prove physical alignment in a
repetitive corridor or across different floors.

The renderer transforms Gaussian means, scales, quaternion frames and SH viewing
directions together. All Gaussians are retained. Depth is **Gaussian expected
radial depth**, not sensor truth. Alpha from both renders and a union of person
segmentation/expanded person boxes mask unreliable comparisons. The supplied
official SimSaC checkpoint loads strictly; weights, source commit, RGB hashes,
camera matrices, masks and raw probabilities are recorded. No new model is trained.

Output files:

| File | Meaning |
| --- | --- |
| `status.json` | Stage, failure or completed execution receipt |
| `inputs.json`, `model.json` | Input/model hashes and actual settings |
| `registration.json` | Fresh global matrix, frozen regions, regional matrices, bias, support |
| `cameras.json` | Supplied path poses transformed into common coordinates |
| `renders/V*/{before,after}.png` | Actual Gaussian RGB inputs to SimSaC |
| `renders/V*/*-geometry.npz` | Depth, alpha, person mask, exact camera pose |
| `renders/V*/inference.npz` | Both probability maps and filtering masks |
| `changes.json`, `changes-3d.npz` | Deduplicated 3D candidate boxes, points and view links |

The output preserves surface-change candidates and adds `object-events.json`.
The lifecycle stage binds unary instance-mask observations to the exact input mesh
when supplied, or uses supplied face instance IDs. Semantic connected components
are a fallback hypothesis, not verified physical IDs. Instance observations can be
provided in the camera JSON as `instance_segmentation`, or through a side's
`instance_observations` file. The packet uses schema `mesh-instance-observations-v1`,
`coordinate_frame: mesh`, the input `mesh_sha256`, unary `observations` and source
`cameras`. Cross-capture event labels and previous pairings are not consumed.
The browser can bind packets from a private administrator registry by exact mesh
hash; an unregistered input never inherits another mesh's segmentation.

Appearance features use native Gaussian RGB renders and the frozen SimSaC VGG
backbone. The original mask-assisted matching gates check shape, displacement
beyond registration uncertainty, shape gain over stationary surfaces, and observed
destination vacancy. A unique match produces one estimated origin; a motion
hypothesis with assignment ambiguity retains every feasible `origin_candidates`
entry and sets `exact_origin_confirmed: false`. The viewer draws those as alternative
origins rather than a certain trajectory. Added/removed estimates require observed
free space, not missing geometry alone. Evidence gaps remain `unresolved` with
rejection reasons. Same-position appearance changes use `modified`; structural
surfaces lacking object support are recorded separately without deleting raw
SimSaC detections. No physical identity ground truth is claimed.

## System API

`spacewatch3d.tasks.task_5a_change_detection.gaussian_backend.analyze(manifest,
output, runtime)` runs the same workflow. The shared task dispatcher additionally
supports `backend: mesh_gaussian_simsac` with `pair_manifest` and `runtime` config
fields and the existing before/after objects lineage manifests. For that shared
contract entry point, `scene_id` and both `capture_id` fields in the pair manifest
must match the lineage manifests. Surface candidates are linked as a separate
artifact instead of being fabricated into confirmed object events.

CPU integration checks:

```sh
python -m unittest discover -s tests -p test_mesh_gaussian_pipeline.py -v
```

Compare two completed runs without modifying their event labels:

```sh
python -m tools.gaussian_change.comparison --baseline BASELINE_OUTPUT --current NEW_OUTPUT
```

The new output receives `comparison.json` and its manifest link. The report checks
mesh/camera input hashes, camera order, checkpoint and classification gates. It
reports both unresolved counts/rates and outcomes for each previously unresolved
region. A missing re-detection is not counted as successful classification. The
browser displays this report alongside the new keyframe change tour. If map and
inference settings change together, this is a combined experiment, not an isolated
causal or ground-truth accuracy evaluation.

Those tests cover coordinate conventions, a known structural offset and duplicate
multi-view fusion. They do not replace real CUDA rendering/inference or accuracy
evaluation. Capture-specific GPU execution receipts remain in the output directory.

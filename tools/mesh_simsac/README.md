# Segmented mesh + original RGB change detection

The viewer now accepts a capture ZIP and runs SimSaC on its **original RGB** on
first submission. It registers the segmented meshes, excludes people in both
images, aggregates independent camera observations, classifies objects and
publishes a job-specific viewer. Gaussian renders and saved change predictions
are not inputs. The shared `structural.py` remains available to the Gaussian
pipeline.

## Start the viewer

```sh
python apps/autotour360/server.py --port 4198 \
  --mesh-change-runtime /path/to/runtime.local.json \
  --mesh-change-jobs-dir /path/to/persistent/jobs
```

Open `/mesh-changes.html` or the main viewer's change-detection button. Select a
ZIP and click **입력하고 변화탐지**. Upload, validation, registration, inference,
classification and result delivery run through `/api/mesh-change-jobs`.
Reloading `?job=<id>` resumes status/result display. There is no default dataset.

When runtime `capture_catalog` points to an administrator-provided JSON catalogue,
the same page also lists existing server captures. Selecting both captures starts
input preparation and the same fresh inference job automatically. Selecting an
input ZIP also starts automatically. `GET /api/mesh-change-jobs/captures` exposes
only display metadata; `POST /api/mesh-change-jobs/from-captures` accepts registered
`before` / `after` IDs, never client-supplied server paths.

The catalogue contains `captures` with `id`, `label`, `mesh`, `scene`, `source_run`,
`coordinates`, `video`, `classes`, `predictions`, `prediction_manifest` and optional
`candidate_regions` / `segmentation_model`. The semantic-scene importer preserves
the complete source triangle map, maps saved OBJ regions to exact original
triangles, verifies video/RGB receipts and samples actual PLY records for depth
support. Original 2D semantic classes gate first-hit mesh associations; these are
explicitly recorded as **surface candidates, not verified video instance tracks**.
People and the reconstruction's supplied person mask are excluded. No previous
change labels or predictions are imported. Catalogue `frame_stride` controls
sampling over the full source video; `point_limit` bounds real PLY record sampling.
`comparison_settings` configures the same pipeline as uploaded bundles. Source
scale assumptions and sampling are retained in the input manifest and audit.

The result cache hashes the declared input files (including the manifest's
camera poses and thresholds), model weights, upstream SimSaC Python sources,
pipeline/viewer sources and execution environment. Only a successful result
with intact published assets is reusable, including after a server restart.
Changed input or settings triggers a new run. Failed/interrupted work does not
produce a result URL. Jobs are serialized per viewer process; use one process
for a jobs directory. An interrupted remote worker can be inspected using the
private `remote.local.json` receipt before resubmitting.

## Input bundle

The ZIP root must contain `manifest.json` with schema `mesh-simsac-input-v1`.
Use [the downloadable example](../../apps/autotour360/mesh-input.example.json).
Replace the example camera matrices with calibrated values for the actual
images. All paths are relative to the ZIP root; external paths and symlinks are
rejected. Only declared source files are transferred to the inference host.

Each `before` / `after` capture contains:

- `id`, `mesh`, `projection`, `frames`. Mesh NPZ arrays are `vertices (N,3)`,
  `faces (M,3)`, `face_instance_id (M,)`; 0/-1 are unlabelled surfaces.
- Each frame has a unique `id`, `image` and `world_from_camera` (OpenCV camera
  axes). `projection=perspective` also requires `K` at the original resolution;
  `erp` creates six perspective directions from the original panorama.
- Optional `objects` JSON provides an `instance_records` array of
  `instance_id` and `label`. Unlabelled instances can still be matched by shape
  and RGB; they are not assumed identical because their IDs match.
- Optional `person_mask` on each frame excludes people. Without complete supplied
  masks, the runtime must have a person detector. Removing the photographer from
  a mesh alone does not prove that their pixels are absent from the photo.
- Optional frame `segmentation` NPZ contains ERP (or perspective) `class_id`,
  `instance_id` and `person` masks. Capture `segmentation_labels` maps class IDs
  1..N to labels, with 0 unknown. Positive instance IDs must match the mesh's
  instance IDs; 0 means no mesh correspondence, not empty space. Bind masks to
  their source image using `segmentation_rgb_sha256`. These original 2D masks
  veto mesh-only object pixels and preserve people omitted from the mesh.
  `import_sam31.py` adapts existing SAM results without running the model again.
- Optional `pointcloud` PLY/NPZ adds independent foreground/depth support.
  Without it, triangle first-hit depth is used and mesh surface samples are used
  only for registration. The provenance explicitly records that no independent
  point-cloud observation was supplied.
- Optional `world_from_mesh`, `world_from_pointcloud` connect coordinate frames.
  Use common units and a world with Y as the upright axis. Length thresholds are
  in input-map units, not automatically metres.

`settings.registration=auto` computes fresh global alignment and regional
wall/floor correction using main's shared mesh registration implementation.
Fitting, translation-bias calibration and residual validation use disjoint
surface samples. The same transforms are applied to the real point clouds,
meshes and per-region RGB camera projection. Whole-cloud bidirectional residual
P90 (without discarding large residuals) and held-out regional error bounds
raise the displacement/absence threshold. Unsupported regions remain
unresolved; these empirical bounds do not account for every possible
calibration, reconstruction or semantic error.

Already registered inputs
use `provided` and an explicit `common_from_world` on both captures. Optional
frozen regional alignment preserves triangle IDs and uses each region's camera
transform for cross-capture projection. At least two independent source frames
are required to establish repeatable change/absence evidence.

## Classification and inspection

- **Moved:** one-to-one identity from rigid shape and masked SimSaC-backbone
  appearance, displacement beyond registration error, and independent vacancy
  observations at both origin and destination.
- **Added / removed:** repeatable own-object observations, no plausible ambiguous
  counterpart and the opposite capture observing the expected location free.
  A separate image-supported path handles thin objects whose coarse mesh lies
  on the background: at least two source masks must support the object's mesh,
  and two unoccluded opposite frames must observe the projected surface with
  at most 2% same-class mask support. The class must be in the supplied label
  catalogue. This is an automatic semantic estimate, not proof that segmentation
  is complete. The recorded reason distinguishes this path from geometric vacancy.
- **Modified:** a stationary, unambiguous matched object with two-frame identity
  support and repeatable RGB change on both sides.
- **Unresolved:** occlusion, incomplete views or ambiguous repeated objects.

ERP results open as draggable 360-degree views of the original panorama frames,
with mouse-wheel zoom and arrow-key rotation. The initial view is original RGB
with one candidate box per photo; whole-image segmentation outlines are confined
to the separate segmentation inspection page. Click a box to zoom and highlight
only that photo's selected object: cyan before, amber after. The opposite
capture's mesh is never painted as an observed object. An absent or unresolved
counterpart retains an expected-location box without a fabricated object fill.
Selecting another event returns to RGB and boxes. The object direction button
restores the initial view. Ordinary perspective photos remain flat images.
The 3D mesh is an expandable detail.
The photo shadows are projected from the original segmented triangles into
that photo's camera; the RGB itself is never synthesized for SimSaC. People are
masked out of those overlays. Cyan denotes before, amber after. Movement keeps
the actual origin and destination and draws an arrow; it is not recentred to
 hide displacement. Before/after photos label the movement origin and destination
separately; the 3D detail retains the movement arrow. Opacity and before/after
controls are available in that detail. Completed jobs use the current renderer
without rewriting their cached inference data or classifications.

These are automatic evidence-based classifications, not guaranteed physical
identity or ground-truth accuracy. Missing reconstruction is not proof of
disappearance.

## Runtime

Configure the inference host separately from uploaded inputs. A local example:

```json
{
  "local_python": "/path/to/python",
  "simsac_source": "/path/to/SimSaC",
  "simsac_checkpoint": "/path/to/simsac_changesim.pth",
  "person_checkpoint": "/path/to/yolo11l-seg.pt",
  "three_package": "/path/to/three/package",
  "device": "cuda",
  "environment": {"CUDA_VISIBLE_DEVICES": "0", "TORCH_HOME": "/path/to/cache"}
}
```

Requires NumPy, SciPy, Pillow, Open3D, PyTorch/torchvision, ultralytics for person
detection and official [SimSaC](https://github.com/SAMMiCA/SimSaC) source and
checkpoint. Three.js supplies the viewer assets and its licence is copied with
them. Keep local runtime files untracked. For an existing authenticated SSH
relay add `remote_session`, `remote_root`, remote `python`, and optional
`dependency_paths`; model paths then refer to that host, while `local_python`
and `three_package` remain local. The worker verifies model hashes again before
inference. Credentials are not accepted from uploaded manifests.

Direct local worker invocation uses the same pipeline as the viewer:

```sh
python -m tools.mesh_simsac.worker --manifest inputs/manifest.json \
  --output outputs/new-run --runtime runtime.local.json
python -m unittest discover -s tests -p 'test_mesh_simsac*.py' -v
python -m unittest discover -s apps/autotour360/tests -p test_mesh_change_jobs.py -v
```

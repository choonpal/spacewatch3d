# Mesh, RGB and point-cloud runtime validation — 2026-10-27

Implementation branch: `codex/mesh-rgb-runtime`, based on `origin/main`
`efef293`. The upload viewer starts registration and SimSaC inference on first
submission. Original RGB, segmented mesh, camera calibration and optional real
point clouds are inputs; Gaussian-rendered RGB and old predictions are not.

Current agreed scope uses already generated meshes, point clouds and calibrated
camera poses with the original video frames. Video-only reconstruction and
segmentation are deferred; the viewer does not claim to generate them from two
uploaded videos. Registration, SimSaC and classification still run on first
submission of these prepared inputs.

## Initial end-to-end execution (superseded classification)

The input ZIP was submitted through the browser's upload control. It contains
two segmented meshes, two real point clouds, instance labels and 11 original
RGB frames per capture. Registration was set to `auto` with regional correction;
previous alignment matrices were removed from the submitted manifest.

Fresh job: `5534716b264343f0bdd4bda7`, completed with `cached=false`.

- Fresh global structural registration and 18 regional corrections.
- 11 regions supported by held-out validation; unsupported evidence is withheld
  from definitive lifecycle classification.
- Independent point-cloud bidirectional overlap: 0.994952 / 0.991321.
- Whole-cloud bidirectional residual P90: 0.091037 / 0.086598 input-map units.
- Original-RGB SimSaC pairs: 261; remote workflow time: 134.37 seconds.
- Results: added 0, moved 0, removed 0, modified 4, unresolved 6.

The modified/unresolved results above exposed classification defects in the
subsequent image/mask audit. They are historical output, not the current verdict.

These inputs are Dorm2 and Dorm3, different floors. This is an execution and
alignment check, not a ground-truth temporal-change benchmark. Overlap is not
change-detection accuracy. Regional residuals, whole-cloud residuals, person
masks, occlusion and repeated observations constrain classification, but do not
cover every calibration, reconstruction or identity error.

## Cache and UI

Submitting the identical ZIP again through the upload control completed as job
`310a6e2d0df84b0aa0d3d5cd` with `cached=true`, the same input/model/source
fingerprint and no new worker-attempt directory. Cache validation took about
four seconds after upload.

Browser checks verified original images shown first, photo bounding-box shadow
toggle, expandable registered 3D mesh and real point-cloud display. Browser
console had no warnings or errors. Cyan shadows represent before and amber
shadows after. The movement origin/destination drawing and classification gates
are covered by automated tests; this real input produced no confirmed movement.

Screenshot: workspace `outputs/mesh-rgb-runtime-20261027/image-comparison.png`.

## Automated verification

- Repository Python suite: 53 passed.
- Application Python suite: 30 passed, 3 skipped because trajectory analysis
  dependencies are not installed in the local test environment.
- Existing viewer JavaScript suite: 80 passed.
- Mesh job HTTP/cache/corruption/failure tests: 5 passed.
- JavaScript syntax checks, Python compilation and `git diff --check` passed.

See `tools/mesh_simsac/README.md` for the input schema, launch command and
classification evidence requirements. Runtime machine paths and credentials
remain outside source control.

## Original-mask and 360-degree audit

The audit reuses the existing SAM masks and source RGB; it does not reconstruct
the videos or rerun segmentation. Each imported mask is bound to its RGB SHA256.
Two reproducible defects were fixed: an unrelated unsupported region inflated
all objects' matching tolerance, and projected mesh IDs were accepted without
agreement with the source-image instance masks. Stationary matches now also
require independent identity observations, an unambiguous match and RGB change
support from both captures.

Coarse geometry can attach a thin object to its background. A wall/floor surface
at that position is no longer treated as proof that the object exists in both
captures. A separate, explicitly recorded semantic rule requires repeated
source-mask support and unoccluded opposite background observations, with the
object class included in the supplied label catalogue. This is an automatic
image-supported addition/removal estimate, not a claim of complete segmentation
or accurate isolated object geometry.

The old unresolved objects were reviewed individually: E1-I13 (chair) remains
unresolved; E1-I27 and E1-I29 (chairs) lack repeated change after source-mask
filtering; E1-I34 and E1-I36 (tables) lose their change support after excluding
missing segmentation and person/projection contamination; E1-I38 (extinguisher)
is added under the semantic-background rule. The two tables are visible in
before RGB despite missing before mesh instances.

ERP evidence is now rendered from the complete original panorama in independent
before/after cameras, with drag/arrow-key rotation and wheel zoom. Boxes follow
rotation. Original-mask and mesh-shadow toggles preserve the viewing direction.
This is extracted-frame inspection, not continuous video playback.

Final fresh job: `beb887b0c1744746b9972681`, `cached=false`, fingerprint
`1dfdffc3a9aa7710602fa51a53ec6cf760514a0681116e6933bb0f7e2af11752`.
The remote workflow took 134.82 seconds, excluding transfer time, and ran 261
SimSaC pairs. Counts: added 1, moved 0, removed 0, modified 0, unresolved 4.
The added extinguisher has change support in after frames 133/139/142 and
semantic background support in before frames 121/124/130/133/136. The remaining
unresolved items are two doors, one chair and one vacuum cleaner. Registration
and whole-cloud residual statistics match the initial execution above.

Current regression checks: 58 repository Python tests and 5 job HTTP/cache tests
passed; JavaScript syntax, Python compilation and whitespace checks passed.
The previously recorded application/viewer suites remain applicable.

Final browser checks confirmed drag rotation, boxes following the camera,
object-view reset/zoom and box-click shadow toggling. Narrow-panel layout was
checked, and the final result console had no warnings or errors. Screenshot:
workspace `outputs/mesh-rgb-runtime-audit-20261027/360-comparison.png`. The
workspace `review.md` in that directory records all six previous unresolved
objects with original RGB/SAM contact sheets.

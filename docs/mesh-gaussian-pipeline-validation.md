# Fresh-input mesh / Gaussian pipeline validation

Validated on 2026-10-27. Execution code and the input contract are in
[`tools/gaussian_change`](../tools/gaussian_change/README.md); the system adapter
is in [`task_5a_change_detection`](../src/spacewatch3d/tasks/task_5a_change_detection/README.md).

## Actually executed

- CPU: seven input/geometry/system-adapter tests passed; nineteen existing shared
  contract tests passed. The shared object-contract adapter test mocks GPU execution
  and checks lineage and the distinction between candidates and object events.
- The wheel built and contains both processing packages and the `src` task adapter.
- An existing pair of triangle meshes, full native Gaussian parameters and saved
  camera requests was submitted through the SSH transport. Fresh mesh global
  registration and 25 regional corrections ran without consuming old detections,
  old inter-map matrices or the historical fixed room partition.
- A four-camera smoke run completed, followed by an unrestricted **27-camera**
  run through the installed task module's CLI. The full GPU worker took **96.31 s**
  (excluding transfer); 14 of 25 regions passed structural support checks.
- Both Gaussian maps were rendered at each common camera. All 27 saved before/after
  camera matrices match exactly; depth, alpha and probabilities are finite. Selected
  masks have no pixels in the union person mask or outside common valid support.
- Bidirectional SimSaC used official source commit
  `43abe282231dc7d8c8cb829a92f9ebc014fa6cb6` and strict-loaded checkpoint SHA256
  `55f28690989603b5b01723d255bbc63157bacc704e635100ae517b0ad4d1f99a`.
  Its same-image control had zero pixels over threshold (maximum probability 0.08470).
- The full run saved **16 3D surface-change candidates**, 37,766 sampled 3D points,
  37,873 selected render pixels, probability maps, masks and per-view evidence.
  Repeated views are fused before stable candidate IDs are assigned.

## What this does not establish

This is an execution/integration check, not a detection-accuracy benchmark or a
claim of correct physical object identity. The pair is from different floors.
The minimum directional global structural overlap was 0.34294 at 0.05 reference
height units; this is a geometric fit diagnostic, not a percentage accuracy.
Unsupported regional comparisons are excluded from candidates. Absolute metric
scale is unverified; depths are Gaussian expected depths.

The generic global adapter uses fresh FPFH/RANSAC/Sim3 ICP. Regional robust ICP
and residual estimation reuse the existing structural functions. It does not
replay the experiment-specific GeoTransformer selection or historical partition.
Object lifecycle classification is not asserted by this candidate-only adapter.

## Browser upload execution

The viewer subsequently received all six raw mesh/Gaussian/camera files and a
coordinate descriptor through its file selectors. Clicking **Run change detection**
created a new job, transferred the uploaded inputs to the GPU host, and ran the
same CLI from global registration onward. No previous changes result was submitted.
The full 27-camera GPU worker completed in 95.80 seconds and saved 16 candidates.
Transfer time is additional. The browser waited for the downloaded output before
marking completion. Keyframe navigation, candidate boxes and after-render blending
were exercised in the browser on this uploaded-input result.

The result inspection screen is a Gaussian-rendered RGB panorama viewer, with 3D
boxes projected through each recorded camera. It is not a new native splat renderer.
Three browser-job service tests, eleven existing HTTP server tests and two existing
Gaussian integration UI-module tests passed. The service tests cover input-file
binding, incomplete-input rejection and waiting for downloaded results.

Input data, model weights, deployment paths, camera-version settings and detailed
run receipts remain outside Git in local configuration/output directories.

## Local automatic classification and shadow viewer validation

The browser workflow additionally connects unary instance-mask observations
by the exact input mesh hash when the browser's Run button is pressed. A new
six-file browser upload was executed from fresh registration through classification
on the GPU. All 27 supplied camera requests ran; 438 unary observations from 481
source camera poses were bound without importing previous event labels or pairs.
Native Gaussian RGB object renders and frozen SimSaC VGG features were recomputed.

That run took 192.09 seconds on the GPU, excluding transfer, and produced 15 raw
surface candidates. Seven structural surface candidates lack supported object
instances and are recorded separately. Three raw regions were grouped into one
motion hypothesis; five candidates remain unresolved. There were no added/removed
estimates on this pair. The motion hypothesis retains three feasible origins and
does **not** establish an exact physical object correspondence. This is an
integration check, not an accuracy claim. No manual candidate overrides were used.

The browser automatically loads the resulting object events. Clicking the fresh
motion box was exercised: a cyan representative origin shadow, an orange destination
shadow, an estimated movement arrow, same-camera before/after/overlay controls and
the XZ location diagram were visible. Alternative origins remain in the original
detector JSON. Change-only navigation groups repeated observations at one
representative keyframe, skips empty cameras and aims at the visible change box.
Full route mode preserves all supplied camera stops. Both modes and box-click
shadows were exercised in the browser. Fourteen processing tests, four job-service
tests, eleven HTTP tests and seven JavaScript navigation/viewer tests passed.

## Native 4K maps with perspective-tile inference

A second fresh GPU run used the existing alternative native Gaussian models
(296,679 and 308,246 Gaussians). The original meshes, 438 unary observations,
27 saved camera requests, model checkpoint and classification gates were held
fixed. The source camera frame-to-mesh scale/axis bridge was also applied to these
native model coordinates. Global and regional registration were recomputed: 14/25
regions remained supported and the maximum global matrix element difference was
0.0000853. No previous registration or detections were supplied to detection.

Each pose produced a new 4096×2048 panorama from six 2048-square native Gaussian
perspective renders. Each face supplied four overlapping 1040-square tiles;
648 paired tiles were inferred bidirectionally over all 27 poses. Audit records
show zero complete-panorama inference calls. The same-image perspective control
had zero pixels over threshold and maximum probability 0.085193. The full GPU
worker completed in 870.46 seconds, excluding result transfer. Complete dense
evidence remains on the GPU host; the viewer download profile fetched images,
3D results and JSON receipts.

| Result | Baseline whole panorama | 4K perspective tiles |
| --- | ---: | ---: |
| Raw surface candidates | 15 | 40 |
| Moved estimates | 1 | 1 |
| Added / removed estimates | 0 / 0 | 0 / 0 |
| Same-position appearance changes | 0 | 3 |
| Unresolved events | 5 | 14 |
| Unresolved share of events | 83.3% | 77.8% |

Unresolved **count increased**. A lower fraction with more detections is not an
accuracy improvement. Twelve new unresolved events lack a linked instance and two
lack sufficient correspondence/observation evidence. Under the comparison gates,
four previous unresolved regions had no corresponding new detection and one was
excluded as a structural surface; none was counted as successfully classified.
The same moved instance pair retained its correspondence, with appearance
similarity 0.921→0.944 and feasible origin hypotheses 3→1. This remains an estimate,
not physical-identity ground truth. Native map and inference changes were combined,
so the run does not isolate their individual effects.

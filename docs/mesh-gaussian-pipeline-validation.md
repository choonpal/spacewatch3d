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

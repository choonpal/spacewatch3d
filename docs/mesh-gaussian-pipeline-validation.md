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

This validation did **not** exercise browser map-upload controls. CLI/API execution
and the existing saved-result viewer are separate surfaces. Input data, model
weights, deployment paths, camera-version settings and detailed run receipts remain
outside Git in local configuration/output directories.

# Integration notes

## Model identity

- Framework: Nerfstudio 1.1.5
- Method: Splatfacto / 3D Gaussian Splatting
- Training steps: 60,000
- Training face resolution: 1024x1024
- Source samples: 256 ERP camera poses, six faces per pose
- Gaussian initialization: supplied colored sparse point cloud
- Camera optimizer: off
- Bilateral grid: on
- Pose preprocessing: shared-frame bounded Savitzky-Golay SE(3) regularization

## Portable loading

Nerfstudio stores paths in `config.yml`. The supplied renderer calls `eval_setup(..., test_mode="inference")` and replaces the saved output/data roots before pipeline creation. The minimum dataparser payload is `inference_data/transforms.json` plus `sparse_pc.ply`; RGB training images are not loaded during inference.

## Rendering contract

Input: one or more camera JSON documents containing `camera_position_glb` and `look_at_glb`.

Output per camera:

- `<name>_360.png`: RGB ERP
- `<name>_360.json`: source camera, model/checkpoint, orientation and initial view metadata
- `render_summary.json`: all output records

The renderer currently invokes Python directly. For a service integration, keep one Nerfstudio pipeline resident on the GPU and reuse `render_cube_faces()` and `stitch_equirectangular()` instead of reloading the 746MB checkpoint for every request.

## Quality limitations

The output is 4096x2048, but apparent detail is bounded by training blur, residual pose error, Gaussian scale and view coverage. Reference sharpening improves matching captured positions but must not be presented as geometry-aware novel-view synthesis. Generative super-resolution is intentionally not included because it can invent text and object boundaries.

## Security and deployment

The bundled viewer binds to `127.0.0.1` by default. If integrated into a remote service, add authentication, request limits, isolated output directories and GPU job serialization before changing the bind address.

# Real before/after instance-map data access

Checked 2026-10-14 (Asia/Seoul). This is an access and file-inventory audit, not a completed data integration.

## Practical result

No immediately downloadable **complete** real pair with triangular instance-labeled mesh, original RGB, depth, camera poses/intrinsics and scan alignment was verified in this bounded audit. Two datasets satisfy the data requirements but their official access routes require a user action: **3RScan** (access form/token) and **MultiScan** (Hugging Face sign-in and license acknowledgment). The directly downloadable **Rescan** release has real before/after instance-labeled surfel point clouds and RGB/depth streams, but the inspected release has no pose/intrinsics sidecars and no triangle meshes. Do not describe a Rescan point-cloud rendering as an original posed RGB keyframe.

## 3RScan — closest match, access required

[Official toolkit](https://github.com/WaldJohannaU/3RScan) documents repeat real indoor scans, persistent object IDs, textured meshes, calibrated RGB-D, camera poses, scan-to-reference transforms and object-change transforms. Per scan files: `mesh.refined.v2.obj`, `.mtl`, texture PNG, `sequence.zip`, `labels.instances.annotated.v2.ply`, over-segmentation JSON and `semseg.v2.json`; `3RScan.json` supplies scan alignment and change metadata. The `sequence.zip` supplies original color/depth, camera pose and intrinsics. See the [official FAQ](https://github.com/WaldJohannaU/3RScan/blob/master/FAQ.md) for projection and visibility rendering.

The [official current download instructions](https://vmnavab26.in.tum.de/3RScan/documentation.php) require the [3RScan Terms of Use form](https://forms.gle/NvL5dvB4tSFrHfQH6). The [official public Git tree](https://api.github.com/repos/WaldJohannaU/3RScan/git/trees/master?recursive=1) has no scan mesh or `sequence.zip` sample: only code, split lists and teaser/frame screenshots under `data/img/`. No form submitted, agreement accepted, access credentials searched, or gated-data URL used.

## MultiScan — complete alternate format, access required

The [official README](https://github.com/smartscenes/multiscan) now directs the main dataset to [3dlg-hcvc/MultiScan on Hugging Face](https://huggingface.co/datasets/3dlg-hcvc/MultiScan). The repository states sign-in and acknowledgment of conditions are required for its data; data license is CC BY-NC 4.0. No acknowledgment submitted.

[Official dataset structure](https://3dlg-hcvc.github.io/multiscan/read-the-docs/dataset/index.html) explicitly identifies `scene_00000_00` / `scene_00000_01` as two states of one real environment and documents, for each scan:

- `scene_xxxxx_xx.ply`: triangular mesh, vertex colors and per-face object/part IDs.
- `scene_xxxxx_xx.annotations.json`: object/part labels, oriented boxes and articulation states.
- `textured_mesh/scene_xxxxx_xx.obj`, `.mtl`, texture PNGs.
- `scene_xxxxx_xx.mp4`: original H.264 RGB stream.
- `scene_xxxxx_xx.depth.zlib` and `.confidence.zlib`: synchronized depth/confidence streams.
- `scene_xxxxx_xx.jsonl`: each frame's timestamp, intrinsics, camera-to-world transform and orientation.
- `scene_xxxxx_xx.align.json`: scan gravity/common-coordinate and reference-rescan alignment transforms.

[Camera format docs](https://3dlg-hcvc.github.io/multiscan/read-the-docs/dataset/files/acquired.html) define column-major 3×3 K and 4×4 camera-to-world matrices, with the ARKit-to-Open3D conversion. [Annotation docs](https://3dlg-hcvc.github.io/multiscan/read-the-docs/dataset/files/annotation.html) define per-triangle `objectId` and `partId`. **Do not assume object IDs persist across scans** without reading real pair annotations: the docs describe object-list indices, not a verified cross-scan identity guarantee. The example [single-scan annotation JSON](https://3dlg-hcvc.github.io/multiscan/read-the-docs/_static/dataset/scene_00000_00.annotations.json) is publicly accessible but alone cannot complete the workflow.

## Rescan — ungated real instance point-cloud pair, incomplete for requested workflow

The [authors' project page](https://rescan.cs.princeton.edu/) links the [direct dataset ZIP](https://rescan.cs.princeton.edu/assets/rescan_dataset.zip). The [official implementation](https://github.com/mhalber/Rescan) calls its 3D representation a point cloud and reconstructs a mesh separately with Poisson reconstruction. The release provides stable instance segmentation of real repeated scans, 45 sequences of 13 spaces, together with RGB H.264 and compressed depth streams. The project page does not show a separate explicit dataset license; the code MIT license should not be automatically treated as the data license.

Read-only HTTP checks returned `200` to HEAD, `Content-Length: 26239218080` (~24.44 GiB), `Accept-Ranges: bytes`, and `206 Partial Content` to bounded Range GET. The ZIP is ZIP64 and has 198 entries. Its central directory is 21,579 bytes at offset 26,239,196,403. It contains only scene folders with `gt_segmentation/*.ply` and occasional ambiguity `.txt`, `color/*.h264`, `depth/*.depth`, plus `scene_list.txt` and `nyu40_classes.txt`. No camera trajectory, intrinsics, triangle mesh, aligned-pose sidecar, or depth decoder file appears in the archive inventory.

A bounded decompression of `scene_i/gt_segmentation/scene_i_0.ply` verifies 185,588 vertices and a header with XYZ, normals, RGB, radius, `class_idx`, `instance_idx`, **no element face**. A bounded `.depth` header inspection shows binary compressed sensor payload, without readable camera metadata. We cannot prove no calibration is embedded anywhere in this opaque stream, but no official complete decoder/pose mapping was found; that uncertainty prevents an evidence-backed RGB-keyframe linkage.

Smallest useful point-cloud candidate is `scene_i_1` / `scene_i_2`; compressed member inventory (offset is local ZIP-header offset, not compressed-payload offset):

| Member | Compressed bytes | Uncompressed bytes | Local-header offset |
| --- | ---: | ---: | ---: |
| scene_i/gt_segmentation/scene_i_1.ply | 4,952,476 | 6,966,524 | 14,620,017,493 |
| scene_i/gt_segmentation/scene_i_2.ply | 4,454,048 | 6,290,147 | 14,624,970,064 |
| scene_i/color/scene_i_1.h264 | 130,587,316 | 130,566,580 | 14,776,184,033 |
| scene_i/color/scene_i_2.h264 | 119,378,892 | 119,360,010 | 14,906,771,435 |
| scene_i/depth/scene_i_1.depth | 110,695,494 | 122,922,665 | 15,026,150,485 |
| scene_i/depth/scene_i_2.depth | 107,108,680 | 119,339,088 | 15,136,846,066 |

To selectively fetch members after choosing this limited fallback: parse the central directory, resolve ZIP64 offsets, GET the 30-byte local header, compute payload start as offset + 30 + filename length + extra length, then fetch compressed bytes and raw-DEFLATE decompress (method 8). Check decompressed size and CRC. This was only inspected in memory; **no scene files or large archives downloaded**.

## RIO10 — complete source data also gated; public helper ZIP is depth-only

The [official RIO10 README](https://github.com/WaldJohannaU/RIO10) describes derived real 3RScan train/validation RGB frames and poses, 3D models `mesh.obj`/`labels.ply`, and 2D instance masks. `seqXX_01` (train) and `seqXX_02` (validation) have released poses; hidden test sequences exclude ground truth. It explicitly routes full downloads through [this RIO Terms of Use form](https://docs.google.com/forms/d/e/1FAIpQLScA-zciAgxMz7r8xirsPQodeQnPk2XA68yBpSxP10167B-M_g/viewform?usp=sf_link), also linked in the [official RIO10 download documentation](https://vmnavab26.in.tum.de/RIO10/documentation.php). No form submitted.

The [official setup.sh](https://github.com/WaldJohannaU/RIO10/blob/master/setup.sh) directly downloads predictions and `rio10_val.zip`; the README says this helper supplies validation **depth images**, not the original RGB or meshes. Checked the official helper URL `https://campar.in.tum.de/public_datasets/RIO10/rio10_val.zip`: HEAD returns 200, 608,909,290 bytes, byte-range supported. ZIP metadata has 34,451 entries; bounded reads of 2,261 central-directory entries from the beginning and 2,259 at the end show only `frame-xxxxxx.rendered.depth.png` plus directories/`.DS_Store`. This confirms the inspected public helper cannot replace the full paired mesh/RGB data. Public GitHub `data/` contains `GT_RIO10.txt`, `intrinsics.txt`, `metadata.json`, and `stats.txt`, but no meshes or color frames.

Rescan ambiguity TXT files were individually retrieved and decompressed in memory: they contain lines such as `4 | 4 7` listing interchangeable object IDs, not poses/calibration. The only remaining release TXT files are the scene-name list and semantic-class mapping. No hidden pose sidecar was found.

## Recommended decision

Use 3RScan when the user supplies approved access/download or an already acquired reference+rescan folder. MultiScan is a valid substitute after the user obtains accepted access; choose a small pair with real changed geometry, read IDs and alignment before selecting camera views. If immediate ungated data is required, Rescan can support an honestly labeled **real point-cloud change demonstration**, but original pose-linked before/after keyframes and actual released mesh maps remain unfulfilled unless the missing calibration/trajectory is obtained or computed.

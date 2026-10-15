# Real Rescan before/after object viewer

The viewer at `/rescan.html` compares a real repeated capture pair with restored
triangle meshes, moved/added/removed object overlays, and original stationary
RGB keyframes. The current three-type example is **scene_e_0 → scene_e_2**.
The previous scene_i_1 → scene_i_2 example remains in its separate local output.

## Data and processing

- Author source: [Rescan](https://rescan.cs.princeton.edu/), [official code](https://github.com/mhalber/Rescan).
- The downloader extracts only selected members of the official ZIP with HTTP
  ranges; it checks member sizes and CRC and records source SHA256 in `source.json`.
- Input PLY contains surfels (XYZ, normals, RGB, radius, semantic class, persistent
  instance annotations), not triangle faces. `prepare_rescan_pair.py` restores
  Poisson meshes, removes unsupported vertices, and creates browser geometry.
- Instance association is provided annotation, not an automatic segmentation
  prediction. A rigid object fit estimates translation/rotation. Appearance and
  disappearance candidates use object annotation presence, excluding missing
  wall/floor/ceiling/unlabelled surfaces.
- Exact vertex/triangle identities and partial-cloud centroid differences are
  not used to classify object movement.
- Original H.264 frames and synchronized depth timestamps are retained. Estimated
  depth-camera poses supply candidate photo links. RGB-to-render feature/PnP
  checks support optional dashed photo boxes; failed RGB checks retain no box.
  These are recovered cameras, not released ground-truth calibration/poses.

## Reproduce the chosen pair

Install `tools/requirements-rescan-review.txt` in an appropriate Python environment.
The raw depth decoder wrapper needs a C++11 compiler and the two official
[ScanNet depth2pgm headers](https://github.com/ScanNet/ScanNet/tree/master/ScannerApp/depth2pgm):
`uplinksimple_image-codecs.h` and `uplinksimple_shift2depth.h`, stored under the
ignored `third_party/rescan-depth/` folder. On Windows compile the wrapper to
`third_party/rescan-depth/decode.exe`, using that folder as the include directory.

```powershell
python tools/download_rescan_pair.py --scene scene_e --epochs 0 2 --depth --output data/rescan/scene_e_0-2
python tools/prepare_rescan_pair.py --data data/rescan/scene_e_0-2 --output outputs/rescan-three-changes
python tools/recover_rescan_keyframes.py --data data/rescan/scene_e_0-2 --depth-data data/rescan/scene_e_0-2 --output outputs/rescan-three-changes --stride 90
python tools/build_rescan_object_changes.py --data data/rescan/scene_e_0-2 --output outputs/rescan-three-changes
python apps/autotour360/server.py --port 8768 --rescan-dir outputs/rescan-three-changes
```

The pair-selection helper samples bounded annotation prefixes of all capture
members. Its shortlist is not an absence annotation or change-detection result.
All displayed events come from the chosen full PLY files.

## Interpretation

`changes.json` preserves classification thresholds, source hashes, estimated
object transforms, correspondence source, and per-frame camera/link status.
The user-facing screen shows the maps, three change types, and linked photographs.
Diagnostic registration and same-capture control panels are omitted.

The official scene_e_2 ambiguity rows are `28 | 19 28` and `29 | 21 29`.
Both groups have one annotated member before and two after: this supports two
addition events at group level, while exact chair identity is ambiguous. These
allowances and group counts are retained in the output. Missing objects remain
annotation-presence candidates; the absence of a PLY label alone does not prove
observed empty space. Original same-location photos let the user inspect them.

Within these 1-before/2-after groups, the least-motion supported rigid fit resolves
the display association so an existing chair at its previous position is not
automatically highlighted as a newly introduced chair. The original annotation IDs
and the inferred association mapping remain separate in `changes.json`.
Presence candidates lacking an opposite-capture photo observation are omitted
from the visible change list and retained as `unlinked_presence_candidates`.

Source videos, depth streams, restored meshes, and output binary buffers stay
in ignored `data/`, `outputs/`, and `third_party/`; they are not committed to Git.

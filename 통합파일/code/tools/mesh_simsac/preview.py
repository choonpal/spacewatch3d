"""Export source mesh/point-cloud geometry and a local 3D candidate review."""
import argparse
from pathlib import Path
import shutil

import numpy as np

from .core import read, sha, transform, write
from .regional import aligned_mesh, load_regional, FreshRegionalAlignment


def export(root, manifest_path, three_package):
    root, manifest_path = Path(root), Path(manifest_path).resolve()
    original, prepared = read(manifest_path), read(root / "manifest.json")
    directory = root / "preview"
    directory.mkdir(exist_ok=True)
    vendor = directory / "vendor"
    vendor.mkdir(exist_ok=True)
    package = Path(three_package)
    for source in ["build/three.module.js", "build/three.core.js", "examples/jsm/controls/OrbitControls.js", "LICENSE"]:
        shutil.copyfile(package / source, vendor / Path(source).name)
    scenes = []
    regional, _ = load_regional(original, manifest_path.parent, prepared['registration']['common_from_world'])
    if prepared['registration'].get('fresh_structural'):
        regional = FreshRegionalAlignment(prepared['registration']['fresh_structural'])
    for epoch, spec in enumerate([original["before"], original["after"]]):
        def resolve(value):
            path = Path(value)
            return path if path.is_absolute() else manifest_path.parent / path
        mesh_path = resolve(spec["mesh"])
        vertices, mesh = aligned_mesh(root, prepared, spec, epoch, manifest_path.parent)
        common = np.array(prepared["registration"]["common_from_world"][epoch])
        matrix = common @ np.array(spec.get("world_from_mesh", np.eye(4)))
        p = vertices[mesh['faces']].reshape(-1, 3).astype('<f4')
        if "rgb" in mesh:
            rgb = mesh["rgb"][mesh["faces"]].reshape(-1, 3).astype("u1")
        else:
            rgb = np.full(p.shape, 160, "u1")
        ids = mesh["face_instance_id"].astype("u4")
        # Deterministic colours show original segmentation boundaries; IDs never cross captures.
        colour = np.column_stack([(ids * 73 + 43) % 190 + 40, (ids * 131 + 71) % 190 + 40,
                                  (ids * 191 + 19) % 190 + 40]).astype("u1")
        colour[ids == 0] = [155, 162, 172]
        colour = np.repeat(colour, 3, axis=0)
        file = directory / f"mesh-{epoch}.bin"
        file.write_bytes(p.tobytes() + rgb.tobytes() + colour.tobytes())
        path = resolve(spec["pointcloud"]) if "pointcloud" in spec else None
        if path is None:
            cloud = np.empty((0, 3))
        elif path.suffix.lower() == ".npz":
            cloud = np.load(path)["points"]
        else:
            import open3d as o3d
            cloud = np.asarray(o3d.io.read_point_cloud(str(path)).points)
        cloud = cloud[np.isfinite(cloud).all(1)]
        cloud = cloud[::max(1, int(np.ceil(len(cloud) / 50000)))]
        world = transform(cloud, np.array(spec.get('world_from_pointcloud', np.eye(4))))
        cloud = transform(world, common)
        if regional:
            cloud = regional.move(cloud, regional.assign(world, epoch), epoch)
        cloud = cloud.astype('<f4')
        (directory / f"cloud-{epoch}.bin").write_bytes(cloud.tobytes())
        scenes.append(dict(epoch=epoch, id=spec["id"], vertices=len(p), mesh=f"mesh-{epoch}.bin",
                           cloud=f"cloud-{epoch}.bin", cloud_points=len(cloud),
                           bounds=np.stack([p.min(0), p.max(0)]).tolist(), source_mesh_sha256=sha(mesh_path),
                           source_pointcloud_sha256=sha(path) if path else None, triangles=len(mesh["faces"])))
    changes = read(root / "results/changes.json")
    write(directory / "scene.json", dict(scenes=scenes, regions=changes["regions"], source="Original segmented meshes and actual point clouds; no Gaussians"))
    shutil.copyfile(Path(__file__).with_name("preview.html"), directory / "index.html")
    return directory / "index.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--three-package", type=Path, required=True)
    args = parser.parse_args()
    print(export(args.root, args.manifest, args.three_package))

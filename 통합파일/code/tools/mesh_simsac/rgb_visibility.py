"""Fail-closed RGB correspondence visibility against the original full meshes.

Image bounds alone do not imply visibility. A first surface nearer than the
candidate blocks its evidence; missing geometry is unknown, never a negative
change vote. Uses existing registration tolerance, not class/ID exceptions.
"""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from .core import project_in_view, transform, camera_transform, similarity, sha

def first_hit_visibility(scene, origin, points, tolerance):
    import open3d as o3d
    points=np.asarray(points);delta=points-np.asarray(origin);distance=np.linalg.norm(delta,axis=-1)
    finite=np.isfinite(delta).all(-1)&(distance>1e-7)
    first=np.full(distance.shape,np.inf)
    if finite.any():
        rays=np.c_[np.broadcast_to(origin,delta[finite].shape),delta[finite]/distance[finite,None]].astype('f4')
        first[finite]=scene.cast_rays(o3d.core.Tensor(rays),nthreads=4)['t_hit'].numpy()
    known=finite&np.isfinite(first)&(first>0)
    gap=distance-first
    epsilon=1e-5*np.maximum(1.,distance)  # float32 ray/mesh numerical precision
    blocked=known&(gap>tolerance)
    uncertain=known&(gap>epsilon)&~blocked
    # Even inside registration tolerance, a point behind the first surface is
    # not evidence of visibility. Defer it rather than see through a thin wall.
    return known&(gap<=epsilon),blocked,(~known)|uncertain

class MeshVisibility:
    def __init__(self, root, prepared, tolerance):
        import open3d as o3d
        self.root=Path(root);self.tolerance=float(tolerance);self.stats=Counter()
        manifests=[Path(p) for p in prepared['source_hashes'] if Path(p).name=='manifest.json']
        specs=[]
        for path in manifests:
            data=json.loads(path.read_text())
            if data.get('schema')=='mesh-simsac-input-v1':specs.append((path,data))
        if len(specs)!=1:raise ValueError('Need unique original RGB/mesh manifest')
        self.path,self.input=specs[0]
        if sha(self.path)!=prepared['source_hashes'][str(self.path)]:raise ValueError('Input receipt changed')
        self.registration=prepared['registration'];self.scenes=[];self.base=[];self.scale=[]
        for epoch,key in enumerate(['before','after']):
            spec=self.input[key];path=self.path.parent/spec['mesh']
            if sha(path)!=prepared['source_hashes'][str(path)]:raise ValueError('Original mesh changed')
            with np.load(path) as z:vertices,faces=z['vertices'],z['faces']
            scene=o3d.t.geometry.RaycastingScene(nthreads=4)
            scene.add_triangles(o3d.core.Tensor(vertices.astype('f4')),o3d.core.Tensor(faces.astype('u4')))
            self.scenes.append(scene)
            base=np.asarray(self.registration['common_from_world'][epoch])@np.asarray(spec.get('world_from_mesh',np.eye(4)))
            self.base.append(base);self.scale.append(similarity(base)[1])

    def test(self, points, reference, regions, epoch):
        points=np.asarray(points);regions=np.asarray(regions);ij,z,inside=project_in_view(points,reference,regions)
        h,w=reference['depth'].shape;x=np.clip(ij[:,0],0,w-1);y=np.clip(ij[:,1],0,h-1)
        depth=reference['depth'][y,x];known=reference['mesh_hit'][y,x]&np.isfinite(depth)&(depth>0)
        same=reference['point_region'][y,x]==regions
        person=reference['person'][y,x]
        depth_blocked=known&(z>depth+self.tolerance)
        visible=inside&known&same&~person&~depth_blocked
        raw=np.empty_like(points);valid_region=np.zeros(len(points),bool)
        for region in np.unique(regions):
            if region<=0 or region>=len(self.registration['region_matrices'][epoch]):continue
            selected=regions==region
            matrix=np.asarray(self.registration['region_matrices'][epoch][int(region)])@self.base[epoch]
            raw[selected]=transform(points[selected],np.linalg.inv(matrix));valid_region[selected]=True
        visible&=valid_region;selected=np.flatnonzero(visible)
        exact_blocked=np.zeros(len(points),bool);unknown=np.zeros(len(points),bool)
        if len(selected):
            raw_pose=camera_transform(reference['pose'],np.linalg.inv(self.base[epoch]))
            passed,blocked,missing=first_hit_visibility(self.scenes[epoch],raw_pose[:3,3],raw[selected],self.tolerance/self.scale[epoch])
            visible[selected]=passed;exact_blocked[selected]=blocked;unknown[selected]=missing
        self.stats.update(pixel_reference_comparisons=len(points),outside_image=int((~inside).sum()),
            missing_first_surface=int((inside&~known).sum()),different_region=int((inside&known&~same).sum()),
            photographer=int((inside&person).sum()),blocked_depth=int((inside&depth_blocked).sum()),
            blocked_exact_ray=int(exact_blocked.sum()),unknown_exact_ray=int(unknown.sum()),visible=int(visible.sum()))
        return ij,visible

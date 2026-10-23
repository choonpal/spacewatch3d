import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dorm_mesh_geometry import transform,refine_similarity,erp_project


class DormGeometryTests(unittest.TestCase):
    def test_low_furniture_surfaces_do_not_replace_floor_or_tall_wall_references(self):
        from align_dorm_mesh_structure import smooth_structural_samples
        x,z=np.meshgrid(np.arange(-2,2,.1),np.arange(-2,2,.1));floor=np.c_[x.ravel(),np.zeros(x.size),z.ravel()]
        y,z=np.meshgrid(np.arange(.3,2.3,.1),np.arange(-2,2,.1));wall=np.c_[np.full(y.size,2.),y.ravel(),z.ravel()]
        # Geometry only: a tabletop and low vertical furniture sides.
        x,z=np.meshgrid(np.arange(-1,1,.1),np.arange(-1,1,.1));table=np.c_[x.ravel(),np.full(x.size,1.1),z.ravel()]
        y,z=np.meshgrid(np.arange(.4,1.1,.1),np.arange(-1,1,.1));side=np.c_[np.zeros(y.size),y.ravel(),z.ravel()]
        points,normals,kind=smooth_structural_samples(np.vstack([floor,wall,table,side]),2.5)
        self.assertGreater((kind==0).sum(),100)
        self.assertGreater((kind==1).sum(),100)
        self.assertTrue(np.all(points[kind==0,1]<.5))
        self.assertTrue(np.all(np.abs(points[kind==1,0]-2.)<.01))

    def test_relative_scale_and_pose_recovered_with_opposing_structural_planes(self):
        rng=np.random.default_rng(42);groups=[];normals=[]
        # A floor and two infinite perpendicular walls alone cannot identify scale:
        # scaling about their intersection preserves all three planes. Add a ceiling.
        for axis,offset in [(0,-2.),(1,0.),(2,3.),(1,3.)]:
            p=rng.uniform(-2,2,(350,3));p[:,axis]=offset;groups.append(p)
            normals.append(np.tile(np.eye(3)[axis],(len(p),1)))
        target=np.vstack(groups);target_n=np.vstack(normals)
        expected=np.eye(4);r=Rotation.from_euler('xyz',[.02,.08,-.03]).as_matrix()
        expected[:3,:3]=1.15*r;expected[:3,3]=[.3,-.15,.4]
        source=transform(target,np.linalg.inv(expected));source_n=target_n@r
        initial=expected.copy();initial[:3,:3]*=.96;initial[:3,3]+=[.05,-.04,.06]
        actual=refine_similarity(source,target,source_n,target_n,initial,4.)
        self.assertLess(np.max(np.linalg.norm(transform(source,actual)-target,axis=1)),.003)

    def test_panorama_world_camera_projection_and_seam(self):
        pose=np.eye(4);pose[:3,3]=[5.,1.,-2.]
        directions=np.array([[0,0,1],[1,0,0],[0,1,0],[.001,0,-1],[-.001,0,-1]])
        pixels,depth=erp_project(directions+pose[:3,3],pose,1000,500)
        np.testing.assert_allclose(pixels[:3],[[499.5,249.5],[749.5,249.5],[499.5,499.5]],atol=1e-4)
        self.assertGreater(abs(pixels[3,0]-pixels[4,0]),990)
        self.assertTrue(np.all(depth>0))

    def test_similarity_scales_camera_center_without_scaling_camera_axes(self):
        source=np.eye(4);source[:3,3]=[1,2,3]
        alignment=np.eye(4);rotation=Rotation.from_euler('y',.3).as_matrix()
        alignment[:3,:3]=2*rotation;alignment[:3,3]=[3,1,-1]
        point=np.array([[.4,.2,2.]])
        world=point@source[:3,:3].T+source[:3,3]
        result=source.copy();result[:3,:3]=rotation@source[:3,:3]
        result[:3,3]=transform(source[None,:3,3],alignment)[0]
        a,_=erp_project(world,source,1000,500)
        b,_=erp_project(transform(world,alignment),result,1000,500)
        np.testing.assert_allclose(a,b,atol=1e-6)

    def test_missing_or_occluded_opposite_mesh_is_not_a_visible_change(self):
        import open3d as o3d
        from run_dorm_mesh_comparison import visibility_gate
        def scene(depth):
            value=o3d.t.geometry.RaycastingScene()
            if depth is not None:
                mesh=o3d.geometry.TriangleMesh.create_box(6,6,.01).translate([-3,-3,depth])
                value.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            return value
        pose=np.eye(4);second=pose.copy();second[0,3]=.2
        cameras=[{'camera_to_world':pose.tolist()} for _ in range(4)]+[{'camera_to_world':second.tolist()}]
        for depth,expected in [(3.,1),(1.,2),(None,2)]:
            with self.subTest(opposite_depth=depth):
                status=np.array([1],np.uint8)
                visibility_gate(np.array([[0.,0.,2.]]),np.array([0]),status,[scene(2.),scene(depth)],[cameras,cameras],2.,.01)
                self.assertEqual(int(status[0]),expected)


if __name__=='__main__':unittest.main()

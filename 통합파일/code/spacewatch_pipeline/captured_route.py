"""Actual-frame candidate graph; existing Task5b objective and greedy selector stay shared."""
from types import MethodType
import numpy as np
from .io import digest

def build(data,scene,config):
    from spacewatch3d.tasks.task_5b_keyframes.routing import Route
    if data['schema']!='spacewatch.captured_trajectory.v1' or data['frame_id']!=scene.reference_frame_id or data['capture_id']!=scene.capture_id:
        raise ValueError('Captured trajectory frame/capture mismatch')
    if config.max_link_m is not None:raise ValueError('Captured-video v1 needs max_link_m=null; no synthetic transition cameras')
    seeds=data['sampled_pose_indices'];frames=data['frames']
    if not seeds or seeds!=sorted(set(seeds)) or seeds[0]<0 or seeds[-1]>=len(frames):raise ValueError('Invalid captured candidate indices')
    # Preserve the entire recorded graph. Sampling chooses seed candidates, not
    # which source poses the existing visibility audit is allowed to discover.
    indices=list(range(len(frames)));n=len(indices)
    positions=np.array([np.asarray(frames[i]['reference_from_camera'])[:3,3] for i in indices])
    adjacent={i:[] for i in range(n)};polylines={};lengths={}
    for a in range(n-1):
        from .existing_path import path_between
        line=np.array(path_between(data,indices[a],indices[a+1]))
        # Original planner/validator graph edges are endpoint segments. The full
        # recorded polyline is exported separately for post-selection display.
        length=float(np.linalg.norm(positions[a+1]-positions[a]))
        adjacent[a].append((a+1,length));adjacent[a+1].append((a,length))
        polylines[a,a+1]=line;polylines[a+1,a]=line[::-1]
        lengths[a,a+1]=length;lengths[a+1,a]=length
    route=Route(origin=np.zeros(2),grid_m=config.grid_m,walkable=np.ones((1,n),bool),
        clearance=np.zeros((1,n)),cells=[(0,i) for i in range(n)],positions=positions,
        components=[1]*n,adjacency=adjacent,candidates=list(seeds),obstacles=None,
        node_ids={j:'frame_'+str(frames[i]['source_frame_index']).zfill(8) for j,i in enumerate(indices)},
        viewpoint_ids={j:'vp_frame_'+str(frames[i]['source_frame_index']).zfill(8) for j,i in enumerate(indices)},
        camera_poses={j:np.array(frames[i]['reference_from_camera']) for j,i in enumerate(indices)},
        candidate_reasons={i:['actual_captured_frame_arc_sample'] for i in seeds},
        candidate_sampling=dict(uniform_seed_count=len(seeds),geometry_added_count=0,visibility_added_count=0,
             complete_source_pose_graph=True,eligible_source_frames=n,
             source='captured_video',spacing_m=data['spacing_m'],sampling=data['sampling']))
    route.edge_polylines=polylines;route.edge_lengths=lengths
    route.capture_metadata={j:{k:frames[i][k] for k in ['pose_index','source_frame_index','timestamp_s','arc_m','pts','time_base',
        'pose_origin','anchor_pose_indices','anchor_frame_indices','interpolation_alpha','pose_independently_estimated'] if k in frames[i]} for j,i in enumerate(indices)}
    base=route.export
    def export(self):
        value=base();value['method']='recorded_capture_trajectory'
        for edge in value['edges']:
            a=next(i for i in range(n) if self.node_id(i)==edge['from_node']);b=next(i for i in range(n) if self.node_id(i)==edge['to_node'])
            edge.update(recorded_polyline_m=polylines[a,b].tolist(),
                        source_pose_indices=[indices[a],indices[b]])
        value['source_frames']=frames;value['source_bindings']={self.node_id(i):self.capture_metadata[i] for i in range(n)}
        value['capture_source']={k:data[k] for k in ['video_sha256','pose_sha256','coordinates_sha256','source_size','spacing_m','sampling']}
        value['physical_safety_verified']=False
        return value
    route.export=MethodType(export,route)
    scene.provenance['capture_trajectory']=dict(identity=digest(data),source=data['video'],
        frames=len(frames),candidate_frames=len(seeds),spacing_m=data['spacing_m'],
        navigation_policy='recorded temporal capture path, not a walkability or physical collision certificate',
        floor_body_clearance_used=False,source_image_pose_exact=data.get('source_image_pose_exact',True),
        pose_origin=data.get('pose_origin','recorded_camera'))
    return route

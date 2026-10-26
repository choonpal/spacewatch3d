"""Minimum image vertices connecting frozen cores, without an itinerary.

All eligible published route nodes are candidates. Geometry-safe straight
chords define an undirected visibility graph. A single-commodity flow MILP
minimizes non-core vertices; selected images expose ALL safe mutual links.
This is not all-branch coverage, preservation of every corridor loop, or an
RGB/human-comfort model. No core observation scores are computed here.
"""
import copy
from itertools import combinations
import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from .models import Config, InputError
from .routing import footprint, safe_segments

DEFAULT_PRIMARY_TIME_LIMIT_S = 120.0
DEFAULT_SECONDARY_TIME_LIMIT_S = 5.0


def _solver_options(limit, name, allow_zero=False):
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, (int, float))
                             or not np.isfinite(limit) or limit < 0 or (limit == 0 and not allow_zero)):
        raise InputError(name + ' must be positive finite seconds (or None for unlimited); secondary also accepts 0 to skip')
    options = {'mip_rel_gap': 0.}
    if limit is not None:
        options['time_limit'] = float(limit)
    return options


def visibility_edges(ids, nodes, floor, obstacles, clearance):
    """Complete finite candidate graph; one vectorized row at a time."""
    points=np.asarray([nodes[n]['position_m'][:2] for n in ids],float)
    edges=[]
    for i in range(len(ids)-1):
        targets=points[i+1:]
        lines=np.stack([np.repeat(points[i:i+1],len(targets),axis=0),targets],axis=1)
        for j in np.flatnonzero(safe_segments(floor,obstacles,lines,clearance))+i+1:
            edges.append((ids[i],ids[int(j)]))
    return edges


def components(ids, edges):
    adjacency={n:set() for n in ids}
    for a,b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    groups=[]
    remaining=set(ids)
    while remaining:
        start=min(remaining)
        reached,stack={start},[start]
        while stack:
            node=stack.pop()
            for other in adjacency[node]-reached:
                reached.add(other)
                stack.append(other)
        remaining-=reached
        groups.append(reached)
    return groups


def minimum_vertices(ids, edges, cores, nodes, *,
                     primary_time_limit_s=DEFAULT_PRIMARY_TIME_LIMIT_S,
                     secondary_time_limit_s=DEFAULT_SECONDARY_TIME_LIMIT_S):
    """Exact finite-graph node-weighted Steiner connectivity via root flow.

    Core nodes cost zero; all others cost one even for existing old images.
    Flow is continuous. Aggregate non-core outgoing capacity requires an
    active image; conservation gives the same bound on incoming flow.
    This preserves feasible integer image sets: any connected set admits
    an acyclic root flow with at most total demand through each node.
    A secondary flow-distance objective only resolves equal image counts;
    it neither removes output edges nor proves an optimal human click tour.
    Solver time limits are computation budgets, not geometry constraints.
    An interrupted valid incumbent is returned with optimality unproven.
    """
    primary_options = _solver_options(primary_time_limit_s, 'primary_time_limit_s')
    secondary_options = _solver_options(secondary_time_limit_s, 'secondary_time_limit_s', True)
    core_set=set(cores)
    assets=sorted(set(ids)-core_set)
    asset_index={n:i for i,n in enumerate(assets)}
    arcs=[pair for a,b in edges for pair in ((a,b),(b,a))]
    offset=len(assets)
    columns=offset+len(arcs)
    incoming={n:[] for n in ids}
    outgoing={n:[] for n in ids}
    rows,cols,values,lower,upper=[],[],[],[],[]
    def row(terms,lo,hi):
        number=len(lower)
        for index,value in terms:
            rows.append(number);cols.append(index);values.append(value)
        lower.append(lo);upper.append(hi)
    demand=len(core_set)-1
    root=min(core_set)
    for index,(a,b) in enumerate(arcs,offset):
        outgoing[a].append(index);incoming[b].append(index)
    for n in ids:
        rhs=demand if n==root else -1 if n in core_set else 0
        row([(i,1) for i in outgoing[n]]+[(i,-1) for i in incoming[n]],rhs,rhs)
        if n not in core_set:
            row([(i,1) for i in outgoing[n]]+[(asset_index[n],-demand)],-np.inf,0)
    def constraints():
        matrix=coo_matrix((values,(rows,cols)),shape=(len(lower),columns)).tocsc()
        return LinearConstraint(matrix,np.asarray(lower),np.asarray(upper))
    high=np.full(columns,float(demand));high[:offset]=1
    bounds=Bounds(np.zeros(columns),high)
    integrality=np.zeros(columns,int);integrality[:offset]=1
    objective=np.zeros(columns);objective[:offset]=1
    started=time.perf_counter()
    result=milp(objective,integrality=integrality,bounds=bounds,constraints=constraints(),options=primary_options)
    info={'solver':'scipy.optimize.milp/HiGHS','status':str(result.message),
          'minimum_movement_images_proven':False,'minimum_movement_image_count':None,
          'candidate_movement_asset_count':len(assets),
          'binary_variable_count':offset,'flow_variable_count':len(arcs),'constraint_count':len(lower),
          'activation_formulation':'aggregate_outgoing_node_capacity',
          'primary_solver_status':str(result.message),'primary_solver_status_code':int(result.status),
          'primary_seconds':time.perf_counter()-started,'primary_time_limit_s':primary_time_limit_s,
          'secondary_time_limit_s':secondary_time_limit_s,'secondary_seconds':0.,
          'secondary_flow_distance_minimum_proven':False,'secondary_solver_status':'not_run'}

    def checked_selection(solution):
        x=getattr(solution,'x',None)
        if x is None:
            return None
        active=np.asarray(x,float)[:offset]
        if len(active)!=offset or not np.all(np.isfinite(active)) or np.any(
                (active < -1e-5) | (active > 1+1e-5) | (np.abs(active-np.rint(active))>1e-5)):
            return None
        selected=core_set|{n for n,i in asset_index.items() if active[i]>.5}
        selected_edges=[(a,b) for a,b in edges if a in selected and b in selected]
        return selected if any(core_set<=group for group in components(selected,selected_edges)) else None

    selected=checked_selection(result) if result.status in (0,1) else None
    if selected is None:
        if result.status==0:
            raise InputError('Free navigation optimizer returned an invalid image graph')
        return None,info
    minimum=len(selected-core_set)
    info['selected_movement_image_count']=minimum
    if result.status!=0:
        info['status']='feasible_not_proven'
        return selected,info
    if abs(float(result.fun)-minimum)>1e-5:
        raise InputError('Free navigation optimizer count disagrees with selected images')
    # Keep this independently checked primary solution if optional tie-breaking
    # is skipped, times out, or fails to return a valid equal-count solution.
    info.update(status='optimal',minimum_movement_images_proven=True,
                minimum_movement_image_count=minimum)
    if secondary_time_limit_s==0 or not arcs:
        info['secondary_solver_status']='disabled' if secondary_time_limit_s==0 else 'no_flow_arcs'
        return selected,info
    row([(i,1) for i in range(offset)],minimum,minimum)
    objective[:offset]=0
    lengths=np.asarray([np.linalg.norm(np.asarray(nodes[a]['position_m'][:2])-nodes[b]['position_m'][:2]) for a,b in arcs])
    objective[offset:]=lengths/max(float(lengths.max()),np.finfo(float).tiny)
    started=time.perf_counter()
    secondary=milp(objective,integrality=integrality,bounds=bounds,constraints=constraints(),options=secondary_options)
    alternative=checked_selection(secondary) if secondary.status==0 else None
    valid=alternative is not None and len(alternative-core_set)==minimum
    if valid:
        selected=alternative
    info.update(secondary_seconds=time.perf_counter()-started,
                secondary_flow_distance_minimum_proven=bool(valid),
                secondary_solver_status=str(secondary.message))
    return selected,info


def create_free_navigation(plan,scene,*,primary_time_limit_s=DEFAULT_PRIMARY_TIME_LIMIT_S,
                           secondary_time_limit_s=DEFAULT_SECONDARY_TIME_LIMIT_S):
    from .navigation import NAVIGATION_IMPLEMENTATION
    _solver_options(primary_time_limit_s, 'primary_time_limit_s')
    _solver_options(secondary_time_limit_s, 'secondary_time_limit_s', True)
    if plan.get('schema_name')!='spacewatch3d.virtual_tour_plan' or plan.get('schema_version')!='0.1.0':
        raise InputError('Expected a Task5b tour plan')
    for key in ('scene_id','capture_id','reference_frame_id'):
        if plan.get(key)!=getattr(scene,key):
            raise InputError('Navigation '+key+' mismatch')
    if plan['provenance']['geometry_sha256']!=scene.provenance['geometry_sha256']:
        raise InputError('Navigation geometry mismatch')
    config=Config.from_dict(plan['provenance']['config'])
    obstacles=footprint(scene,config)
    nodes={n['node_id']:n for n in plan['route_graph']['nodes']}
    source={v['viewpoint_id']:copy.deepcopy(v) for v in plan['viewpoints']}
    for v in source.values():
        pose=np.asarray(v['reference_from_camera'],float)
        if pose.shape!=(4,4) or not np.allclose(pose[:3,3],nodes[v['route_node_id']]['position_m'],rtol=0,atol=1e-8):
            raise InputError('Navigation source image pose disagrees with route position')
    cores={vid:v for vid,v in source.items() if 'content' in v['roles']}
    core_nodes={v['route_node_id'] for v in cores.values()}
    by_node={v['route_node_id']:vid for vid,v in source.items()}
    # Content takes priority if a legacy source also lists an image at its node.
    by_node.update({v['route_node_id']:vid for vid,v in cores.items()})
    excluded=set()
    for record in plan.get('excluded_viewpoints',[]):
        node=record.get('route_node_id')
        if node is None and record['viewpoint_id'] in source:
            node=source[record['viewpoint_id']]['route_node_id']
        if node is None and record['viewpoint_id'].startswith('vp_'):
            node=record['viewpoint_id'][3:]
        if node not in nodes:
            raise InputError('Cannot locate an excluded image in this legacy plan; regenerate the source plan')
        excluded.add(node)
    ids=sorted(nodes)
    points=np.asarray([nodes[n]['position_m'][:2] for n in ids],float).reshape((-1,2))
    valid=safe_segments(scene.floor,obstacles,np.repeat(points[:,None,:],2,axis=1),config.clearance_m)
    eligible=[n for n,ok in zip(ids,valid) if ok and n not in excluded]
    failures=[]
    if not cores:
        failures.append({'code':'no_content_viewpoints'})
    unavailable=sorted(core_nodes-set(eligible))
    if unavailable:
        failures.append({'code':'frozen_core_unavailable','route_node_ids':unavailable})
    core_edges=visibility_edges(sorted(core_nodes),nodes,scene.floor,obstacles,config.clearance_m)
    zero=bool(core_nodes) and any(core_nodes<=group for group in components(core_nodes,core_edges))
    selected=set(core_nodes)
    optimizer={'solver':'zero_movement_connectivity_certificate','status':'optimal' if zero and not failures else 'infeasible',
               'minimum_movement_images_proven':zero and not failures,'minimum_movement_image_count':0 if zero else None}
    graph_edges=core_edges
    if not failures and not zero:
        graph_edges=visibility_edges(eligible,nodes,scene.floor,obstacles,config.clearance_m)
        group=next((group for group in components(eligible,graph_edges) if core_nodes<=group),None)
        if group is None:
            failures.append({'code':'core_connectivity_unavailable','core_route_node_ids':sorted(core_nodes)})
        else:
            chosen,optimizer=minimum_vertices(sorted(group),[(a,b) for a,b in graph_edges if a in group and b in group],core_nodes,nodes,
                                              primary_time_limit_s=primary_time_limit_s,secondary_time_limit_s=secondary_time_limit_s)
            if chosen is None:
                failures.append({'code':'navigation_constraints_unsolved','optimizer':optimizer})
            else:
                selected=chosen
                if not optimizer['minimum_movement_images_proven']:
                    failures.append({'code':'navigation_minimum_unproven',
                                     'detail':'A connected safe incumbent is available, but minimum image count is not proven.'})
    views=dict(cores)
    for node in sorted(selected-core_nodes):
        if node in by_node:
            vid=by_node[node]
            views[vid]=source[vid]
        else:
            vid='vp_'+node
            if vid in source or vid in views:
                raise InputError('Navigation viewpoint ID collision')
            pose=np.eye(4);pose[:3,:3]=[[1,0,0],[0,0,1],[0,-1,0]]
            pose[:3,3]=nodes[node]['position_m']
            views[vid]={'viewpoint_id':vid,'route_node_id':node,'component_id':nodes[node]['component_id'],
                        'reference_from_camera':pose.tolist(),'roles':['transition'],
                        'reason':'minimum_free_navigation_connector' if optimizer['minimum_movement_images_proven'] else 'feasible_free_navigation_connector',
                        'image_ref':None,'render_status':'pending','score_basis':'navigation_graph_geometry_only'}
            by_node[node]=vid
    # Preserve all safe edges among final assets, not only the flow certificate.
    # This intentionally permits branches and cycles without extra image cost.
    ordered=sorted(views)
    final_points={vid:np.asarray(views[vid]['reference_from_camera'])[:3,3] for vid in ordered}
    pairs=list(combinations(ordered,2))
    safe=safe_segments(scene.floor,obstacles,[[final_points[a][:2],final_points[b][:2]] for a,b in pairs],config.clearance_m)
    adjacency={vid:[] for vid in ordered}
    links=[]
    for (a,b),ok in zip(pairs,safe):
        if not ok or views[a]['route_node_id'] in unavailable or views[b]['route_node_id'] in unavailable:
            continue
        pa,pb=final_points[a],final_points[b]
        delta=pb[:2]-pa[:2]
        links.append({'link_id':'link_'+str(len(links)),'from_viewpoint':a,'to_viewpoint':b,'bidirectional':True,
                      'safe_chord_m':[pa.tolist(),pb.tolist()],'length_m':float(np.linalg.norm(delta)),
                      'geometry_safe':True,'world_heading_rad':float(np.arctan2(delta[1],delta[0])),
                      'image_overlap_validated':False})
        adjacency[a].append(b);adjacency[b].append(a)
    groups=components(ordered,[(l['from_viewpoint'],l['to_viewpoint']) for l in links])
    connected=bool(cores) and any(set(cores)<=group for group in groups)
    if not connected and not failures:
        failures.append({'code':'output_core_connectivity_failed'})
    movement=[v for vid,v in views.items() if vid not in cores]
    new=[v for v in movement if v['viewpoint_id'] not in source]
    requests=[{'viewpoint_id':v['viewpoint_id'],'capture_id':scene.capture_id,'reference_frame_id':scene.reference_frame_id,
               'reference_from_camera':v['reference_from_camera'],'projection':'equirectangular','width':config.width,'height':config.height,
               'media_origin':'novel_view','render_required':True,'purpose':'navigation'} for v in movement if not v.get('image_ref')]
    return {'schema_name':'spacewatch3d.tour_navigation','schema_version':'0.3.0','implementation':NAVIGATION_IMPLEMENTATION,
            'navigation_mode':'free_exploration_graph','scene_id':scene.scene_id,'capture_id':scene.capture_id,
            'reference_frame_id':scene.reference_frame_id,'source_route_id':plan['route_id'],
            'source_core_implementation':plan['provenance'].get('implementation','unreported'),
            'source_planning_status':plan['planning_status'],'source_input_coverage_status':plan.get('input_coverage_status','unreported'),
            'source_failures':copy.deepcopy(plan.get('failures',[])),
            'navigation_status':'incomplete' if failures or plan['planning_status']!='complete' else 'complete',
            'navigation_status_scope':'geometry_safe_core_connectivity_and_minimum_image_certification_with_source_planning_status',
            'image_readability_validated':False,'publication_ready':False,'viewpoints':list(views.values()),'links':links,
            'adjacency':[{'viewpoint_id':vid,'neighbors':sorted(neighbors)} for vid,neighbors in adjacency.items()],
            'render_requests':requests,'failures':failures,'optimizer':optimizer,
            'report':{'source_image_count':len(source),'frozen_content_count':len(cores),'movement_image_count':len(movement),
                      'added_image_count':len(new),'total_image_count':len(views),'link_count':len(links),'connected_components':len(groups),
                      'cycle_rank':len(links)-len(views)+len(groups),'all_core_reachable':connected,
                      'released_source_transition_count':sum(vid not in cores and vid not in views for vid in source),
                      'reused_source_transition_count':len(movement)-len(new),'forced_distance_spacing_m':None,
                      'source_tour_order_required':False,'all_safe_selected_links_exposed':True,
                      'candidate_route_node_count':len(eligible),'visibility_graph_built':not zero and not unavailable and bool(cores),
                      'minimum_movement_images_proven':optimizer['minimum_movement_images_proven'] and not failures,
                      'minimum_scope':'eligible_published_route_nodes_and_all_geometry_safe_straight_chords',
                      'method':'node_weighted_steiner_connectivity_root_flow_milp',
                      'continuous_space_or_human_comfort_optimum_guaranteed':False,'source_content_changed':False,'actual_media_bytes':None},
            'warnings':['Free exploration replaces ordered-path and per-crossing gateway obligations; old tour order is not a goal.',
                        'All safe links between selected images are exposed; a UI may declutter arrows without changing the asset graph.',
                        'Unused skeleton branches and every alternative physical corridor loop are not coverage obligations.',
                        'Solver time budgets do not limit geometry. A safe incumbent without minimum-count proof remains incomplete; distance tie-breaking is optional.',
                        'Geometry safety is not RGB overlap, comfortable displacement, readability or publication approval.',
                        'Core scores, coverage targets and source planning/input failures are not repaired or rescored.']}

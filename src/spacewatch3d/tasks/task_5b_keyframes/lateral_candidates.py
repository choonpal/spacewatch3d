"""Optional coverage-preserving off-center proposals on existing safe raster.

The centerline remains the backbone. Only useful observation endpoints and
their verified raster paths become graph spurs. Probes get NO object credit;
selected endpoints must subsequently receive real object scores in the caller.
"""
from copy import copy, deepcopy
import heapq
import hashlib
import numpy as np
from scipy.sparse import csr_matrix

from .models import InputError
from .routing import neighbors
from .selection import AttainableGoals
from .coverage_repair import JointObjective


def propose_lateral_selection(route, objectives, selected, candidate_nodes, config,
                              evaluate, required=(), banned=()):
    """Pure proposal, never mutate input route/models or use final holdouts.

evaluate(draw_index, positions) returns only original surface visibility.
Original observable masks, object thresholds and impossible goals stay frozen.
"""
    report={'status':'disabled','original_count':len(selected),'selected_count':len(selected),
            'object_probe_credit':'none','uses_final_holdout':False,'probe_count':0}
    if not config.lateral_candidate_refinement or not selected:
        return report
    if route.raster_edges is None or route.raster_labels is None:
        return {**report,'status':'missing_verified_raster_topology'}
    joint=JointObjective(objectives);goals=AttainableGoals(joint)
    if not goals.met(joint.measure(selected)[1]):
        return {**report,'status':'original_attainable_goals_not_met'}
    known={route.cells[n] for n in candidate_nodes}
    excluded={route.cells[n] for n in banned}
    components={route.components[candidate_nodes[i]] for i in selected}
    forbidden_original=[i for i,node in enumerate(candidate_nodes) if route.components[node] not in components]
    cells=[tuple(map(int,c)) for c in np.argwhere(route.walkable)
           if tuple(c) not in known and tuple(c) not in excluded
           and int(route.raster_labels[tuple(c)]) in components]
    report['probe_count']=len(cells)
    if len(cells)>config.max_visibility_probes:
        return {**report,'status':'probe_budget_deferred'}
    if not cells:
        return {**report,'status':'no_new_safe_positions'}
    height=float(route.positions[candidate_nodes[selected[0]],2])
    positions=np.array([[route.origin[0]+(c+.5)*route.grid_m,
                         route.origin[1]+(r+.5)*route.grid_m,height] for r,c in cells])
    # Do not displace an existing exact canonical pose sharing this raster cell.
    aligned={cell for i,cell in enumerate(route.cells) if np.allclose(route.positions[i,:2],
        route.origin+(np.array(cell[::-1])+.5)*route.grid_m,rtol=0,atol=1e-8)}
    reserved=set(route.cells)-aligned
    keep=[i for i,cell in enumerate(cells) if cell not in reserved]
    cells=[cells[i] for i in keep];positions=positions[keep]
    if not cells:
        return {**report,'status':'no_new_safe_positions'}
    report['probe_count']=len(cells)
    original_best=[m.quality[selected].max(axis=0) for m in objectives]
    protected=[(best>=m.object_thresholds) | np.array([o['must_see'] for o in m.objects],bool)
               for m,best in zip(objectives,original_best)]
    anchors=list(selected);required=set(required)
    def object_safe(rows):
        for m,best,mask in zip(objectives,original_best,protected):
            current=m.quality[rows].max(axis=0) if rows else np.zeros(len(m.objects))
            target=np.minimum(best,m.object_thresholds)
            if np.any(current[mask]<target[mask]):return False
        return True
    for row in reversed(list(anchors)):
        trial=[i for i in anchors if i!=row]
        if row not in required and object_safe(trial):anchors=trial
    models=[];matrices=[];regional=[]
    for draw,m in enumerate(objectives):
        measured=np.asarray(evaluate(draw,positions),bool)
        if measured.shape!=(len(cells),len(m.weights)):
            raise InputError('Lateral visibility shape differs from original surface targets')
        model=copy(m)
        model.visible=np.concatenate([m.visible,measured])
        model.quality=np.concatenate([m.quality,np.zeros((len(cells),len(m.objects)))])
        # Do NOT redefine observable/possible_objects from the probe union.
        models.append(model);matrices.append(csr_matrix(model.visible,dtype=float))
        _,inverse=np.unique(m.regions,return_inverse=True)
        possible=np.bincount(inverse,weights=m.weights*m.observable)
        norm=np.divide(m.weights,possible[inverse],out=np.zeros_like(m.weights),where=possible[inverse]>0)*m.observable
        regional.append((inverse,possible,norm,int(np.count_nonzero(possible))))
    expanded=JointObjective(models)
    chosen=list(anchors);covered=[np.any(m.visible[chosen],axis=0) if chosen else np.zeros(len(m.weights),bool) for m in models]
    while not goals.met(expanded.measure(chosen)[1]):
        gain=np.zeros(len(models[0].visible))
        for i,(m,matrix,(inv,area,norm,nregions)) in enumerate(zip(models,matrices,regional)):
            ratios=np.bincount(inv,weights=norm*covered[i],minlength=len(area))
            short=(area>0)&(ratios+1e-10<config.region_target)
            coefficient=(norm/nregions if nregions and ratios[area>0].mean()+1e-10<config.space_target else np.zeros_like(norm))+norm*short[inv]
            gain+=matrix@((~covered[i])*coefficient)
        gain[chosen]=-np.inf
        gain[forbidden_original]=-np.inf
        row=int(np.argmax(gain))
        if gain[row]<=0:return {**report,'status':'no_safe_completion'}
        chosen.append(row)
        for i,m in enumerate(models):covered[i]|=m.visible[row]
    for row in reversed(list(chosen)):
        if row in anchors:continue
        trial=[i for i in chosen if i!=row]
        if goals.met(expanded.measure(trial)[1]):chosen=trial
    report.update(object_anchor_count=len(anchors),selected_count=len(chosen))
    if len(chosen)>=len(selected):
        return {**report,'status':'no_reduction'}
    offset=len(candidate_nodes)
    report.update(status='proposed',selected_original_indices=[i for i in chosen if i<offset],
                  selected_extra_cells=[cells[i-offset] for i in chosen if i>=offset],
                  original_goals_preserved=True,
                  selected_extra_positions_m=[positions[i-offset].tolist() for i in chosen if i>=offset])
    return report


def attach_lateral_cells(route, cells, config):
    """Copy route; attach endpoints by shortest previously validated raster paths."""
    if route.raster_edges is None or route.raster_labels is None:
        raise InputError('Lateral attachment requires verified raster topology')
    result=copy(route);result.cells=list(route.cells);result.positions=route.positions.copy()
    result.components=list(route.components);result.adjacency={n:list(edges) for n,edges in route.adjacency.items()}
    result.candidates=list(route.candidates);result.candidate_reasons=deepcopy(route.candidate_reasons)
    result.candidate_sampling=deepcopy(route.candidate_sampling)
    result.node_ids=dict(route.node_ids);result.viewpoint_ids=dict(route.viewpoint_ids);result.camera_poses=dict(route.camera_poses)
    lookup={cell:i for i,cell in enumerate(route.cells)}
    seeds={cell for cell,i in lookup.items() if np.allclose(route.positions[i,:2],
        route.origin+(np.array(cell[::-1])+.5)*route.grid_m,rtol=0,atol=1e-8)}
    reserved=set(lookup)-seeds
    if not seeds:raise InputError('No grid-aligned backbone available for verified lateral attachment')
    distance={cell:0. for cell in seeds};previous={};queue=[(0.,cell) for cell in sorted(seeds)]
    heapq.heapify(queue);targets=set(cells)-seeds
    while queue and targets:
        cost,cell=heapq.heappop(queue)
        if cost!=distance[cell]:continue
        targets.discard(cell)
        for other,step in neighbors(cell,route.walkable,route.raster_edges):
            if other in reserved:continue
            trial=cost+step*route.grid_m
            if trial<distance.get(other,float('inf'))-1e-12:
                distance[other]=trial;previous[other]=cell;heapq.heappush(queue,(trial,other))
    if targets:raise InputError('No verified raster path for lateral observation pose')
    marked=set(lookup)
    for endpoint in cells:
        cell=endpoint
        while cell not in marked:
            marked.add(cell);cell=previous[cell]
    extra=sorted(marked-set(lookup))
    if len(route.positions)+len(extra)>config.max_route_nodes:
        raise InputError('max_route_nodes exceeded by optional observation spurs')
    for cell in extra:lookup[cell]=len(result.cells);result.cells.append(cell)
    height=float(route.positions[lookup[min(seeds)],2])
    if extra:
        result.positions=np.concatenate([route.positions,np.array([[route.origin[0]+(c+.5)*route.grid_m,
            route.origin[1]+(r+.5)*route.grid_m,height] for r,c in extra])])
        result.components.extend(int(route.raster_labels[cell]) for cell in extra)
    if route.node_ids:
        reserved_ids={route.node_id(i) for i in range(len(route.positions))}
        for cell in extra:
            node=lookup[cell]
            key=hashlib.sha256(np.asarray(result.positions[node],dtype='<f8').tobytes()).hexdigest()[:24]
            nid='n_lateral_'+key
            if nid in reserved_ids:raise InputError('New lateral/reference node identity collision')
            result.node_ids[node]=nid;reserved_ids.add(nid)
    for cell in extra:result.adjacency[lookup[cell]]=[]
    # Only parent-tree edges are needed. Keep all original centerline edges.
    for cell in extra:
        parent=previous[cell];a,b=lookup[cell],lookup[parent]
        length=float(np.linalg.norm(result.positions[a]-result.positions[b]))
        result.adjacency.setdefault(a,[]).append((b,length));result.adjacency[b].append((a,length))
    result.candidate_sampling['lateral_backbone_node_count']=len(route.positions)
    result.candidate_sampling['lateral_spur_node_count']=len(extra)
    return result,[lookup[cell] for cell in cells]


def refine_lateral_plan(route, datasets, objectives, selected, candidate_nodes, objects,
                        config, required=(), banned=()):
    """Validate a proposal with actual object scores and expanded goal masks.

Only endpoints proposed for use are added/scored. A rejected proposal returns
the original inputs, including their original observable denominators.
"""
    from .visibility import evaluate_positions
    from .routing import add_candidates, connect_tour
    from .coverage_repair import make_objectives
    from .selection_guard import guard_objectives
    def spatial(draw,positions):
        return evaluate_positions(positions,datasets[draw]['surface'],[],config)[0]
    proposal=propose_lateral_selection(route,objectives,selected,candidate_nodes,config,
                                      spatial,required,banned)
    if proposal['status']!='proposed':return route,datasets,selected,proposal
    try:
        updated,endpoints=attach_lateral_cells(route,proposal['selected_extra_cells'],config)
        if len(updated.candidates)+len(endpoints)>config.max_candidates:
            return route,datasets,selected,{**proposal,'status':'candidate_budget_deferred'}
        added=add_candidates(updated,endpoints,config.grid_m,'lateral_observation',config,force=True)
    except InputError as exc:
        return route,datasets,selected,{**proposal,'status':'attachment_deferred','reason':str(exc)}
    scored=[]
    for dataset in datasets:
        item=dict(dataset)
        visible,quality=evaluate_positions(updated.positions[added],dataset['surface'],objects,config)
        if dataset.get('quality_refiner') is not None:
            quality=dataset['quality_refiner'].evaluate(updated.positions[added],objects)
        item['visible']=np.concatenate([dataset['visible'],visible])
        item['quality']=np.concatenate([dataset['quality'],quality]);scored.append(item)
    allowed=[i for i,n in enumerate(updated.candidates) if n not in set(banned)]
    candidates=[updated.candidates[i] for i in allowed];lookup={node:i for i,node in enumerate(candidates)}
    chosen=proposal['selected_original_indices']+[lookup[n] for n in endpoints]
    declared=objectives[0].declared_regions
    models=guard_objectives(make_objectives(scored,allowed,objects,config,declared))
    objective=JointObjective(models);goals=AttainableGoals(objective)
    actual=objective.measure(chosen)[1]
    original_nodes=[candidate_nodes[i] for i in selected]
    new_nodes=[candidates[i] for i in chosen]
    _,old_links,old_transitions,old_fail=connect_tour(route,original_nodes,config,set(banned))
    _,new_links,new_transitions,new_fail=connect_tour(updated,new_nodes,config,set(banned))
    old_cost=len(set(original_nodes)|set(old_transitions));new_cost=len(set(new_nodes)|set(new_transitions))
    preserved=(goals.met(actual) and set(f['code'] for f in new_fail)<=set(f['code'] for f in old_fail)
               and new_cost<old_cost
               and {updated.components[n] for n in new_nodes} <= {route.components[n] for n in original_nodes})
    if not preserved:
        return route,datasets,selected,{**proposal,'status':'rejected_after_full_observation_check',
            'expanded_goals_met':goals.met(actual),'original_routed_image_count':old_cost,
            'proposed_routed_image_count':new_cost}
    proposal.update(status='accepted',actual_object_scores_checked=True,expanded_goals_met=True,
        original_routed_image_count=old_cost,refined_routed_image_count=new_cost,
        added_candidate_ids=[updated.candidate_id(n) for n in added],
        added_candidate_count=len(added),backbone_node_count=len(route.positions),
        spur_node_count=len(updated.positions)-len(route.positions),
        original_collisions_and_observation_surfaces_unchanged=True,
        newly_discovered_target_denominators_preserved=True)
    updated.candidate_sampling['lateral_added_count']=len(added)
    return updated,scored,chosen,proposal

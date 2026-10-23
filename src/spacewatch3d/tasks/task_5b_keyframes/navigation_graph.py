"""Minimum transit assets on frozen paths; structural gateway witnesses only.

Adaptation of decision-point navigation, not an optimal human-wayfinding claim.
Unused raw skeleton branches are not patrol goals. No meter/angle thresholds.
"""
from collections import defaultdict
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from .models import InputError
from .routing import safe_segments


def core_paths(plan, views, cores):
    """Release old transition stops, preserving their complete route paths."""
    result, failures, pending = [], [], None
    for index, link in enumerate(plan['tour_links']):
        ids = list(link['route_node_ids'])
        a, b = link['from_viewpoint'], link['to_viewpoint']
        if not ids or views[a]['route_node_id'] != ids[0] or views[b]['route_node_id'] != ids[-1]:
            raise InputError('Navigation path endpoints disagree with source images')
        if pending is not None and pending['to_viewpoint'] != a:
            failures.append({'code':'interrupted_transition_chain','source_link_indices':pending['source_link_indices']})
            pending = None
        if pending is None:
            if a not in cores:
                failures.append({'code':'unanchored_transition_chain','source_link_indices':[index]})
                continue
            pending = {'from_viewpoint':a,'to_viewpoint':b,'route_node_ids':ids,'source_link_indices':[index]}
        else:
            pending['route_node_ids'].extend(ids[1:])
            pending['source_link_indices'].append(index)
            pending['to_viewpoint'] = b
        if b in cores:
            if len(pending['route_node_ids']) >= 2:
                result.append(pending)
            pending = None
    if pending is not None:
        failures.append({'code':'unfinished_transition_chain','source_link_indices':pending['source_link_indices']})
    return result, failures


def decision_events(paths, nodes, core_nodes, floor, obstacles, clearance):
    """Group adjacent junctions, trace degree-two chains to visible gateways."""
    adjacency = defaultdict(set)
    for path in paths:
        for a,b in zip(path['route_node_ids'],path['route_node_ids'][1:]):
            if a != b:
                adjacency[a].add(b)
                adjacency[b].add(a)
    remaining = {n for n,neighbors in adjacency.items() if len(neighbors) >= 3}
    events = []
    while remaining:
        cluster,stack = set(),[min(remaining)]
        while stack:
            node = stack.pop()
            if node not in remaining:
                continue
            remaining.remove(node)
            cluster.add(node)
            stack.extend(adjacency[node] & remaining)
        portals = set()
        for node in sorted(cluster):
            for first in sorted(adjacency[node]-cluster):
                chain,previous,current,seen = [],node,first,{node}
                while current not in seen:
                    chain.append(current)
                    seen.add(current)
                    if current in core_nodes or len(adjacency[current]) != 2:
                        break
                    previous,current = current,next(iter(adjacency[current]-{previous}))
                point = np.asarray(nodes[node]['position_m'][:2])
                targets = np.asarray([nodes[n]['position_m'][:2] for n in chain])
                lines = np.stack([np.repeat(point[None,:],len(chain),axis=0),targets],axis=1)
                visible = safe_segments(floor,obstacles,lines,clearance)
                # Do not skip an occluded bend to a later reappearance.
                prefix = int(np.argmax(~visible)) if not np.all(visible) else len(chain)
                # An unavailable geometric obligation remains explicit.
                portals.add(chain[prefix-1] if prefix else first)
        events.append({'event_id':'decision_'+str(len(events)), 'kind':'routed_junction',
                       'junction_node_ids':sorted(cluster),'gateway_node_ids':sorted(portals)})
    return events, adjacency


def prepare_paths(paths, events, nodes, eligible, floor, obstacles, clearance):
    prepared,failures = [],[]
    for source_index,path in enumerate(paths):
        ids = path['route_node_ids']
        points = np.asarray([nodes[n]['position_m'] for n in ids],float)
        allowed = np.asarray([n in eligible for n in ids],bool)
        applicable = [e for e in events if set(ids) & set(e['junction_node_ids'])]
        requirements = []
        for event in applicable:
            access = np.zeros(len(ids),bool)
            for node in event['junction_node_ids']:
                target = np.asarray(nodes[node]['position_m'][:2])
                lines = np.stack([points[:,:2],np.repeat(target[None,:],len(ids),axis=0)],axis=1)
                access |= safe_segments(floor,obstacles,lines,clearance)
            for portal in event['gateway_node_ids']:
                target = np.asarray(nodes[portal]['position_m'][:2])
                lines = np.stack([points[:,:2],np.repeat(target[None,:],len(ids),axis=0)],axis=1)
                witnesses = np.flatnonzero(access & allowed & safe_segments(floor,obstacles,lines,clearance)).tolist()
                requirements.append({'event_id':event['event_id'],'gateway_node_id':portal,'witness_indices':witnesses})
        if not allowed[0] or not allowed[-1] or any(not r['witness_indices'] for r in requirements):
            failures.append({'code':'navigation_decision_unavailable','source_path_index':source_index,
                             'source_link_indices':path['source_link_indices']})
            continue
        direct = bool(safe_segments(floor,obstacles,[[points[0,:2],points[-1,:2]]],clearance)[0])
        witnessed = all(0 in r['witness_indices'] or len(ids)-1 in r['witness_indices'] for r in requirements)
        arcs = []
        if direct and witnessed:
            # Zero movement cost and one hop is already globally optimal here.
            arcs = [(0,len(ids)-1)]
        else:
            reached = {0}
            for i in range(len(ids)-1):
                if i not in reached or not allowed[i]:
                    continue
                targets = np.flatnonzero(allowed[i+1:])+i+1
                lines = np.stack([np.repeat(points[i:i+1,:2],len(targets),axis=0),points[targets,:2]],axis=1)
                for j in targets[safe_segments(floor,obstacles,lines,clearance)]:
                    arcs.append((i,int(j)))
                    reached.add(int(j))
            if len(ids)-1 not in reached:
                failures.append({'code':'navigation_image_gap','source_path_index':source_index,
                                 'source_link_indices':path['source_link_indices']})
                continue
        prepared.append({**path,'source_path_index':source_index,'points':points,'requirements':requirements,'arcs':arcs})
    return prepared,failures


def solve_minimum(paths, core_nodes):
    """Binary shared image activation + ordered unit flows + gateway witnesses.

    Lexicographic objective: unique movement images, then traversal hops.
    Optimality applies ONLY to the given paths/candidates/events/safe chords.
    """
    if not paths:
        return [],{'status':'optimal','minimum_movement_images_proven':True,
                   'minimum_movement_image_count':0,'traversal_hops':0,'solver':'scipy.optimize.milp/HiGHS'}
    assets = sorted({p['route_node_ids'][j] for p in paths for _,j in p['arcs']}-core_nodes)
    asset_index = {n:i for i,n in enumerate(assets)}
    edge_index,offset = [],len(assets)
    for path in paths:
        index = {edge:offset+i for i,edge in enumerate(path['arcs'])}
        edge_index.append(index)
        offset += len(index)
    rows,cols,values,low,high = [],[],[],[],[]
    def row(terms,lo,hi):
        number = len(low)
        for column,value in terms.items():
            rows.append(number)
            cols.append(column)
            values.append(value)
        low.append(lo)
        high.append(hi)
    for path,index in zip(paths,edge_index):
        incoming,outgoing = defaultdict(list),defaultdict(list)
        for (i,j),column in index.items():
            incoming[j].append(column)
            outgoing[i].append(column)
            target = path['route_node_ids'][j]
            if target not in core_nodes:
                row({column:1,asset_index[target]:-1},-np.inf,0)
        for i in range(len(path['route_node_ids'])):
            terms = {column:1 for column in outgoing[i]}
            terms.update({column:-1 for column in incoming[i]})
            rhs = 1 if i == 0 else -1 if i == len(path['route_node_ids'])-1 else 0
            row(terms,rhs,rhs)
        for requirement in path['requirements']:
            if 0 not in requirement['witness_indices']:
                row({column:1 for j in requirement['witness_indices'] for column in incoming[j]},1,np.inf)
    base_rows = len(low)
    def constraints():
        matrix = coo_matrix((values,(rows,cols)),shape=(len(low),offset)).tocsc()
        return LinearConstraint(matrix,np.asarray(low),np.asarray(high))
    objective = np.zeros(offset)
    objective[:len(assets)] = 1
    result = milp(objective,integrality=np.ones(offset),bounds=Bounds(0,1),constraints=constraints(),
                  options={'mip_rel_gap':0.})
    info = {'solver':'scipy.optimize.milp/HiGHS','status':str(result.message),
            'minimum_movement_images_proven':result.status == 0,'candidate_movement_asset_count':len(assets),
            'binary_variable_count':offset,'constraint_count':base_rows}
    if result.status != 0:
        return None,info
    optimum = int(round(result.fun))
    row({i:1 for i in range(len(assets))},optimum,optimum)
    objective[:len(assets)] = 0
    objective[len(assets):] = 1
    secondary = milp(objective,integrality=np.ones(offset),bounds=Bounds(0,1),constraints=constraints(),
                     options={'mip_rel_gap':0.})
    chosen = secondary if secondary.status == 0 else result
    info.update(status='optimal',minimum_movement_image_count=optimum,hop_minimum_proven=secondary.status == 0,
                secondary_solver_status=str(secondary.message))
    selected = []
    for path,index in zip(paths,edge_index):
        next_index = {i:j for (i,j),column in index.items() if chosen.x[column] > .5}
        visits,current = [0],0
        while current != len(path['route_node_ids'])-1:
            if current not in next_index or next_index[current] <= current:
                raise InputError('Navigation solver returned an invalid ordered path')
            current = next_index[current]
            visits.append(current)
        if any(not set(visits).intersection(r['witness_indices']) for r in path['requirements']):
            raise InputError('Navigation solver omitted a required decision witness')
        selected.append(visits)
    actual = {p['route_node_ids'][i] for p,visits in zip(paths,selected) for i in visits}-core_nodes
    if len(actual) != optimum:
        raise InputError('Navigation asset minimum disagrees with selected path assets')
    info['traversal_hops'] = sum(len(visits)-1 for visits in selected)
    return selected,info

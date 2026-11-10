"""Exact routed-selection summary and immutable objective metadata caching.

Only the planner's internal candidate-set cost evaluation uses summary links.
Final tours still use the original full geometry-producing connector.
"""
import math
from types import MappingProxyType
import numpy as np
from concurrent.futures import ThreadPoolExecutor

POOL=ThreadPoolExecutor(max_workers=4,thread_name_prefix='visibility')


def save_validation_draft(plan,requests):
    """Durable unvalidated receipt; never masquerades as a successful plan."""
    from pathlib import Path
    import json
    folder=Path(__file__).resolve().parent/'validation_draft'
    folder.mkdir(exist_ok=False)
    for name,value in [('UNVALIDATED_plan.json',plan),('UNVALIDATED_requests.json',requests)]:
        (folder/name).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False))


def parallel_visibility(original,positions,surface,objects,config):
    """Independent pose rows, same operations and stable row order, four workers."""
    positions=np.asarray(positions)
    if len(positions)<32:return original(positions,surface,objects,config)
    import native_executor
    bvhs=[surface.bvh]+list((surface.object_bvhs or {}).values())
    # Populate shared immutable BVH buffers on the caller thread, not concurrently.
    for bvh in bvhs:
        if bvh.root[2] is None and bvh not in native_executor.PACKED:
            native_executor.PACKED[bvh]=native_executor.packed(bvh)
    visible=np.empty((len(positions),len(surface.points)),bool)
    quality=np.empty((len(positions),len(objects)),float)
    spans=[(i,min(i+64,len(positions))) for i in range(0,len(positions),64)]
    def one(span):
        a,b=span
        v,q=original(positions[a:b],surface,objects,config)
        return a,b,v,q
    for a,b,v,q in POOL.map(one,spans):visible[a:b]=v;quality[a:b]=q
    return visible,quality


def install_parallel_visibility():
    from spacewatch3d.tasks.task_5b_keyframes import visibility
    import sys
    original=visibility.evaluate_positions
    def evaluate(positions,surface,objects,config):
        return parallel_visibility(original,positions,surface,objects,config)
    evaluate.__wrapped__=original
    visibility.evaluate_positions=evaluate
    for name in ['planner','coverage_repair','coverage_validation','lateral_candidates']:
        m=sys.modules.get('spacewatch3d.tasks.task_5b_keyframes.'+name)
        if m is not None and getattr(m,'evaluate_positions',None) is original:m.evaluate_positions=evaluate


def connect_tour_summary(route,selected,config,banned,start=None,paths_from=None,hop_paths_from=None):
    from spacewatch3d.tasks.task_5b_keyframes.routing import connect_tour,shortest,restore_path
    if config.max_link_m is not None:
        return connect_tour(route,selected,config,banned,start,paths_from,hop_paths_from)
    signature=(id(route.positions),id(route.adjacency),len(route.positions))
    if getattr(route,'_fast_summary_signature',None)!=signature:
        route._fast_summary_signature=signature
        route._fast_pair_lengths={}
        route._fast_source_paths={}
    pair_lengths=route._fast_pair_lengths
    source_paths=route._fast_source_paths
    visits=[];links=[];components=sorted({route.components[n] for n in selected})
    for component in components:
        remaining={n for n in selected if route.components[n]==component}
        current=min(remaining) if start is None else min(remaining,key=lambda n:(float(np.linalg.norm(route.positions[n,:2]-start)),n))
        visits.append(current);remaining.remove(current)
        while remaining:
            if current not in source_paths:
                source_paths[current]=paths_from(current) if paths_from else shortest(route.adjacency,current)
            distances,previous=source_paths[current]
            target=min(remaining,key=lambda n:(distances.get(n,math.inf),n))
            key=(current,target)
            if key not in pair_lengths:
                path=restore_path(previous,current,target)
                length=0.0
                # Keep the reference operation and summation order exactly.
                for a,b in zip(path,path[1:]):length+=float(np.linalg.norm(route.positions[a]-route.positions[b]))
                pair_lengths[key]=length
            links.append({'length_m':pair_lengths[key]})
            visits.append(target);current=target;remaining.remove(target)
    failures=[{'code':'disconnected_tour','message':'Separate tours required; no teleport links created'}] if len(components)>1 else []
    return visits,links,[],failures


def install_objective_metadata_cache():
    """Freeze owned static arrays, then avoid repeated full-array byte copies.

New/reassigned arrays invalidate the identity key. In-place mutation is rejected,
not silently ignored. Visibility, quality and thresholds are never memoized here.
"""
    import objective_executor
    original_static=objective_executor.static_data
    original_init=objective_executor.Objective.__init__
    fields=('weights','regions','observable','importance','possible_objects')
    def initialize(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        for name in fields:
            owned=np.array(getattr(self,name),copy=True)
            owned.flags.writeable=False
            setattr(self,name,owned)
    def static(self):
        key=(tuple(self.region_ids),frozenset(self.declared_regions),
             tuple((id(getattr(self,n)),getattr(self,n).shape,getattr(self,n).dtype.str) for n in fields))
        if getattr(self,'_fast_static_identity',None)!=key:
            # New arrays assigned by a caller are frozen on refresh as well.
            for name in fields:
                a=getattr(self,name)
                if a.flags.writeable:
                    a=np.array(a,copy=True);a.flags.writeable=False;setattr(self,name,a)
            result=original_static(self)
            self._fast_static_identity=(tuple(self.region_ids),frozenset(self.declared_regions),
                tuple((id(getattr(self,n)),getattr(self,n).shape,getattr(self,n).dtype.str) for n in fields))
            self._fast_static_result=result
        return self._fast_static_result
    objective_executor.Objective.__init__=initialize
    objective_executor.static_data=static

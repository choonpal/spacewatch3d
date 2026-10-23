"""Explicit fragment ownership; never infer physical instances from class labels."""
import math
import numpy as np
from .models import Geometry,InputError

def group_physical_objects(objects,geometries,provenance,warnings):
    groups={}
    for obj in objects:
        owner=obj.get('physical_object_id',obj['instance_id'])
        if not isinstance(owner,str) or not owner:
            raise InputError('physical_object_id must be a nonempty string when supplied')
        groups.setdefault(owner,[]).append(obj)
    explicit={o['instance_id']:owner for owner,members in groups.items() for o in members
              if owner!=o['instance_id'] or len(members)>1}
    if not explicit:
        return objects,geometries
    targets=[]
    for owner,members in groups.items():
        if len(members)==1 and members[0]['instance_id'] not in explicit:
            targets.append(members[0])
            continue
        names={m['class_name'] for m in members}
        def score(obj):
            v=obj.get('importance')
            if isinstance(v,dict) and not isinstance(v.get('reason',''),str):
                raise InputError('Grouped manual importance reason must be a string')
            v=v.get('score') if isinstance(v,dict) else v
            if v is None:return -1.
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1:
                raise InputError('Grouped manual importance must be in [0,1]')
            return v
        highest=max(members,key=score)
        target={**highest,'instance_id':owner,'class_name':members[0]['class_name'] if len(names)==1 else None,
                'must_see':any(m['must_see'] for m in members),
                'description':'\n'.join(m['description'] for m in members if m['description']),
                'source_member_instance_ids':[m['instance_id'] for m in members],
                'identity_source':'explicit_physical_object_id'}
        target.pop('physical_object_id',None)
        if len(names)>1:
            warnings.append({'code':'grouped_object_class_conflict','instance_id':owner,
                             'message':'Explicit physical ownership retained; conflicting class names remain unknown'})
        targets.append(target)
    # Keep every original collision shell/solid separately. A merged surface
    # must not fill the space between disjoint parts or change solid clipping.
    output=[]
    pieces={}
    for g in geometries:
        if g.object_id not in explicit:
            output.append(g);continue
        owner=explicit[g.object_id]
        pieces.setdefault(owner,[]).append(g.triangles)
        collision=g.collision_triangles if g.collision_triangles is not None else g.triangles
        output.append(Geometry(g.geometry_id+':collision',np.empty((0,3,3)),None,
                               obstacle=g.obstacle,solid=g.solid,collision_triangles=collision))
    for owner,items in pieces.items():
        output.append(Geometry('physical_object:'+owner,np.concatenate(items),owner,obstacle=False,solid=False))
    provenance['physical_object_groups']={owner:[m['instance_id'] for m in members]
        for owner,members in groups.items() if any(m['instance_id'] in explicit for m in members)}
    provenance['physical_object_grouping_policy']='explicit_identity_only_max_manual_priority_any_must_see_original_collision'
    return targets,output

def record_observation_identity(objects,provenance,warnings):
    context=provenance.get('source_context',{})
    unverified=isinstance(context,dict) and context.get('physical_object_identity_verified') is False
    record={'unit':'semantic_region' if unverified else 'input_object',
            'physical_identity_verified':False if unverified else None,
            'observation_unit_count':len(objects),'automatic_class_based_merging':False}
    if provenance.get('physical_object_groups'):
        record['explicit_physical_group_count']=len(provenance['physical_object_groups'])
    provenance['observation_identity']=record
    if unverified:
        warnings.append({'code':'object_identity_unverified','observation_unit_count':len(objects),
            'message':'Producer supplied semantic regions, not verified physical instances. Targets and failures are retained; object counts are not physical inventory counts.'})

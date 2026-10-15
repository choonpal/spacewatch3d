"""Classify actual Rescan instances as moved/added/removed using both captures.

Correspondence uses the dataset's persistent instance annotations. Motion is
estimated by rigid surface registration, not by partial-scan centroid changes.
Missing structural surfaces are not counted as removed objects.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from build_rescan_review import read_cloud, cloud_geometry, PACKED, TO_DISPLAY, digest
from link_rescan_review import project
from recover_rescan_keyframes import render

NAMES = {3:'수납장',4:'침대',5:'의자',6:'소파',7:'테이블',8:'문',9:'창문',10:'책장',11:'액자',12:'카운터',13:'블라인드',14:'책상',15:'선반',16:'커튼',17:'서랍장',18:'베개',19:'거울',20:'매트',21:'옷',23:'책',24:'냉장고',25:'TV',26:'종이',27:'수건',28:'샤워 커튼',29:'상자',30:'화이트보드',31:'사람',32:'협탁',33:'변기',34:'세면대',35:'조명',36:'욕조',37:'가방',38:'구조물',39:'기타 가구',40:'소품'}


def object_cloud(cloud, instance):
    indices = np.flatnonzero(cloud["instance"] == instance)
    return {k: v[indices] for k, v in cloud.items()}, indices


def rigid_motion(before, after):
    source = cloud_geometry(before).voxel_down_sample(.015)
    target = cloud_geometry(after).voxel_down_sample(.015)
    a, b = np.asarray(source.points), np.asarray(target.points)
    center_a, center_b = np.median(a, axis=0), np.median(b, axis=0)
    tree = cKDTree(b)
    reg = o3d.pipelines.registration
    results = []
    # Rotation about gravity; ICP can refine all axes. Identity is included so
    # stable objects can win over large motion on repeated/symmetric surfaces.
    initial = [np.eye(4)]
    for angle in np.arange(0, 360, 30):
        r = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0., np.deg2rad(angle), 0.]))
        t = np.eye(4)
        t[:3, :3], t[:3, 3] = r, center_b - r @ center_a
        initial.append(t)
    for seed in initial:
        fitted = reg.registration_icp(source, target, .12, seed, reg.TransformationEstimationPointToPoint(),
                                     reg.ICPConvergenceCriteria(max_iteration=80))
        fitted = reg.registration_icp(source, target, .045, fitted.transformation,
                                     reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=.025)),
                                     reg.ICPConvergenceCriteria(max_iteration=50))
        transform = fitted.transformation
        transformed = a @ transform[:3,:3].T + transform[:3,3]
        forward = tree.query(transformed)[0]
        backward = cKDTree(transformed).query(b)[0]
        coverage = float((np.mean(forward < .04) + np.mean(backward < .04)) / 2)
        median = float((np.median(forward) + np.median(backward)) / 2)
        rotation = float(np.rad2deg(np.arccos(np.clip((np.trace(transform[:3,:3])-1)/2,-1,1))))
        displacement = float(np.linalg.norm(transform[:3,:3] @ center_a + transform[:3,3] - center_a))
        score = coverage - 3 * median - .002 * rotation / 180 - .002 * displacement
        results.append((score, transform, coverage, median, rotation, displacement))
    _, transform, coverage, median, rotation, displacement = max(results, key=lambda r: r[0])
    unchanged_a = cKDTree(b).query(a)[0]
    unchanged_b = cKDTree(a).query(b)[0]
    unchanged_fraction = float((np.mean(unchanged_a > .06) + np.mean(unchanged_b > .06)) / 2)
    return {"before_to_after": transform.tolist(), "surface_overlap": coverage,
            "surface_residual_m": median, "rotation_deg": rotation, "displacement_m": displacement,
            "unmoved_surface_difference_fraction": unchanged_fraction,
            "before_reference_m": center_a.tolist(),
            "after_reference_m": (transform[:3,:3] @ center_a + transform[:3,3]).tolist()}


def bounds(points):
    p = points @ TO_DISPLAY.T
    return {"min": p.min(0).tolist(), "max": p.max(0).tolist()}


def photo_links(points, epoch, manifest, meshes, output):
    points = points[np.linspace(0,len(points)-1,min(len(points),1200)).astype(int)]
    links=[]
    for f in manifest["keyframes"]:
        if f["epoch"] != epoch:
            continue
        checked = f["status"] == "estimated_rgb_checked"
        pose = np.array(f["camera_to_world"] if checked else f.get("depth_camera_to_world",[]))
        if pose.shape != (4,4):
            continue
        focal = f["focal_px"] if checked else 585.225
        pixels,z = project(points,pose,focal)
        inside=(z>.25)&(pixels[:,0]>=20)&(pixels[:,0]<628)&(pixels[:,1]>=20)&(pixels[:,1]<464)
        if inside.sum()<8:
            continue
        cache=output/"frames-cache"/(f["id"]+"-visibility.npy")
        if cache.exists():
            depth=np.load(cache)
        else:
            _,_,depth=render(meshes[epoch],pose,focal)
            np.save(cache,depth)
        xy=np.rint(pixels[inside]).astype(int)
        surface=depth[xy[:,1],xy[:,0]]
        visible=np.isfinite(surface)&(surface>=z[inside]-.10)
        fraction=float(visible.mean()*inside.mean())
        if visible.sum()<8 or fraction<.05:
            continue
        relevant=pixels[inside][visible]
        area=float(np.prod(np.maximum(np.ptp(relevant,axis=0),1))/(648*484))
        links.append({"frame_id":f["id"],"score":float(fraction*np.sqrt(area)),"visible_sample_fraction":fraction,
                      "bbox_px":np.r_[relevant.min(0),relevant.max(0)].tolist() if checked else None,
                      "camera_check":"rgb_feature_correspondence" if checked else "depth_registration_only",
                      "relation":"same_instance_view_candidate", "absence_confirmed":False})
    links.sort(key=lambda x:(-int(x["camera_check"]=="rgb_feature_correspondence"),-x["score"]))
    return links[:5]


def build(data, output):
    review=json.loads((output/"review.json").read_text(encoding="utf-8"))
    clouds=[read_cloud(data/f"{review['scene']}_{e}.ply") for e in review['source_epochs']]
    t=np.array(review["registration"]["after_to_before"])
    clouds[1]["xyz"]=clouds[1]["xyz"]@t[:3,:3].T+t[:3,3]
    clouds[1]["normals"]=clouds[1]["normals"]@t[:3,:3].T
    meshes=[o3d.io.read_triangle_mesh(str(output/f"cache/{e}-depth8.ply")) for e in (0,1)]
    motion_cache_path=output/'cache/object-motion.json'
    cache_key=[s['sha256'] for s in review['source_files']]+[review['registration']['after_to_before']]
    motion_cache=json.loads(motion_cache_path.read_text()) if motion_cache_path.exists() else {}
    if motion_cache.get('sources')!=cache_key:
        motion_cache={'sources':cache_key,'motions':{}}
    ambiguity=[]
    for epoch in review['source_epochs']:
        annotation=data/f"{review['scene']}_{epoch}.txt"
        if annotation.exists():
            for row in annotation.read_text().splitlines():
                base,members=row.split('|')
                ambiguity.append({'capture':epoch,'base_instance':int(base),'equivalent_instance_ids':sorted(set([int(base)]+[int(i) for i in members.split()]))})
    objects=[]
    ids=[set(int(v) for v in np.unique(c["instance"])) for c in clouds]
    # Author-allowed identity exchanges must not label an existing chair at its
    # old location as an addition. Resolve 1-before/2-after groups by the least
    # supported motion; retain the ambiguity and original IDs in the manifest.
    original_after_ids=clouds[1]['instance'].copy()
    after_to_association={i:i for i in ids[1]}
    resolved_groups=set()
    for group in ambiguity:
        members=set(group['equivalent_instance_ids'])
        before_members=members&ids[0]
        after_members=members&ids[1]
        if group['capture']!=review['source_epochs'][1] or len(before_members)!=1 or len(after_members)!=2 or tuple(sorted(members)) in resolved_groups:
            continue
        before_id=next(iter(before_members))
        fits=[]
        for after_id in sorted(after_members):
            key=f'{before_id}->{after_id}'
            motion=motion_cache['motions'].get(key)
            if motion is None:
                motion=rigid_motion(object_cloud(clouds[0],before_id)[0],object_cloud(clouds[1],after_id)[0])
                motion_cache['motions'][key]=motion
                motion_cache_path.write_text(json.dumps(motion_cache,indent=2),encoding='utf8')
            cost=motion['displacement_m']+.4*(1-motion['surface_overlap'])+4*motion['surface_residual_m']+.04*motion['rotation_deg']/180
            fits.append((cost,after_id))
        matched_after=min(fits)[1]
        added_after=next(iter(after_members-{matched_after}))
        new_association=next(iter(after_members-before_members))
        after_to_association[matched_after]=before_id
        after_to_association[added_after]=new_association
        resolved_groups.add(tuple(sorted(members)))
        print('Ambiguous group',sorted(members),'before',before_id,'→ after',matched_after,'extra after',added_after,flush=True)
    association_to_after={v:k for k,v in after_to_association.items()}
    for original,association in after_to_association.items():
        clouds[1]['instance'][original_after_ids==original]=association
    for instance in sorted(ids[0]|ids[1]):
        if instance==1024:
            continue
        available=[instance in group for group in ids]
        parts=[object_cloud(c,instance)[0] if exists else None for c,exists in zip(clouds,available)]
        semantic=int(np.median(next(p for p in parts if p is not None)["class"]))
        if semantic in (0,1,2,22):
            continue
        original_after=association_to_after.get(instance) if available[1] else None
        display_instance=instance if available[0] else original_after
        name=f"{NAMES.get(semantic,'객체')} #{display_instance}"
        if all(available):
            key=f'{instance}->{original_after}'
            motion=motion_cache['motions'].get(key) or (motion_cache['motions'].get(str(instance)) if original_after==instance else None)
            if motion is None:
                motion=rigid_motion(*parts)
                motion_cache['motions'][key]=motion
                motion_cache_path.write_text(json.dumps(motion_cache,indent=2),encoding='utf8')
            moved=(motion["displacement_m"]>.08 or motion["rotation_deg"]>15) and motion["surface_overlap"]>.55 and motion["surface_residual_m"]<.04 and motion["unmoved_surface_difference_fraction"]>.12
            change="moved" if moved else "unchanged"
            print(name,change,round(motion["displacement_m"],3),round(motion["rotation_deg"],1),round(motion["surface_overlap"],2),flush=True)
        else:
            change="removed" if available[0] else "added"
            motion=None
            print(name,change,"annotation-presence candidate; absence not confirmed",flush=True)
        objects.append({"instance":instance,"before_instance_id":instance if available[0] else None,
                        "after_instance_id":original_after,"semantic_class":semantic,"name":name,"type":change,
                        "motion":motion,"available":available,"parts":parts})
    changes=[]
    unlinked=[]
    for obj in objects:
        if obj["type"]=="unchanged":
            continue
        parts=obj["parts"]
        bb=[bounds(p["xyz"]) if p is not None else None for p in parts]
        both=[b for b in bb if b is not None]
        union={"min":np.min([b["min"] for b in both],axis=0).tolist(),"max":np.max([b["max"] for b in both],axis=0).tolist()}
        epoch=0 if parts[0] is not None else 1
        links=[photo_links((parts[e] or parts[1-e])["xyz"],e,review,meshes,output) for e in (0,1)]
        if obj['type'] in ('added','removed') and review['keyframes'] and not all(links):
            unlinked.append({'instance_id':obj['instance'],'name':obj['name'],'annotation_presence_type':obj['type'],
                             'status':'counterpart_observation_unavailable','keyframes':links,
                             'before_bounds':bb[0],'after_bounds':bb[1]})
            continue
        groups=[g for g in ambiguity if obj['instance'] in g['equivalent_instance_ids']]
        group_counts=[{'instance_ids':g['equivalent_instance_ids'], 'before':len(ids[0]&set(g['equivalent_instance_ids'])),
                       'after':len(ids[1]&set(g['equivalent_instance_ids']))} for g in groups]
        display_name=obj['name']
        if obj['type']=='added' and groups:
            display_name=f"{NAMES.get(obj['semantic_class'],'객체')} {chr(65+sum(e['type']=='added' for e in changes))}"
        changes.append({"id":len(changes)+1,"instance_id":obj["instance"],
                        "before_instance_id":obj['before_instance_id'],"after_instance_id":obj['after_instance_id'],
                        "name":display_name,"type":obj["type"],
                        "status":"estimated_motion" if obj["motion"] else "presence_candidate",
                        "epoch":epoch,"bounds":union,"before_bounds":bb[0],"after_bounds":bb[1],
                        "motion":obj["motion"],"point_count":[len(p["xyz"]) if p is not None else 0 for p in parts],
                        "context_instance_ids":[obj["instance"]],"keyframes":links,"paired_photos":all(links),
                        "paired_rgb_checked":all(any(f["camera_check"]=="rgb_feature_correspondence" for f in group) for group in links),
                        "absence_confirmed":False, 'identity_ambiguous':bool(groups),'ambiguous_group_counts':group_counts})
    manifest=copy.deepcopy(review)
    if len(changes)>63:
        raise ValueError('Too many object changes for the display encoding')
    manifest["regions"]=changes
    manifest["kind"]="rescan_object_changes"
    manifest["method"]="Persistent annotated instance association + multi-start rigid surface registration"
    manifest["object_matching_source"]="Rescan persistent instance annotations; permitted ambiguous identities resolved by supported rigid-fit cost; not automatic instance recognition or identity ground truth"
    manifest['annotation_identity_ambiguities']=ambiguity
    manifest['after_annotation_to_association_id']=after_to_association
    manifest["change_summary"]={kind:sum(o["type"]==kind for o in changes) for kind in ("moved","added","removed")}
    manifest['change_summary']['unchanged']=sum(o['type']=='unchanged' for o in objects)
    manifest['unlinked_presence_candidates']=unlinked
    manifest["classification"]={"translation_threshold_m":.08,"rotation_threshold_deg":15,"minimum_overlap":.55,
                                "maximum_surface_residual_m":.04,"minimum_original_difference_fraction":.12,
                                "addition_removal":"object annotation presence; missing wall/floor/unknown surfaces excluded"}
    manifest["all_objects"]=[{k:v for k,v in o.items() if k!="parts"} for o in objects]
    # Keep restored geometry; transfer change type + event number to byte 27.
    # Bits 6..7: moved=1,added=2,removed=3; bits 0..5: event ID.
    for epoch in (0,1):
        for kind in ("mesh","points"):
            original=output/Path(review["displays"][epoch][kind]["url"]).name
            packed=np.fromfile(original,dtype=PACKED)
            packed["region"]=0
            for event in changes:
                source_instance=event['before_instance_id'] if epoch==0 else event['after_instance_id']
                if source_instance is not None:
                    packed["region"][packed["instance"]==source_instance]=event["id"]+64*{"moved":1,"added":2,"removed":3}[event["type"]]
            name=f"{epoch}-object-{kind}.bin"
            packed.tofile(output/name)
            manifest["displays"][epoch][kind].update(url="/rescan/"+name,sha256=digest(output/name))
    manifest.setdefault('link_summary',{})
    manifest["link_summary"]["regions_with_before_after_photo_candidates"]=sum(e["paired_photos"] for e in changes)
    manifest["link_summary"]["regions_with_both_rgb_checked"]=sum(e["paired_rgb_checked"] for e in changes)
    (output/"changes.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Result:",manifest["change_summary"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    build(args.data,args.output)

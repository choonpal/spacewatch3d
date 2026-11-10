"""v5 input preservation, target policy, plan and genuine render contracts."""
from pathlib import Path
import numpy as np
from plyfile import PlyData
from PIL import Image
from .io import read,save,result,sha
from . import __version__

def validate(job):
    from spacewatch3d.tasks.task_5b_keyframes.validation import validate_outputs
    deps=job['dependencies'];out=Path(job['output']);catalog=read(deps['discovery']['vocabulary']);allowed={r['id'] for r in catalog}|{255};scenes=[]
    for capture in job['config']['captures']:
        prefix=capture['id'];source=deps[prefix+'.reconstruct'];inst=deps[prefix+'.instances']
        original=np.load(deps[prefix+'.semantic']['arrays']);d=np.load(inst['arrays']);labels,ids=d['face_class_id'],d['face_instance_id']
        rid,kind=d['face_region_id'],d['face_region_kind'];regions=read(inst['regions']);scene=read(deps[prefix+'.scene']['scene'])
        candidates=[r for r in regions if r['kind']=='object_candidate'];ply=PlyData.read(inst['instance_ply'])['face'].data
        checks=dict(semantic_ids_valid=set(np.unique(labels)).issubset(allowed),
            every_face_has_region=rid.shape==(len(d['faces']),) and bool((rid>0).all()),
            region_face_counts_exact=sum(r['faces'] for r in regions)==len(d['faces']),
            candidate_ids_only=np.array_equal(ids,np.where(kind==2,rid,0)),
            no_unknown_candidates=not bool(ids[labels==255].any()),
            no_cross_class_candidates=all(len(np.unique(labels[ids==i]))==1 for i in np.unique(ids) if i),
            ply_ids_exact=np.array_equal(ply['instance_id'],ids),ply_regions_exact=np.array_equal(ply['region_id'],rid),
            ply_kinds_exact=np.array_equal(ply['region_kind'],kind),
            all_candidate_targets={o['instance_id'] for o in scene['objects']}=={r.get('instance_id','region_'+str(r['id'])) for r in candidates},
            scene_face_count=len(d['faces'])==scene['source_context']['original_mesh_faces'],
            scene_residual_count=int((kind!=2).sum())==scene['source_context']['residual_faces'])
        for key in ['vertices','faces','rgb','face_class_id']:
            checks[key+'_preserved']=np.array_equal(original[key],d[key])
        for key in ['vote_agreement','mean_winning_support','observations','geometric_observations','source_face_index']:
            if key in original:checks[key+'_preserved']=np.array_equal(original[key],d[key])
        if job['config']['segmentation']['source']=='florence':
            root=Path(deps[prefix+'.objects']['data_root']);inference=read(root/'inference_summary.json');prepared=read(root/'prepared.json')['scenes'][0]
            manifest=read(Path(source['source_run'])/'input/manifest.json');observations=read(root/'observations.json')
            checks.update(single_segmentation_model=set(job['config']['discovery']['models'])=={'florence'},
                all_selected_views_inferred=inference['views']==len(prepared['views']),
                source_frame_hashes=all(manifest['frames'][r['frame']]['sha256']==r['source_sha256'] for r in prepared['views']),
                semantic_multiframe_support=bool((original['observations'][labels!=255]>=2).all()))
            for r in observations:
                path=root/'predictions'/(r['key']+'.json');pred=read(path)
                if not pred['quality']['usable'] or pred['result']['output_tokens']>=inference['max_tokens']+1 or pred['quality']['outside_box_fraction']>.1:raise ValueError('Rejected Florence mask was projected')
                if sha(path.with_suffix('.npz'))!=pred['mask_sha256']:raise ValueError('Florence mask changed')
        expected=capture.get('expected_candidates')
        if expected is not None:checks['expected_candidate_count']=len(candidates)==expected
        plan=read(deps[prefix+'.plan']['plan']);requests=read(deps[prefix+'.plan']['requests'])
        if job['config'].get('media',{}).get('mode')=='original_erp':
            from .capture519 import validate_plan
            validate_plan(job,capture,deps[prefix+'.plan']['plan'],deps[prefix+'.plan']['requests'])
        elif capture.get('planning_profile'):
            from .target_profile import validate_plan
            validate_plan(job,capture,deps[prefix+'.plan']['plan'],deps[prefix+'.plan']['requests'])
        else:validate_outputs(plan,requests)
        renders=read(deps[prefix+'.render']['renders'])['results'];render_map={r['viewpoint_id']:r for r in renders}
        checks['render_ids_exact']=len(render_map)==len(renders)==len(requests['requests']) and set(render_map)=={r['viewpoint_id'] for r in requests['requests']}
        for request in requests['requests']:
            r=render_map[request['viewpoint_id']]
            native=job['config'].get('media',{}).get('mode')=='original_erp'
            origin='original_video_frame' if native else '3DGS_novel_view'
            if r['render_status']!='succeeded' or r['media_origin']!=origin or r['reference_from_camera']!=request['reference_from_camera']:raise ValueError('Render pose/origin mismatch')
            with Image.open(r['image_ref']) as im:
                expected=tuple(r['source_size']) if native else (request['width'],request['height'])
                if im.size!=expected:raise ValueError('Render dimensions mismatch')
                im.verify()
        checks['render_files_valid']=True
        checks={k:bool(v) for k,v in checks.items()}
        if not all(checks.values()):raise ValueError('v5 contract validation failed: '+str(checks))
        holdout=plan['coverage_report']['independent_validation'];robust={r.get('instance_id','region_'+str(r['id'])) for r in candidates}
        for run in holdout['runs']:
            robust&={oid for oid,q in run['coverage_report']['object_best_quality'].items() if q>=plan['provenance']['config']['object_quality_threshold']}
        scenes.append(dict(capture=prefix,checks=checks,total_faces=len(labels),candidate_count=len(candidates),
            residual_faces=int((kind!=2).sum()),keyframes=len(plan['viewpoints']),rendered_images=len(renders),
            observed_candidates_both_holdouts=len(robust),planning_status=plan['planning_status'],
            holdout_status=holdout['status'],input_coverage_status=plan['input_coverage_status'],
            whole_surface_coverage_min=min(r['coverage_report']['coverage_total'] for r in holdout['runs']),
            selected_connected_components=len({r['component_id'] for r in plan['viewpoints']}),instance_summary=inst['summary']))
    report=dict(status='passed',pipeline_version=__version__,scenes=scenes,segmentation_source=job['config']['segmentation']['source'],
        ground_truth_accuracy_measured=False,metric_scale_verified=all(deps[c['id']+'.reconstruct']['metric_scale_verified'] for c in job['config']['captures']),
        contract_success_is_not_complete_observation=True)
    save(out/'validation.json',report);result(out,report=str(out/'validation.json'),**report)

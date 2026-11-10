"""Machine-verifiable checks on the actual pipeline outputs; no ground-truth claim."""
from pathlib import Path
import numpy as np
from .common import read,save,sha


def run(config,output):
    from plyfile import PlyData
    output=Path(output);classes=read(output/'classes.json');priors=read(output/'size_priors.json')
    allowed={c['id'] for c in classes}|{255};results=[]
    assert [c['id'] for c in classes]==list(range(len(classes)))
    for capture in config['captures']:
        directory=output/'meshes'/capture['id'];before=np.load(capture['mesh']);after=np.load(directory/'mesh_arrays.npz')
        checks={}
        for key in ('vertices','faces','rgb'):checks[key+'_exactly_preserved']=bool(np.array_equal(before[key],after[key]))
        checks['valid_semantic_ids']=set(np.unique(after['face_class_id'])).issubset(allowed)
        checks['one_instance_id_per_original_face']=after['face_instance_id'].shape==(len(before['faces']),)
        checks['no_ids_on_unknown_faces']=not bool(after['face_instance_id'][after['face_class_id']==255].any())
        checks['instances_do_not_cross_semantic_classes']=all(len(np.unique(after['face_class_id'][after['face_instance_id']==i]))==1 for i in np.unique(after['face_instance_id']) if i)
        ply=PlyData.read(directory/'mesh_instances.ply')
        checks['ply_ids_roundtrip']=bool(np.array_equal(ply['face'].data['instance_id'],after['face_instance_id']))
        checks['ply_labels_roundtrip']=bool(np.array_equal(ply['face'].data['class_id'],after['face_class_id']))
        checks['new_vocabulary_consumed_by_ooops']=read(output/'segmentation'/capture['id']/'complete.json')['vocabulary_sha256']==sha(output/'classes.json')
        instances=after['face_instance_id'];baseline=after['baseline_face_instance_id']
        for row in read(directory/'splits.json'):
            if row['status']=='split':
                checks[f'parent_{row["parent_id"]}_exact_partition']=bool(np.array_equal(np.isin(instances,row['children']),baseline==row['parent_id']))
            elif row['status']=='abstained':
                checks[f'parent_{row["parent_id"]}_abstention_preserved']=bool(np.array_equal(instances[baseline==row['parent_id']],baseline[baseline==row['parent_id']]))
        if not all(checks.values()):raise AssertionError(checks)
        results.append(dict(scene=capture['id'],checks=checks,summary=read(directory/'summary.json')))
    refs={r['instance_id']:r for r in read(output/'instances.json')}
    for row in priors['classes'].values():
        for family in row['families']:
            assert family['references'] and all(refs[i]['reference_eligible'] for i in family['references'])
            assert all(len(refs[i]['complete_frames'])>=2 for i in family['references'])
    report=dict(status='passed',scenes=results,prior_classes=[k for k,v in priors['classes'].items() if v['families']],
                caveats=['No human-labeled 3D instance ground truth is available.',
                         'Geometric consistency checks are not semantic accuracy or AP.',
                         'Measured dimensions use reconstruction units; metric scale is unverified unless calibrated.',
                         'Size splitting supports elongated rows of similar objects; other layouts abstain.'])
    save(output/'validation.json',report)
    # Consumable config for the existing integration, with no stale vocabulary caches.
    base=Path(config['runtime']['pipeline_root'])/'configs/dorm2_dorm3.json'
    if base.exists():
        pipeline=read(base);pipeline['vocabulary']=str(output/'classes.json');captures=[]
        for c in config['captures']:
            source=Path(c['source_run']);m=read(source/'input/manifest.json')
            path=output/'segmentation'/c['id']/'prediction_manifest.json'
            save(path,dict(source_video_sha256=m['source_sha256'],frames=[dict(source_sha256=r['sha256']) for r in m['frames']]))
            captures.append(dict(id=c['id'],source_run=c['source_run'],coordinates=c['coordinates'],
                                 predictions=str(output/'segmentation'/c['id']),prediction_manifest=str(path)))
        pipeline['captures']=captures
        save(output/'spacewatch_config.json',pipeline)
    print('VALIDATION_PASSED',len(results),'scenes; size priors:',report['prior_classes'],flush=True)


if __name__=='__main__':
    import sys
    run(read(sys.argv[1]),sys.argv[2])

"""Survey presentation metadata only; never changes plans, targets or geometry."""
from collections import Counter
from pathlib import Path
from .io import read, verify


def metadata(job):
    config=job['config'];requested=config.get('requested_inputs',{})
    ratings=[];noise=[];producer_times=[];rating_sources=set()
    for capture in config['captures']:
        cid=capture['id']
        if capture.get('importance_cache'):
            cache=read(capture['importance_cache']);ref=cache.get('rating_source')
            if ref and ref['path'] not in rating_sources:
                data=read(verify(ref));rating_sources.add(ref['path'])
                if data['manual']!=requested.get('manual') or set(data['class_names'])!=set(requested.get('object_classes',[])):
                    raise ValueError('Survey ratings do not match the actual requested inputs')
                ratings.extend(dict(class_name=name,ratings=row['ratings'],mean=row['mean'],sd=row['sd'],
                    status='UNVERIFIED',cached=True) for name,row in sorted(data['classes'].items()))
        profile=capture.get('planning_profile',{}).get('path')
        if profile:
            info=read(Path(profile)/'filter.json')
            counts={r['class_name']:r for r in info['classes']}
            # Include absent requested classes too, not just tuned/present classes.
            names=sorted(set(requested.get('object_classes',[]))|set(counts))
            for name in names:
                r=counts.get(name,{})
                noise.append([cid,name,r.get('source_components',0),r.get('retained',0),r.get('excluded',0)])
        source=Path(capture['source_run'])
        for filename,stage,field in [('inference_status.json','원래 점군 모델 추론','wall_s'),
                                     ('export_status.json','원래 점군 저장·복셀 처리','seconds')]:
            path=source/filename
            if path.exists():
                row=read(path)
                if isinstance(row.get(field),(int,float)):
                    producer_times.append(dict(stage=cid+' · '+stage,seconds=row[field],reused=True,scope='original_producer'))
    run=Path(job['run_root']);state=read(run/'status.json')
    timings=[dict(stage=name,seconds=row.get('wall_seconds',row.get('seconds')),reused=row.get('resumed',False),
        scope='current_run') for name,row in state['stages'].items() if row.get('status')=='complete']
    return dict(version='survey519-native-v1',ratings=ratings,noise=noise,timings=timings+producer_times,
        original_pixels=True,new_gaussian_images=False,physical_instances_verified=False,
        object_recognition_status='2D box/umbrella detections present; fused3D missing/insufficient, not repaired',
        general_new_video_entrypoint_complete=False)

"""Reuse the original Gaussian-RGB SimSaC regions throughout the keyframe tour.

No model inference or candidate geometry is changed here. Existing rendered
expected depth is used only to suppress regions hidden from a keyframe.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import numpy as np

from prepare import read, save, sha


def support(bounds, camera, geometry, height):
    """Count distinct ERP rays whose expected surface intersects the region."""
    points=np.stack(np.meshgrid(*[np.linspace(bounds[0][i],bounds[1][i],9) for i in range(3)],indexing='ij'),-1).reshape(-1,3)
    camera=np.asarray(camera);local=(points-camera[:3,3])@camera[:3,:3]
    distance=np.linalg.norm(local,axis=1);valid=distance>1e-8
    longitude=np.arctan2(local[:,0],local[:,2]);latitude=np.arcsin(np.clip(local[:,1]/np.maximum(distance,1e-8),-1,1))
    rows,cols=geometry['depth'].shape
    x=np.floor((longitude/(2*np.pi)+.5)*cols).astype(int)%cols
    y=np.clip(np.floor((latitude/np.pi+.5)*rows).astype(int),0,rows-1)
    depth=geometry['depth'][y,x];alpha=geometry['alpha'][y,x]
    tolerance=.08*height+.02*distance
    valid&=(alpha>=.85)&np.isfinite(depth)&(depth>.02*height)&(depth<40*height)&(np.abs(distance-depth)<=tolerance)
    return int(len(np.unique(y[valid]*cols+x[valid])))


def build(run, source, destination):
    original=read(source/'gaussian-data.json');prediction=read(source/'simsac-candidates.json')
    data=read(run/'published/gaussian-data.json')
    # Candidate coordinates must refer to exactly these registered maps.
    for key in ['checkpoint_sha256','video_sha256','native_to_mesh']:
        if [s[key] for s in data['scenes']]!=[s[key] for s in original['scenes']]:
            raise ValueError('Original candidates and tour refer to different maps or frames')
    for phase in ['before','after']:
        if [(a['sha256'],a['matrix']) for a in data['models'][phase]]!=[(a['sha256'],a['matrix']) for a in original['models'][phase]]:
            raise ValueError('Registered Gaussian model transforms differ')
    visibility={}
    for view in data['zones']:
        view_result={'raw':{},'filtered':{}}
        geometry={}
        for phase in ['before','after']:
            record=data['renders'][view['id']][phase];path=run/'published'/record['geometry']
            if sha(path)!=record['geometry_sha256']:raise ValueError('Keyframe depth changed')
            with np.load(path) as archive:geometry[phase]={k:archive[k] for k in ['depth','alpha']}
        for mode in ['raw','filtered']:
            for b in prediction['boxes'][mode]:
                counts={phase:support(b['bounds'],view['camera_to_common_cv'],geometry[phase],data['height_u']) for phase in geometry}
                view_result[mode][b['id']]={'visible':max(counts.values())>=3,'support_rays':counts}
        visibility[view['id']]=view_result
    # Preserve every original candidate, including its id, source zone and bounds.
    result={'schema':'frozen-gaussian-simsac-tour-overlay-v1','boxes':copy.deepcopy(prediction['boxes']),
            'source_prediction':'../gaussian-map-comparison-20261025/simsac-candidates.json',
            'source_prediction_sha256':sha(source/'simsac-candidates.json'),
            'source_data_sha256':sha(source/'gaussian-data.json'),'source_zones':original['zones'],
            'source_renders':original['renders'],'source_stats':prediction['zones'],
            'source_asset_prefix':'../gaussian-map-comparison-20261025/',
            'visibility':visibility,'candidate_geometry_changed':False,'new_inference':False,
            'visibility_policy':{'method':'9^3 points per fixed candidate box tested against saved keyframe ERP expected depth',
                'alpha_min':.85,'distance_tolerance_height':.08,'distance_tolerance_relative':.02,
                'minimum_distinct_rays':3,'epochs':'either before or after',
                'during_transition':'hide until exact destination keyframe; no depth at intermediate poses',
                'accuracy_claimed':False}}
    save(destination,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=Path('outputs/gaussian-keyframes-20261026-v2'))
    p.add_argument('--source',type=Path,default=Path('outputs/scenediff-mesh-rgb-20261024/gaussian-map-comparison-20261025'))
    p.add_argument('--destination',type=Path,default=Path('outputs/scenediff-mesh-rgb-20261024/gaussian-keyframe-comparison-20261026/tour-candidates.json'))
    a=p.parse_args();r=build(a.run,a.source,a.destination)
    print({'fixed_filtered_regions':len(r['boxes']['filtered']),'views':len(r['visibility']),
           'visible_filtered':{k:[b for b,s in v['filtered'].items() if s['visible']] for k,v in r['visibility'].items()}})

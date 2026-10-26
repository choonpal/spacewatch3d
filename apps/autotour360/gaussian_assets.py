"""Serve the published Gaussian change tour without exposing private inputs."""
import json
import re
from pathlib import Path

TOUR='gaussian-uncapped-tour-20261026'
SHARED='gaussian-map-comparison-20261025'
PAGE=f'/gaussian-change/{TOUR}/index.html?embed=1'
PUBLIC={'index.html','viewer.js','viewer.css','tour.js','tour_navigation.js','object_evidence.js','change_regions.js',
        'gaussian-data.json','simsac-candidates.json','simsac-model.json','object-evidence.json','automatic-matching.json',
        'tour-candidates.json','tour-routes.json','change-regions.json','tour-validation.json','automatic-replay-audit.json',
        'route-overview.png','REPORT.md','README.md','gaussian-validation.json','shared-change-validation.json'}


def asset_path(root,route):
    prefix='/gaussian-change/'
    if not route.startswith(prefix):return None
    relative=route[len(prefix):];parts=relative.split('/')
    if len(parts)<2 or parts[0] not in {TOUR,SHARED} or any(p in {'','.','..'} for p in parts):return None
    name='/'.join(parts[1:])
    allowed=name in PUBLIC or re.fullmatch(r'renders/(?:K\d+|R\d{2})/(?:before|after)\.jpg',name)
    allowed=allowed or re.fullmatch(r'object-meshes/S[01]-\d{3}(?:@I[01]-\d{3})?\.json',name)
    if parts[0]==SHARED:
        allowed=allowed or re.fullmatch(r'models/[A-Za-z0-9_-]+\.ply',name)
        allowed=allowed or re.fullmatch(r'vendor/[A-Za-z0-9_./-]+\.(?:js|wasm|bin|txt)',name)
    if not allowed:return None
    path=(Path(root)/relative).resolve();root=Path(root).resolve()
    return path if path.is_relative_to(root) else None


def status(root):
    root=Path(root);folder=root/TOUR
    required=['index.html','gaussian-data.json','tour-routes.json','tour-candidates.json','change-regions.json','object-evidence.json']
    missing=[name for name in required if not (folder/name).is_file()]
    result={'available':False,'viewer_url':PAGE,'missing':missing,'source':'gaussian_simsac_saved_automatic_results',
            'new_inference_on_navigation':False,'new_maps_end_to_end_available':False}
    if missing:return result
    try:
        data=json.loads((folder/'gaussian-data.json').read_text(encoding='utf-8'))
        regions=json.loads((folder/'change-regions.json').read_text(encoding='utf-8'))
        for phase in ['before','after']:
            for asset in data['models'][phase]:
                path=(folder/asset['file']).resolve()
                if not path.is_relative_to(root.resolve()) or not path.is_file():missing.append(asset['file'])
        vendor=root/SHARED/'vendor'
        for name in ['three.module.js','spark.module.js']:
            if not (vendor/name).is_file():missing.append('vendor/'+name)
        result.update(available=not missing,keyframes=len(data['zones']),counts=regions['modes']['filtered']['counts'])
    except (OSError,ValueError,KeyError,TypeError) as error:
        result['error']='Gaussian 변화 결과의 형식을 확인해 주세요.'
    return result

"""Bridge the user's exact 21-node Dorm3 plan, without rerunning change detection."""
from pathlib import Path
import argparse
import numpy as np
from prepare import read,save,sha
from prepare_keyframes import build as prepare
from build_tour_routes import build as routes


def build(base,source,output):
    receipt=read(source/'source-receipt.json');report=read(source/'map_report.json')
    if sha(source/'plan/tour_plan.json')!='a705fc726dea288bdc04566d725a760c802b9aa2a8ae22b6f8f06fed73d5663c':
        raise ValueError('Not the exact user-selected uncapped plan')
    if sha(source/'dorm3-uncapped-overview.png')!=receipt['image_sha256']:
        raise ValueError('Source overview differs from the attached route')
    ids={v['viewpoint_id']:v['label'] for v in report['viewpoints']}
    if set(ids.values())!={f'K{i}' for i in range(1,22)}:raise ValueError('Expected K1..K21')
    protocol=prepare(base,source,output,source_sets=[('U3','dorm3',source)],view_ids=ids,
                     remote_root='/workspace/gaussian_uncapped_tour_20261026')
    for view in protocol['views']:
        view['label']=f'{view["id"]} · Dorm3 평지·상한 해제 · {view["registration_zone"]}'
    protocol['decisions'].update(change_detection_mode='reuse_existing_gaussian_rgb_simsac',
        new_simsac_inference=False,view_selection='Exact user-selected Dorm3 flat-floor uncapped 21-camera plan',
        candidate_geometry_changed=False)
    save(output/'protocol.json',protocol)
    graph=routes(output,output/'tour-routes.json')
    if graph['total_nodes']!=21 or graph['total_links']!=20 or len(graph['groups'][0]['segments'])!=1:
        raise ValueError('Expected one saved connected tour of 21 views and 20 links')
    for view in protocol['views']:
        source_point=next(v['position'] for v in report['viewpoints'] if v['viewpoint_id']==view['viewpoint_id'])
        np.testing.assert_allclose(np.asarray(view['source_reference_from_camera'])[:3,3],source_point,atol=1e-10)
    print('Exact K labels, source positions, cameras and all 20 link endpoints verified')
    return protocol


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=Path('outputs/gaussian-change-20261025/protocol.json'))
    p.add_argument('--source',type=Path,default=Path('outputs/route-versions-20261026/dorm3-fast-uncapped'))
    p.add_argument('--output',type=Path,default=Path('outputs/gaussian-uncapped-tour-20261026'))
    a=p.parse_args();build(a.base,a.source,a.output)

"""Prepare one chosen real Rescan pair's meshes for object-change/photo viewing.

Only data processing needed for the viewer: no attention regions or same-scan
benchmark controls. Original files remain unchanged and hashes are recorded.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from build_rescan_review import read_cloud, register, reconstruct, pack_geometry, TO_DISPLAY, digest


def prepare(data, output):
    source=json.loads((data/'source.json').read_text(encoding='utf8'))
    paths=[data/f"{source['scene']}_{e}.ply" for e in source['epochs']]
    clouds=[read_cloud(p) for p in paths]
    registration=register(clouds)
    output.mkdir(parents=True,exist_ok=True)
    cache=output/'cache';cache.mkdir(exist_ok=True)
    all_xyz=np.vstack([c['xyz']@TO_DISPLAY.T for c in clouds])
    origin=(all_xyz.min(0)+all_xyz.max(0))/2
    displays=[]
    for epoch,cloud in enumerate(clouds):
        print(f"Restoring {source['scene']} capture {source['epochs'][epoch]} ({len(cloud['xyz']):,} surfels)",flush=True)
        mesh=reconstruct(cloud,cache/f'{epoch}-depth8.ply',depth=8)
        ids=np.zeros(len(cloud['xyz']),np.uint8)
        displays.append({'epoch':epoch,'source_points':len(cloud['xyz']),
            'mesh':pack_geometry(mesh,cloud,ids,origin,output/f'{epoch}-mesh.bin',True),
            'points':pack_geometry(None,cloud,ids,origin,output/f'{epoch}-points.bin',False),
            'full_mesh_vertices':len(mesh.vertices),'full_mesh_triangles':len(mesh.triangles)})
    manifest={'version':1,'dataset':'Rescan','scene':source['scene'],'source_epochs':source['epochs'],
        'synthetic':False,'source_url':'https://rescan.cs.princeton.edu/','code_url':'https://github.com/mhalber/Rescan',
        'source_files':[{'path':str(p.resolve()),'sha256':digest(p),'bytes':p.stat().st_size} for p in paths],
        'unit':'m','display_origin':origin.tolist(),'display_rotation':TO_DISPLAY.tolist(),
        'bounds':{'min':all_xyz.min(0).tolist(),'max':all_xyz.max(0).tolist()},
        'registration':registration,'displays':displays,'regions':[], 'keyframes':[],
        'mesh_reconstruction':{'method':'Poisson','depth':8,'scale':1.1,'maximum_source_support_distance_m':.045},
        'camera_status':'pending_recovery'}
    (output/'review.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print('Prepared',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();prepare(args.data,args.output)

from pathlib import Path
import json,numpy as np
from runtime import R,C
out=R/'viewer'
with np.load(R/'result/viewer_sample.npz') as z:xyz=z['xyz'].copy();rgb=z['rgb']
flip=np.array([1,-1,-1],np.float32);xyz*=flip
rows=json.loads((R/'result/camera_poses.json').read_text())
trajectory=np.array([r['camera_to_world'] for r in rows])[:,:3,3]*flip
center=np.median(xyz,axis=0);xyz-=center;trajectory-=center
xyz.astype('<f4').tofile(out/'cloud_0_xyz.bin');rgb.astype('u1').tofile(out/'cloud_0_rgb.bin')
v=json.loads((R/'result/validation.json').read_text())
meta=dict(bounds=np.quantile(xyz,[.005,.995],axis=0).tolist(),cut_height=float(np.quantile(xyz[:,1],.73)),datasets=[dict(points=v['points'],displayed=len(xyz),trajectory=trajectory.tolist())],unverified_bridges=[dict(a=b['a'],b=b['b'],trajectory=trajectory[[b['a'],b['b']]].tolist()) for b in v.get('unverified_bridges',[])])
(out/'viewer_metadata.json').write_text(json.dumps(meta,indent=2))
print(json.dumps({'preview_points':len(xyz),'full_points':v['points'],'viewer':str(out/'index.html')}))

meta.update(name=C['name'],width=C['width'],height=C['height'],frames=C['frames'],interval=C['sample_interval_s'],pixel_step=C['pixel_step'],bridge_times=[b['timestamp_a'] for b in v.get('unverified_bridges',[])])
(out/'viewer_metadata.json').write_text(json.dumps(meta,indent=2))

meta['component_filter'] = {k: v['component_filter'][k] for k in ['input_points', 'kept_points', 'removed_points', 'removed_percent', 'kept_components', 'options']}
(out/'viewer_metadata.json').write_text(json.dumps(meta,indent=2))

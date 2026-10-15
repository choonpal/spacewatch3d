"""Inspect bounded prefixes of Rescan surfel annotations to shortlist real pairs.

Sampling is for pair selection only. The selected full PLY files must supply
the displayed changes; no prefix sample is used as absence ground truth.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import struct
import zipfile
import zlib
import numpy as np
from download_rescan_pair import RemoteZip, get_range

DTYPE=np.dtype([(n,'<f4') for n in ('x','y','z','nx','ny','nz')]+[(n,'u1') for n in ('red','green','blue')]+[('radius','<f4'),('class_idx','<i4'),('instance_idx','<i4')])


def inspect(info):
    header=get_range(info.header_offset,info.header_offset+29)
    name_len,extra_len=struct.unpack_from('<HH',header,26)
    start=info.header_offset+30+name_len+extra_len
    compressed=get_range(start,start+min(info.compress_size,768*1024)-1)
    raw=zlib.decompressobj(-15).decompress(compressed) if info.compress_type==8 else compressed
    end=raw.index(b'end_header\n')+11
    header=raw[:end].decode('ascii')
    if 'property int instance_idx' not in header or 'element face' in header:
        raise ValueError('Unexpected Rescan annotation layout')
    records=np.frombuffer(raw[end:end+((len(raw)-end)//DTYPE.itemsize)*DTYPE.itemsize],dtype=DTYPE)
    instances=[]
    for i in np.unique(records['instance_idx']):
        points=records[records['instance_idx']==i]
        semantic=int(np.median(points['class_idx']))
        if i==1024 or semantic in (0,1,2,22) or len(points)<8:
            continue
        xyz=np.column_stack([points[n] for n in ('x','y','z')])
        instances.append({'id':int(i),'class':semantic,'sample_points':len(points),'centroid':xyz.mean(0).tolist()})
    result={'member':info.filename,'sampled_points':len(records),'compressed_bytes':info.compress_size,
            'uncompressed_bytes':info.file_size,'instances':instances}
    print(info.filename,'sample',len(records),'objects',[i['id'] for i in instances],flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    archive=zipfile.ZipFile(RemoteZip(26239218080))
    infos=[i for i in archive.infolist() if '/gt_segmentation/' in i.filename and i.filename.endswith('.ply')]
    with ThreadPoolExecutor(max_workers=4) as pool:
        captures=list(pool.map(inspect,infos))
    pairs=[]
    for a in captures:
        for b in captures:
            if a['member'].split('/')[0]!=b['member'].split('/')[0]:
                continue
            epoch=lambda c:int(Path(c['member']).stem.rsplit('_',1)[1])
            if epoch(a)>=epoch(b):
                continue
            aa={o['id']:o for o in a['instances']};bb={o['id']:o for o in b['instances']}
            added=sorted(set(bb)-set(aa));removed=sorted(set(aa)-set(bb))
            if not added or not removed:
                continue
            moved=[i for i in set(aa)&set(bb) if np.linalg.norm(np.array(aa[i]['centroid'])-bb[i]['centroid'])>.12]
            pairs.append({'before':Path(a['member']).stem,'after':Path(b['member']).stem,
                          'added_candidates':added,'removed_candidates':removed,'moved_centroid_candidates':sorted(moved),
                          'compressed_cloud_bytes':a['compressed_bytes']+b['compressed_bytes']})
    pairs.sort(key=lambda x:(-min(len(x['added_candidates']),len(x['removed_candidates'])),x['compressed_cloud_bytes']))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'scope':'annotation-prefix shortlist only','captures':captures,'pairs':pairs},indent=2),encoding='utf8')
    print(json.dumps(pairs,indent=2),flush=True)


if __name__=='__main__':
    main()

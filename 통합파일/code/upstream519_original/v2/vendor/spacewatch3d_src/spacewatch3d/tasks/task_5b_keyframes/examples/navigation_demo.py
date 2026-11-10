"""CPU-rendered synthetic panoramas for the image-only navigation example.

Content endpoints are prescribed for transition testing, NOT selected by an LLM
or claimed as an optimal coverage solution. No real scene synthesis is done.
"""
import argparse
import base64
from dataclasses import asdict
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from shapely.ops import triangulate

from ..models import Config, Scene
from ..navigation import create_navigation


def panorama(floor, pose, width=768, height=384):
    faces=[]
    boundary=list(floor.exterior.coords)
    for a,b in zip(boundary,boundary[1:]):
        faces.extend([[[*a,0],[*b,0],[*b,3]],[[*a,0],[*b,3],[*a,3]]])
    for t in triangulate(floor):
        if floor.covers(t):
            xy=list(t.exterior.coords)[:3]
            faces.extend([[[*p,z] for p in xy] for z in (0,3)])
    lon=(np.arange(width)+.5)/width*2*np.pi-np.pi
    lat=np.pi/2-(np.arange(height)+.5)/height*np.pi
    lon,lat=np.meshgrid(lon,lat)
    rays=np.stack([np.sin(lon)*np.cos(lat),-np.sin(lat),np.cos(lon)*np.cos(lat)],axis=-1).reshape(-1,3)
    pose=np.asarray(pose);rays=rays@pose[:3,:3].T;origin=pose[:3,3]
    best=np.full(len(rays),np.inf);hit=np.zeros(len(rays),int)
    for i,tri in enumerate(np.asarray(faces)):
        a,b,c=tri;e1,e2=b-a,c-a;h=np.cross(rays,e2);det=h@e1
        active=np.abs(det)>1e-10;inv=np.divide(1.,det,out=np.zeros_like(det),where=active)
        s=origin-a;u=(h@s)*inv;q=np.cross(s,e1);v=(rays@q)*inv;t=float(q@e2)*inv
        ok=active&(u>=0)&(v>=0)&(u+v<=1)&(t>1e-8)&(t<best)
        best[ok]=t[ok];hit[ok]=i
    finite=np.isfinite(best);points=origin+rays*np.where(finite,best,0)[:,None]
    palette=np.array([[210,140,90],[110,163,202],[145,181,156],[205,184,125]])
    colors=palette[(hit//2)%4].astype(float)
    grid=np.mod(np.floor(points[:,0]*2)+np.floor(points[:,1]*2)+np.floor(points[:,2]*2),2)
    colors*=np.where(grid[:,None]==0,.72,1.)
    colors[~finite]=[35,45,60]
    result=Image.fromarray(np.uint8(np.clip(colors,0,255)).reshape(height,width,3))
    data=io.BytesIO();result.save(data,format='JPEG',quality=82)
    return 'data:image/jpeg;base64,'+base64.b64encode(data.getvalue()).decode()


def make_case(bent):
    floor=Polygon([(0,0),(6,0),(6,6),(4,6),(4,2),(0,2)]) if bent else Polygon([(0,0),(8,0),(8,2),(0,2)])
    xy=[[1,1],[3,1],[5,1],[5,3],[5,5]] if bent else [[1,1],[3,1],[5,1],[7,1]]
    points=[[*p,1.5] for p in xy]
    scene=Scene('synthetic_bend' if bent else 'synthetic_straight','demo',True,'world',floor,0.,[],[],[],None,{'geometry_sha256':'synthetic_demo'})
    nodes=[{'node_id':str(i),'position_m':p,'component_id':'component_001'} for i,p in enumerate(points)]
    def view(i):
        pose=np.eye(4);pose[:3,:3]=[[1,0,0],[0,0,1],[0,-1,0]];pose[:3,3]=points[i]
        return {'viewpoint_id':'v'+str(i),'route_node_id':str(i),'reference_from_camera':pose.tolist(),'roles':['content'],'image_ref':None}
    plan={'schema_name':'spacewatch3d.virtual_tour_plan','schema_version':'0.1.0',
          'scene_id':scene.scene_id,'capture_id':'demo','reference_frame_id':'world','route_id':'demo',
          'planning_status':'complete','route_graph':{'nodes':nodes},'viewpoints':[view(0),view(len(xy)-1)],
          'excluded_viewpoints':[],'provenance':{'config':asdict(Config()),'geometry_sha256':'synthetic_demo'},
          'tour_links':[{'from_viewpoint':'v0','to_viewpoint':'v'+str(len(xy)-1),'route_node_ids':[str(i) for i in range(len(xy))],'polyline_m':points}]}
    nav=create_navigation(plan,scene)
    media={v['viewpoint_id']:panorama(floor,v['reference_from_camera']) for v in nav['viewpoints']}
    return {'name':'L자 복도 · 필요한 이동 1장' if bent else '직선 복도 · 이동 추가 없음','navigation':nav,'media':media}


def generate(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    cases=[make_case(True),make_case(False)]
    source=Path(__file__).with_name('navigation_viewer.html').read_text(encoding='utf-8')
    encoded=json.dumps({'cases':cases},ensure_ascii=False).replace('<','\\u003c')
    source=source.replace('{"cases":[]}',encoded)
    (output/'navigation-demo.html').write_text(source,encoding='utf-8')
    for i,c in enumerate(cases):
        (output/f'case-{i}.navigation.json').write_text(json.dumps(c['navigation'],ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'html':str(output/'navigation-demo.html'),'cases':[c['navigation']['report'] for c in cases]},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    generate(parser.parse_args().output)

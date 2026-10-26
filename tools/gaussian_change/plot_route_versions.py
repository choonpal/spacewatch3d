"""Plot actual saved floor polygons, keyframes and links, in planner coordinates."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as effects
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

OUT=Path('outputs/route-versions-20261026')


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def panel(ax,folder,title,color,subfolder=''):
    scene=read(folder/'scene.json');plan=read(folder/subfolder/'tour_plan.json')
    for component in scene['floor']['components']:
        polygon=np.asarray(component['polygon'])
        ax.add_patch(Polygon(polygon,facecolor='#d9dee4',edgecolor='#a9b4bf',linewidth=.4,zorder=0))
        for hole in component.get('holes',[]):ax.add_patch(Polygon(hole,facecolor='white',edgecolor='#a9b4bf',linewidth=.4))
    for edge in plan['route_graph']['edges']:
        p=np.asarray(edge['polyline_m']);ax.plot(p[:,0],p[:,1],color='#879bab',lw=.55,alpha=.55,zorder=1)
    for link in plan['tour_links']:
        p=np.asarray(link['polyline_m']);ax.plot(p[:,0],p[:,1],color=color,lw=1.4,alpha=.85,zorder=2)
    views={v['viewpoint_id']:v for v in plan['viewpoints']}
    coords=[]
    for number,visit in enumerate(plan['tour_visits'],1):
        pose=np.asarray(views[visit['viewpoint_id']]['reference_from_camera']);x,y=pose[:2,3];coords.append([x,y])
        ax.scatter(x,y,s=27,color=color,edgecolors='white',linewidth=.7,zorder=3)
        # Adjacent numbered points can be very close. Alternate text offsets;
        # retain the original stored tour order instead of sorting by position.
        offset=(6,5) if number%2 else (-7,-10)
        text=ax.annotate(str(number),(x,y),xytext=offset,textcoords='offset points',ha='left' if number%2 else 'right',fontsize=8.5,color=color,weight='bold',zorder=4)
        text.set_path_effects([effects.withStroke(linewidth=2.5,foreground='white')])
    all_points=np.concatenate([np.asarray(c['polygon']) for c in scene['floor']['components']])
    minimum=all_points.min(0);maximum=all_points.max(0)
    ax.set_xlim(minimum[0]-2,maximum[0]+2);ax.set_ylim(minimum[1]-2,maximum[1]+2)
    ax.set_aspect('equal');ax.set_xlabel('X (계획 좌표)');ax.set_ylabel('Y (계획 좌표)')
    ax.grid(alpha=.18,lw=.5);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    gaps=sum(1 for a,b in zip(plan['tour_visits'],plan['tour_visits'][1:]) if not any({l['from_viewpoint'],l['to_viewpoint']}=={a['viewpoint_id'],b['viewpoint_id']} for l in plan['tour_links']))
    ax.set_title(f'{title}\n{len(plan["tour_visits"])}지점 · 저장 이동선 {len(plan["tour_links"])}개 · 연속 방문 구간 {gaps+1}개',fontsize=12,pad=12,weight='bold')
    return {'title':title,'folder':str(folder),'plan':str(folder/subfolder/'tour_plan.json'),
            'keyframes':len(plan['tour_visits']),'stored_links':len(plan['tour_links']),
            'segments':gaps+1,'planning_status':plan['planning_status'],'publication_ready':plan['publication_ready'],
            'numbering':'saved tour_visits order','coordinates':'each capture native planner XY; not cross-capture registered'}


def figure(name,title,items):
    fig,axes=plt.subplots(1,len(items),figsize=(7.2*len(items),12),squeeze=False)
    records=[]
    for ax,item in zip(axes.flat,items):records.append(panel(ax,*item))
    fig.suptitle(title,fontsize=19,weight='bold',y=.985)
    handles=[Line2D([],[],color='#d9dee4',lw=8,label='입력 바닥 영역'),Line2D([],[],color='#879bab',lw=1,label='저장 경로 그래프'),Line2D([],[],color=items[0][2],lw=2,marker='o',label='선정 키프레임·실제 저장 이동선')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.948),ncol=3,frameon=False,fontsize=10)
    fig.text(.5,.028,'번호: 저장된 투어 방문 순서. 끊긴 이동선은 임의로 연결하지 않았습니다.\n각 촬영의 독립 좌표계이며, 모든 원본 계획은 검증 미완료 상태입니다.',ha='center',fontsize=10,color='#495664',linespacing=1.6)
    fig.subplots_adjust(left=.09,right=.96,top=.88,bottom=.095,wspace=.25)
    target=OUT/(name+'.png');fig.savefig(target,dpi=170,facecolor='white');plt.close(fig)
    return {'image':str(target.absolute()),'routes':records}


def main():
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False})
    OUT.mkdir(parents=True,exist_ok=True)
    v1=Path('outputs/gaussian-v1-tour-20261026/source');v5=Path('outputs/gaussian-keyframes-20261026/source')
    result=[figure('v1-dorm2-dorm3','사진 속 통합 파이프라인 v1 · 2026-10-24',[
        (v1/'dorm2','Dorm2 · v1','#21638b'),(v1/'dorm3','Dorm3 · v1','#167650')]),
        figure('v5-dorm2-dorm3','현재 화면에 연결했던 v5 기준 계획',[
        (v5/'production/dorm2','Dorm2 · v5 (P2)','#21638b'),(v5/'production/dorm3','Dorm3 · v5 (P3)','#167650')]),
        figure('statistical-dorm3','최신 Dorm3 통계 실험 · 2026-10-26',[
        (v5/'task5b','Dorm3 · 통계 실험 (S3)','#8053a3','plan')])]
    (OUT/'index.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

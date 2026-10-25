"""Show actual original-input and external-input SceneDiff candidate masks."""
from __future__ import annotations
import argparse, json, html, hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw, ImageFilter

ORIGINAL=np.array([255,160,55]);MESH=np.array([45,205,245]);COMMON=np.array([100,230,135])

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def overlay(rgb,mask,color):
    result=rgb.copy()
    result[mask]=(.5*result[mask]+.5*color).astype('u1')
    edge=np.array(Image.fromarray(mask.astype('u1')*255).filter(ImageFilter.FIND_EDGES))>0
    result[edge]=color
    return result

def match_candidates(a,b):
    ai=np.unique(a[a>=0]);bi=np.unique(b[b>=0])
    mat=np.zeros((len(ai),len(bi)))
    for i,x in enumerate(ai):
        for j,y in enumerate(bi):
            am,bm=a==x,b==y
            mat[i,j]=np.count_nonzero(am&bm)/max(1,np.count_nonzero(am|bm))
    pairs=[]
    if mat.size:
        rows,cols=linear_sum_assignment(-mat)
        pairs=[dict(original=int(ai[i]),mesh=int(bi[j]),iou=float(mat[i,j])) for i,j in zip(rows,cols) if mat[i,j]>=.5]
    return dict(original=len(ai),mesh=len(bi),common=len(pairs),
                original_only=len(ai)-len(pairs),mesh_only=len(bi)-len(pairs),matches=pairs,
                criterion='One-to-one matching of final candidate masks over all 10 shared RGB views; IoU >= 0.5. Counts indicate agreement, not correctness.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('outputs/scenediff-mesh-rgb-20261024/paired'))
    args=p.parse_args();root=args.root;source=root.parent/'prepared'
    data=read(root/'results.json');assets=root/'images';assets.mkdir(exist_ok=True)
    cards=[];zones=[];best=[];coverage={}
    old=read(root.parent/'results.json')
    names={r['id']:r.get('label','') for r in old.get('objects',[])}
    for zone in ['R01','R07','R08']:
        inputs=np.load(source/zone/'inputs.npz');meta=read(source/zone/'manifest.json')
        a=np.load(root/'results'/zone/'original/maps.npz');b=np.load(root/'results'/zone/'mesh/maps.npz')
        sa=read(root/'results'/zone/'original/summary.json');sb=read(root/'results'/zone/'mesh/summary.json')
        assert sa['input_rgb_sha256']==sb['input_rgb_sha256']
        assert sa['input_rgb_sha256']==[hashlib.sha256(inputs[f'rgb_{i}'].tobytes()).hexdigest() for i in range(len(meta['views']))]
        for maps,summary in [(a,sa),(b,sb)]:
            assert np.all(maps['cluster'][maps['labels']>=0]>summary['threshold'])
            assert np.all(maps['masks'][maps['labels']>=0]>=0)
        labels_a,labels_b=a['labels'],b['labels'];am,bm=labels_a>=0,labels_b>=0
        pixels=dict(common=int((am&bm).sum()),original_only=int((am&~bm).sum()),mesh_only=int((bm&~am).sum()),
                    neither=int((~am&~bm).sum()),union=int((am|bm).sum()))
        pixels['iou']=pixels['common']/max(1,pixels['union'])
        stats=match_candidates(labels_a,labels_b)
        zones.append(dict(zone=zone,original=sa,mesh=sb,candidates=stats,pixels=pixels))
        zone_cards=[]
        for i,v in enumerate(meta['views']):
            rgb=inputs[f'rgb_{i}'];ma,mb=am[i],bm[i]
            for region,objid in v['region_to_input_instance'].items():
                region_mask=inputs[f'mask_{i}']==int(region)
                key=(zone,objid)
                entry=coverage.setdefault(key,dict(zone=zone,id=objid,label=names.get(objid,''),pixels=0,original_pixels=0,mesh_pixels=0))
                entry['pixels']+=int(region_mask.sum())
                entry['original_pixels']+=int((region_mask&ma).sum())
                entry['mesh_pixels']+=int((region_mask&mb).sum())
            diff=rgb.copy()
            for mask,color in [(ma&~mb,ORIGINAL),(mb&~ma,MESH),(ma&mb,COMMON)]:
                diff=overlay(diff,mask,color)
            path=f'{zone}-{i:02d}'
            images=[rgb,overlay(rgb,ma,ORIGINAL),overlay(rgb,mb,MESH),diff]
            for name,img in zip(['rgb','original','mesh','diff'],images):
                Image.fromarray(img).resize((448,448)).save(assets/f'{path}-{name}.png')
            c=dict(zone=zone,index=i,epoch=v['epoch'],frame=v['map_frame'],yaw=v['yaw_degrees'],
                original_ids=[int(x) for x in np.unique(labels_a[i][ma])],mesh_ids=[int(x) for x in np.unique(labels_b[i][mb])],
                original_pixels=int(ma.sum()),mesh_pixels=int(mb.sum()),common_pixels=int((ma&mb).sum()),
                original_only_pixels=int((ma&~mb).sum()),mesh_only_pixels=int((mb&~ma).sum()),
                path='images/'+path)
            c['different']=c['original_only_pixels']+c['mesh_only_pixels']
            c['candidate']=bool(ma.any() or mb.any())
            cards.append(c);zone_cards.append((c,images))
        best.append(max(zone_cards,key=lambda item:item[0]['different']))
    totals={key:sum(z['candidates'][key] for z in zones) for key in ['original','mesh','common','original_only','mesh_only']}
    for entry in coverage.values():
        entry['original_fraction']=entry['original_pixels']/max(1,entry['pixels'])
        entry['mesh_fraction']=entry['mesh_pixels']/max(1,entry['pixels'])
    report=dict(zones=zones,totals=totals,views=cards,existing_instance_coverage=list(coverage.values()),paired_rgb_identical=True,resolution=[224,224],
        source_commit=data['upstream_commit'],input_comparison='RGB -> Pi3 + original SAM vs identical RGB -> existing mesh-ray geometry + existing instance masks',
        note='Different floors; candidate disagreement is not a temporal-change accuracy metric.')
    save(root/'comparison.json',report)
    payload=json.dumps(report,ensure_ascii=False).replace('</','<\\/')
    template='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SceneDiff — 원본 입력 vs 메쉬 입력</title>
<style>*{box-sizing:border-box}body{margin:0;background:#101821;color:#edf3fa;font:15px system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:28px}h1{font-size:28px;margin:0 0 10px}p{color:#b6c7d9;line-height:1.7;margin:8px 0}section,article{background:#192634;border:1px solid #304253;border-radius:12px;padding:18px;margin:17px 0}.paths{display:grid;grid-template-columns:1fr 1fr;gap:16px}.paths div{padding:12px;background:#223346;border-radius:8px}.a{color:#ffa037}.b{color:#2dcdf5}.c{color:#64e687}.stats{display:flex;gap:12px;flex-wrap:wrap}.stats span{flex:1;min-width:150px;background:#223346;padding:14px;border-radius:8px}.stats b{display:block;font-size:28px}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:10px;border-bottom:1px solid #33475a}.legend span{display:inline-block;margin-right:22px}select,button{background:#293e53;color:white;padding:9px;border:1px solid #506779;border-radius:6px;margin:5px}a{color:#91cfff}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.tile img{width:100%;display:block;border-radius:6px}.tile b{display:block;margin-bottom:8px}.caption{font-size:13px;color:#b1c4d7;margin-top:8px}h3{font-size:17px;margin:0 0 15px}small{color:#afc3d5}.sticky{position:sticky;top:0;z-index:3;box-shadow:0 7px 20px #101821}details{line-height:1.7}code{overflow-wrap:anywhere}@media(max-width:900px){main{padding:14px}.grid{grid-template-columns:1fr 1fr}.paths{grid-template-columns:1fr}h1{font-size:22px}.sticky{position:static}}</style>
<main><h1>SceneDiff: 원본 입력과 메쉬 입력의 변화 후보 차이</h1><p>같은 RGB 30장 · 공통 224×224 입력 · 같은 DINOv3 H+/16 · 같은 후속 후보 탐지·병합·대응 코드</p>
<section class="paths"><div><strong class="a">A · 원본 모델 입력 경로</strong><p>RGB → 실제 Pi3 깊이·카메라·신뢰도 + 실제 SAM ViT-H 마스크 → SceneDiff 변화 후보</p></div><div><strong class="b">B · 기존 메쉬 입력 경로</strong><p>동일 RGB → 기존 메쉬에서 투영한 깊이·카메라 + 기존 SAM3.1 인스턴스 → SceneDiff 변화 후보</p></div></section>
<div class="stats" id="stats"></div><section><table><thead><tr><th>구역</th><th>원본 후보</th><th>메쉬 후보</th><th>후보 픽셀 IoU</th><th>유효 비교 영역 A / B</th><th>자동 기준 A / B</th></tr></thead><tbody id="summary"></tbody></table><p><small>후보 수는 SceneDiff 최종 통합 ID 수입니다. 후보 픽셀 IoU는 두 방식이 선택한 영역의 겹침 정도입니다. 후보 경계와 ID 병합 방식이 달라 수가 비슷해도 위치는 다릅니다. 서로 다른 층을 비교한 파일럿이며 시간 변화의 정확도 평가가 아닙니다. 원본 기본 입력 크기는 최대 518입니다.</small></p></section>
<section class="sticky"><label>구역 <select id="zone"><option value="">전체</option><option>R01</option><option>R07</option><option>R08</option></select></label><label>촬영 <select id="epoch"><option value="">전체</option><option value="0">Dorm2</option><option value="1">Dorm3</option></select></label><label>보기 <select id="filter"><option value="all">전체 30장</option><option value="different">후보가 다른 장면</option><option value="candidate">후보가 있는 장면</option><option value="original">원본에만 잡힌 픽셀 있음</option><option value="mesh">메쉬에만 잡힌 픽셀 있음</option></select></label><label>정렬 <select id="sort"><option value="different">차이가 큰 순</option><option value="input">입력 순서</option></select></label><small id="count"></small><p class="legend"><span class="a">■ 원본만 후보</span><span class="b">■ 메쉬만 후보</span><span class="c">■ 두 방식 공통 후보 픽셀</span></p></section><div id="views"></div>
<section><details><summary>실험 조건·재현 자료</summary><p>회차별 16초 기존 인스턴스 결과에서 R01·R07·R08, 각 촬영 5장씩 선택했습니다. 두 경로의 RGB 바이트 해시가 전부 일치합니다. 224×224 공통 입력을 사용했습니다. 원본 로더의 기본 최대 크기는 518이므로 논문 기본 설정의 전체 해상도 재현과는 구분해야 합니다.</p><p>두 방식 모두 기하 + 0.1×영역 대응 + 0.5×외형 비용, max-entropy 자동 임계값, 구역별 독립 탐지, 원본의 객체 병합·회차 간 대응을 실행했습니다. 자동 임계값의 수치는 입력별로 달라집니다. 메쉬의 유효 ray hit 신뢰도는 1이며 Pi3의 학습 신뢰도와 다릅니다. 원본의 스케일 정규화·voxel 크기 반환 규칙도 두 경로에 동일 적용했습니다.</p><p>빈 영역 처리와 빈 임계값 입력 보호만 공통 적용하고, voxel 평균은 메모리 절약을 위해 같은 평균을 스트리밍 계산했습니다. 기존 인스턴스 ID를 먼저 변화 후보로 지정하지 않았습니다.</p><p>Dorm2와 Dorm3는 서로 다른 층입니다. 아래 결과는 공간 차이 후보 비교이며 시간 변화의 정확도·개선율을 뜻하지 않습니다.</p><a href="comparison.json">후보·픽셀 비교 JSON</a> · <a href="results.json">전체 실행 설정</a> · <a href="native-model-audit.json">Pi3·SAM 모델 검증</a> · <a href="REPORT.md">실험 설명</a></details></section></main>
<script>const d=__DATA__,$=s=>document.querySelector(s),pixelIou=d.zones.reduce((s,z)=>s+z.pixels.common,0)/Math.max(1,d.zones.reduce((s,z)=>s+z.pixels.union,0));$('#stats').innerHTML=[['원본 최종 후보',d.totals.original],['메쉬 최종 후보',d.totals.mesh],['전체 후보 픽셀 IoU',(100*pixelIou).toFixed(1)+'%'],['후보 영역이 다른 장면',d.views.filter(v=>v.different>0).length+' / 30']].map(([k,v])=>`<span>${k}<b>${v}</b></span>`).join('');$('#summary').innerHTML=d.zones.map(z=>`<tr><td>${z.zone}</td><td class="a">${z.candidates.original}</td><td class="b">${z.candidates.mesh}</td><td>${(100*z.pixels.iou).toFixed(1)}%</td><td>${(100*z.original.visible_fraction).toFixed(1)}% / ${(100*z.mesh.visible_fraction).toFixed(1)}%</td><td>${z.original.threshold.toFixed(4)} / ${z.mesh.threshold.toFixed(4)}</td></tr>`).join('');
function draw(){let rows=d.views.filter(v=>(!$('#zone').value||v.zone==$('#zone').value)&&(!$('#epoch').value||v.epoch==$('#epoch').value));let f=$('#filter').value;rows=rows.filter(v=>f==='all'||f==='different'&&v.different>0||f==='candidate'&&v.candidate||f==='original'&&v.original_only_pixels>0||f==='mesh'&&v.mesh_only_pixels>0);if($('#sort').value==='different')rows.sort((a,b)=>b.different-a.different);$('#count').textContent=rows.length+'장';$('#views').innerHTML=rows.map(v=>`<article><h3>${v.zone} · Dorm${v.epoch+2} · frame ${v.frame} · yaw ${v.yaw}°</h3><div class="grid">${[['rgb','동일 RGB 입력','마스크를 덮지 않은 원본'],['original','A · Pi3 + SAM',v.original_ids.length+'개 후보 ID · '+v.original_pixels+'px'],['mesh','B · 메쉬 + 기존 인스턴스',v.mesh_ids.length+'개 후보 ID · '+v.mesh_pixels+'px'],['diff','후보 영역 차이','원본만 '+v.original_only_pixels+'px / 메쉬만 '+v.mesh_only_pixels+'px / 공통 '+v.common_pixels+'px']].map(([key,title,caption])=>`<div class="tile"><b>${title}</b><img loading="lazy" src="${v.path}-${key}.png"><div class="caption">${caption}</div></div>`).join('')}</div></article>`).join('')}for(const e of ['zone','epoch','filter','sort'])$('#'+e).onchange=draw;draw();</script></html>'''
    for field,label in [('zone','구역'),('epoch','촬영'),('filter','보기'),('sort','정렬')]:
        template=template.replace(f'id="{field}"',f'id="{field}" aria-label="{label}"')
    (root/'index.html').write_text(template.replace('__DATA__',payload),encoding='utf-8')
    sheet=Image.new('RGB',(4*448,3*485),(16,24,33));draw=ImageDraw.Draw(sheet)
    for row,(card,images) in enumerate(best):
        for col,(image,title) in enumerate(zip(images,['RGB','A: Pi3 + SAM','B: mesh + instances','orange=A / cyan=B / green=both'])):
            x,y=col*448,row*485
            draw.text((x+8,y+7),f"{card['zone']} Dorm{card['epoch']+2} frame{card['frame']} | {title}",fill='white')
            sheet.paste(Image.fromarray(image).resize((448,448)),(x,y+29))
    sheet.save(root/'overview.jpg',quality=94)
    lines=['# SceneDiff 입력 경로 A/B 비교','','동일 RGB에서 원본 Pi3+SAM 경로와 기존 메쉬+인스턴스 경로를 비교했다.',
           '두 경로 모두 원본 후속 변화 후보 탐지·객체 병합·회차 간 대응을 실행했다. 기존 ID에 점수만 부여하는 실험과 다르다.','',
           '| 구역 | 원본 최종 후보 | 메쉬 최종 후보 | IoU≥0.5 대응 | 후보 픽셀 IoU |','|---|---:|---:|---:|---:|']
    for z in zones:lines.append(f"| {z['zone']} | {z['candidates']['original']} | {z['candidates']['mesh']} | {z['candidates']['common']} | {100*z['pixels']['iou']:.1f}% |")
    lines+=['',f'통합 후보 합계: 원본 {totals["original"]}, 메쉬 {totals["mesh"]}. 구역 간 중복 제거는 하지 않았다.','',
            '입력: 같은 224×224 RGB 30장(3구역×2촬영×5장). 원본 기본 loader는 최대518이므로 본 결과는 해상도를 통제한 파일럿이다.',
            'Dorm2와 Dorm3는 서로 다른 층이며 시간 변화 정답이 없다. 후보 감소가 정확도 향상을 뜻하지 않는다.',
            'DINOv3 H+/16 실제 가중치와 SceneDiff의 동일 비용 가중치·max-entropy 방식 사용. 자동 임계값의 수치는 입력별로 결정된다.',
            '메쉬 입력은 ray-cast depth, 기존 camera pose/intrinsics, 가림을 고려한 기존 SAM3.1 인스턴스 마스크이며 valid ray hit confidence=1.',
            '원본의 스케일·voxel 반환 규칙을 유지했다. 입력별 voxel_size와 자동 임계값은 summary.json에 기록했다.',
            '공통 호환 처리: 빈 영역 대응 보호, 빈/상수 임계값 입력 보호, 동일 voxel 평균의 스트리밍 계산, native PyTorch scatter_mean.',
            f'Upstream commit: {data["upstream_commit"]}','',
            '원본 출력 RLE, 마스크·비용·가시성 배열, 모델 SHA256, RGB 해시는 이 폴더의 JSON/NPZ에 보관했다.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(root/'validation.json',dict(paired_rgb_identical=True,local_rgb_hashes_verified=True,candidate_threshold_and_input_region_checks=True,views=len(cards),image_count=len(list(assets.glob('*.png'))),
          all_candidate_pixels_accounted=all(z['pixels']['common']+z['pixels']['original_only']+z['pixels']['mesh_only']==z['pixels']['union'] for z in zones)))
    print(json.dumps(dict(totals=totals,zones=[dict(zone=z['zone'],candidates=z['candidates'],pixels=z['pixels']) for z in zones]),ensure_ascii=False,indent=2))

if __name__=='__main__':main()

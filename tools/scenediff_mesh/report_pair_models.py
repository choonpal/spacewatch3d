"""Compare measured masks; do not equate 2D components with physical objects."""
from __future__ import annotations
import argparse,json,shutil,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFilter
from scipy.ndimage import label

COLORS=dict(scenediff_original=np.array([255,160,55]),scenediff_mesh=np.array([45,205,245]),viewdelta=np.array([90,225,135]),simsac=np.array([210,135,255]))
TITLES=dict(scenediff_original='SceneDiff · Pi3/SAM',scenediff_mesh='SceneDiff · 메쉬',viewdelta='ViewDelta',simsac='SimSaC')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
def overlay(rgb,mask,color):
    result=rgb.copy();result[mask]=(.48*result[mask]+.52*color).astype('u1')
    edge=np.array(Image.fromarray(mask.astype('u1')*255).filter(ImageFilter.FIND_EDGES))>0
    result[edge]=color
    return result
def stats(mask,reference):
    inter=int((mask&reference).sum());union=int((mask|reference).sum())
    components=0
    for image in mask:
        labels,n=label(image)
        sizes=np.bincount(labels.ravel())[1:]
        components+=int((sizes>=100).sum())
    return dict(pixels=int(mask.sum()),fraction=float(mask.mean()),intersection_pixels=inter,union_pixels=union,iou_with_scenediff_mesh=inter/union if union else None,components_100px=components)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('outputs/scenediff-mesh-rgb-20261024/model-comparison-20261025'))
    root=p.parse_args().root;root.mkdir(exist_ok=True)
    manifest=read(root/'inputs/manifest.json');rgb=np.load(root/'inputs/rgb.npz');views=manifest['views'];lookup={v['key']:v for v in views}
    images=root/'images';images.mkdir(exist_ok=True)
    probabilities={m:np.load(root/'results'/m/'common_view_probabilities.npz') for m in ['viewdelta','simsac']}
    erp_prob={m:np.load(root/'results'/m/'erp_probabilities.npz') for m in ['viewdelta','simsac']}
    models={m:dict(audit=read(root/'results'/m/'model.json'),status=read(root/'results'/m/'status.json'),control=read(root/'results'/m/'self-control.json')) for m in probabilities}
    assert all(x['status']['state']=='complete' and x['audit']['loaded_strict'] for x in models.values())
    scene={}
    old_inputs={z:np.load(root.parent/'prepared'/z/'inputs.npz') for z in ['R01','R07','R08']}
    for v in views:
        assert hashlib.sha256(rgb[v['key']].tobytes()).hexdigest()==v['rgb_sha256']
        assert np.array_equal(rgb[v['key']],old_inputs[v['zone']][f"rgb_{v['index']}"])
    for mode in ['original','mesh']:
        scene['scenediff_'+mode]={z:np.load(root.parent/'paired/results'/z/mode/'maps.npz') for z in ['R01','R07','R08']}
    data=dict(views=[],modes={},models=models,pairing=manifest['pairing'],scenediff_baseline='Existing original and mesh SceneDiff outputs on the same 30 perspective views are fixed in both input-mode comparisons. SceneDiff was not rerun directly on full ERP.',
        conditions='30 common views from 20 full ERP frames, two different floors, no temporal change ground truth. All new-model probability thresholds fixed at 0.5 before inference.',
        comparison='Candidate pixel agreement, not precision/recall or object detection accuracy. SceneDiff merged IDs differ from per-image connected components.')
    for v in views:
        key=v['key'];Image.fromarray(rgb[key]).resize((448,448)).save(images/(key+'-rgb.jpg'),quality=94)
        Image.fromarray(rgb[v['erp']]).save(images/(key+'-erp-rgb.jpg'),quality=92)
        first=lookup[v['refs'][0]]
        Image.fromarray(rgb[first['erp']]).save(images/(key+'-erp-reference.jpg'),quality=92)
        entry=dict(v,stats={})
        for m in ['viewdelta','simsac']:
            prob=erp_prob[m][key].astype('f4')
            Image.fromarray(overlay(rgb[v['erp']],prob>.5,COLORS[m])).save(images/(key+'-erp-full-'+m+'.jpg'),quality=94)
        data['views'].append(entry)
    for mode in ['erp','view']:
        masks={m:[] for m in COLORS}
        for v,entry in zip(views,data['views']):
            key=v['key'];local={}
            for method,color in COLORS.items():
                if method.startswith('scenediff'):
                    mask=scene[method][v['zone']]['labels'][v['index']]>=0
                else:
                    prob=probabilities[method][mode+'__'+key]
                    assert prob.shape==(224,224) and np.isfinite(prob).all()
                    mask=prob>.5
                local[method]=mask;masks[method].append(mask)
                Image.fromarray(overlay(rgb[key],mask,color)).resize((448,448)).save(images/(key+'-'+mode+'-'+method+'.jpg'),quality=95)
            entry['stats'][mode]={m:stats(local[m][None],local['scenediff_mesh'][None]) for m in COLORS}
            entry['stats'][mode]['disagreement']=int((local['viewdelta']^local['scenediff_mesh']).sum()+(local['simsac']^local['scenediff_mesh']).sum())
        masks={m:np.stack(a) for m,a in masks.items()}
        reference=masks['scenediff_mesh']
        data['modes'][mode]=dict(total={m:stats(a,reference) for m,a in masks.items()},zones={})
        for zone in ['R01','R07','R08']:
            ind=[i for i,v in enumerate(views) if v['zone']==zone]
            data['modes'][mode]['zones'][zone]={m:stats(a[ind],reference[ind]) for m,a in masks.items()}
    save(root/'comparison.json',data)
    text=json.dumps(data,ensure_ascii=False).replace('</','<\\/')
    template='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>파노라마 변화 후보 · ViewDelta / SimSaC / SceneDiff</title>
<style>*{box-sizing:border-box}body{margin:0;background:#101821;color:#eaf2fa;font:15px system-ui,sans-serif}main{max-width:1800px;margin:auto;padding:25px}h1{font-size:26px;margin:0 0 12px}p{line-height:1.65;color:#b6c9db}section,article{margin:16px 0;padding:18px;border:1px solid #34475b;background:#192634;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}.pano{display:grid;grid-template-columns:1fr 1fr;gap:14px}.tile img{width:100%;border-radius:7px}.tile b{display:block;margin:5px 0 10px;font-size:14px}small,.caption{font-size:13px;color:#afc3d5;line-height:1.6}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:9px;border-bottom:1px solid #35495c}select{background:#2a4157;color:white;border:1px solid #627a8e;border-radius:6px;padding:8px;margin:4px 12px 4px 0}.stats{display:flex;gap:12px;flex-wrap:wrap}.stats span{background:#22384c;padding:13px;border-radius:8px;flex:1;min-width:140px}.stats b{display:block;font-size:24px;margin-top:5px}a{color:#8bceff}.note{border-left:4px solid #627e97;padding-left:12px}.controls{position:sticky;top:0;z-index:2;box-shadow:0 7px 18px #101821}details{margin-top:15px}summary{cursor:pointer;color:#96cef7}h3{font-size:16px;margin:0 0 14px}.tablewrap{overflow:auto}@media(max-width:1100px){.grid{grid-template-columns:repeat(3,1fr)}}@media(max-width:700px){main{padding:13px}.grid,.pano{grid-template-columns:1fr 1fr}.controls{position:static}h1{font-size:22px}}</style>
<main><h1>우리 파노라마의 변화 후보: ViewDelta · SimSaC · SceneDiff</h1><p>공개 학습 가중치 실제 추론 · 같은 장소 대응 · SceneDiff와 동일한 30개 방향에서 비교</p>
<section><strong>두 입력 방식을 따로 실행했습니다.</strong><p><b>전체 파노라마:</b> 원본 ERP 프레임 전체(1024×512)를 모델에 입력하고, 결과를 기존 30개 뷰에 투영합니다.<br><b>동일 방향 뷰:</b> SceneDiff가 사용한 224×224 RGB 바이트를 그대로 사용합니다.</p><p class="note">ViewDelta는 내부에서 256×256과 DINOv2 전처리, SimSaC는 520×520 / 256×256 두 입력을 사용합니다. 두 모델은 구면 ERP 전용이므로 실행한 것이 아니라 일반 모델의 파노라마 적용 실험입니다. 서로 다른 층을 비교한 후보이며 변화 정답이 없어 정확도 순위를 매길 수 없습니다.</p></section>
<section class="controls"><label>입력 <select aria-label="입력" id="mode"><option value="erp">전체 파노라마 → 결과 투영</option><option value="view">SceneDiff와 동일 방향 뷰</option></select></label><label>구역 <select aria-label="구역" id="zone"><option value="">전체</option><option>R01</option><option>R07</option><option>R08</option></select></label><label>촬영 <select aria-label="촬영" id="epoch"><option value="">전체</option><option value="0">Dorm2</option><option value="1">Dorm3</option></select></label><label>정렬 <select aria-label="정렬" id="sort"><option value="difference">SceneDiff와 차이가 큰 순</option><option value="input">입력 순서</option></select></label><small id="count"></small></section>
<div class="stats" id="stats"></div><section class="tablewrap"><table><thead><tr><th>구역</th><th>SceneDiff Pi3/SAM</th><th>SceneDiff 메쉬</th><th>ViewDelta</th><th>SimSaC</th></tr></thead><tbody id="summary"></tbody></table><p><small>각 셀: 후보 픽셀 비율 / SceneDiff 메쉬 후보와의 IoU. 수치가 높다고 정답을 더 잘 찾았다는 뜻은 아닙니다. SceneDiff는 기존 비용·영역·객체 병합 결과, 두 새 모델은 비교 상대별 변화 확률 평균 &gt; 0.5입니다. 모델 간 영역 크기와 후처리가 다릅니다.</small></p></section>
<section><b>동일 프레임을 자기 자신과 비교한 대조</b><p id="controls"></p><small>자기 비교에서도 후보가 나오면 해당 입력·모델의 불필요한 반응을 확인할 단서입니다. 이 수치를 보고 임계값을 재조정하지 않았습니다.</small></section><div id="views"></div>
<section><details><summary>CYWS와 동영상 두 회차 비교 조사</summary><p>CYWS는 원본 DINOv1·U-Net 구조와 검출 가중치를 전량 복원하면 ERP 기하를 연결해 시험할 경로가 있습니다. 현재 CYWS-360의 기본 DINOv3·SCSE 구조에 원본 가중치를 일부 읽히는 것은 학습 완료 모델 실행이 아닙니다. 원본 point-splat renderer인 PyTorch3D도 필요합니다.</p><p>VSCDNet(ICML 2026)은 비동기 실내 영상 두 개의 정렬·다중 reference 변화 탐지 코드가 있지만 학습 완료 변화 head 가중치를 확인하지 못했습니다. O-SCD(CVPR 2026)는 공개 SAM2.1을 사용하지만 첫 회차의 reference 3DGS와 테스트 시 변화장 최적화가 필요합니다. 둘 다 전체 ERP 동영상 전용이라는 근거는 없습니다.</p><a href="research.md">공식 출처·가중치·CYWS 호환성 조사</a></details><details><summary>재현·실행 자료</summary><p>RGB 20개 ERP 프레임에서 30개 방향 뷰를 사용했습니다. 장소 대응은 변경 점수를 사용하지 않고 메쉬 입력 SceneDiff의 원본 view-overlap 코드로 고정했습니다. 모든 비영 대응을 사용하며 ERP의 중복 프레임 쌍은 한 번만 추론합니다. 새 모델의 reference별 확률을 평균한 다음 0.5로 후보를 결정합니다.</p><a href="comparison.json">전체 비교 JSON</a> · <a href="inputs/manifest.json">입력·대응</a> · <a href="results/viewdelta/model.json">ViewDelta 가중치 검사</a> · <a href="results/simsac/model.json">SimSaC 가중치 검사</a> · <a href="REPORT.md">실험 설명</a> · <a href="../paired/index.html">앞선 SceneDiff 입력 비교</a></details></section></main>
<script>const d=__DATA__,$=s=>document.querySelector(s),titles=__TITLES__,colors={scenediff_original:'#ffa037',scenediff_mesh:'#2dcdf5',viewdelta:'#5ae187',simsac:'#d287ff'},methods=Object.keys(titles);function fmt(v){return v==null?'—':(v*100).toFixed(1)+'%'}function draw(){const mode=$('#mode').value,z=$('#zone').value,e=$('#epoch').value;const stats=z?d.modes[mode].zones[z]:d.modes[mode].total;$('#stats').innerHTML=methods.map(m=>`<span style="color:${colors[m]}">${titles[m]} 候보 픽셀<b>${fmt(stats[m].fraction)}</b></span>`).join('');$('#summary').innerHTML=Object.entries(d.modes[mode].zones).filter(([k])=>!z||z===k).map(([k,s])=>`<tr><td>${k}</td>${methods.map(m=>`<td style="color:${colors[m]}">${fmt(s[m].fraction)} / ${fmt(s[m].iou_with_scenediff_mesh)}</td>`).join('')}</tr>`).join('');$('#controls').innerHTML=['viewdelta','simsac'].map(m=>`${titles[m]}: 후보 픽셀 ${fmt(d.models[m].control[mode].candidate_fraction)}`).join(' · ');let rows=d.views.filter(v=>(!z||v.zone===z)&&(!e||v.epoch==e));if($('#sort').value==='difference')rows.sort((a,b)=>b.stats[mode].disagreement-a.stats[mode].disagreement);$('#count').textContent=rows.length+'장';$('#views').innerHTML=rows.map(v=>`<article><h3>${v.zone} · Dorm${v.epoch+2} · frame ${v.frame} · yaw ${v.yaw}°</h3><div class="grid"><div class="tile"><b>동일 관측 RGB</b><img loading="lazy" src="images/${v.key}-rgb.jpg"><div class="caption">공통 비교 좌표 · 224×224</div></div>${methods.map(m=>`<div class="tile"><b style="color:${colors[m]}">${titles[m]}</b><img loading="lazy" src="images/${v.key}-${mode}-${m}.jpg"><div class="caption">후보 ${fmt(v.stats[mode][m].fraction)} · 메쉬 SceneDiff와 IoU ${fmt(v.stats[mode][m].iou_with_scenediff_mesh)}</div></div>`).join('')}</div><details><summary>전체 파노라마의 후보와 비교 상대 보기</summary><p><small>ERP 입력 결과입니다. 비교 상대 ${v.refs.length}개 방향(중복 프레임 제외 후 추론). 모든 상대: ${v.refs.join(', ')}. 아래 상대 영상은 그중 첫 번째 예시입니다.</small></p><div class="pano">${[['erp-rgb','전체 ERP 입력'],['erp-reference','비교 상대 ERP 예시'],['erp-full-viewdelta','전체 ERP · ViewDelta 후보'],['erp-full-simsac','전체 ERP · SimSaC 후보']].map(([suffix,title])=>`<div class="tile"><b>${title}</b><img loading="lazy" src="images/${v.key}-${suffix}.jpg"></div>`).join('')}</div></details></article>`).join('')}for(const id of ['mode','zone','epoch','sort'])$('#'+id).onchange=draw;draw();</script></html>'''
    aggregate='''function aggregate(rows,mode){return Object.fromEntries(methods.map(m=>{const values=rows.map(v=>v.stats[mode][m]),pixels=values.reduce((s,v)=>s+v.pixels,0),inter=values.reduce((s,v)=>s+v.intersection_pixels,0),union=values.reduce((s,v)=>s+v.union_pixels,0);return [m,{pixels,fraction:pixels/Math.max(1,rows.length*224*224),iou_with_scenediff_mesh:union?inter/union:null}]}))}'''
    template=template.replace('<strong>두 입력 방식을 따로 실행했습니다.</strong>','<strong>ViewDelta·SimSaC를 두 입력 방식으로 실행했습니다.</strong><p>SceneDiff 두 결과는 기존 30개 방향 뷰의 출력을 고정 비교군으로 사용합니다.</p>')
    template=template.replace('function draw(){',aggregate+'function draw(){')
    template=template.replace("const stats=z?d.modes[mode].zones[z]:d.modes[mode].total;","const selected=d.views.filter(v=>(!z||v.zone===z)&&(!e||v.epoch==e));const stats=aggregate(selected,mode);")
    template=template.replace("Object.entries(d.modes[mode].zones).filter(([k])=>!z||z===k)","Object.entries(Object.fromEntries([...new Set(selected.map(v=>v.zone))].map(k=>[k,aggregate(selected.filter(v=>v.zone===k),mode)])))")
    template=template.replace("let rows=d.views.filter(v=>(!z||v.zone===z)&&(!e||v.epoch==e));","let rows=selected.slice();")
    template=template.replace("${fmt(stats[m].fraction)}</b></span>","${fmt(stats[m].fraction)}</b><small>${stats[m].pixels.toLocaleString()} px</small></span>")
    template=template.replace('候보','후보').replace('두 모델은 구면 ERP 전용이므로 실행한 것이 아니라 일반 모델의 파노라마 적용 실험입니다.','일반 이미지용 모델을 파노라마에 적용한 실험입니다.').replace('__DATA__',text).replace('__TITLES__',json.dumps(TITLES,ensure_ascii=False))
    (root/'index.html').write_text(template,encoding='utf-8')
    best=[]
    for zone in ['R01','R07','R08']:
        best.append(max((v for v in data['views'] if v['zone']==zone),key=lambda v:v['stats']['erp']['disagreement']))
    for preview_mode in ['erp','view']:
        sheet=Image.new('RGB',(5*320,3*350),(16,24,33));draw=ImageDraw.Draw(sheet)
        for row,v in enumerate(best):
            for col,method in enumerate(['rgb',*COLORS]):
                title='RGB' if method=='rgb' else TITLES[method].replace(' · 메쉬',' mesh').replace(' · Pi3/SAM',' Pi3/SAM')
                filename=f"{v['key']}-rgb.jpg" if method=='rgb' else f"{v['key']}-{preview_mode}-{method}.jpg"
                x,y=col*320,row*350
                draw.text((x+5,y+7),f"{preview_mode} {v['zone']} D{v['epoch']+2} f{v['frame']} | {title}",fill='white')
                sheet.paste(Image.open(images/filename).resize((320,320)),(x,y+27))
        sheet.save(root/f'overview-{preview_mode}.jpg',quality=95)
    shutil.copy2(root/'overview-erp.jpg',root/'overview.jpg')
    shutil.copy2('docs/360-video-change-comparison-20261025.md',root/'research.md')
    lines=['# ViewDelta·SimSaC·SceneDiff 실제 후보 비교','','파노라마 전체 입력과 동일 방향 뷰 입력을 별도로 실행했다. 두 새 모델은 원본 공개 가중치를 strict=True로 로드했다.','',
           'SceneDiff 후보 수(통합 ID)와 새 모델의 2D 연결요소 수는 의미가 다르므로 같은 객체 수로 나란히 평가하지 않는다.','',
           '## 전체 ERP 입력을 공통 뷰에 투영한 후보 픽셀 비율','', '| 구역 | SceneDiff 원본 | SceneDiff 메쉬 | ViewDelta | SimSaC |','|---|---:|---:|---:|---:|']
    for zone,s in data['modes']['erp']['zones'].items():lines.append('| '+zone+' | '+' | '.join(f"{100*s[m]['fraction']:.2f}%" for m in COLORS)+' |')
    lines+=['','## 새 모델 입력 방식에 따른 공통 30뷰의 후보 영역','',
            'SceneDiff는 ERP 전체로 다시 추론한 결과가 아니다. 기존 30개 방향 뷰의 원본 후보9.03%, 메쉬 후보10.66%를 두 입력 방식 모두의 고정 비교군으로 사용했다.','',
            '| 모델 | ERP 전체 입력 | 동일 방향 뷰 입력 |','|---|---:|---:|']
    for method in ['viewdelta','simsac']:
        erp=data['modes']['erp']['total'][method];view=data['modes']['view']['total'][method]
        lines.append(f"| {TITLES[method]} | {100*erp['fraction']:.4f}% ({erp['pixels']:,}px) | {100*view['fraction']:.2f}% ({view['pixels']:,}px) |")
    lines+=['','## 관찰과 실행 범위','',
            'ERP 전체 입력에서 ViewDelta는 공통 30뷰에 6픽셀만 후보로 남겼다. 전체 ERP의 다른 방향에 후보가 없었다는 뜻은 아니다.',
            '동일 방향 뷰 입력에서는 ViewDelta와 SimSaC가 R07의 촬영자 머리를 후보로 선택한 장면도 확인됐다. 후보를 곧바로 가구·시설물 재배치로 확정하지 않았다.',
            'CYWS는 이 실행 비교에 포함하지 않았다. 원본 가중치 재사용 조건과 현재 환경의 PyTorch3D 부재는 research.md에 따로 기록했다.',
            '','## 조건','',
            '- 동일 SceneDiff 30뷰 / 대응 ERP 20프레임. 기존 인스턴스 파일럿 구간에 한정한다.',
            '- ERP 전체는1024×512. ViewDelta의 원본256×256 전처리와SimSaC520×520/256×256 두 해상도를 사용한다. 구면 전용 구조로 바꾸지 않았다.',
            '- 방향 뷰 대조에서는 기존SceneDiff의224×224 RGB 바이트를 그대로 사용한다.',
            '- 기존메쉬SceneDiff의view-overlap 코드로상대를고정. 모든nonzero상대의예측확률을평균하고0.5로판정한다.',
            '- 후보마스크를기존객체ID로사전제한하지않았다. 두모델은pixelmask를직접예측한다.',
            '- 후보픽셀IoU는모델간동의정도다. 서로다른층이며변화정답이없어성능개선율이나정확도로해석하지않는다.',
            '- ViewDelta의DeepSpeed는사용하지않는import만제외했고, SimSaC의CUDA/CuPycorrelation은같은9×9/mean연산을nativeTorch로실행했다.',
            '- 실제가중치SHA256·sourcecommit·strictload·동일프레임대조·모든쌍예측확률을보존한다.',
            '', '## 동일 입력 대조', '',json.dumps({m:x['control'] for m,x in models.items()},ensure_ascii=False,indent=2)]
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(root/'validation.json',dict(model_checkpoints_loaded_strict=True,views=len(views),images=len(list(images.glob('*.jpg'))),modes=['erp','view'],source_view_hashes_preserved=True))
    print(json.dumps({mode:obj['total'] for mode,obj in data['modes'].items()},indent=2))

if __name__=='__main__':main()

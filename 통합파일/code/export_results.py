#!/usr/bin/env python3
"""Export a finished v5.1 run as an offline report + meshes/plans/render images."""
import argparse,hashlib,html,json,shutil
from pathlib import Path

def read(path):return json.loads(Path(path).read_text())
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024**2),b''):h.update(part)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--zip',action='store_true');a=p.parse_args()
    state=read(a.run/'status.json')
    if state['status']!='complete':raise ValueError('Finish the pipeline before exporting')
    result=read(a.run/'results.json');validation=read(result['validate']['report'])
    if validation['status']!='passed':raise ValueError('Validation has not passed')
    a.output.mkdir(parents=True,exist_ok=False);files=[]
    def copy(source,relative):
        source=Path(source);target=a.output/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        if digest(source)!=digest(target):raise ValueError('Export copy mismatch: '+relative)
        files.append(dict(file=relative,sha256=digest(target),bytes=target.stat().st_size))
    for name in ['status.json','results.json','config.json','identity.json']:copy(a.run/name,name)
    copy(result['validate']['report'],'validation.json')
    if 'pair.changes' in result:copy(result['pair.changes']['changes'],'changes.json')
    parts=['<h1>SpaceWatch v5.1 실행 결과</h1><p>모든 메시 면을 보존한 라벨+크기 분할, 키프레임 선정, 실제 3DGS 렌더 결과입니다. 처리 완료와 모든 관측 목표 달성은 별개입니다.</p>', '<p>개별 필수 관측 대상은 크기 조건을 통과한 후보입니다. 나머지 면도 가림·충돌·표면 관측에 포함됩니다. 물리적 객체 정답·미터 스케일·렌더 화질은 미검증입니다.</p><p><a href="validation.json">검증 JSON</a> · <a href="config.json">실행 설정</a></p>']
    if 'pair.changes' in result:
        change=result['pair.changes'];parts.append(f'<p>회차 정합: {html.escape(change["alignment_status"])} · 변화 후보 {change["candidates"]} / 판단 보류 {change["deferred"]} · <a href="changes.json">변화 결과</a></p>')
    for row in validation['scenes']:
        name=row['capture'];inst=result[name+'.instances'];plan=result[name+'.plan'];render=result[name+'.render']
        for key in ['arrays','regions','assumptions','instance_mesh','instance_ply']:
            source=Path(inst[key]);copy(source,name+'/'+source.name)
        for source in Path(inst['arrays']).parent.glob('*_regions.json'):copy(source,name+'/'+source.name)
        copy(Path(inst['arrays']).with_name('summary.json'),name+'/segmentation_summary.json')
        copy(plan['plan'],name+'/tour_plan.json');copy(plan['requests'],name+'/render_requests.json');copy(render['renders'],name+'/renders.json')
        parts.append(f'<section><h2>{html.escape(name)}</h2><p>후보 {row["candidate_count"]}개 · 독립 검증에서 관측 {row["observed_candidates_both_holdouts"]}개 · 키프레임 {row["keyframes"]}개 · 전체 표면 최소 {row["whole_surface_coverage_min"]:.2%}</p><p>계획 {html.escape(row["planning_status"])} · 연결 성분 {row["selected_connected_components"]}개 · 보존 {row["total_faces"]:,}면</p>')
        parts.append(f'<p><a href="{name}/mesh_instances.glb">후보 메시 GLB</a> · <a href="{name}/mesh_arrays.npz">전체 면/비교 결과 NPZ</a> · <a href="{name}/regions.json">전체 영역</a> · <a href="{name}/size_assumptions.json">크기 가정</a> · <a href="{name}/tour_plan.json">계획과 좌표</a></p><div class="grid">')
        for image in read(render['renders'])['results']:
            relative=name+'/images/'+Path(image['image_ref']).name;copy(image['image_ref'],relative)
            parts.append(f'<figure><a href="{relative}"><img loading="lazy" src="{relative}" alt="{html.escape(image["viewpoint_id"])} 3DGS panorama"></a><figcaption>{html.escape(image["viewpoint_id"])}</figcaption></figure>')
        parts.append('</div></section>')
    style='body{font:16px system-ui;background:#111c20;color:#eef4ee;max-width:1280px;margin:32px auto;padding:0 24px}p{line-height:1.65;color:#c1d1ce}a{color:#b5e4c5}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:18px}figure{margin:0}img{width:100%;border-radius:8px}figcaption{padding:8px}section{margin:40px 0}'
    (a.output/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SpaceWatch v5.1 결과</title><style>'+style+'</style>'+''.join(parts)+'</html>')
    files.append(dict(file='index.html',sha256=digest(a.output/'index.html'),bytes=(a.output/'index.html').stat().st_size))
    (a.output/'EXPORT_MANIFEST.json').write_text(json.dumps(dict(source_run=str(a.run.resolve()),files=files),indent=2)+'\n')
    if a.zip:print(shutil.make_archive(str(a.output),'zip',a.output.parent,a.output.name))
    print(a.output/'index.html')

if __name__=='__main__':main()

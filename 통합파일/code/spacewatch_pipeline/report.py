import html
import os
from pathlib import Path
from .io import read,save,result

def publish(job):
    if job['config'].get('media',{}).get('mode')=='original_erp':
        from .capture519_report import publish
        return publish(job)
    root=Path(job['run_root']);out=Path(job['output']);deps=job['dependencies'];cards=[];summaries=[]
    for c in job['config']['captures']:
        prefix=c['id'];p=deps[prefix+'.plan'];g=deps[prefix+'.gs_train'];r=deps[prefix+'.render']
        rows=read(r['renders'])['results']
        title=html.escape(prefix)
        check=next(r for r in deps['validate']['scenes'] if r['capture']==prefix)
        cards.append(f'<p>{title}: 후보 {check["candidate_count"]}개 / 독립 검증 관측 {check["observed_candidates_both_holdouts"]}개 / 전체 표면 {check["whole_surface_coverage_min"]:.1%} / 키프레임 {check["keyframes"]}개</p>')
        cards.append(f'<h2>{title}</h2><p>계획 상태: {html.escape(p["planning_status"])} · 3DGS {g["steps"]:,}회 학습 · 검증 PSNR {g["validation_psnr"]:.2f} dB</p>')
        cards.append(f'<p>전체입력 관측 목표: {html.escape(p.get("input_coverage_status","unknown"))} · 선정 완료와 전체 관측 목표 충족은 다릅니다.</p>')
        for row in rows:
            src=html.escape(os.path.relpath(row['image_ref'],out))
            cards.append(f'<figure><a href="{src}"><img loading="lazy" src="{src}"></a><figcaption>{html.escape(row["viewpoint_id"])} · α coverage {row["alpha_coverage"]:.1%}</figcaption></figure>')
        summaries.append(dict(capture=prefix,planning=p,training=g,rendering=r))
    change=deps.get('pair.changes')
    if change:cards.insert(0,f'<p>변화 의심 {change["candidates"]}구역 · 판단 보류 {change["deferred"]}구역 · 정합 {html.escape(change["alignment_status"])}</p>')
    doc='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SpaceWatch 통합 결과</title>
<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 24px;background:#101820;color:#eee}img{width:100%;border-radius:8px}figure{margin:24px 0}a{color:#7dd3fc}p,figcaption{color:#b8c5d1}</style>
<h1>SpaceWatch v5 통합 결과</h1><p>관측 대상과 개별 필수 관측 여부는 각 입력의 설정을 따릅니다. 잔여 영역·미분류도 삭제하지 않고 가림·충돌·전체 표면 관측에 포함합니다. 실행 complete는 모든 객체 관측이나 경로 연결 완료를 뜻하지 않습니다.</p><p>키프레임 위치에서 3DGS로 생성한 360° 이미지입니다. 관측 목표 충족 여부와 이미지 품질 검증은 별개입니다. 학습 영상 밖의 위치에는 흐림이나 빈 영역이 생길 수 있습니다.</p>'''+''.join(cards)+'</html>'
    (out/'index.html').write_text(doc)
    summary=dict(captures=summaries,changes=change,execution_complete=True,
                 pipeline_version=job['config'].get('pipeline_version','1.1'),discovery=deps.get('discovery'),validation=deps.get('validate'),
                 all_planning_complete=all(s['planning']['planning_status']=='complete' for s in summaries),
                 image_quality_validated=False,report=str(out/'index.html'))
    save(out/'summary.json',summary)
    result(out,**summary)

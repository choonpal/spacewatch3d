"""Native-media report; no fabricated Gaussian training statistics."""
import html
import os
from pathlib import Path
from .io import read,save,result


def publish(job):
    out=Path(job['output']);deps=job['dependencies'];summaries=[];cards=[]
    for capture in job['config']['captures']:
        prefix=capture['id'];p=deps[prefix+'.plan'];r=deps[prefix+'.render']
        check=next(row for row in deps['validate']['scenes'] if row['capture']==prefix)
        summaries.append(dict(capture=prefix,planning=p,rendering=r,training=None))
        cards.append(f'<h2>{html.escape(prefix)}</h2><p>키프레임 {check["keyframes"]}개 / 독립 검증 관측 {check["observed_candidates_both_holdouts"]}/{check["candidate_count"]} / 전체 표면 {check["whole_surface_coverage_min"]:.1%}</p><p>계획 {html.escape(p["planning_status"])} / 전체입력 목표 {html.escape(p["input_coverage_status"])}</p>')
        for row in read(r['renders'])['results']:
            src=html.escape(os.path.relpath(row['image_ref'],out))
            cards.append(f'<p><a href="{src}">{html.escape(row["viewpoint_id"])}</a> · 원본 영상 프레임 {row["source_frame"]["source_frame_index"]}</p>')
    doc='<!doctype html><html lang="ko"><meta charset="utf-8"><title>SpaceWatch 원본 촬영 프레임</title><h1>5.1.9 기반 촬영 프레임 결과</h1><p>Gaussian 없음. 원본 영상 픽셀을 보존합니다. 보간 자세·물리 안전·의미 정답은 검증 완료가 아닙니다.</p>'+''.join(cards)+'</html>'
    (out/'index.html').write_text(doc)
    value=dict(captures=summaries,changes=deps.get('pair.changes'),execution_complete=True,
        pipeline_version=job['config']['pipeline_version'],discovery=deps.get('discovery'),validation=deps['validate'],
        all_planning_complete=all(s['planning']['planning_status']=='complete' for s in summaries),
        image_quality_validated=False,image_origin='original_video_frame',report=str(out/'index.html'))
    save(out/'summary.json',value);result(out,**value)

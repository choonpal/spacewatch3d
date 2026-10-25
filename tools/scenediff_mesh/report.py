"""Build a local, self-contained comparison viewer from measured results."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/scenediff-mesh-rgb-20261024"))
    args = parser.parse_args()
    root = args.root
    data, prepared = read(root / "results.json"), read(root / "prepared/manifest.json")
    proofs = root / "proofs"
    proofs.mkdir(exist_ok=True)
    rows = sorted(data["objects"], key=lambda r: (not (r["combined_candidate"] != r["geometry_candidate"]), not r["combined_candidate"], r.get("zone") or "ZZ", r["id"]))
    for row in rows:
        if not row["views"]:
            row["proof"] = None
            continue
        best = max(row["views"], key=lambda v: (v["supported"], v["visible_pixels"], v["pixels"]))
        folder = root / "prepared" / row["zone"]
        meta, arrays = read(folder / "manifest.json"), np.load(folder / "inputs.npz")
        index = best["prepared_index"]
        view = meta["views"][index]
        local = next(int(k) for k,v in view["region_to_input_instance"].items() if v == row["id"])
        rgb, mask = arrays[f"rgb_{index}"], arrays[f"mask_{index}"] == local
        overlay = rgb.copy()
        color = np.array([250, 165, 25] if row["combined_candidate"] else [30, 190, 220])
        overlay[mask] = (.55 * overlay[mask] + .45 * color).astype("u1")
        edge = np.asarray(Image.fromarray(mask.astype("u1") * 255).filter(ImageFilter.FIND_EDGES)) > 0
        overlay[edge] = color
        image = Image.fromarray(overlay).resize((336, 336))
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 310, 335, 335], fill=(12, 20, 30))
        draw.text((7, 316), f"{row['id']} | frame {best['map_frame']} | {best['yaw_degrees']} deg", fill="white")
        target = proofs / f"{row['id']}.jpg"
        image.save(target, quality=94)
        row["proof"] = f"proofs/{target.name}"
        row["representative_view"] = best
    matches = {}
    for pair in sorted(data["matches"], key=lambda m: -m["similarity"]):
        matches.setdefault(pair["reference"], pair)
        matches.setdefault(pair["source"], pair)
    lookup = {r["id"]: r for r in rows}
    for row in rows:
        pair = matches.get(row["id"])
        if pair:
            oid = pair["source"] if pair["reference"] == row["id"] else pair["reference"]
            row["match"] = {**pair, "other": oid, "proof": lookup[oid].get("proof")}
    raw = json.dumps({"objects": rows, "counts": data["counts"], "zones": [{"zone": z["zone"], "thresholds": z["thresholds"]} for z in data["zones"]]}, ensure_ascii=False).replace("</", "<\\/")
    template = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>SceneDiff · 기존 메쉬 + RGB 비교</title>
<style>body{margin:0;background:#101722;color:#edf3fb;font:15px system-ui,sans-serif}main{max-width:1350px;margin:auto;padding:32px}h1{margin:0 0 12px;font-size:27px}p{line-height:1.7;color:#b7c6d9}a{color:#80c5ff}section,article{background:#182331;border:1px solid #2a3b50;border-radius:13px;padding:18px;margin:16px 0}.stats{display:flex;gap:12px;flex-wrap:wrap}.stats b{font-size:25px;display:block}.stats span{flex:1;min-width:140px;background:#1c2b3e;padding:16px;border-radius:10px}select,label{margin-right:14px}select{background:#263a52;color:white;padding:8px;border:0;border-radius:6px}.grid{display:grid;grid-template-columns:260px 1fr 1fr;gap:16px}img{max-width:100%;border-radius:8px}small{color:#acbdd1}strong.yes{color:#ffbc58}strong.no{color:#77cce3}.tag{border:1px solid #526578;padding:3px 7px;border-radius:5px;display:inline-block;margin-bottom:8px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:7px;text-align:left;border-bottom:1px solid #304357}details{margin-top:15px}pre{white-space:pre-wrap;word-break:break-word;color:#a6bad1}@media(max-width:850px){.grid{grid-template-columns:1fr 1fr}.info{grid-column:1/-1}}</style>
<main><h1>기존 객체 메쉬·인스턴스 + RGB로 SceneDiff 비교</h1>
<p>기존 SAM3.1의 객체 ID와 메쉬를 카메라로 투영해 Pi3와 SAM 입력을 대체했습니다. DINOv3 ViT-H+/16의 실제 특징과 SceneDiff 원본의 기하·외형·영역 대응 비용을 사용했습니다. 기하 단독과 RGB 결합은 동일 입력·정합·관측 조건을 사용합니다.</p>
<section><strong>자료 범위</strong><p>Dorm2와 Dorm3는 서로 다른 층입니다. 이 결과는 공간의 객체 차이 후보이며 시간에 따른 이동·추가·삭제의 정답 평가가 아닙니다. 기존 인스턴스 결과는 회차별 16초 파일럿입니다. 가림, 부분 메쉬, 대응 불확실성은 남습니다. 점수는 확률이 아니며 자동 임계값은 구역·방식별로 다릅니다.</p><a href="results.json">전체 측정 JSON</a> · <a href="self-comparison-control.json">동일 입력 대조</a> · <a href="model.json">모델·가중치</a> · <a href="prepared/manifest.json">입력·기하 검증</a></section>
<div class="stats" id="stats"></div><section><label>구역 <select id="zone"><option value="">전체</option></select></label><label>결과 <select id="filter"><option value="all">전체 객체</option><option value="different">RGB로 후보 여부가 달라진 객체</option><option value="combined">RGB 결합 후보</option><option value="compared">관측 충분</option><option value="unknown">관측 부족 / 메쉬 없음</option></select></label><small id="count"></small></section><div id="items"></div>
<script>const data=__DATA__;const $=s=>document.querySelector(s),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const fmt=x=>x==null?'—':x.toFixed(4);let zones=Object.fromEntries(data.zones.map(z=>[z.zone,z]));
const stats=[['입력 ID',data.counts.input_instances],['관측 충분',data.counts.compared],['기하 후보',data.counts.geometry_candidates],['RGB 결합 후보',data.counts.combined_candidates],['RGB로 추가 / 제외',`${data.counts.rgb_promoted} / ${data.counts.rgb_demoted}`]];$('#stats').innerHTML=stats.map(([k,v])=>`<span>${k}<b>${v}</b></span>`).join(''); [...new Set(data.objects.map(o=>o.zone).filter(Boolean))].sort().forEach(z=>$('#zone').add(new Option(z,z)));
function draw(){const z=$('#zone').value,f=$('#filter').value,rows=data.objects.filter(o=>(!z||o.zone===z)&&(f==='all'||f==='different'&&o.geometry_candidate!==o.combined_candidate||f==='combined'&&o.combined_candidate||f==='compared'&&o.status==='compared'||f==='unknown'&&o.status!=='compared'));$('#count').textContent=`${rows.length}개`;$('#items').innerHTML=rows.map(o=>{const t=zones[o.zone]?.thresholds||{};return `<article id="${esc(o.id)}"><div class="grid"><div class="info"><span class="tag">${esc(o.zone||'구역 없음')} · Dorm${o.epoch+2}</span><h3>${esc(o.id)}</h3><p>${esc(o.label)} · ${o.faces}면<br>유효 관측 ${o.supported_views}개<br>${o.status==='compared'?'비교 가능':esc(o.status)}</p><table><tr><th></th><th>점수 / 기준</th></tr><tr><td>기하 단독</td><td><strong class="${o.geometry_candidate?'yes':'no'}">${o.geometry_candidate?'후보':'보류/기준 이하'}</strong><br>${fmt(o.geometry_score)} / ${fmt(t.geometry)}</td></tr><tr><td>RGB 결합</td><td><strong class="${o.combined_candidate?'yes':'no'}">${o.combined_candidate?'후보':'보류/기준 이하'}</strong><br>${fmt(o.combined_score)} / ${fmt(t.combined)}</td></tr></table></div><div>${o.proof?`<img src="${esc(o.proof)}"><p><small>원본 RGB + 해당 메쉬 인스턴스의 보이는 면</small></p>`:'<p>유효한 메쉬/촬영 증거 없음</p>'}</div><div>${o.match?.proof?`<img src="${esc(o.match.proof)}"><p><a href="#${esc(o.match.other)}">${esc(o.match.other)}</a><br><small>RGB 특징 대응 가설: ${fmt(o.match.similarity)} · 차선과 차이 ${fmt(o.match.margin)}<br>${o.match.supported?'대응 기준 통과':'반복 외형 / 대응 모호'} · 물리적 동일성·이동 확정 아님</small></p>`:'<p>다른 회차의 대응 가설 없음<br><small>이 사실만으로 추가·삭제로 판정하지 않습니다.</small></p>'}</div></div><details><summary>관측별 수치</summary><pre>${esc(JSON.stringify(o.views,null,2))}</pre></details></article>`}).join('')};$('#zone').onchange=draw;$('#filter').onchange=draw;draw();</script></main></html>'''
    (root / "index.html").write_text(template.replace("__DATA__", raw), encoding="utf-8")
    fields = ["id", "epoch", "zone", "label", "faces", "status", "supported_views", "geometry_score", "combined_score", "appearance_score", "region_match_score", "geometry_candidate", "combined_candidate"]
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    selected = [r for r in rows if r.get("proof") and (r["geometry_candidate"] != r["combined_candidate"])]
    if len(selected) < 6:
        selected += [r for r in rows if r.get("proof") and r not in selected][:6-len(selected)]
    selected = selected[:8]
    sheet = Image.new("RGB", (336*4, 376*max(1, (len(selected)+3)//4)), (16, 23, 34))
    draw = ImageDraw.Draw(sheet)
    for n, row in enumerate(selected):
        x, y = n % 4 * 336, n // 4 * 376
        sheet.paste(Image.open(root / row["proof"]), (x, y))
        draw.text((x+6, y+341), f"{row['label']} | geo={row['geometry_candidate']} RGB={row['combined_candidate']}", fill="white")
        draw.text((x+6, y+358), f"scores {row.get('geometry_score')} / {row.get('combined_score')}", fill=(170,190,210))
    sheet.save(root / "overview.jpg", quality=94)
    assert len(list(proofs.glob("*.jpg"))) == sum(bool(r.get("proof")) for r in rows)
    (root / "report-validation.json").write_text(json.dumps({"all_input_ids_retained": len(rows)==prepared["objects"], "proof_images": len(list(proofs.glob('*.jpg'))), "pi3_inference": data["pi3_inference"], "sam_inference": data["sam_inference"], "counts": data["counts"]}, indent=2), encoding="utf-8")
    print(json.dumps(data["counts"], indent=2))


if __name__ == "__main__":
    main()

"""Attach reviewed original-photo pairs and a frame-based alignment diagnostic.

Visual notes are explanations only. They never edit the geometric decisions.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_dorm_surface_changes import decode, read, write
from dorm_surface_change import compare_surfaces


def panel(root, views, output, title):
    image = Image.new('RGB', (1100, 314), '#111923'); draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
    draw.text((12, 9), title, font=font, fill='white')
    for e, view in enumerate(views):
        original = Image.open(root/view['file']).convert('RGB'); w, h = original.size
        x0, y0, x1, y1 = view['bbox_px']; cx, cy = (x0+x1)/2, (y0+y1)/2
        width = max(270, min(480, (x1-x0)*2.3)); height = max(170, min(260, (y1-y0)*1.8))
        left = int(cx-width/2) % w; top = int(max(0, min(h-height, cy-height/2)))
        tiled = Image.new('RGB', (w*3, h))
        for j in range(3): tiled.paste(original, (j*w, 0))
        crop = tiled.crop((left+w, top, left+w+int(width), top+int(height)))
        box_x = (x0-left) % w
        if box_x > width: box_x -= w
        ImageDraw.Draw(crop).rectangle((box_x, y0-top, box_x+x1-x0, y1-top), outline=('#ffac70' if e==0 else '#70caff'), width=2)
        crop = crop.resize((535, 246)); image.paste(crop, (10+e*550, 61))
        draw.text((12+e*550, 35), f'Dorm{e+2} | frame {view["frame"]} | {view["time_s"]:.1f}s', font=font, fill='white')
    image.save(output, quality=94)


def frame_diagnostic(before, after, old, audit):
    item = next(x for x in old['events'] if x['id']=='P0012')
    surfaces = [decode((before/old['object_meshes'][mid]['file']).read_bytes()) for mid in item['meshes']]
    planes = sorted([p for p in audit['materials']['panes'] if p['zone']=='R05' and p.get('material')!='static_wall'], key=lambda p:p['epoch'])
    x, y = np.mean(np.concatenate([p for p, _ in surfaces]), axis=0)[:2]
    z = [(-p['offset']-p['normal'][0]*x-p['normal'][1]*y)/p['normal'][2] for p in planes]
    delta = z[1]-z[0]; corrected = surfaces[1][0]-[0, 0, delta]
    raw = []; moved = []
    for e in [0, 1]:
        raw.append(compare_surfaces(*surfaces[e], *surfaces[1-e])[0].ravel())
    moved.append(compare_surfaces(*surfaces[0], corrected, surfaces[1][1])[0].ravel())
    moved.append(compare_surfaces(corrected, surfaces[1][1], *surfaces[0])[0].ravel())
    return {'id':'P0012','zone':'R05','frame_offset_u':float(abs(delta)),
            'raw_surface_difference_u':float(np.median(np.concatenate(raw))),
            'after_frame_correction_u':float(np.median(np.concatenate(moved))),
            'threshold_u':item['threshold_u'],'map_modified':False,'object_fit':False,
            'method':'창면이 아닌 창틀·벽 테두리에서 평면을 추정한 국소 진단. 그 기준면 차이만 보정해 창면을 다시 비교했습니다.',
            'interpretation':'주변 구조까지 함께 어긋나 있으며, 실제 객체 변화보다 국소 정합·재구성 오차가 주원인으로 보입니다. 전역 정합 오차와 복원 오차를 완전히 분해한 결과는 아닙니다.'}


def main(before, after, notes, output):
    if output.exists(): raise ValueError('Choose a new output folder')
    shutil.copytree(after, output)
    old = read(before/'tour.json'); data = read(after/'tour.json'); audit = read(after/'surface-audit.json'); review = read(notes)
    current = {r['id']:r for r in data['events']+data['records']}; gallery = []
    for note in review['items']:
        item = next(r for r in old['events'] if r['id']==note['id'])
        pair = [item['views'][note['pair_indices'][e]][e] for e in [0,1]]
        name = f'review-{item["id"]}.jpg'; panel(before, pair, output/name, f'R08 / {item["id"]} | distance {item["distance_p50_u"]:.4f} u | threshold {item["threshold_u"]:.4f} u')
        for v in pair:
            if not (output/v['file']).exists(): shutil.copy2(before/v['file'], output/v['file'])
        detail = {**note, 'ground_truth':False,'affects_detector':False,
                  'file':name,'views':pair,'original_distance_u':item['distance_p50_u'],
                  'map_error_u':item['map_error_u'],'threshold_u':item['threshold_u'],
                  'original_ratio':item['threshold_ratio'],'original_triangles':item['triangles'],
                  'own_view_counts':item['own_view_counts'],'free_view_counts':item['opposite_free_view_counts']}
        if item['id'] in current: current[item['id']]['photo_review'] = detail
        gallery.append(detail)
    diagnostic = frame_diagnostic(before, after, old, audit)
    item = next(r for r in old['events'] if r['id']=='P0012')
    name = 'review-P0012.jpg'; panel(before, item['views'][0], output/name, 'R05 / P0012 | window frame diagnostic')
    diagnostic['file'] = name
    if 'P0012' in current: current['P0012']['registration_diagnostic'] = diagnostic
    data['review_gallery'] = gallery; data['registration_diagnostics'] = [diagnostic]
    write(output/'tour.json', data)
    write(output/'photo-review-audit.json', {'review':review,'diagnostic':diagnostic,'geometric_decisions_modified':False})
    print(json.dumps({'gallery':len(gallery),'diagnostic':diagnostic,'summary':data['summary']}, ensure_ascii=False), flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ['before','after','notes','output']: parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();main(args.before,args.after,args.notes,args.output)

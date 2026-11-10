"""Link every unchanged event to actual evidence photos and nearby CAMERA stops."""
import numpy as np


def build_catalog(events,captures,recorded,overlay):
    output=[]
    for event in events['events']:
        pairs=[];targets={};shown=[]
        for pair in event.get('photo_sets',[]):
            row={}
            for side,p in pair.items():
                image=p['image']
                if image.startswith('/') or '..' in image.split('/'):raise ValueError('Unsafe source photo')
                row[side]={k:p[k] for k in ['capture_id','timestamp_s','source_frame_index','width','height','role']}
                row[side]['image']='/pipeline-assets/keypoint-current/'+image
                row[side]['source_image_sha256']=p['image_sha256']
                box=p.get('bbox');w=p['width'];h=p['height']
                row[side]['bbox']=[max(0.,min(w,float(box[0]))),max(0.,min(h,float(box[1]))),
                    max(0.,min(w,float(box[2]))),max(0.,min(h,float(box[3])))] if box else None
            if row:pairs.append(row)
        if not pairs:raise ValueError('Event has no actual evidence photos')
        for cap in captures:
            cid=cap['id'];order=cap['tour_order'];candidates={v['id']:v for v in cap['views']};choices=[]
            for row in overlay['captures'][cid]['views']:
                if any(i['id']==event['id'] for i in row['items']):
                    shown.append(dict(capture=cid,viewpoint_id=row['viewpoint_id'],keypoint=order.index(row['viewpoint_id'])+1))
            for pair in pairs:
                for photo in pair.values():
                    if photo['capture_id']!=cid:continue
                    anchor=recorded[cid].get(photo['source_frame_index'])
                    if anchor is None:raise ValueError('Evidence photo has no exact recorded camera pose')
                    position=np.asarray(anchor['reference_from_camera'])[:3,3]
                    for identifier in order:
                        v=candidates[identifier];distance=float(np.linalg.norm(position-np.asarray(v['position'])))
                        choices.append((photo['role']!='observed_object',distance,identifier,photo))
            if not choices:continue
            _,gap,identifier,photo=min(choices,key=lambda r:(r[0],r[1],r[2]))
            targets[cid]=dict(capture=cid,viewpoint_id=identifier,keypoint=order.index(identifier)+1,
                camera_gap_m=gap,evidence_source_frame_index=photo['source_frame_index'],
                relation='Nearest selected CAMERA to exact evidence CAMERA; NOT object position')
        if not targets:raise ValueError('Event is not reachable from any capture')
        note=('3D형상이 없는 RGB 후보입니다. 아래 실제 검출 사진의 박스를 확인합니다. 현재 키포인트 사진에 3D박스를 만들어 붙이지 않습니다.'
              if event.get('rgb_only') else '일부 키포인트에서 가시성·거리·화면 크기를 만족하는 표면 박스가 보입니다.' if shown
              else '선정된 키포인트 사진에서는 표시 조건을 만족하지 못했습니다. 아래 원래 검출 사진에서 후보를 확인할 수 있습니다.')
        output.append(dict(id=event['id'],label=event['label'],type=event['type'],rgb_only=bool(event.get('rgb_only')),
            pairs=pairs,targets=targets,projected_at=shown,display_note=note))
    return dict(schema='source-change-evidence-catalog-v1',events=output,
        interpretation='Twelve source candidates, not twelve confirmed events; camera links are not object localization')

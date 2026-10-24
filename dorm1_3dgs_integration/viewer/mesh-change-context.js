// Event types and photo associations provide context, not detector outputs.
export const CHANGE_TYPES = {
  added: {label:'추가', color:[.35,.75,1.], css:'#59bfff'},
  removed: {label:'소실', color:[1.,.57,.28], css:'#ff9147'},
  moved: {label:'이동', color:[.75,.58,1.], css:'#bf94ff'},
  suspected: {label:'유형 미정', color:[.96,.76,.32], css:'#f4c65c'},
};
export const CHANGE_SOURCES={m3c2_independent:'의심 구역',existing:'기존 Rescan 비교'};
export const CHANGE_FILTERS=['all','added','removed','moved'];
const originalPhotoUrl=/^\/(?:rescan|m3c2-independent|datasets\/bonn-removal)\/frame-[01]-\d{6}\.jpg$/;
export function regionName(event) {
  return event.source==='m3c2_independent'?event.name.replace(/^M3C2 (?:의심 구역|추가|소실|이동) /,'구역 '):event.name;
}
export function evidenceSummary(event) {
  const evidence=event.classification_evidence;
  if(evidence.noise_filter)return '표면 차이 '+(evidence.noise_filter.median_distance_m*100).toFixed(1)+
    'cm · 요철·정합 오차 범위 '+(evidence.noise_filter.median_envelope_m*100).toFixed(1)+
    'cm · 다중 크기 기준 충족 '+(evidence.noise_filter.persistent_fraction*100).toFixed(1)+'% (필요 55%)';
  return '거리 차이 '+(evidence.m3c2_median_distance_m*100).toFixed(1)+'cm · 변화 기준 충족 표본 '+
    (evidence.m3c2_significant_fraction*100).toFixed(1)+'% (종류 판단 기준 55%)';
}
export function contextPhotoCandidates(reference, epoch) {
  const frames=reference.keyframes.filter(frame=>frame.epoch===epoch&&
    ['estimated_rgb_checked','estimated_rgb_tracked','dataset_pose'].includes(frame.status)&&
    originalPhotoUrl.test(frame.url));
  return frames.slice(0,1).map(frame=>({frame,link:{relation:'scene_context',bbox_px:null}}));
}
export function displayBounds(event, epoch, origin, padding=0) {
  const bounds=(epoch===0?event.before_bounds:event.after_bounds)||event.bounds;
  if(!bounds?.min||!bounds?.max||origin.length!==3)throw new Error('선택 위치의 좌표가 없습니다.');
  return {min:bounds.min.map((v,i)=>v-origin[i]-padding),max:bounds.max.map((v,i)=>v-origin[i]+padding)};
}
export function photoCandidates(reference, event, epoch) {
  const frames=new Map(reference.keyframes.map(frame=>[frame.id,frame])),used=new Set();
  return (event.keyframes?.[epoch]||[]).flatMap(link=>{
    const frame=frames.get(link.frame_id);
    if(!frame||frame.epoch!==epoch||used.has(frame.id)||!originalPhotoUrl.test(frame.url))return [];
    used.add(frame.id);return [{frame,link}];
  });
}
export function photoTitle(event, epoch) {
  if(!event)return epoch===0?'이전 원본 · 공간 참고':'이후 원본 · 공간 참고';
  if(event.type==='suspected')return epoch===0?'이전 원본 · 같은 위치':'이후 원본 · 같은 위치';
  if(event.type==='added')return epoch===0?'추가 전 · 같은 위치':'추가 후 · 대상 위치';
  if(event.type==='removed')return epoch===0?'소실 전 · 대상 위치':'소실 후 · 같은 위치';
  return epoch===0?'이동 전':'이동 후';
}
export function validateReference(comparison, reference) {
  if(!['rescan_object_changes','rgbd_reconstruction_pair'].includes(reference.kind)||!comparison.scene.includes(reference.scene)||
      !comparison.scene.includes(reference.source_epochs.join(' → ')))throw new Error('메쉬와 원본 사진의 촬영 쌍이 다릅니다.');
  if(reference.kind==='rgbd_reconstruction_pair'&&reference.dataset_id!==comparison.dataset_id)throw new Error('메쉬와 사진의 데이터셋이 다릅니다.');
  return reference.regions.filter(event=>CHANGE_TYPES[event.type]&&event.bounds&&event.keyframes);
}
export function mergeIndependent(comparison, reference, independent) {
  if(independent.kind!=='m3c2_independent_changes'||independent.scene!==reference.scene||
      independent.source_epochs.join(',')!==reference.source_epochs.join(',')||
      independent.label_independent!==true||independent.classification_uses_instance_ids!==false||
      JSON.stringify(independent.source_hashes)!==JSON.stringify(comparison.input_hashes))throw new Error('독립 탐지 결과의 촬영 쌍 또는 입력 메쉬가 다릅니다.');
  const oldFrames=new Set(reference.regions.flatMap(event=>event.keyframes.flat().map(link=>link.frame_id)));
  const oldIds=new Set(reference.regions.map(event=>event.id));
  for(const event of independent.regions){
    if(oldIds.has(event.id)||event.source!=='m3c2_independent'||!CHANGE_TYPES[event.type]||
        event.keyframes.flat().some(link=>oldFrames.has(link.frame_id)))throw new Error('독립 탐지에 기존 항목 또는 원본 프레임이 중복되었습니다.');
  }
  const keyframes=[...new Map([...reference.keyframes,...independent.keyframes].map(frame=>[frame.id,frame])).values()];
  return {reference:{...reference,keyframes},events:[...reference.regions.map(event=>({...event,source:'existing'})),...independent.regions]};
}
export function filteredEvents(events, source, type='all') {
  return events.filter(event=>event.source===source&&(type==='all'||event.type===type));
}
export function surfaceAssessment(bytes, bounds, method, regionIds=null, candidateId=0) {
  const view=new DataView(bytes),areas=[0,0,0,0],values=[],offset=method==='m3c2'?28:44;
  for(let base=0;base+180<=bytes.byteLength;base+=180){
    const p=[0,60,120].map(step=>[0,4,8].map(k=>view.getFloat32(base+step+k,true)));
    const ab=p[1].map((v,i)=>v-p[0][i]),ac=p[2].map((v,i)=>v-p[0][i]);
    const area=Math.hypot(ab[1]*ac[2]-ab[2]*ac[1],ab[2]*ac[0]-ab[0]*ac[2],ab[0]*ac[1]-ab[1]*ac[0])/6;
    for(let i=0;i<3;i++){
      if(candidateId&&regionIds&&regionIds[base/60+i]!==candidateId)continue;
      if(!p[i].every((v,k)=>v>=bounds.min[k]&&v<=bounds.max[k]))continue;
      const field=base+i*60+offset,category=Math.round(view.getFloat32(field+8,true));
      if(category<0||category>3)continue;
      areas[category]+=area;
      if(category!==3)values.push(method==='m3c2'?Math.abs(view.getFloat32(field+4,true)):view.getFloat32(field,true));
    }
  }
  const total=areas.reduce((sum,v)=>sum+v,0),fraction=k=>total?areas[k]/total:0;
  values.sort((a,b)=>a-b);
  const median=values.length?values[Math.floor(values.length/2)]:null;
  const status=!total||fraction(3)>.7?'사진 확인 필요':fraction(2)>=.12?'변화 감지':fraction(1)+fraction(2)>=.1?'변화 의심':'표면 차이 작음';
  return {status,changed:fraction(2),suspected:fraction(1),unsupported:fraction(3),median};
}

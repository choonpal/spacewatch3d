const TYPES={possible_relocation:'이동',possible_added:'추가',possible_removed:'소실'};
export const CLASS_NAMES={chair:'의자','swivel chair':'의자',table:'테이블','dining table':'테이블',cabinet:'수납장',door:'문',windowpane:'창문',blind:'블라인드',refrigerator:'냉장고·정수기',lamp:'조명',light:'조명',sofa:'소파',couch:'소파',bench:'벤치',shelf:'선반',television:'TV',tv:'TV',painting:'액자',curtain:'커튼',plant:'화분',screen:'화면',board:'게시판'};
export function className(value){return CLASS_NAMES[value]||value;}
export function decisionLabel(item){
  if(item.decision==='deferred')return '판단 보류';
  if(item.decision!=='candidate')return '후보 제외';
  const labels=Object.keys(TYPES).filter(k=>item.alternatives.includes(k)).map(k=>TYPES[k]);
  return labels.length?labels.join('·')+' 가능':'배치 차이 후보';
}
export function itemsForFilter(data,filter){
  if(filter==='deferred')return data.records.filter(r=>r.decision==='deferred');
  if(filter==='excluded')return data.records.filter(r=>r.decision==='suppressed');
  const alternative={moved:'possible_relocation',added:'possible_added',removed:'possible_removed'}[filter];
  return data.events.filter(e=>!alternative||e.alternatives.includes(alternative));
}
export function validateDormTour(data){
  if(data.kind!=='dorm_change_tour'||data.units!=='reference_units'||!Array.isArray(data.zones)||data.zones.length!==13||data.maps?.length!==2)throw new Error('Dorm 맵 자료 형식이 다릅니다.');
  if(!Array.isArray(data.events)||!Array.isArray(data.records))throw new Error('Dorm 변화 결과가 없습니다.');
  const zones=new Set(data.zones.map(z=>z.id)),ids=new Set();
  for(const item of [...data.events,...data.records]){
    if(ids.has(item.id)||!zones.has(item.zone)||item.confirmed!==false||!Array.isArray(item.views))throw new Error('변화 항목 연결이 올바르지 않습니다.');
    ids.add(item.id);
    for(const pair of item.views)for(const v of pair)if(v&&(!/^Dorm[23]-\d{5}\.jpg$/.test(v.file)||v.width!==1036||v.height!==518))throw new Error('원본 사진 연결 형식이 다릅니다.');
    for(const id of item.meshes)if(id&&!data.object_meshes[id])throw new Error('객체 메시 연결이 없습니다.');
  }
  if(data.records.filter(r=>r.decision==='deferred').length!==data.summary.record_dispositions.deferred)throw new Error('판단 보류 항목 수가 원본 결과와 다릅니다.');
  return data;
}
export function decodeIndexedMesh(bytes){
  if(bytes.byteLength<8)throw new Error('메시가 비어 있습니다.');
  const [nv,nf]=new Uint32Array(bytes,0,2);
  if(bytes.byteLength!==8+nv*24+nf*12||nv>2000000||nf>4000000)throw new Error('메시 자료 크기가 다릅니다.');
  const positions=new Float32Array(bytes,8,nv*3),normals=new Float32Array(bytes,8+nv*12,nv*3),indices=new Uint32Array(bytes,8+nv*24,nf*3);
  if(indices.some(i=>i>=nv))throw new Error('메시 정점 연결이 올바르지 않습니다.');
  return {positions,normals,indices,nv,nf};
}
export function combineBounds(values){
  const valid=values.filter(Boolean);return [0,1].map(side=>[0,1,2].map(k=>(side?Math.max:Math.min)(...valid.map(b=>b[side][k]))));
}
export function panoramaView(box,width=1036,height=518,aspect=1.7){
  if(!box)return {yaw:0,pitch:0,fov:90};
  const [x0,y0,x1,y1]=box,w=(x1-x0)/width*360,h=(y1-y0)/height*180;
  return {yaw:((x0+x1)/2/width-.5)*360,pitch:(.5-(y0+y1)/2/height)*180,
    fov:Math.max(35,Math.min(100,Math.max(h,w/Math.max(.5,aspect))*1.65))};
}

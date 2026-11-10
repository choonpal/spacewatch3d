// Color only the original faces named by the detector. No counterpart is inferred.
export const patchPalette={candidate:[64,185,172],deferred:[202,138,58]};
export function patchColors(data,events,epoch,selected,mode){
  if(mode==='rgb')return data.rgb;
  const colors=new Uint8Array(data.rgb.length);
  for(let i=0;i<colors.length;i++)colors[i]=Math.round(data.rgb[i]*.42+20);
  const matching=events.filter(e=>e.epoch===epoch&&(mode==='all'||e.id===selected?.id));
  for(const event of matching){
    const tint=event.id===selected?.id?(event.status==='candidate'?[143,255,229]:[255,224,160]):patchPalette[event.status];
    if(!tint)throw Error('지원하지 않는 표면 판정입니다.');
    for(const face of event.face_ids){
      if(!Number.isInteger(face)||face<0||face>=data.count/3)throw Error('구역 면 번호가 메시 범위를 벗어났습니다.');
      for(let vertex=0;vertex<3;vertex++)colors.set(tint,face*9+vertex*3);
    }
  }
  return colors;
}
export function boxSegments(bounds,mesh,width,height){
  const basis=mesh.basis(),center=mesh.center,radius=mesh.radius;
  const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
  const corners=Array.from({length:8},(_,i)=>{
    const p=[bounds[i&1?1:0][0],bounds[i&4?1:0][2],-bounds[i&2?1:0][1]].map((v,j)=>v-center[j]);
    return [width/2+dot(p,basis.right)*height/(2*radius),height/2-dot(p,basis.up)*height/(2*radius)];
  });
  const lines=[];
  for(let i=0;i<8;i++)for(const bit of [1,2,4])if(!(i&bit))lines.push([corners[i],corners[i|bit]]);
  return lines;
}
export function validateComparison(data){
  if(data.schema_version!==1||data.scope!=='surface_regions_not_object_events'||data.no_object_identity_claim!==true||!Array.isArray(data.events))throw Error('지원하지 않는 변화 결과입니다.');
  if(data.available&&(data.maps?.length!==2||!data.bounds||!data.alignment.baseline_from_current))throw Error('비교 좌표 정보가 없습니다.');
  if(!data.available&&(data.events.length||data.maps?.length))throw Error('비교 불가 결과에 메시 또는 구역이 포함되어 있습니다.');
  return data;
}

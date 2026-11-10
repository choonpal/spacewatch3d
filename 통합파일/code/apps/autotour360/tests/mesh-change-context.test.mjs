import test from 'node:test';
import assert from 'node:assert/strict';
import {displayBounds,photoCandidates,photoTitle,validateReference,surfaceAssessment,mergeIndependent,filteredEvents,CHANGE_TYPES,CHANGE_SOURCES,CHANGE_FILTERS,regionName,evidenceSummary,contextPhotoCandidates} from '../mesh-change-context.js';

test('photo and mesh context use the same epoch and coordinate origin',()=>{
  const event={type:'added',bounds:{min:[10,20,30],max:[11,22,33]},before_bounds:null,after_bounds:{min:[10,20,30],max:[11,22,33]}};
  assert.deepEqual(displayBounds(event,0,[1,2,3]),{min:[9,18,27],max:[10,20,30]});
  assert.deepEqual(displayBounds(event,0,[1,2,3]),displayBounds(event,1,[1,2,3]));
  assert.equal(photoTitle(event,0),'추가 전 · 같은 위치');
  assert.equal(photoTitle({type:'removed'},1),'소실 후 · 같은 위치');
  assert.equal(event.bounds.min[0],10);
});

test('photo links never borrow a frame from another epoch or an external URL',()=>{
  const reference={keyframes:[{id:'a',epoch:0,url:'/rescan/frame-0-000010.jpg'},{id:'b',epoch:1,url:'/rescan/frame-1-000020.jpg'},
    {id:'c',epoch:0,url:'https://example.com/frame.jpg'}]};
  const event={keyframes:[[{frame_id:'b'},{frame_id:'a'},{frame_id:'a'},{frame_id:'missing'},{frame_id:'c'}],[]]};
  assert.deepEqual(photoCandidates(reference,event,0).map(item=>item.frame.id),['a']);
});

test('an unrelated capture pair is rejected instead of showing misleading photos',()=>{
  const reference={kind:'rescan_object_changes',scene:'scene_e',source_epochs:[0,2],regions:[]};
  assert.throws(()=>validateReference({scene:'Rescan scene_i · 0 → 2'},reference),/촬영 쌍/);
  assert.throws(()=>validateReference({scene:'Rescan scene_e · 0 → 1'},reference),/촬영 쌍/);
});

function triangle(categories){
  const bytes=new ArrayBuffer(180),view=new DataView(bytes);
  for(let i=0;i<3;i++){
    const p=[[0,0,0],[1,0,0],[0,1,0]][i];
    p.forEach((v,k)=>view.setFloat32(i*60+k*4,v,true));
    for(const field of [28,44]){view.setFloat32(i*60+field,.2,true);view.setFloat32(i*60+field+4,.1,true);view.setFloat32(i*60+field+8,categories[i],true);}
  }
  return bytes;
}
const bounds={min:[-1,-1,-1],max:[2,2,2]};
test('surface evidence is measured in the selected box without classifying event types',()=>{
  const result=surfaceAssessment(triangle([2,2,1]),bounds,'m3c2');
  assert.equal(result.status,'변화 감지');assert.ok(Math.abs(result.changed-2/3)<1e-8);
  assert.ok(Math.abs(result.suspected-1/3)<1e-8);assert.equal('type' in result,false);
});
test('a missing counterpart requests photo review and does not claim disappearance',()=>{
  assert.equal(surfaceAssessment(triangle([3,3,3]),bounds,'gls').status,'사진 확인 필요');
  assert.equal(surfaceAssessment(triangle([0,0,0]),bounds,'gls').status,'표면 차이 작음');
});

test('independent suspects have their own source and use different original frames',()=>{
  const comparison={input_hashes:['a','b']};
  const reference={scene:'scene_e',source_epochs:[0,2],keyframes:[{id:'old',epoch:0}],
    regions:[{id:1,type:'added',keyframes:[[{frame_id:'old'}],[]]}]};
  const candidate={id:1001,type:'suspected',source:'m3c2_independent',keyframes:[[{frame_id:'new0'}],[{frame_id:'new1'}]]};
  const independent={kind:'m3c2_independent_changes',scene:'scene_e',source_epochs:[0,2],label_independent:true,
    classification_uses_instance_ids:false,source_hashes:['a','b'],regions:[candidate],keyframes:[{id:'new0',epoch:0},{id:'new1',epoch:1}]};
  const merged=mergeIndependent(comparison,reference,independent);
  assert.equal(filteredEvents(merged.events,'m3c2_independent','suspected').length,1);
  assert.equal(filteredEvents(merged.events,'m3c2_independent','added').length,0);
  assert.equal(filteredEvents(merged.events,'existing','added').length,1);
  assert.equal(merged.reference.keyframes.length,3);
  assert.equal(photoTitle(candidate,0),'이전 원본 · 같은 위치');
  assert.throws(()=>mergeIndependent(comparison,reference,{...independent,regions:[{...candidate,keyframes:[[{frame_id:'old'}],[]]}]}),/중복/);
  assert.throws(()=>mergeIndependent(comparison,reference,{...independent,classification_uses_instance_ids:true}),/입력 메쉬/);
});

test('independent surface metrics use the exact candidate footprint rather than the full box',()=>{
  const values=new Uint16Array([0,1,0]);
  const measured=surfaceAssessment(triangle([2,1,2]),bounds,'m3c2',values,1);
  assert.equal(measured.changed,0);assert.equal(measured.suspected,1);
});

test('suspected regions are a source with three change filters; weak floor evidence remains untyped',()=>{
  assert.equal(CHANGE_SOURCES.m3c2_independent,'의심 구역');
  assert.deepEqual(CHANGE_FILTERS,['all','added','removed','moved']);
  const floor={id:1002,name:'M3C2 의심 구역 02',type:'suspected',source:'m3c2_independent',classification_evidence:{
    m3c2_median_distance_m:.054025,m3c2_significant_fraction:8/49,m3c2_median_ratio:1.2988,
    epoch_observations:[{epoch:1,opposite:{free:0}}]}};
  assert.equal(CHANGE_TYPES[floor.type].label,'유형 미정');
  assert.equal(regionName(floor),'구역 02');
  assert.match(evidenceSummary(floor),/거리 차이 5\.4cm/);
  assert.match(evidenceSummary(floor),/16\.3%.*55%/);
  assert.equal(filteredEvents([floor],'m3c2_independent','added').length,0);
  assert.equal(filteredEvents([floor],'m3c2_independent','removed').length,0);
  assert.equal(filteredEvents([floor],'m3c2_independent','moved').length,0);
  assert.equal(regionName({name:'테이블 #9',type:'moved',source:'existing'}),'테이블 #9');
});

test('empty detections keep context originals separate from change evidence',()=>{
  const reference={keyframes:[{id:'wrong',epoch:0,status:'unlocalized',url:'/rescan/frame-0-000001.jpg'},
    {id:'before',epoch:0,status:'estimated_rgb_checked',url:'/rescan/frame-0-000010.jpg'},
    {id:'after',epoch:1,status:'estimated_rgb_checked',url:'/rescan/frame-1-000020.jpg'},
    {id:'external',epoch:1,status:'estimated_rgb_checked',url:'https://example.com/img.jpg'}]};
  assert.deepEqual(contextPhotoCandidates(reference,0).map(item=>item.frame.id),['before']);
  assert.deepEqual(contextPhotoCandidates(reference,1).map(item=>item.frame.id),['after']);
  assert.equal(contextPhotoCandidates(reference,0)[0].link.bbox_px,null);
  assert.equal(photoTitle(null,0),'이전 원본 · 공간 참고');
  assert.equal(photoTitle(null,1),'이후 원본 · 공간 참고');
  assert.equal(filteredEvents([{type:'added',source:'existing'}],'m3c2_independent').length,0);
});

test('screened scores show robust surface spread and persistence rather than nominal confidence',()=>{
  const text=evidenceSummary({classification_evidence:{noise_filter:{median_distance_m:.12,median_envelope_m:.08,persistent_fraction:.8}}});
  assert.match(text,/12\.0cm/);assert.match(text,/오차 범위 8\.0cm/);assert.match(text,/80\.0%/);
});

test('Bonn originals require the matching dataset and remain in the correct epoch',()=>{
  const comparison={scene:'Bonn box removal · 0 → 1',dataset_id:'bonn-removal'};
  const reference={kind:'rgbd_reconstruction_pair',scene:'Bonn box removal',dataset_id:'bonn-removal',source_epochs:[0,1],regions:[],
    keyframes:[{id:'b',epoch:0,url:'/datasets/bonn-removal/frame-0-000030.jpg',status:'dataset_pose'},
      {id:'a',epoch:1,url:'/datasets/bonn-removal/frame-1-000400.jpg',status:'dataset_pose'}]};
  assert.deepEqual(validateReference(comparison,reference),[]);
  assert.throws(()=>validateReference({...comparison,dataset_id:'other'},reference),/데이터셋/);
  assert.deepEqual(photoCandidates(reference,{keyframes:[[{frame_id:'b'},{frame_id:'a'}],[]]},0).map(p=>p.frame.id),['b']);
  assert.equal(contextPhotoCandidates(reference,1)[0].frame.id,'a');
});

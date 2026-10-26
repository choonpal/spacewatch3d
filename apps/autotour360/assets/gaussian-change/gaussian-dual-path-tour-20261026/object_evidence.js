import * as THREE from 'three';

const $=q=>document.querySelector(q);
const label=(parent,tag,text,className)=>{const e=document.createElement(tag);e.textContent=text;if(className)e.className=className;parent.append(e);return e;};

export function createObjectEvidence(scene,evidence,height){
 let selected=null,group=null,token=0;const cache=new Map();
 const stage=$('#stage'),overlay=document.createElement('div');overlay.className='movement-overlay';overlay.hidden=true;stage.append(overlay);
 overlay.innerHTML='<svg class="movement-arrow" viewBox="0 0 1000 1000" preserveAspectRatio="none" aria-hidden="true"><defs><linearGradient id="move-gradient" gradientUnits="userSpaceOnUse"><stop stop-color="#59dbf2"/><stop offset="1" stop-color="#ff9e45"/></linearGradient><marker id="move-head" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,4 L0,8 Z" fill="#ff9e45"/></marker></defs><line stroke="url(#move-gradient)" stroke-width="4" vector-effect="non-scaling-stroke" marker-end="url(#move-head)"/></svg><span class="movement-label movement-origin"></span><span class="movement-label movement-destination"></span><span class="movement-key"><b class="object-before">청록: 출발</b> → <b class="object-after">주황: 도착</b> · 위치 대응 표시</span>';
 const record=(box,mode)=>evidence.regions[mode][box.id];
 function focusBounds(box,mode){const e=record(box,mode)?.event;if(e?.type!=='moved')return null;return [...e.movement.origin_instances,e.movement.destination_instance].map(id=>evidence.objects[id].bounds);}
 function draw(camera){
  const movement=selected?.event.movement;overlay.hidden=!movement||!group||!$('#show-object-meshes').checked;if(overlay.hidden)return;
  const ids=[movement.origin_instances[0],movement.destination_instance];
  const points=ids.map(id=>{const obj=evidence.objects[id],p=new THREE.Vector3().fromArray(obj.center);p.y=obj.bounds[1][1]+height*.08;const local=camera.worldToLocal(p.clone()),v=p.project(camera);const behind=local.z>=0;if(behind){v.x=-v.x;v.y=-v.y;}const outside=behind||Math.abs(v.x)>.94||Math.abs(v.y)>.88;return{x:Math.max(6,Math.min(94,(v.x+1)*50)),y:Math.max(12,Math.min(82,(1-v.y)*50)),outside};});
  for(const[i,selector]of ['.movement-origin','.movement-destination'].entries()){const node=overlay.querySelector(selector),p=points[i];node.textContent=(i?movement.destination_label:movement.origin_label)+(p.outside?' · 시야 밖':'');node.style.left=p.x+'%';node.style.top=p.y+'%';}
  const line=overlay.querySelector('line');for(const[i,p]of points.entries()){line.setAttribute('x'+(i+1),p.x*10);line.setAttribute('y'+(i+1),p.y*10);}const gradient=overlay.querySelector('linearGradient');for(const[i,p]of points.entries()){gradient.setAttribute('x'+(i+1),p.x*10);gradient.setAttribute('y'+(i+1),p.y*10);}line.setAttribute('stroke-dasharray',movement.exact_origin_confirmed?'':'10 5');
 }
 function dispose(){if(!group)return;scene.remove(group);group.traverse(o=>{o.geometry?.dispose();if(o.material)for(const m of Array.isArray(o.material)?o.material:[o.material])m.dispose();});group=null;}
 async function updateMeshes(){
  const ticket=++token;dispose();if(!selected||!$('#show-object-meshes').checked)return;
  const record=selected,pairIds=(record.event.selected_pair||[]).filter(Boolean);
  const ids=new Set($('#show-object-context').checked||!pairIds.length?[...pairIds,...record.linked_objects.map(o=>o.id)]:pairIds);
  const loaded=await Promise.all([...ids].map(async id=>{
   const obj=evidence.objects[id];if(!cache.has(id))cache.set(id,fetch(obj.mesh_file).then(r=>{if(!r.ok)throw new Error('Object mesh HTTP '+r.status);return r.json();}));return{obj,mesh:await cache.get(id)};
  }));
  if(ticket!==token||selected!==record||!$('#show-object-meshes').checked)return;
  group=new THREE.Group();group.name='segmented-object-review';
  for(const{obj,mesh}of loaded){
   const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(mesh.positions,3));geometry.setIndex(mesh.indices);
   const color=obj.epoch?0xffbd66:0x59dbf2;
   const moving=record.event.type==='moved';
   const solid=new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({color,transparent:true,opacity:moving?(obj.epoch ? .72 : .36):.22,depthTest:false,depthWrite:false,side:THREE.DoubleSide}));solid.renderOrder=100;group.add(solid);
   const lines=new THREE.LineSegments(new THREE.EdgesGeometry(geometry,35),new THREE.LineBasicMaterial({color,transparent:true,opacity:.95,depthTest:false,depthWrite:false}));lines.renderOrder=101;group.add(lines);
   if(moving){const box=new THREE.Box3Helper(new THREE.Box3(new THREE.Vector3().fromArray(obj.bounds[0]),new THREE.Vector3().fromArray(obj.bounds[1])),color);box.material.depthTest=false;box.material.depthWrite=false;box.material.transparent=true;box.material.opacity=.65;box.renderOrder=102;group.add(box);}
  }
  scene.add(group);
 }
 function clear(){++token;selected=null;dispose();overlay.hidden=true;$('#object-evidence-panel').hidden=true;}
 function show(box,mode){
  selected=evidence.regions[mode][box.id];const root=$('#object-evidence-content');root.replaceChildren();$('#object-evidence-panel').hidden=false;
  const event=selected.event;
  if(event.type==='moved'){$('#show-object-meshes').checked=true;$('#toggle-object-meshes')?.setAttribute('aria-pressed','true');}
  label(root,'h2',`${event.label}${event.class_label?' · '+event.class_label:''}`,'event-title event-'+event.type);
  label(root,'p',event.reason);
  label(root,'p',event.type==='moved'?'청록 출발 영역과 주황 도착 객체를 색 그림자로 표시합니다. 화살표는 전→후 위치의 대응이며 실제 이동 궤적은 아닙니다.':'기본 겹침은 유형 판단에 사용한 대응 조각을 표시합니다. 연결된 조각 모두를 켜면 주변 분할 메쉬도 확인할 수 있습니다.','note');
  if(event.type==='added'||event.type==='removed')label(root,'p','다른 위치에서 이동해 왔거나 이동해 갔을 가능성도 남습니다.','evidence-caution');
  if(event.type==='moved'){
   const pair=event.automatic_match||selected.correspondence_proposals.find(p=>p.reference===event.selected_pair?.[0]&&p.source===event.selected_pair?.[1]);
   if(pair)label(root,'p',`${pair.reference} → ${pair.source} · 자동 1:1 대응 · 형상 F1 ${pair.shape_f1.toFixed(2)} · 중심 차이 ${pair.displacement_u.toFixed(2)} 모델 좌표 단위. 출발/도착의 반대 회차 빈 공간 관측 ${pair.origin_free_views} / ${pair.destination_vacancy_views}회.`,'note');
   const origin=evidence.objects[event.selected_pair?.[0]];
   if(origin?.automatic_partition)label(root,'p',`${origin.parent_instance}에서 영상 인스턴스 ${origin.image_track}의 3D 마스크 표본으로 원본 ${origin.faces}면을 자동 분리했습니다.`, 'note');
   if(!event.automatic_match&&selected.correspondence_proposals.some(p=>p.reference===pair?.reference&&p.source!==pair?.source&&p.shape_f1>=pair?.shape_f1))label(root,'p','같거나 더 높은 형상 점수의 다른 대응 제안이 있습니다. 아래 대응 후보를 함께 확인하세요.','evidence-caution');
  }
  const table=document.createElement('table');table.className='object-table';const head=document.createElement('tr');for(const t of ['회차','분할 객체','영역과 겹침','관측 근거'])label(head,'th',t);table.append(head);
  const linkedIds=new Set(selected.linked_objects.map(o=>o.id)),originIds=event.movement?.origin_instances||[];
  const shown=[...originIds.filter(id=>!linkedIds.has(id)).map(id=>evidence.objects[id]),...selected.linked_objects];
  for(const obj of shown){const row=document.createElement('tr');label(row,'td',obj.epoch?'후':'전',obj.epoch?'object-after':'object-before');label(row,'td',`${obj.class_label} · ${obj.id}`);label(row,'td',linkedIds.has(obj.id)?`${obj.intersecting_faces}면 / ${(obj.intersection_fraction*100).toFixed(0)}%`:'박스 밖 출발 객체 · '+obj.faces+'면');
   const frames=Math.max(0,...obj.tracks.map(t=>t.frames)),vacancy=Math.max(0,...obj.tracks.filter(t=>t.supported).map(t=>t.vacancy_views));label(row,'td',`다중 관측 면 ${(obj.multi_observation_fraction*100).toFixed(0)}%${frames?' · 영상 '+frames+'프레임':''}${vacancy?' · 반대 빈 공간 '+vacancy+'회':''}`);table.append(row);}
  if(selected.linked_objects.length)root.append(table);else label(root,'p','현재 변화 박스와 직접 겹치는 분할 객체가 없습니다.');
  if(selected.correspondence_proposals.length){
   const details=document.createElement('details');label(details,'summary',`기존 알고리즘의 형상 대응 제안 · 상위 ${Math.min(8,selected.correspondence_proposals.length)} / ${selected.correspondence_proposals.length}`);
   for(const p of selected.correspondence_proposals.slice(0,8))label(details,'p',`${p.reference} (${p.reference_zone}) → ${p.source} (${p.source_zone}) · F1 ${p.shape_f1.toFixed(2)}${p.already_paired_with_other?' · 다른 객체와 기존 대응됨':''}`);
   root.append(details);
  }
  label(root,'p','자동 분류 결과입니다. 합쳐진 메쉬는 기존 영상 인스턴스 트랙으로 자동 분리하며, 대응이 모호하거나 관측이 부족하면 유형 불확실로 남깁니다. 같은 물리 객체임을 보증하는 정답 판정은 아닙니다.','note');
  updateMeshes().catch(e=>{$('#error').textContent=e.stack||String(e);});
 }
 const update=()=>{$('#toggle-object-meshes')?.setAttribute('aria-pressed',String($('#show-object-meshes').checked));updateMeshes().catch(e=>{$('#error').textContent=e.stack||String(e);});};
 $('#show-object-meshes').addEventListener('change',update);
 $('#show-object-context').addEventListener('change',update);
 $('#toggle-object-meshes')?.addEventListener('click',()=>{$('#show-object-meshes').checked=!$('#show-object-meshes').checked;update();});
 return{show,clear,record,draw,focusBounds};
}

import * as THREE from 'three';
import {capturePhase} from './tour_navigation.js';
import {SparkRenderer,SplatMesh,SplatEdit,SplatEditSdf,SplatEditSdfType} from '@sparkjsdev/spark';

const embedded=new URLSearchParams(location.search).get('embed')==='1';
if(embedded)document.body.classList.add('embedded-tour');
let renderingEnabled=true;
const $=q=>document.querySelector(q),n=x=>x.toLocaleString('ko-KR'),names={baseline:'기존',simsac:'SimSaC'};
let data,prediction,frozenCandidates=null,objectEvidence=null,objectEvidenceView=null,changeRegions=null,regionHelpers=null,scene,camera,renderer,spark,tour=null,zone,phase='before',active=null,allBoxes=[],boxes=[],yaw=0,pitch=0,selectionToken=0;
const assets=new Map(),meshes=new Map(),edits=new Map(),boxNodes=new Map(),pointers=new Map();
let minimapWasOpen=null,listKey=null;
const matrix=rows=>new THREE.Matrix4().fromArray(rows.flat()).transpose();
const vector=a=>new THREE.Vector3().fromArray(a);
const separatePaths=()=>data?.decisions.tour_route_mode==='separate_capture_paths';
const defaultPhase=view=>capturePhase(view,separatePaths());
const routeContains=id=>!separatePaths()||!tour||tour.contains(id);
const methods=()=>$('#method').value==='both'?['baseline','simsac']:[$('#method').value];
const mode=()=>$('#exclude-person').checked?'filtered':'raw';
const eventRecord=b=>objectEvidence?.regions[mode()]?.[b.id];
const candidates=()=>[...data.baseline_boxes,...(frozenCandidates||prediction).boxes[mode()]].filter(b=>(b.method!=='simsac'||b.before_pixels+b.after_pixels>=Number($('#minimum-pixels')?.value||0))&&(!$('#event-type')||$('#event-type').value==='all'||eventRecord(b)?.event.type===$('#event-type').value));
const regionIndex=()=>changeRegions?.modes[mode()];
const eventKey=b=>regionIndex()?.candidate_to_event[b.id]||b.id;
const regionGroups=()=>regionHelpers?.visibleRegions(regionIndex(),candidates())||[];
const candidatesAt=id=>regionHelpers?regionHelpers.uniqueForView(candidates(),regionIndex(),frozenCandidates.visibility[id]?.[mode()]):candidates().filter(b=>frozenCandidates?frozenCandidates.visibility[id]?.[mode()]?.[b.id]?.visible:b.zone===id);
const overlaps=(a,b)=>[0,1,2].every(i=>a[0][i]<=b[1][i]&&a[1][i]>=b[0][i]);
function fail(error){$('#error').textContent=error?.stack||String(error);if(embedded)window.parent.postMessage({type:'gaussian-tour:state',error:error?.message||String(error)},location.origin);}
function counts(z){const list=candidatesAt(z);return{baseline:list.filter(b=>b.method==='baseline').length,simsac:list.filter(b=>b.method==='simsac').length};}
function refreshZones(){
 if(tour){tour.refresh();return;}
 const previous=$('#zone').value;const group=$('#plan-group')?.value;const zones=data.zones.filter(z=>(!group||group==='all'||z.plan_group===group)&&(!$('#only-candidates').checked||candidates().some(b=>b.zone===z.id&&methods().includes(b.method))));
 $('#zone').innerHTML=zones.map(z=>{const c=counts(z.id);return`<option value="${z.id}">${z.label||z.id} · ${data.viewpoint_mode?'후보 '+c.simsac:'기존 '+c.baseline+' / SimSaC '+c.simsac}</option>`;}).join('');
 if(zones.some(z=>z.id===previous))$('#zone').value=previous;else if(zones.some(z=>z.id==='R08'))$('#zone').value='R08';
 if(!zones.length){$('#summary').textContent='현재 선택한 방식에서 남은 후보 구역이 없습니다. 후보 있는 구역만을 끄면 맵을 둘러볼 수 있습니다.';boxes=[];renderBoxList();return;}
 showZone();
}
function updateDirection(){const y=THREE.MathUtils.degToRad(yaw),p=THREE.MathUtils.degToRad(pitch);camera.lookAt(camera.position.clone().add(new THREE.Vector3(Math.sin(y)*Math.cos(p),Math.sin(p),Math.cos(y)*Math.cos(p))));camera.updateMatrixWorld();}
function lookAt(point){camera.lookAt(vector(point));camera.updateMatrixWorld();const f=camera.getWorldDirection(new THREE.Vector3());yaw=THREE.MathUtils.radToDeg(Math.atan2(f.x,f.z));pitch=THREE.MathUtils.radToDeg(Math.asin(f.y));}
function frameMovement(bounds){
 // Rotate and widen the current keyframe view; never translate its camera.
 const directions=bounds.map(b=>vector(b[0]).add(vector(b[1])).multiplyScalar(.5).sub(camera.position).normalize());
 const direction=directions.reduce((sum,p)=>sum.add(p),new THREE.Vector3()).normalize();if(direction.lengthSq()<.01)return;
 lookAt(camera.position.clone().add(direction).toArray());let required=65;
 for(const p of bounds.flatMap(corners)){const local=camera.worldToLocal(p.clone());if(local.z>=-.01)continue;const vertical=Math.atan2(Math.abs(local.y),-local.z),horizontal=Math.atan2(Math.abs(local.x),-local.z);required=Math.max(required,THREE.MathUtils.radToDeg(2*vertical)*1.12,THREE.MathUtils.radToDeg(2*Math.atan(Math.tan(horizontal)/camera.aspect))*1.12);}
 camera.fov=Math.min(100,required);camera.updateProjectionMatrix();
}
function resetCamera(){if(!zone)return;camera.fov=65;camera.updateProjectionMatrix();if(zone.reset_from_pose){const gl=matrix(zone.camera_to_common_cv).multiply(new THREE.Matrix4().makeScale(1,-1,-1));gl.decompose(camera.position,camera.quaternion,new THREE.Vector3());camera.updateMatrixWorld(true);const f=camera.getWorldDirection(new THREE.Vector3());yaw=THREE.MathUtils.radToDeg(Math.atan2(f.x,f.z));pitch=THREE.MathUtils.radToDeg(Math.asin(f.y));}else{camera.position.copy(vector(zone.camera_to_common_cv.map(row=>row[3]).slice(0,3)));lookAt(zone.target);}}
function showZone(preserveCamera=false){
 const next=data.zones.find(z=>z.id===$('#zone').value);clearSelection();zone=next;if(!zone)return;if(separatePaths()&&!preserveCamera)phase=defaultPhase(zone);updateVisibility();
 boxes=candidatesAt(zone.id).filter(b=>methods().includes(b.method));allBoxes=boxes;
 renderBoxList();if(!preserveCamera)resetCamera();updateSummary();
 updateInputImages();
}
function updateInputImages(candidate=null){
 if(!$('#input-before')||!zone)return;
 const source=candidate&&frozenCandidates,records=source?frozenCandidates.source_renders[candidate.zone]:data.renders[zone.id],prefix=source?frozenCandidates.source_asset_prefix:'';
 for(const phase of ['before','after']){$('#input-'+phase).src=prefix+records[phase].image;$('#download-'+phase).href=prefix+records[phase].image;}
 if($('#render-input-note'))$('#render-input-note').textContent=source?`${candidate.id}의 원본 분석 입력입니다. 기존 구역 ${candidate.zone}에서 렌더링한 Gaussian RGB를 SimSaC가 비교해 얻은 영역이며, 투어 위치가 바뀌어도 같은 3D 영역을 사용합니다.`:`${zone.id}의 시점 확인용 전후 Gaussian 렌더입니다. 투어 박스는 기존 13개 구역의 Gaussian RGB SimSaC 결과를 재사용하며, 이 시점의 별도 추론 결과로 교체하지 않습니다. 박스를 선택하면 그 후보의 원본 분석 입력이 표시됩니다.`;
}
function boxText(b){const event=eventRecord(b)?.event;return b.method==='baseline'?`${b.zone} 기존 · ${b.source_entry}`:`${event?event.label+' · ':''}${b.zone} SimSaC · 전 ${n(b.before_pixels)} / 후 ${n(b.after_pixels)} px`;}
function selectionButtons(){for(const node of $('#candidates').querySelectorAll('[data-candidate]'))node.setAttribute('aria-pressed',String(!!active&&node.dataset.candidate===eventKey(active)));}
function activateEvent(id){
 if(!tour||tour.moving)return;
 const current=candidatesAt(zone.id).find(b=>eventKey(b)===id);
 if(current){selectBox(current).catch(fail);return;}
 const event=regionIndex().events[id],allowed=new Set(candidates().map(b=>b.id));
 const target=routeContains(event.best_view)&&event.observing_keyframes.includes(event.best_view)&&event.candidate_ids.some(alias=>allowed.has(alias)&&frozenCandidates.visibility[event.best_view][mode()][alias].visible)?event.best_view:event.observing_keyframes.find(v=>routeContains(v)&&event.candidate_ids.some(alias=>allowed.has(alias)&&frozenCandidates.visibility[v][mode()][alias].visible));
 if(target)tour.go(target,()=>{const box=candidatesAt(target).find(b=>eventKey(b)===id);if(box)selectBox(box).catch(fail);});
}
function renderBoxList(){
 const groups=regionGroups();
 if(changeRegions){
  const key=JSON.stringify([mode(),tour?.groupId,groups.map(g=>[g.id,g.events.map(e=>[e.id,e.candidate_ids])])]);
  if(key!==listKey){
   listKey=key;$('#candidates').replaceChildren();
   const source=new Map(candidates().map(b=>[b.id,b]));
   for(const group of groups){
    const card=document.createElement('section');card.className='change-region';card.dataset.changeRegion=group.id;
    const heading=document.createElement('button');heading.className='region-heading';const representative=routeContains(group.representative_view)?group.representative_view:group.observing_keyframes.find(routeContains);heading.dataset.regionView=representative||'';heading.type='button';heading.textContent=`${group.label} · 고유 후보 ${group.events.length}개`;heading.addEventListener('click',()=>representative&&tour?.go(representative));card.append(heading);
    const observations=document.createElement('small');observations.className='region-observations';observations.textContent='같은 후보 관측: '+(group.observing_keyframes.join(' · ')||'현재 경로에서 미관측');card.append(observations);
    for(const event of group.events){
     const box=source.get(event.candidate_ids[0]),type=eventRecord(box)?.event,button=document.createElement('button');button.type='button';button.className='simsac shared-candidate';button.dataset.candidate=event.id;button.dataset.event=type?.type||'unresolved';button.setAttribute('aria-pressed','false');
     const name=document.createElement('span');name.textContent=(type?type.label+' · ':'')+event.id.replace('SimSaC-','');const note=document.createElement('small');note.className='observation-state';button.append(name,note);button.addEventListener('click',()=>activateEvent(event.id));card.append(button);
    }
    $('#candidates').append(card);
   }
  }
  const count=groups.reduce((n,g)=>n+g.events.length,0);$('#candidate-count').textContent=`· ${groups.length}구역 / 고유 ${count}개`;$('#empty').hidden=count>0;
  const visible=new Set(boxes.map(eventKey));
  for(const button of $('#candidates').querySelectorAll('[data-candidate]')){const seen=visible.has(button.dataset.candidate);button.classList.toggle('observed-here',seen);const available=regionIndex().events[button.dataset.candidate].observing_keyframes.some(routeContains);button.dataset.routeUnavailable=String(!available);button.querySelector('.observation-state').textContent=seen?'현재 시점에서 관측':available?'관측 시점으로 이동':'이 경로에서 미관측';button.disabled=!!tour?.moving||!available;}
  for(const card of $('#candidates').children){const group=groups.find(g=>g.id===card.dataset.changeRegion);card.classList.toggle('current-area',!!group?.observing_keyframes.includes(zone?.id));}
 }else{
  $('#candidates').replaceChildren();$('#candidate-count').textContent=`· ${boxes.length}개`;$('#empty').hidden=boxes.length>0;
  for(const box of boxes){const button=document.createElement('button');button.type='button';button.className=box.method;button.dataset.candidate=box.id;button.setAttribute('aria-pressed','false');button.textContent=boxText(box);button.addEventListener('click',()=>selectBox(box).catch(fail));$('#candidates').append(button);}
 }
 $('#box-layer').replaceChildren();boxNodes.clear();
 const ordinals={baseline:0,simsac:0};boxes.forEach(b=>{const event=eventRecord(b)?.event;const title=(event?event.label+' · ':'')+(frozenCandidates?eventKey(b).replace('SimSaC-',''):names[b.method]+' '+(++ordinals[b.method]));
  const hit=document.createElement('button');hit.type='button';hit.className='box-hit '+b.method;hit.dataset.box=b.id;hit.dataset.sharedEvent=eventKey(b);if(event)hit.dataset.event=event.type;hit.setAttribute('aria-label',boxText(b)+' 후보 박스');hit.title=b.title;const label=document.createElement('span');label.textContent=title;hit.append(label);hit.addEventListener('click',e=>{if(e.detail===0)selectBox(b).catch(fail);});$('#box-layer').append(hit);boxNodes.set(b.id,{button:hit,corners:corners(b.bounds)});
 });selectionButtons();
}

function corners(bounds){const out=[];for(const x of bounds.map(b=>b[0]))for(const y of bounds.map(b=>b[1]))for(const z of bounds.map(b=>b[2]))out.push(new THREE.Vector3(x,y,z));return out;}
function drawBoxes(){
 if(!camera)return;const forward=camera.getWorldDirection(new THREE.Vector3());$('#box-layer').hidden=!$('#show-boxes').checked||!data.models[phase].some(a=>meshes.has(a.file));
 for(const b of boxes){const node=boxNodes.get(b.id);const points=node.corners.filter(p=>p.clone().sub(camera.position).dot(forward)>.01).map(p=>p.clone().project(camera));
  if(!points.length){node.button.hidden=true;continue;}let left=Math.min(...points.map(p=>p.x)),right=Math.max(...points.map(p=>p.x)),bottom=Math.min(...points.map(p=>p.y)),top=Math.max(...points.map(p=>p.y));
  const hidden=right< -1||left>1||top< -1||bottom>1;node.button.hidden=hidden;if(hidden)continue;left=Math.max(-1,left);right=Math.min(1,right);bottom=Math.max(-1,bottom);top=Math.min(1,top);
  Object.assign(node.button.style,{left:(left+1)*50+'%',top:(1-top)*50+'%',width:Math.max(1,(right-left)*50)+'%',height:Math.max(2,(top-bottom)*50)+'%',zIndex:String(10000-Math.round((right-left)*(top-bottom)*1000))});node.button.classList.toggle('selected',active?.id===b.id);
 }
}
function updateLoad(){const before=data.models.before.reduce((s,a)=>s+(meshes.has(a.file)?a.gaussians:0),0),after=data.models.after.reduce((s,a)=>s+(meshes.has(a.file)?a.gaussians:0),0);$('#load-state').textContent=`가우시안 로드 · 전 ${n(before)} / ${n(data.scenes[0].gaussians)} · 후 ${n(after)} / ${n(data.scenes[1].gaussians)}`;$('#load-state').dataset.beforeLoaded=before;$('#load-state').dataset.afterLoaded=after;}
async function loadAsset(asset){
 if(!assets.has(asset.file))assets.set(asset.file,(async()=>{
  const mesh=new SplatMesh({url:asset.file,extSplats:true,raycastable:false,onProgress:event=>{$('#load-state').textContent=`${asset.source_epoch?'후':'전'} 가우시안 맵 로드 ${event.total?Math.round(event.loaded/event.total*100)+'%':n(event.loaded)+' bytes'}`;}});
  mesh.name=asset.file;mesh.visible=false;const m=matrix(asset.matrix);m.decompose(mesh.position,mesh.quaternion,mesh.scale);if(Math.max(mesh.scale.x,mesh.scale.y,mesh.scale.z)-Math.min(mesh.scale.x,mesh.scale.y,mesh.scale.z)>1e-5)throw new Error('가우시안 변환이 균일 축척이 아닙니다.');scene.add(mesh);await mesh.initialized;
  if(mesh.numSplats!==asset.gaussians)throw new Error(`가우시안 개수 불일치: ${asset.file} ${mesh.numSplats}/${asset.gaussians}`);meshes.set(asset.file,mesh);mesh.updateMatrixWorld(true);updateLoad();return mesh;
 })().catch(error=>{$('#load-state').textContent='가우시안 맵 파일을 읽지 못했습니다. README의 입력 파일 설정을 확인하세요.';throw error;}));return assets.get(asset.file);
}
function removeEdits(){for(const[mesh,record]of edits){mesh.remove(record.edit);mesh.updateGenerator();}edits.clear();}
function applyBoxEdit(mesh,bounds,invert,opacity){
 const center=vector(bounds[0]).add(vector(bounds[1])).multiplyScalar(.5),size=vector(bounds[1]).sub(vector(bounds[0])).multiplyScalar(.5);
 const edit=new SplatEdit({softEdge:0});const sdf=new SplatEditSdf({type:SplatEditSdfType.BOX,invert,opacity});edit.add(sdf);mesh.add(edit);mesh.updateMatrixWorld(true);
 const world=new THREE.Matrix4().compose(center,new THREE.Quaternion(),size);const local=mesh.matrixWorld.clone().invert().multiply(world);local.decompose(sdf.position,sdf.quaternion,sdf.scale);sdf.updateMatrixWorld(true);mesh.updateGenerator();edits.set(mesh,{edit,sdf,invert});
}
function updateVisibility(){for(const a of data.models.before){const m=meshes.get(a.file);if(m){m.visible=phase==='before';m.opacity=1;}}for(const a of data.models.after){const m=meshes.get(a.file);if(m){m.visible=phase==='after';m.opacity=1;}}$('#map-before').setAttribute('aria-pressed',String(phase==='before'));$('#map-after').setAttribute('aria-pressed',String(phase==='after'));$('#stage-title').textContent=`${frozenCandidates&&zone?zone.id+' · ':''}${phase==='before'?'전 · Dorm2':'후 · Dorm3'} · 실제 3D Gaussian 맵`;}
function clearSelection(){++selectionToken;active=null;delete $('#stage').dataset.selectionKind;if(minimapWasOpen!==null){$('.tour-minimap').open=minimapWasOpen;minimapWasOpen=null;}removeEdits();objectEvidenceView?.clear();if(separatePaths())phase=defaultPhase(zone);$('#ghost-controls').hidden=true;$('#selection').hidden=true;selectionButtons();if(data)updateVisibility();updateInputImages();}
function setWeight(value){if(!active||eventRecord(active)?.event.type==='moved')return;const weight=Number(value)/100;$('#opacity').value=value;$('#opacity-value').textContent=value+'%';for(const[mesh,record]of edits){if(record.invert){mesh.visible=weight>0;mesh.opacity=weight;}else record.sdf.opacity=1-weight;}$('#stage-title').textContent=`${frozenCandidates&&zone?zone.id+' · ':''}전 · Dorm2${weight?' + 선택 영역의 후 · Dorm3 '+value+'%':''} · 실제 3DGS`;}
async function selectBox(b){
 if(tour?.moving)return;
 tour?.pause();
 if(active?.id===b.id){clearSelection();return;}clearSelection();phase=defaultPhase(zone);updateVisibility();const token=++selectionToken;active=b;lookAt(b.center);updateInputImages(b);objectEvidenceView?.show(b,mode());
 $('#selection').hidden=false;$('#selection-title').textContent=boxText(b);$('#selection-note').textContent='선택 영역의 실제 후 가우시안을 불러오는 중…';
 const event=eventRecord(b)?.event;
 if(event?.type==='moved'){
  const minimap=$('.tour-minimap');if(minimap){minimapWasOpen=minimap.open;minimap.open=false;}
  $('#stage').dataset.selectionKind='movement';const focus=objectEvidenceView.focusBounds(b,mode());if(focus)frameMovement(focus);
  $('#stage-title').textContent=`${zone.short_label||zone.id} · ${phase==='after'?'후':'전'} Gaussian 맵 + 출발→도착 색 그림자`;
  $('#ghost-controls').hidden=false;$('#ghost-title').textContent='이동 위치 · 청록 → 주황';
  $('#selection-note').textContent=event.reason+' 현재 Gaussian 맵 위에 기존 분할 메쉬를 색 그림자로 표시합니다.';
  selectionButtons();drawBoxes();return;
 }
 phase='before';updateVisibility();const afterAssets=data.models.after.filter(a=>overlaps(a.bounds,b.bounds));const loaded=await Promise.all(afterAssets.map(loadAsset));if(token!==selectionToken)return;
 for(const a of data.models.before){const mesh=meshes.get(a.file);if(mesh)applyBoxEdit(mesh,b.bounds,false,.45);}
 for(const mesh of loaded)applyBoxEdit(mesh,b.bounds,true,0);
 $('#ghost-controls').hidden=false;$('#ghost-title').textContent=names[b.method]+' 후보';$('#selection-note').textContent=b.method==='baseline'?'기존 메쉬 판정과 후보 원본 면을 그대로 유지한 박스입니다. 선택 영역에서만 전·후 가우시안을 겹칩니다.':frozenCandidates?`기존 Gaussian RGB 분석의 ${b.id}를 그대로 표시합니다. 현재 키프레임 ${zone.id}에서도 같은 3D 범위의 후 맵을 겹칩니다. 키프레임별 새 검출로 교체하지 않습니다.`:'동일 정합 시점의 3DGS 렌더에서 새로 검출했습니다. 가우시안 예상 깊이로 후보 영역을 3D에 연결했으며, 재구성·정합 잔차가 포함될 수 있습니다.';
 selectionButtons();setWeight(55);drawBoxes();
}
async function showPhase(value){clearSelection();phase=value;if(value==='after')await Promise.all(data.models.after.map(loadAsset));else await Promise.all(data.models.before.map(loadAsset));updateVisibility();}
function reportParent(){if(!embedded||!data)return;const groups=regionGroups();window.parent.postMessage({type:"gaussian-tour:state",keyframe:zone?.id,keyframes:data.zones.length,regions:groups.length,events:groups.reduce((n,g)=>n+g.events.length,0),error:$("#error").textContent},location.origin);}
function renderFrame(now){tour?.tick(now);renderer.render(scene,camera);drawBoxes();objectEvidenceView?.draw(camera);}
function updateSummary(){
 reportParent();
 const c=counts(zone.id),excluded=$('#exclude-person').checked;
 $('#audit').innerHTML=data.scenes.map((s,i)=>`<p><b>${i?'후 · Dorm3':'전 · Dorm2'}</b> · ${n(s.gaussians)} Gaussians · ${n(s.step)} step<br>원본 영상 SHA-256: ${s.video_sha256}<br>체크포인트 SHA-256: ${s.checkpoint_sha256}<br>메쉬↔3DGS 카메라 ${data.camera_checks[s.capture].cameras}개 좌표 일치 확인</p>`).join('');
 if(frozenCandidates){
  if(objectEvidence){const totals={added:0,removed:0,moved:0,unresolved:0};for(const id of (regionIndex()?Object.keys(regionIndex().events):Object.keys(objectEvidence.regions[mode()])))totals[objectEvidence.regions[mode()][id].event.type]++;$('#event-counts').textContent=`전체 영역 유형: 추가 후보 ${totals.added} · 이동 추정 ${totals.moved} · 소실 추정 ${totals.removed} · 유형 불확실 ${totals.unresolved} (확정된 객체 변화 건수 아님)`;}
  const groups=regionGroups(),observed=groups.filter(g=>g.observing_keyframes.includes(zone.id));
  $('#summary').textContent=changeRegions?`${zone.id} · ${observed.map(g=>g.label).join(' / ')||'경유 시점'} · 등록된 같은 후보 ${c.simsac}개를 관측 중 · 전체 ${groups.length}구역 / 고유 ${groups.reduce((n,g)=>n+g.events.length,0)}개`:`${zone.id} · 기존 Gaussian RGB SimSaC 영역 중 이 위치에서 관측 가능한 ${c.simsac}영역 · 사람 제외 ${excluded?'적용':'해제'}`;
  if($('#valid-render'))$('#valid-render').textContent=`같은 3D 변화는 구역에 한 번만 등록하며 여러 키프레임은 관측 링크로만 연결합니다. 원본 사람 제외 후보 ${frozenCandidates.boxes.filtered.length}개를 공통 3D 좌표에 고정했습니다. 키프레임의 전후 Gaussian 예상 깊이로 가림을 검사하며 현재 시야에 들어오는 박스를 표시합니다. 이동 중에는 중간 깊이 검증이 없어 도착까지 박스를 숨깁니다.`;
  $('#audit').insertAdjacentHTML('beforeend',`<p>원본 키프레임: ${zone.viewpoint_id} · 기준 촬영: ${zone.capture_id} · 정합 구역: ${zone.registration_zone}<br>투어 표시 기준: 기존 Gaussian RGB SimSaC 후보. 원본 SHA-256: ${frozenCandidates.source_prediction_sha256}<br>후보 id·3D 범위 보존. 새 SimSaC 추론 없이 저장된 후보를 키프레임에 연결했습니다. 양쪽 맵을 이 키프레임의 같은 포즈에서 렌더링한 깊이는 가림 검사에만 사용합니다.</p>`);
  return;
 }
 const s=prediction.zones[zone.id],pixelKey=excluded?'filtered_pixels':'observable_without_person_filter';
 $('#summary').textContent=`${zone.id} · 기존 메쉬 후보 ${c.baseline} / 새 SimSaC 후보 ${c.simsac} · ${excluded?'사람 제외 적용':'사람 제외 해제'} · 후보 박스를 눌러 후 맵 확인`;
 $('#audit').insertAdjacentHTML('beforeend',`<p>${zone.id} 새 SimSaC 전 ${n(s.stats.before[pixelKey])} / 후 ${n(s.stats.after[pixelKey])} px · 사람 제외 ${n(excluded?s.stats.before.removed_person_pixels+s.stats.after.removed_person_pixels:0)} px · 양쪽 맵의 유효 렌더 겹침 ${(s.valid_overlap_pixels/(1024*512)*100).toFixed(1)}%</p>`);
 if(data.viewpoint_mode==='tour_keyframes'){
  $('#summary').textContent=`${zone.label} · SimSaC ${c.simsac}영역 · 사람 제외 ${excluded?'적용':'해제'} · ${zone.registration_supported?'기존 구역 정합 적용':'기존 구역 정합의 관측 지원 부족'}`;
  if($('#valid-render'))$('#valid-render').textContent=`분석에 사용한 양쪽 유효 렌더 영역: ${(s.valid_overlap_pixels/(1024*512)*100).toFixed(1)}%. 나머지는 변화 없음으로 판정하지 않습니다. 이 비율은 정합 정확도가 아닙니다.`;
 }
}
function bindControls(){
 const stage=$('#stage'),canvas=$('#gaussian-canvas');
 stage.addEventListener('pointerdown',e=>{if(tour?.moving||e.target.closest('.stage-tools,.ghost-controls,[data-tour-ui]'))return;e.preventDefault();canvas.focus({preventScroll:true});stage.setPointerCapture(e.pointerId);pointers.set(e.pointerId,{x:e.clientX,y:e.clientY,moved:0,button:e.button,id:e.target.closest('[data-box]')?.dataset.box});});
 stage.addEventListener('pointermove',e=>{const p=pointers.get(e.pointerId);if(!p)return;const dx=e.clientX-p.x,dy=e.clientY-p.y;p.moved+=Math.hypot(dx,dy);if(p.moved<4)return;p.x=e.clientX;p.y=e.clientY;
  if(p.button===2){if(tour)return;const right=new THREE.Vector3(1,0,0).applyQuaternion(camera.quaternion),up=new THREE.Vector3(0,1,0);camera.position.addScaledVector(right,-dx*data.height_u*.002);camera.position.addScaledVector(up,dy*data.height_u*.002);updateDirection();}
  else{yaw-=dx*.18;pitch=Math.max(-85,Math.min(85,pitch-dy*.18));updateDirection();}});
 stage.addEventListener('pointerup',e=>{const p=pointers.get(e.pointerId);pointers.delete(e.pointerId);if(p?.id&&p.moved<4){const b=boxes.find(b=>b.id===p.id);if(b)selectBox(b).catch(fail);}});for(const name of ['pointercancel','lostpointercapture'])stage.addEventListener(name,e=>pointers.delete(e.pointerId));stage.addEventListener('contextmenu',e=>e.preventDefault());
 stage.addEventListener('wheel',e=>{if(tour?.moving||document.activeElement!==canvas)return;e.preventDefault();camera.fov=Math.max(30,Math.min(100,camera.fov+e.deltaY*.035));camera.updateProjectionMatrix();},{passive:false});
 canvas.addEventListener('keydown',e=>{if(tour?.moving)return;const step=data.height_u*.08*(e.shiftKey?3:1),f=camera.getWorldDirection(new THREE.Vector3()),r=new THREE.Vector3(1,0,0).applyQuaternion(camera.quaternion);f.y=0;f.normalize();const key=e.key.toLowerCase();if(tour&&['w','a','s','d','q','e','pageup','pagedown'].includes(key)){e.preventDefault();if(['w','d','pagedown'].includes(key))tour.step(1);else if(['s','a','pageup'].includes(key))tour.step(-1);return;}if(key==='w')camera.position.addScaledVector(f,step);else if(key==='s')camera.position.addScaledVector(f,-step);else if(key==='a')camera.position.addScaledVector(r,-step);else if(key==='d')camera.position.addScaledVector(r,step);else if(key==='q')camera.position.y-=step;else if(key==='e')camera.position.y+=step;else if(key==='arrowleft')yaw+=8;else if(key==='arrowright')yaw-=8;else if(key==='arrowup')pitch=Math.min(85,pitch+8);else if(key==='arrowdown')pitch=Math.max(-85,pitch-8);else return;e.preventDefault();updateDirection();});
 for(const id of ['method','exclude-person','only-candidates'])$('#'+id).addEventListener('change',refreshZones);$('#zone').addEventListener('change',()=>tour?tour.go($('#zone').value):showZone());$('#show-boxes').addEventListener('change',drawBoxes);$('#reset').addEventListener('click',resetCamera);
 $('#plan-group')?.addEventListener('change',()=>tour?tour.selectGroup($('#plan-group').value):refreshZones());
 $('#minimum-pixels')?.addEventListener('change',refreshZones);
 $('#event-type')?.addEventListener('change',refreshZones);
 for(const[id,delta]of [['zoom-in',-10],['zoom-out',10]])$('#'+id).addEventListener('click',()=>{camera.fov=Math.max(30,Math.min(100,camera.fov+delta));camera.updateProjectionMatrix();});
 $('#fullscreen').addEventListener('click',()=>{(document.fullscreenElement?document.exitFullscreen():stage.requestFullscreen()).catch(fail);});$('#map-before').addEventListener('click',()=>showPhase('before').catch(fail));$('#map-after').addEventListener('click',()=>showPhase('after').catch(fail));
 $('#opacity').addEventListener('input',()=>setWeight($('#opacity').value));$('#before-only').addEventListener('click',()=>setWeight(0));$('#ghost-half').addEventListener('click',()=>setWeight(55));$('#after-only').addEventListener('click',()=>setWeight(100));$('#close-ghost').addEventListener('click',clearSelection);
}
async function init(){
 [data,prediction]=await Promise.all(['gaussian-data.json','simsac-candidates.json'].map(async path=>{const r=await fetch(path);if(!r.ok)throw new Error(path+' HTTP '+r.status);return r.json();}));
 if(data.scenes[0].checkpoint_sha256===data.scenes[1].checkpoint_sha256||data.scenes[0].video_sha256===data.scenes[1].video_sha256)throw new Error('서로 다른 두 입력 맵이 확인되지 않았습니다.');
 $('#sources').innerHTML=data.scenes.map((s,i)=>`<div class="source"><b>${i?'후 · Dorm3':'전 · Dorm2'}</b> · ${n(s.gaussians)} Gaussians<span>별도 원본 영상 · 별도 60,000-step 학습 · ${s.checkpoint_sha256.slice(0,12)}…</span></div>`).join('');
 scene=new THREE.Scene();scene.background=new THREE.Color('#101820');camera=new THREE.PerspectiveCamera(65,1,.01,1000);camera.up.set(0,1,0);
 renderer=new THREE.WebGLRenderer({canvas:$('#gaussian-canvas'),antialias:false});renderer.setPixelRatio(Math.min(devicePixelRatio||1,1.5));spark=new SparkRenderer({renderer});scene.add(spark);
 const resize=()=>{const rect=$('#stage').getBoundingClientRect();renderer.setSize(rect.width,rect.height,false);camera.aspect=rect.width/rect.height;camera.updateProjectionMatrix();if(active&&eventRecord(active)?.event.type==='moved'){const focus=objectEvidenceView.focusBounds(active,mode());if(focus)frameMovement(focus);}};new ResizeObserver(resize).observe($('#stage'));resize();
 if($('#event-type')){
  const [r,module]=await Promise.all([fetch('object-evidence.json').then(r=>{if(!r.ok)throw new Error('object-evidence.json HTTP '+r.status);return r.json();}),import('./object_evidence.js?v=automatic-1')]);
  objectEvidence=r;objectEvidenceView=module.createObjectEvidence(scene,r,data.height_u);
 }
 if(data.viewpoint_mode==='tour_keyframes'&&$('#tour-strip')){
  const [module,routes,fixed,shared]=await Promise.all([import('./tour.js?v=shared-2'),...['tour-routes.json','tour-candidates.json'].map(path=>fetch(path).then(r=>{if(!r.ok)throw new Error(path+' HTTP '+r.status);return r.json();})), $('#change-region-panel')?Promise.all([fetch('change-regions.json').then(r=>{if(!r.ok)throw new Error('change-regions.json HTTP '+r.status);return r.json();}),import('./change_regions.js?v=2')]):null]);
  frozenCandidates=fixed;if(shared){[changeRegions,regionHelpers]=shared;}
  tour=module.createTour({routes,data,camera,candidatesAt,regionsFor:shared?regionGroups:null,ready:()=>(separatePaths()?[...data.models.before,...data.models.after]:data.models.before).every(a=>meshes.has(a.file)),
   depart:()=>{clearSelection();phase=defaultPhase(zone);updateVisibility();boxes=[];renderBoxList();pointers.clear();$('#summary').textContent=changeRegions?'이동 중 · 기존 구역의 같은 후보를 관측 시점에 연결합니다.':'키프레임 사이를 이동 중입니다. 도착한 시점의 후보 박스를 자동으로 표시합니다.';},
   arrive:(id,preserve)=>{if(!preserve)phase=defaultPhase(data.zones.find(v=>v.id===id));$('#zone').value=id;showZone(preserve);}});
 }
 bindControls();if(!tour)refreshZones();else if(separatePaths())renderBoxList();
 renderer.setAnimationLoop(renderingEnabled?renderFrame:null);await Promise.all((separatePaths()?[...data.models.before,...data.models.after]:data.models.before).map(loadAsset));updateVisibility();
}
if(embedded)window.addEventListener('message',event=>{
 if(event.source!==window.parent||event.origin!==location.origin||event.data?.type!=='gaussian-tour:command')return;
 const{action,value}=event.data;
 if(action==='navigate'&&[-1,1].includes(value))tour?.step(value);
 else if(action==='clear')clearSelection();
 else if(action==='suspend'){renderingEnabled=false;tour?.pause();renderer?.setAnimationLoop(null);}
 else if(action==='resume'){renderingEnabled=true;renderer?.setAnimationLoop(renderFrame);}
});
init().catch(fail);

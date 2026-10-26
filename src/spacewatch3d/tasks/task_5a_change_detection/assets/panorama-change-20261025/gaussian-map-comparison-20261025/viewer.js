import * as THREE from 'three';
import {SparkRenderer,SplatMesh,SplatEdit,SplatEditSdf,SplatEditSdfType} from '@sparkjsdev/spark';

const $=q=>document.querySelector(q),n=x=>x.toLocaleString('ko-KR'),names={baseline:'기존',simsac:'SimSaC'};
let data,prediction,scene,camera,renderer,spark,zone,phase='before',active=null,allBoxes=[],boxes=[],yaw=0,pitch=0,selectionToken=0;
const assets=new Map(),meshes=new Map(),edits=new Map(),boxNodes=new Map(),pointers=new Map();
const matrix=rows=>new THREE.Matrix4().fromArray(rows.flat()).transpose();
const vector=a=>new THREE.Vector3().fromArray(a);
const methods=()=>$('#method').value==='both'?['baseline','simsac']:[$('#method').value];
const mode=()=>$('#exclude-person').checked?'filtered':'raw';
const candidates=()=>[...data.baseline_boxes,...prediction.boxes[mode()]].filter(b=>b.method!=='simsac'||b.before_pixels+b.after_pixels>=Number($('#minimum-pixels')?.value||0));
const overlaps=(a,b)=>[0,1,2].every(i=>a[0][i]<=b[1][i]&&a[1][i]>=b[0][i]);
function fail(error){$('#error').textContent=error?.stack||String(error);}
function counts(z){const list=candidates().filter(b=>b.zone===z);return{baseline:list.filter(b=>b.method==='baseline').length,simsac:list.filter(b=>b.method==='simsac').length};}
function refreshZones(){
 const previous=$('#zone').value;const group=$('#plan-group')?.value;const zones=data.zones.filter(z=>(!group||group==='all'||z.plan_group===group)&&(!$('#only-candidates').checked||candidates().some(b=>b.zone===z.id&&methods().includes(b.method))));
 $('#zone').innerHTML=zones.map(z=>{const c=counts(z.id);return`<option value="${z.id}">${z.label||z.id} · ${data.viewpoint_mode?'후보 '+c.simsac:'기존 '+c.baseline+' / SimSaC '+c.simsac}</option>`;}).join('');
 if(zones.some(z=>z.id===previous))$('#zone').value=previous;else if(zones.some(z=>z.id==='R08'))$('#zone').value='R08';
 if(!zones.length){$('#summary').textContent='현재 선택한 방식에서 남은 후보 구역이 없습니다. 후보 있는 구역만을 끄면 맵을 둘러볼 수 있습니다.';boxes=[];renderBoxList();return;}
 showZone();
}
function updateDirection(){const y=THREE.MathUtils.degToRad(yaw),p=THREE.MathUtils.degToRad(pitch);camera.lookAt(camera.position.clone().add(new THREE.Vector3(Math.sin(y)*Math.cos(p),Math.sin(p),Math.cos(y)*Math.cos(p))));camera.updateMatrixWorld();}
function lookAt(point){camera.lookAt(vector(point));camera.updateMatrixWorld();const f=camera.getWorldDirection(new THREE.Vector3());yaw=THREE.MathUtils.radToDeg(Math.atan2(f.x,f.z));pitch=THREE.MathUtils.radToDeg(Math.asin(f.y));}
function resetCamera(){if(!zone)return;camera.fov=65;camera.updateProjectionMatrix();if(zone.reset_from_pose){const gl=matrix(zone.camera_to_common_cv).multiply(new THREE.Matrix4().makeScale(1,-1,-1));gl.decompose(camera.position,camera.quaternion,new THREE.Vector3());camera.updateMatrixWorld(true);const f=camera.getWorldDirection(new THREE.Vector3());yaw=THREE.MathUtils.radToDeg(Math.atan2(f.x,f.z));pitch=THREE.MathUtils.radToDeg(Math.asin(f.y));}else{camera.position.copy(vector(zone.camera_to_common_cv.map(row=>row[3]).slice(0,3)));lookAt(zone.target);}}
function showZone(){
 clearSelection();zone=data.zones.find(z=>z.id===$('#zone').value);if(!zone)return;
 boxes=candidates().filter(b=>b.zone===zone.id&&methods().includes(b.method));allBoxes=boxes;
 renderBoxList();resetCamera();updateSummary();
 if($('#input-before'))for(const phase of ['before','after']){$('#input-'+phase).src=data.renders[zone.id][phase].image;$('#download-'+phase).href=data.renders[zone.id][phase].image;}
}
function boxText(b){return b.method==='baseline'?`${b.zone} 기존 · ${b.source_entry}`:`${b.zone} SimSaC · 전 ${n(b.before_pixels)} / 후 ${n(b.after_pixels)} px`;}
function renderBoxList(){
 $('#candidates').replaceChildren();$('#box-layer').replaceChildren();boxNodes.clear();$('#candidate-count').textContent=`· ${boxes.length}개`;$('#empty').hidden=boxes.length>0;
 const ordinals={baseline:0,simsac:0};boxes.forEach(b=>{const title=names[b.method]+' '+(++ordinals[b.method]);const button=document.createElement('button');button.type='button';button.className=b.method;button.dataset.candidate=b.id;button.setAttribute('aria-pressed','false');button.textContent=boxText(b);button.addEventListener('click',()=>selectBox(b).catch(fail));$('#candidates').append(button);
  const hit=document.createElement('button');hit.type='button';hit.className='box-hit '+b.method;hit.dataset.box=b.id;hit.setAttribute('aria-label',boxText(b)+' 후보 박스');hit.title=b.title;const label=document.createElement('span');label.textContent=title;hit.append(label);hit.addEventListener('click',e=>{if(e.detail===0)selectBox(b).catch(fail);});$('#box-layer').append(hit);boxNodes.set(b.id,{button:hit,corners:corners(b.bounds)});
 });
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
function updateVisibility(){for(const a of data.models.before){const m=meshes.get(a.file);if(m){m.visible=phase==='before';m.opacity=1;}}for(const a of data.models.after){const m=meshes.get(a.file);if(m){m.visible=phase==='after';m.opacity=1;}}$('#map-before').setAttribute('aria-pressed',String(phase==='before'));$('#map-after').setAttribute('aria-pressed',String(phase==='after'));$('#stage-title').textContent=`${phase==='before'?'전 · Dorm2':'후 · Dorm3'} · 실제 3D Gaussian 맵`;}
function clearSelection(){++selectionToken;active=null;removeEdits();$('#ghost-controls').hidden=true;$('#selection').hidden=true;for(const b of $('#candidates').children)b.setAttribute('aria-pressed','false');if(data)updateVisibility();}
function setWeight(value){if(!active)return;const weight=Number(value)/100;$('#opacity').value=value;$('#opacity-value').textContent=value+'%';for(const[mesh,record]of edits){if(record.invert){mesh.visible=weight>0;mesh.opacity=weight;}else record.sdf.opacity=1-weight;}$('#stage-title').textContent=`전 · Dorm2${weight?' + 선택 영역의 후 · Dorm3 '+value+'%':''} · 실제 3DGS`;}
async function selectBox(b){
 if(active?.id===b.id){clearSelection();return;}clearSelection();phase='before';updateVisibility();const token=++selectionToken;active=b;lookAt(b.center);
 $('#selection').hidden=false;$('#selection-title').textContent=boxText(b);$('#selection-note').textContent='선택 영역의 실제 후 가우시안을 불러오는 중…';
 const afterAssets=data.models.after.filter(a=>overlaps(a.bounds,b.bounds));const loaded=await Promise.all(afterAssets.map(loadAsset));if(token!==selectionToken)return;
 for(const a of data.models.before){const mesh=meshes.get(a.file);if(mesh)applyBoxEdit(mesh,b.bounds,false,.45);}
 for(const mesh of loaded)applyBoxEdit(mesh,b.bounds,true,0);
 $('#ghost-controls').hidden=false;$('#ghost-title').textContent=names[b.method]+' 후보';$('#selection-note').textContent=b.method==='baseline'?'기존 메쉬 판정과 후보 원본 면을 그대로 유지한 박스입니다. 선택 영역에서만 전·후 가우시안을 겹칩니다.':'동일 정합 시점의 3DGS 렌더에서 새로 검출했습니다. 가우시안 예상 깊이로 후보 영역을 3D에 연결했으며, 재구성·정합 잔차가 포함될 수 있습니다.';
 for(const node of $('#candidates').children)node.setAttribute('aria-pressed',String(node.dataset.candidate===b.id));setWeight(55);drawBoxes();
}
async function showPhase(value){clearSelection();phase=value;if(value==='after')await Promise.all(data.models.after.map(loadAsset));else await Promise.all(data.models.before.map(loadAsset));updateVisibility();}
function updateSummary(){const c=counts(zone.id),s=prediction.zones[zone.id],pixelKey=$('#exclude-person').checked?'filtered_pixels':'observable_without_person_filter';$('#summary').textContent=`${zone.id} · 기존 메쉬 후보 ${c.baseline} / 새 SimSaC 후보 ${c.simsac} · ${$('#exclude-person').checked?'사람 제외 적용':'사람 제외 해제'} · 후보 박스를 눌러 후 맵 확인`;
 $('#audit').innerHTML=data.scenes.map((s,i)=>`<p><b>${i?'후 · Dorm3':'전 · Dorm2'}</b> · ${n(s.gaussians)} Gaussians · ${n(s.step)} step<br>원본 영상 SHA-256: ${s.video_sha256}<br>체크포인트 SHA-256: ${s.checkpoint_sha256}<br>메쉬↔3DGS 카메라 ${data.camera_checks[s.capture].cameras}개 좌표 일치 확인</p>`).join('')+`<p>${zone.id} 새 SimSaC 전 ${n(s.stats.before[pixelKey])} / 후 ${n(s.stats.after[pixelKey])} px · 사람 제외 ${n($('#exclude-person').checked?s.stats.before.removed_person_pixels+s.stats.after.removed_person_pixels:0)} px · 양쪽 맵의 유효 렌더 겹침 ${(s.valid_overlap_pixels/(1024*512)*100).toFixed(1)}%</p>`;
 if(data.viewpoint_mode==='tour_keyframes'){$('#summary').textContent=`${zone.label} · SimSaC ${c.simsac}영역 · 사람 제외 ${$('#exclude-person').checked?'적용':'해제'} · ${zone.registration_supported?'기존 구역 정합 적용':'기존 구역 정합의 관측 지원 부족'}`;if($('#valid-render'))$('#valid-render').textContent=`분석에 사용한 양쪽 유효 렌더 영역: ${(s.valid_overlap_pixels/(1024*512)*100).toFixed(1)}%. 나머지는 변화 없음으로 판정하지 않습니다. 이 비율은 정합 정확도가 아닙니다.`;$('#audit').insertAdjacentHTML('beforeend',`<p>원본 키프레임: ${zone.viewpoint_id} · 기준 촬영: ${zone.capture_id} · 정합 구역: ${zone.registration_zone}<br>전후는 이 키프레임의 동일 위치·방향에서 렌더링했습니다. 원본 투어 계획은 경로·커버리지 검증 미완료 상태입니다. 시점 선택에 변화 후보를 사용하지 않았습니다.</p>`);}
}
function bindControls(){
 const stage=$('#stage'),canvas=$('#gaussian-canvas');
 stage.addEventListener('pointerdown',e=>{if(e.target.closest('.stage-tools,.ghost-controls'))return;e.preventDefault();canvas.focus({preventScroll:true});stage.setPointerCapture(e.pointerId);pointers.set(e.pointerId,{x:e.clientX,y:e.clientY,moved:0,button:e.button,id:e.target.closest('[data-box]')?.dataset.box});});
 stage.addEventListener('pointermove',e=>{const p=pointers.get(e.pointerId);if(!p)return;const dx=e.clientX-p.x,dy=e.clientY-p.y;p.moved+=Math.hypot(dx,dy);if(p.moved<4)return;p.x=e.clientX;p.y=e.clientY;
  if(p.button===2){const right=new THREE.Vector3(1,0,0).applyQuaternion(camera.quaternion),up=new THREE.Vector3(0,1,0);camera.position.addScaledVector(right,-dx*data.height_u*.002);camera.position.addScaledVector(up,dy*data.height_u*.002);updateDirection();}
  else{yaw-=dx*.18;pitch=Math.max(-85,Math.min(85,pitch-dy*.18));updateDirection();}});
 stage.addEventListener('pointerup',e=>{const p=pointers.get(e.pointerId);pointers.delete(e.pointerId);if(p?.id&&p.moved<4){const b=boxes.find(b=>b.id===p.id);if(b)selectBox(b).catch(fail);}});for(const name of ['pointercancel','lostpointercapture'])stage.addEventListener(name,e=>pointers.delete(e.pointerId));stage.addEventListener('contextmenu',e=>e.preventDefault());
 stage.addEventListener('wheel',e=>{if(document.activeElement!==canvas)return;e.preventDefault();camera.fov=Math.max(30,Math.min(100,camera.fov+e.deltaY*.035));camera.updateProjectionMatrix();},{passive:false});
 canvas.addEventListener('keydown',e=>{const step=data.height_u*.08*(e.shiftKey?3:1),f=camera.getWorldDirection(new THREE.Vector3()),r=new THREE.Vector3(1,0,0).applyQuaternion(camera.quaternion);f.y=0;f.normalize();const key=e.key.toLowerCase();if(key==='w')camera.position.addScaledVector(f,step);else if(key==='s')camera.position.addScaledVector(f,-step);else if(key==='a')camera.position.addScaledVector(r,-step);else if(key==='d')camera.position.addScaledVector(r,step);else if(key==='q')camera.position.y-=step;else if(key==='e')camera.position.y+=step;else if(key==='arrowleft')yaw+=8;else if(key==='arrowright')yaw-=8;else if(key==='arrowup')pitch=Math.min(85,pitch+8);else if(key==='arrowdown')pitch=Math.max(-85,pitch-8);else return;e.preventDefault();updateDirection();});
 for(const id of ['method','exclude-person','only-candidates'])$('#'+id).addEventListener('change',refreshZones);$('#zone').addEventListener('change',showZone);$('#show-boxes').addEventListener('change',drawBoxes);$('#reset').addEventListener('click',resetCamera);
 $('#plan-group')?.addEventListener('change',refreshZones);
 $('#minimum-pixels')?.addEventListener('change',refreshZones);
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
 const resize=()=>{const rect=$('#stage').getBoundingClientRect();renderer.setSize(rect.width,rect.height,false);camera.aspect=rect.width/rect.height;camera.updateProjectionMatrix();};new ResizeObserver(resize).observe($('#stage'));resize();bindControls();refreshZones();
 renderer.setAnimationLoop(()=>{renderer.render(scene,camera);drawBoxes();});await Promise.all(data.models.before.map(loadAsset));updateVisibility();
}
init().catch(fail);

import {PanoramaCanvas,normalize,sub,rotate,inverseRotate} from './panorama_canvas.js';
const $=s=>document.querySelector(s),n=x=>x.toLocaleString('ko-KR'),sides=['before','after'],phase={before:'전',after:'후'};
const images=new Map(),depths=new Map(),viewers={};let data,selected=[],current,anchor,loadId=0;
function image(path){if(!images.has(path))images.set(path,new Promise((resolve,reject)=>{const i=new Image();i.onload=()=>resolve(i);i.onerror=()=>reject(new Error('이미지를 읽지 못했습니다: '+path));i.src=path;}));return images.get(path);}
function depth(path){if(!depths.has(path))depths.set(path,fetch(path).then(r=>{if(!r.ok)throw new Error('깊이 정보 HTTP '+r.status);return r.arrayBuffer();}).then(b=>new Float32Array(b)));return depths.get(path);}
const excluded=()=>$('#exclude-person').checked;
function matches(c){return(!$('#zone').value||c.zone===$('#zone').value)&&($('#filter').value!=='person'||sides.some(s=>c.stats[s].removed_person_pixels>0))&&($('#filter').value!=='candidate'||c.has_existing_candidate)&&($('#filter').value!=='close'||c.camera_gap_h<=.5);}
function setProjection(value){$('#projection').value=value;document.body.dataset.projection=value;for(const v of Object.values(viewers)){v.full=value==='full';v.draw();}}
function settings(){for(const v of Object.values(viewers)){v.exclude=excluded();v.review=$('#review').checked;v.overlay=$('#overlay').checked;v.draw();}updateNumbers();}
function aim(point,fov=72){anchor=point;for(const side of sides){const direction=sub(point,current.cameras[side].center);for(const method of ['baseline','simsac'])viewers[method+'-'+side].lookAt(direction,fov);}}
function pointFor(side,direction){
 const f=data.frames[current[side]],cam=current.cameras[side],local=normalize(inverseRotate(cam.rotation,direction)),W=f.depth_width,H=f.depth_height;
 const u=((Math.round((Math.atan2(local[0],local[2])/(2*Math.PI)+.5)*W-.5)%W)+W)%W;
 const v=Math.max(0,Math.min(H-1,Math.round((Math.asin(Math.max(-1,Math.min(1,local[1])))/Math.PI+.5)*H-.5)));
 const distance=f.loadedDepth[v*W+u]*cam.scale;if(!Number.isFinite(distance)||distance<=0)return null;
 return cam.center.map((x,i)=>x+direction[i]*distance);
}
function onInput(driver){
 if(!current)return;const side=driver.side,other=side==='before'?'after':'before';
 for(const method of ['baseline','simsac']){const v=viewers[method+'-'+side];if(v!==driver)v.setView(driver.yaw,driver.pitch,driver.fov);}
 if($('#linked').checked){const point=pointFor(side,driver.forward());
  if(point){anchor=point;for(const method of ['baseline','simsac'])viewers[method+'-'+other].lookAt(sub(point,current.cameras[other].center),driver.fov);}
  else for(const method of ['baseline','simsac'])viewers[method+'-'+other].setView(driver.yaw,driver.pitch,driver.fov);
 }
}
function onFocus(driver,direction){const point=pointFor(driver.side,direction);setProjection('look');if(point)aim(point);else{for(const v of Object.values(viewers))v.lookAt(direction);} }
function updateNumbers(){
 if(!current)return;
 for(const side of sides){const f=data.frames[current[side]],s=current.stats[side],b=excluded()?f.candidate_pixels_filtered:f.candidate_pixels,model=excluded()?s.filtered_pixels:s.raw_pixels;
  $('#baseline-'+side+'-stat').textContent=`후보 표면 ${n(b)} px`;
  $('#simsac-'+side+'-stat').textContent=`후보 ${n(model)} px`;
 }
 $('#numbers').innerHTML=sides.map(side=>{const f=data.frames[current[side]],s=current.stats[side];return `<div><b>${phase[side]} · Dorm${f.epoch+2} · frame ${f.frame}</b><p>SimSaC 원본 ${n(s.raw_pixels)} px → 표시 ${n(excluded()?s.filtered_pixels:s.raw_pixels)} px<br>사람 제외 ${n(excluded()?s.removed_person_pixels:0)} px</p></div>`;}).join('');
}
async function show(){
 const i=selected.findIndex(c=>c.id===$('#pair').value),c=selected[i];$('#prev').disabled=i<=0;$('#next').disabled=i<0||i>=selected.length-1;if(!c)return;
 current=c;const id=++loadId;$('#case').classList.add('loading');$('#case-title').textContent=c.zone+' · 원본 360° 전후';$('#case-count').textContent=`${i+1} / ${selected.length}쌍`;
 $('#alignment-status').textContent='원본 360° 파노라마를 읽는 중…';
 try{
  const loaded=await Promise.all(sides.map(async side=>{const f=data.frames[c[side]];const [rgb,base,sim,dep]=await Promise.all([image(f.rgb),image(f.baseline),image(c.model_masks[side]),depth(f.depth)]);return{side,rgb,base,sim,dep};}));
  if(id!==loadId)return;
  for(const {side,rgb,base,sim,dep}of loaded){const f=data.frames[c[side]];f.loadedDepth=dep;viewers['baseline-'+side].setScene(rgb,base,c.cameras[side]);viewers['simsac-'+side].setScene(rgb,sim,c.cameras[side]);}
  $('#phases').innerHTML=sides.map(side=>{const f=data.frames[c[side]];return `<div class="phase"><span><b>${phase[side]}</b>Dorm${f.epoch+2} · frame ${f.frame}</span><a href="${f.rgb}" target="_blank" rel="noopener">360° 원본 전체</a></div>`;}).join('');
  aim(c.anchor);settings();
  const blocked=sides.filter(s=>c.anchor_person[s]).map(s=>phase[s]).join('·');
  $('#alignment-status').textContent=`같은 3D 지점 중심으로 정렬 · 카메라 위치 차이 ${c.camera_gap_u.toFixed(3)} 맵 단위 (기준높이의 ${(c.camera_gap_h*100).toFixed(0)}%) · 원근·가림 차이는 남습니다.${blocked?' 초기 지점이 '+blocked+'의 사람 제외 영역과 겹칩니다.':''}`;
 }catch(error){$('#error').textContent=error.stack;}finally{if(id===loadId)$('#case').classList.remove('loading');}
}
function refresh(){const previous=$('#pair').value;selected=data.cases.filter(matches);$('#pair').innerHTML=selected.map(c=>{const b=data.frames[c.before],a=data.frames[c.after];return `<option value="${c.id}">${c.zone} · 전 ${b.frame} ↔ 후 ${a.frame} · 위치 차이 ${(c.camera_gap_h*100).toFixed(0)}%</option>`;}).join('');if(selected.some(c=>c.id===previous))$('#pair').value=previous;$('#empty').hidden=selected.length>0;$('#case').hidden=selected.length===0;show();}
async function init(){
 const response=await fetch('panorama-review.json');if(!response.ok)throw new Error('360° 비교 자료 HTTP '+response.status);data=await response.json();
 for(const method of ['baseline','simsac'])for(const side of sides){const v=new PanoramaCanvas($('#'+method+'-'+side),method==='simsac'?1:0,onInput,onFocus,setProjection);v.side=side;viewers[method+'-'+side]=v;}
 for(const id of ['zone','filter'])$('#'+id).addEventListener('change',refresh);$('#pair').addEventListener('change',show);
 for(const id of ['exclude-person','overlay','review'])$('#'+id).addEventListener('change',settings);
 $('#projection').addEventListener('change',()=>setProjection($('#projection').value));$('#linked').addEventListener('change',()=>{if(current&&$('#linked').checked)aim(anchor||current.anchor);});
 $('#object-focus').addEventListener('click',()=>{setProjection('look');if(current)aim(anchor||current.anchor);});$('#reset').addEventListener('click',()=>{if(current)aim(current.anchor);});
 $('#nearest').addEventListener('click',()=>{if(!selected.length)return;const best=selected.reduce((a,b)=>a.camera_gap_u<=b.camera_gap_u?a:b);$('#pair').value=best.id;show();});
 for(const [id,delta]of [['prev',-1],['next',1]])$('#'+id).addEventListener('click',()=>{const i=selected.findIndex(c=>c.id===$('#pair').value)+delta;if(selected[i]){$('#pair').value=selected[i].id;show();}});
 refresh();
}
init().catch(error=>{$('#error').textContent=error.stack;});

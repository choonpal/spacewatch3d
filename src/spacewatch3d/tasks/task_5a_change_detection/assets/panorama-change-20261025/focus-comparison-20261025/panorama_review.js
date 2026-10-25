import {PanoramaCanvas,normalize,sub,rotate} from './panorama_canvas.js';
const $=s=>document.querySelector(s),n=x=>x.toLocaleString('ko-KR'),sides=['before','after'],names={baseline:'기존',simsac:'SimSaC'};
const images=new Map();let data,interaction,selected=[],current,viewer,activeRegion=null,regions=[],loadId=0;
const excluded=()=>$('#exclude-person').checked;
const methods=()=>$('#method')?.value==='baseline'?['baseline']:$('#method')?.value==='simsac'?['simsac']:['baseline','simsac'];
function image(path){if(!images.has(path))images.set(path,new Promise((resolve,reject)=>{const i=new Image();i.onload=()=>resolve(i);i.onerror=()=>reject(new Error('이미지를 읽지 못했습니다: '+path));i.src=path;}));return images.get(path);}
function counts(c){
 const result={baseline:{before:0,after:0},simsac:{before:0,after:0}},saved=interaction?.cases[c.id]?.regions[excluded()?'filtered':'raw'];
 for(const method of ['baseline','simsac'])for(const side of sides){
  if(saved)result[method][side]=saved[method].reduce((sum,r)=>sum+r[side+'_pixels'],0);
  else result[method][side]=method==='baseline'?data.frames[c[side]][excluded()?'candidate_pixels_filtered':'candidate_pixels']:c.stats[side][excluded()?'filtered_pixels':'raw_pixels'];
 }return result;
}
function matches(c){const value=$('#filter').value,stats=counts(c),has=methods().some(m=>sides.some(s=>stats[m][s]>0));
 return(!$('#zone').value||c.zone===$('#zone').value)&&(value!=='changes'||has)&&(value!=='person'||sides.some(s=>c.stats[s].removed_person_pixels>0))&&(value!=='candidate'||sides.some(s=>stats.baseline[s]>0))&&(value!=='close'||has&&c.camera_gap_h<=.5);
}
function setProjection(value){$('#projection').value=value;document.body.dataset.projection=value;if(viewer){viewer.full=value==='full';viewer.draw();}}
function aim(point,fov=65){viewer.lookAt(sub(point,current.cameras.before.center),fov);}
function regionText(r){return`${names[r.method]} ${r.number} · 전 ${n(r.before_pixels)} / 후 ${n(r.after_pixels)} px`;}
function directionAtUV(u,v){const lo=(u-.5)*2*Math.PI,la=(v-.5)*Math.PI;return rotate(current.cameras.before.rotation,[Math.sin(lo)*Math.cos(la),Math.sin(la),Math.cos(lo)*Math.cos(la)]);}
function rectsFor(r){const [center,half,v0,v1]=r.bounds,points=[];
 for(let y=0;y<=8;y++)for(let x=0;x<=12;x++){const p=viewer.projectDirection(directionAtUV(center-half+half*2*x/12,v0+(v1-v0)*y/8));if(p.visible)points.push(p);}
 if(!points.length)return[];let groups=[points];if(viewer.full&&Math.max(...points.map(p=>p.x))-Math.min(...points.map(p=>p.x))>1.5)groups=[points.filter(p=>p.x<0),points.filter(p=>p.x>=0)];
 return groups.filter(p=>p.length).map(group=>{let x0=Math.min(...group.map(p=>p.x)),x1=Math.max(...group.map(p=>p.x)),y0=Math.min(...group.map(p=>p.y)),y1=Math.max(...group.map(p=>p.y));
  if(x1< -1||x0>1||y1< -1||y0>1)return null;x0=Math.max(-1,x0);x1=Math.min(1,x1);y0=Math.max(-1,y0);y1=Math.min(1,y1);
  return{left:(x0+1)*50,top:(1-y1)*50,width:Math.max(.8,(x1-x0)*50),height:Math.max(1.4,(y1-y0)*50)};}).filter(Boolean);
}
function drawBoxes(){
 const host=$('#boxes');if(!current||!viewer||!host)return;host.hidden=!$('#show-boxes').checked;host.replaceChildren();if(host.hidden)return;
 for(const r of regions)for(const rect of rectsFor(r)){const button=document.createElement('button');button.type='button';button.dataset.regionBox=r.id;button.className='candidate-box '+r.method+(activeRegion?.id===r.id?' selected':'');button.setAttribute('aria-label',regionText(r)+' 후보 박스');button.title=regionText(r)+' · 클릭하면 후 영상 겹침';for(const[k,v]of Object.entries(rect))button.style[k]=v+'%';const label=document.createElement('span');label.textContent=names[r.method]+' '+r.number;button.append(label);button.addEventListener('click',event=>{if(event.detail===0)selectRegion(r);});host.append(button);}
}
function clearRegion(){activeRegion=null;viewer.ghostEnabled=false;$('#ghost-panel').hidden=true;$('#after-label').hidden=true;for(const b of $('#region-list').children)b.setAttribute('aria-pressed','false');viewer.draw();}
function setOpacity(value){if(!activeRegion)return;const opacity=Math.max(0,Math.min(100,Number(value)));$('#ghost-opacity').value=opacity;$('#ghost-value').textContent=opacity+'%';viewer.ghostOpacity=opacity/100;viewer.ghostEnabled=opacity>0;$('#after-label').hidden=opacity===0;viewer.draw();}
function selectRegion(r){
 activeRegion=r;viewer.ghostBounds=r.bounds;setProjection('look');aim(r.anchor);$('#ghost-panel').hidden=false;
 $('#selection-title').textContent=regionText(r);const coverage=excluded()?r.ghost_coverage_filtered:r.ghost_coverage;
 $('#selection-details').textContent=`선택한 박스 안에 후 영상을 겹칩니다. 전 깊이·사람 제외 후 유효 표면 ${(coverage*100).toFixed(0)}%.${r.before_pixels===0?' 후에서만 검출된 영역을 전 시점에 투영한 후보입니다.':''}${r.after_rotation_fallback_pixels?' 일부 후 후보는 깊이가 없어 방향만 맞춘 근사 위치입니다.':''}${coverage===0?' 이 영역에는 겹쳐 볼 수 있는 깊이 표면이 없습니다.':''} 촬영 위치·가림·정합 잔차는 남습니다.`;
 const a=data.frames[current.after];$('#after-label').textContent=`후 · Dorm${a.epoch+2} · frame ${a.frame} · 선택 영역 겹침`;
 for(const b of $('#region-list').children)b.setAttribute('aria-pressed',String(b.dataset.region===r.id));setOpacity(55);
}
function updateRegions(){
 const saved=interaction.cases[current.id].regions[excluded()?'filtered':'raw'];regions=methods().flatMap(m=>saved[m].map((r,i)=>({...r,number:i+1}))).sort((a,b)=>(b.before_pixels+b.after_pixels)-(a.before_pixels+a.after_pixels));
 $('#region-summary').textContent=`· ${regions.length}개`;$('#region-list').replaceChildren();
 for(const r of regions){const button=document.createElement('button');button.type='button';button.className=r.method;button.dataset.region=r.id;button.setAttribute('aria-pressed','false');button.textContent=regionText(r);button.addEventListener('click',()=>{selectRegion(r);$('#panorama-stage').scrollIntoView({behavior:'smooth',block:'center'});});$('#region-list').append(button);}
 clearRegion();viewer.exclude=excluded();const first=regions.find(r=>r.method==='baseline')||regions[0];aim(first?.anchor||current.anchor);updateNumbers();
}
function updateNumbers(){
 if(!current)return;const c=counts(current),has=methods().some(m=>sides.some(s=>c[m][s]>0)),status=$('#candidate-status');status.classList.toggle('empty-candidates',!has);
 status.textContent=has?`전 영상에 후보 박스 ${regions.length}개 · ${methods().map(m=>`${names[m]} 전 ${n(c[m].before)} / 후 ${n(c[m].after)} px`).join(' · ')} · 박스를 눌러 후 영상 확인`:`현재 설정에서 변화 후보가 없습니다. SimSaC 원본 전 ${n(current.stats.before.raw_pixels)} / 후 ${n(current.stats.after.raw_pixels)} px 중 사람 제외 전 ${n(excluded()?current.stats.before.removed_person_pixels:0)} / 후 ${n(excluded()?current.stats.after.removed_person_pixels:0)} px.`;
 $('#numbers').innerHTML=sides.map(side=>{const f=data.frames[current[side]],s=current.stats[side];return`<p><b>${side==='before'?'전':'후'} · Dorm${f.epoch+2} · frame ${f.frame}</b><br>SimSaC 원본 ${n(s.raw_pixels)} → 사람 제외 후 ${n(excluded()?s.filtered_pixels:s.raw_pixels)} px · 전 시점 박스에 포함 ${n(c.simsac[side])} px<br>기존 후보 표면 ${n(excluded()?f.candidate_pixels_filtered:f.candidate_pixels)} px · 전 시점 박스에 포함 ${n(c.baseline[side])} px</p>`;}).join('');
}
async function show(){
 const i=selected.findIndex(c=>c.id===$('#pair').value),c=selected[i];$('#prev').disabled=i<=0;$('#next').disabled=i<0||i>=selected.length-1;if(!c)return;
 current=c;const token=++loadId;$('#case').classList.add('loading');$('#error').textContent='';$('#case-title').textContent=c.zone+' · 전 영상의 변화 후보';$('#case-count').textContent=`${i+1} / ${selected.length}쌍`;activeRegion=null;viewer.ghostEnabled=false;regions=[];$('#boxes').replaceChildren();$('#ghost-panel').hidden=true;$('#after-label').hidden=true;
 try{const b=data.frames[c.before],a=data.frames[c.after],entry=interaction.cases[c.id];const [rgb,mask,ghost,valid]=await Promise.all([image(b.rgb),image(b.baseline),image(entry.ghost_rgb),image(entry.ghost_mask)]);if(token!==loadId)return;
  viewer.setScene(rgb,mask,c.cameras.before);viewer.setGhost(ghost,valid);$('#phase-label').textContent=`전 · Dorm${b.epoch+2} · frame ${b.frame} · 드래그로 둘러보기`;
  $('#original-links').innerHTML=`<a href="${b.rgb}" target="_blank" rel="noopener">전 360° 원본 전체</a><a href="${a.rgb}" target="_blank" rel="noopener">후 360° 원본 전체</a>`;
  $('#alignment-status').textContent=`비교 쌍: 전 ${b.frame} ↔ 후 ${a.frame} · 카메라 간격 ${c.camera_gap_u.toFixed(3)} 맵 단위 (기준높이의 ${(c.camera_gap_h*100).toFixed(0)}%). 후 겹침은 실제 후 RGB를 깊이로 재투영하며 가림·정합 잔차가 남습니다.`;updateRegions();
 }catch(error){$('#error').textContent=error.stack;}finally{if(token===loadId)$('#case').classList.remove('loading');}
}
function refresh(){
 const previous=$('#pair').value;selected=data.cases.filter(matches);$('#pair').innerHTML=selected.map(c=>{const b=data.frames[c.before],a=data.frames[c.after],s=counts(c);return`<option value="${c.id}">${c.zone} · 전 ${b.frame} ↔ 후 ${a.frame} · ${methods().map(m=>`${names[m]} ${n(s[m].before)}/${n(s[m].after)}px`).join(' · ')}</option>`;}).join('');if(selected.some(c=>c.id===previous))$('#pair').value=previous;
 const scoped=data.cases.filter(c=>!$('#zone').value||c.zone===$('#zone').value);$('#list-status').textContent=`${scoped.length}쌍 중 ${selected.length}쌍 표시 · ${excluded()?'사람 제외 적용':'사람 제외 해제'} · 후보 수는 전 / 후 순서입니다. 후보는 확정 변화가 아닙니다.`;$('#empty').hidden=selected.length>0;$('#case').hidden=selected.length===0;if(!selected.length){++loadId;current=null;regions=[];return;}show();
}
async function init(){
 [data,interaction]=await Promise.all(['panorama-review.json','panorama-interactions.json'].map(async path=>{const r=await fetch(path);if(!r.ok)throw new Error('360° 비교 자료 HTTP '+r.status+' · '+path);return r.json();}));
 viewer=new PanoramaCanvas($('#panorama-before'),0,()=>drawBoxes(),(driver,direction)=>{setProjection('look');driver.lookAt(direction,Math.max(35,driver.fov*.75));},setProjection);viewer.onDraw=drawBoxes;viewer.onBoxClick=id=>{const r=regions.find(r=>r.id===id);if(r)selectRegion(r);};setProjection($('#projection').value);
 for(const id of ['zone','filter','method','exclude-person'])$('#'+id).addEventListener('change',refresh);$('#pair').addEventListener('change',show);$('#projection').addEventListener('change',()=>setProjection($('#projection').value));$('#show-boxes').addEventListener('change',drawBoxes);
 $('#ghost-opacity').addEventListener('input',()=>setOpacity($('#ghost-opacity').value));$('#show-before').addEventListener('click',()=>setOpacity(0));$('#show-ghost').addEventListener('click',()=>setOpacity(55));$('#show-after').addEventListener('click',()=>setOpacity(100));$('#close-selection').addEventListener('click',clearRegion);
 $('#reset').addEventListener('click',()=>{setProjection('look');if(current)aim((regions.find(r=>r.method==='baseline')||regions[0])?.anchor||current.anchor);});
 for(const[id,delta]of [['zoom-in',-12],['zoom-out',12]])$('#'+id).addEventListener('click',()=>{setProjection('look');viewer.setView(viewer.yaw,viewer.pitch,viewer.fov+delta);});
 $('#fullscreen').addEventListener('click',async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await $('#panorama-stage').requestFullscreen();}catch(error){$('#error').textContent=error.message;}});
 for(const[id,delta]of [['prev',-1],['next',1]])$('#'+id).addEventListener('click',()=>{const i=selected.findIndex(c=>c.id===$('#pair').value)+delta;if(selected[i]){$('#pair').value=selected[i].id;show();}});refresh();
}
init().catch(error=>{$('#error').textContent=error.stack;});

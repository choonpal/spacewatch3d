import {inspectVideoFile,uploadVideo} from './autotour.js';
import {validateTrajectory} from './trajectory-math.js';
import {Panorama} from './panorama.js';
import {TrajectoryController} from './trajectory.js';
import {uid,clamp,escapeHTML as esc,formatTime,newScene,emptyProject,validateProject,removeScene} from './core.js';

const paths={
  compass:'<circle cx="12" cy="12" r="9"/><path d="m16 8-2.5 5.5L8 16l2.5-5.5Z"/>',
  edit:'<path d="m14 5 5 5M4 20l5-1 11-11a2 2 0 0 0-5-5L4 14Z"/>',
  folder:'<path d="M3 7V5h6l2 2h10v13H3Z"/>',download:'<path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5"/>',
  help:'<circle cx="12" cy="12" r="9"/><path d="M9.5 9a2.5 2.5 0 1 1 4 2c-1 .5-1.5 1-1.5 2M12 16h.01"/>',
  plus:'<path d="M12 5v14M5 12h14"/>',minus:'<path d="M5 12h14"/>',shield:'<path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6ZM8 12l3 3 5-6"/>',
  sidebar:'<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16"/>',panorama:'<path d="M3 5q9 4 18 0v14q-9-4-18 0ZM3 15l5-5 5 5 3-3 5 4"/><circle cx="16" cy="9" r="1"/>',
  reset:'<path d="M4 10a8 8 0 1 1 1 7M4 4v6h6"/>',fullscreen:'<path d="M8 3H3v5M16 3h5v5M3 16v5h5M21 16v5h-5"/>',
  drag:'<path d="M8 12V5a2 2 0 0 1 4 0v7-4a2 2 0 0 1 4 0v4-2a2 2 0 0 1 4 0v5c0 4-2 6-6 6h-1c-3 0-4-1-6-4l-3-4a2 2 0 0 1 3-2Z"/>',
  play:'<path d="m8 5 11 7-11 7Z" fill="currentColor" stroke="none"/>',pause:'<path d="M8 5v14M16 5v14" stroke-width="4"/>',
  mute:'<path d="m11 4-5 5H3v6h3l5 5ZM16 9l6 6m0-6-6 6"/>',sound:'<path d="m11 4-5 5H3v6h3l5 5ZM15 8a6 6 0 0 1 0 8m3-11a10 10 0 0 1 0 14"/>',
  rotate:'<path d="M20 10a8 8 0 0 0-14-5L3 8m0-5v5h5M4 14a8 8 0 0 0 14 5l3-3m0 5v-5h-5"/>',
  bookmark:'<path d="M6 3h12v18l-6-4-6 4ZM9 9h6m-3-3v6"/>',close:'<path d="m6 6 12 12M6 18 18 6"/>',upload:'<path d="M12 17V3m-5 5 5-5 5 5M4 15v6h16v-6"/>',
  arrow:'<path d="m8 4 8 8-8 8"/>',link:'<path d="m7 10 5-5 5 5M7 17l5-5 5 5"/>',info:'<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7h.01"/>',
  trash:'<path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7m4-7v7"/>',check:'<path d="m5 12 4 4L19 6"/>',camera:'<path d="M3 7h5l2-3h4l2 3h5v14H3Z"/><circle cx="12" cy="13" r="4"/>',file:'<path d="M5 3h9l5 5v13H5ZM14 3v6h5M8 13h8m-8 4h6"/>'};
const icon=name=>`<svg viewBox="0 0 24 24" aria-hidden="true">${paths[name]||paths.panorama}</svg>`;
const $=id=>document.getElementById(id);
document.querySelectorAll('[data-icon]').forEach(el=>el.innerHTML=icon(el.dataset.icon));
const KEY='autotour360.project.v1';
let libraryTimer,libraryBusy=false,adding=false,uploadController;
const activeJobs=new Set();
let project=emptyProject(),activeId=null,editing=false,renderer,media=null,mediaAssetId=null,loadVersion=0,config={media:[]};
let saveTimer,toastTimer,dbPromise,muted=true,volume=.7,rate=1,relinkAssetId=null,sceneLoading=false,trajectory;
const sources=new Map(),objectURLs=new Set();
const current=()=>project.scenes.find(s=>s.id===activeId);
const descriptor=id=>project.assets.find(a=>a.id===id);
function toast(text,error=false){clearTimeout(toastTimer);$('toast').textContent=text;$('toast').classList.toggle('error',error);$('toast').hidden=false;toastTimer=setTimeout(()=>$('toast').hidden=true,error?6500:3500);}
function storageStatus(text,error=false){$('save-status').textContent=text;$('save-status').classList.toggle('error',error);}
function saveNow(){clearTimeout(saveTimer);try{localStorage.setItem(KEY,JSON.stringify(project));storageStatus('이 브라우저에 자동 저장됨');}catch{storageStatus('자동 저장 불가 · 투어 파일로 저장해 주세요',true);}}
function changed(){storageStatus('저장 중…');clearTimeout(saveTimer);saveTimer=setTimeout(saveNow,250);updateHeader();}
function removePresetNavigation(tour){
  let removed=false;
  for(const scene of tour.scenes){
    const previous=scene.hotspots.length;
    scene.hotspots=scene.hotspots.filter(h=>h.type!=='link'||!['이전 지점으로','다음 지점으로'].includes(h.title.trim()));
    removed ||= scene.hotspots.length!==previous;
  }
  return removed;
}
window.addEventListener('beforeunload',()=>{saveNow();for(const url of objectURLs)URL.revokeObjectURL(url);});
function database(){if(!dbPromise)dbPromise=new Promise((resolve,reject)=>{const r=indexedDB.open('autotour360-media',1);r.onupgradeneeded=()=>r.result.createObjectStore('files');r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);});return dbPromise;}
function fingerprint(a){return `${a.name}:${a.size}:${a.lastModified||0}`;}
async function storedFile(a){try{const db=await database();return await new Promise((resolve,reject)=>{const r=db.transaction('files').objectStore('files').get(fingerprint(a));r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);});}catch{return null;}}
async function keepFile(a,file){try{const db=await database();await new Promise((resolve,reject)=>{const tx=db.transaction('files','readwrite');tx.objectStore('files').put(file,fingerprint(a));tx.oncomplete=resolve;tx.onerror=()=>reject(tx.error);tx.onabort=()=>reject(tx.error);});}catch{toast('브라우저 보관 공간이 부족합니다. 투어 설정은 저장되며, 다음 실행 때 원본 파일을 다시 선택해 주세요.',true);}}
function fileSource(file){const url=URL.createObjectURL(file);objectURLs.add(url);return {url,file};}
async function resolveSources(){
  for(const url of objectURLs)URL.revokeObjectURL(url);objectURLs.clear();
  sources.clear();
  for(const a of project.assets){
    const server=config.media.find(m=>(a.trajectoryKey&&m.trajectoryKey===a.trajectoryKey)||(m.name===a.name&&m.size===a.size));
    if(server){a.trajectoryKey=server.trajectoryKey;sources.set(a.id,{url:server.url,trajectoryKey:server.trajectoryKey});}
    else{const file=await storedFile(a);if(file)sources.set(a.id,fileSource(file));}
  }
}
function updateHeader(){
  $('project-title').textContent=project.title;
  const count=project.scenes.reduce((n,s)=>n+s.hotspots.length,0);
  $('tour-summary').textContent=`영상 ${project.assets.length}개 · 연결 및 설명 지점 ${count}개`;
  const scene=current();if(scene){$('active-title').textContent=scene.title;$('active-description').textContent=['촬영 경로를 클릭해 공간을 둘러보세요.','바닥 화살표를 눌러 동선을 이동하세요.','방향 버튼을 눌러 동선을 이동하세요.'].includes(scene.description)?'바닥 화살표를 눌러 동선을 이동하세요.':scene.description;$('scene-kicker').textContent='360° VIRTUAL TOUR';}
}
function renderHotspots(){
  $('hotspots').replaceChildren();
  for(const h of current()?.hotspots||[]){const b=document.createElement('button');b.className=`hotspot ${h.type}`;b.dataset.id=h.id;b.setAttribute('aria-label',`${h.type==='link'?'이동':'설명'}: ${h.title}`);b.innerHTML=`<span class="hotspot-disc">${icon(h.type)}</span><span class="hotspot-label">${esc(h.title)}</span>`;b.onclick=()=>editing?hotspotDialog(h.type,h):activateHotspot(h);$('hotspots').append(b);}
  updateView();
}
function updateView(){
  if(!renderer)return;
  const view=renderer.view;$('heading-value').textContent=`${String(Math.round((view.yaw+360)%360)).padStart(3,'0')}°`;$('compass-needle').style.transform=`rotate(${-view.yaw}deg)`;$('auto-rotate').setAttribute('aria-pressed',String(renderer.autoRotate));
  for(const el of $('hotspots').children){const h=current()?.hotspots.find(x=>x.id===el.dataset.id);const p=h&&renderer.project(h.yaw,h.pitch);el.hidden=!p || !media || sceneLoading;if(p){el.style.left=`${p.x}px`;el.style.top=`${p.y}px`;}}
  if($('live-yaw')){$('live-yaw').textContent=`${Math.round(view.yaw)}°`;$('live-pitch').textContent=`${Math.round(view.pitch)}°`;$('live-fov').textContent=`${Math.round(view.fov)}°`;}
}
new ResizeObserver(updateView).observe($('viewport'));
function activateHotspot(h){if(h.type==='link')loadScene(h.targetId);else openDialog(h.title,`<p>${esc(h.body||'아직 등록된 설명이 없습니다.')}</p><div class="dialog-actions"><button class="button primary" data-close>확인</button></div>`);}
function showMessage(title,text,action='미디어 선택'){$('message-title').textContent=title;$('message-text').textContent=text;$('message-action').textContent=action;$('viewer-message').hidden=false;$('loading').hidden=true;}
function setLoading(value){$('loading').hidden=!value;}
function disposeMedia(){
  renderer?.setMedia(null);
  if(media instanceof HTMLVideoElement){media.pause();media.removeAttribute('src');media.load();}
  media=null;mediaAssetId=null;updatePlayer();
}
function waitFor(target,event,signal){return new Promise((resolve,reject)=>{
  const cleanup=()=>{clearTimeout(timeout);target.removeEventListener(event,done);target.removeEventListener('error',error);signal?.removeEventListener('abort',cancel);};
  const done=()=>{cleanup();resolve();};const error=()=>{cleanup();reject(new Error('미디어를 열 수 없습니다. 파일 연결과 브라우저의 코덱 지원을 확인해 주세요.'));};const cancel=()=>{cleanup();reject(new DOMException('Aborted','AbortError'));};
  const timeout=setTimeout(()=>{cleanup();reject(new Error('미디어 응답이 지연되고 있습니다. 원본 파일을 다시 연결해 주세요.'));},30000);
  target.addEventListener(event,done,{once:true});target.addEventListener('error',error,{once:true});signal?.addEventListener('abort',cancel,{once:true});
});}
let loadController;
async function loadScene(id){
  const version=++loadVersion;loadController?.abort();loadController=new AbortController();const signal=loadController.signal;
  sceneLoading=false;
  const scene=project.scenes.find(s=>s.id===id);if(!scene){disposeMedia();activeId=null;void trajectory?.syncAsset();renderHotspots();renderInspector();$('media-name').textContent='미디어 없음';$('resolution').textContent='EQUIRECTANGULAR';showMessage('공간을 연결하는 새로운 방법','360도 영상을 추가하면 촬영 경로를 자동으로 만듭니다.\n원본 영상은 외부로 전송되지 않습니다.');return;}
  activeId=id;showWelcome(false);const a=descriptor(scene.assetId),source=sources.get(a.id);
  void trajectory?.syncAsset();
  sceneLoading=true;renderer.autoRotate=false;renderer.setView(scene.view);$('viewer-message').hidden=true;setLoading(true);updatePlayer();updateHeader();renderHotspots();renderInspector();
  if(!source){sceneLoading=false;disposeMedia();showMessage('원본 파일을 연결해 주세요',`${a.name}\n투어 설정은 준비되어 있습니다. 이 장면에서 사용할 원본 파일을 선택해 주세요.`,'원본 파일 연결');return;}
  try{
    if(mediaAssetId!==a.id || !media){
      disposeMedia();mediaAssetId=a.id;
      if(a.type==='video'){
        const video=document.createElement('video');media=video;video.preload='auto';video.playsInline=true;video.muted=muted;video.volume=volume;video.playbackRate=rate;
        for(const event of ['timeupdate','durationchange','play','pause','ended','volumechange'])video.addEventListener(event,()=>{if(media===video)updatePlayer();});
        video.addEventListener('waiting',()=>{if(media===video)setLoading(true);});video.addEventListener('playing',()=>{if(media===video&&!sceneLoading)setLoading(false);});
        video.addEventListener('seeked',()=>{if(media===video){renderer.frameDirty=true;if(!sceneLoading)setLoading(false);updatePlayer();}});
        video.addEventListener('error',()=>{if(media===video)showMessage('영상을 재생할 수 없습니다','원본 파일과 H.264/AAC 코덱 지원을 확인해 주세요.','원본 파일 연결');});
        video.src=source.url;video.load();
      }else{
        const image=new Image();media=image;image.src=source.url;
      }
    }
    const loadedMedia=media;
    if(loadedMedia instanceof HTMLVideoElement){
      // Metadata is enough to initiate a new seek, even while an older seek is pending.
      // Waiting for loadeddata here can deadlock repeated selections of the same video.
      if(loadedMedia.readyState<1)await waitFor(loadedMedia,'loadedmetadata',signal);
      if(version!==loadVersion)return;
      loadedMedia.pause();const time=clamp(scene.startTime,0,Math.max(0,(loadedMedia.duration||0)-.05));
      if(loadedMedia.seeking||Math.abs(loadedMedia.currentTime-time)>.025){const sought=waitFor(loadedMedia,'seeked',signal);loadedMedia.currentTime=time;await sought;}
      if(version!==loadVersion)return;
      if(loadedMedia.readyState<2)await waitFor(loadedMedia,'canplay',signal);
    }else if(!loadedMedia.complete){await waitFor(loadedMedia,'load',signal);}
    if(version!==loadVersion)return;
    renderer.setMedia(loadedMedia);renderer.frameDirty=true;renderer.render();
    const w=loadedMedia.videoWidth||loadedMedia.naturalWidth,h=loadedMedia.videoHeight||loadedMedia.naturalHeight;
    $('resolution').textContent=`${w} × ${h}`;$('media-name').textContent=a.name;sceneLoading=false;setLoading(false);updatePlayer();renderHotspots();renderAnalysis();
    if(Math.abs(w/h-2)>.15)toast('2:1 비율 영상이 아닙니다. 완성된 360° 등거리 원통도법 영상인지 확인해 주세요.',true);
  }catch(error){if(version!==loadVersion||error.name==='AbortError')return;sceneLoading=false;disposeMedia();showMessage('장면을 열 수 없습니다',error.message,'원본 파일 연결');}
}
function updatePlayer(){
  const video=media instanceof HTMLVideoElement?media:null,ready=video&&video.readyState>=2&&!sceneLoading;
  $('play').disabled=!ready;$('timeline').disabled=!ready;$('mute').disabled=!video;$('volume').disabled=!video;$('speed').disabled=!video;
  $('play').innerHTML=icon(video&&!video.paused?'pause':'play');$('play').setAttribute('aria-label',video&&!video.paused?'일시정지':'재생');
  const duration=Number.isFinite(video?.duration)?video.duration:0,time=video?.currentTime||0;$('timeline').max=duration||1;$('timeline').value=time;$('timeline').style.setProperty('--progress',`${duration?time/duration*100:0}%`);$('current-time').textContent=formatTime(time);$('duration').textContent=video?formatTime(duration):'사진';
  $('mute').innerHTML=icon(muted||volume===0?'mute':'sound');$('mute').setAttribute('aria-label',muted?'음소거 해제':'음소거');$('volume').style.setProperty('--progress',`${muted?0:volume*100}%`);
}
function seekVideo(time){
  if(!(media instanceof HTMLVideoElement)||sceneLoading||media.readyState<1||!Number.isFinite(time)||!Number.isFinite(media.duration))return;
  media.currentTime=clamp(time,0,media.duration);renderer.frameDirty=true;updatePlayer();
}
async function togglePlay(){if(!(media instanceof HTMLVideoElement))return;try{if(media.paused){if(media.ended)media.currentTime=current()?.startTime||0;await media.play();}else media.pause();}catch{toast('재생 버튼을 다시 눌러 주세요. 브라우저가 자동 재생을 제한했거나 코덱을 지원하지 않습니다.',true);}}
function toggleMute(){muted=!muted;if(media instanceof HTMLVideoElement)media.muted=muted;updatePlayer();}
function setEditing(value){editing=value;document.body.classList.toggle('is-editing',value);$('inspector').hidden=!value;$('crosshair').hidden=!value;$('browse-mode').classList.toggle('selected',!value);$('edit-mode').classList.toggle('selected',value);$('browse-mode').setAttribute('aria-pressed',String(!value));$('edit-mode').setAttribute('aria-pressed',String(value));renderInspector();renderHotspots();}
function renderInspector(){
  const s=current(),el=$('inspector-content');if(!s){el.innerHTML='<div class="inspector-block"><p class="hint">미디어를 추가하면 장면을 편집할 수 있습니다.</p></div>';return;}
  const a=descriptor(s.assetId);
  el.innerHTML=`<div class="inspector-block"><label class="field">장면 이름<input id="scene-name" value="${esc(s.title)}" maxlength="80"></label><label class="field">장면 설명<textarea id="scene-description" maxlength="2000" rows="2" placeholder="이 공간을 소개해 주세요.">${esc(s.description)}</textarea></label><div class="field-row"><label class="field">시작 시각 (초)<input id="scene-start" type="number" min="0" max="864000" step="0.1" value="${s.startTime.toFixed(1)}" ${a.type==='image'?'disabled':''}></label></div><p class="hint">${esc(a.name)}</p><button id="relink-media" class="text-button">${icon('folder')}원본 파일 다시 연결</button></div>
    <div class="inspector-block"><h3>시작 시점</h3><p class="hint">장면에 들어왔을 때 처음 보이는 방향입니다.</p><div class="view-values"><span><b id="live-yaw">0°</b>좌우</span><span><b id="live-pitch">0°</b>상하</span><span><b id="live-fov">75°</b>시야각</span></div><button id="save-view" class="button subtle small-button wide">${icon('camera')}현재 시각·방향으로 설정</button></div>
    <div class="inspector-block"><h3>이동 및 설명 지점 <span style="color:var(--accent)">${s.hotspots.length}</span></h3><div>${s.hotspots.map(h=>`<button class="hotspot-row" data-hotspot="${esc(h.id)}"><span>${icon(h.type)}</span><span><strong>${esc(h.title)}</strong><small>${h.type==='link'?'장면으로 이동':'공간 설명'}</small></span><span>${icon('edit')}</span></button>`).join('')||'<p class="hint">원하는 곳을 화면 가운데에 놓고<br>새 지점을 추가하세요.</p>'}</div><div class="hotspot-add"><button id="add-link" class="button subtle small-button">${icon('plus')}이동</button><button id="add-info" class="button subtle small-button">${icon('plus')}설명</button></div><p class="hint" style="margin-top:12px">편집 중 지점을 누르면 수정할 수 있습니다.<br>둘러보기 모드에서 이동을 확인하세요.</p></div>
    <div class="inspector-block"><button id="delete-scene" class="delete-scene danger">${icon('trash')}이 장면 삭제</button></div>`;
  $('scene-name').oninput=e=>{s.title=e.target.value;changed();};$('scene-name').onblur=()=>{if(!s.title.trim()){s.title='새 장면';$('scene-name').value=s.title;changed();}};
  $('scene-description').oninput=e=>{s.description=e.target.value;changed();};
  $('scene-start').onchange=e=>{s.startTime=clamp(Number(e.target.value)||0,0,media instanceof HTMLVideoElement&&Number.isFinite(media.duration)?Math.max(0,media.duration-.05):864000);e.target.value=s.startTime.toFixed(1);changed();};
  $('relink-media').onclick=()=>chooseMedia(a.id);
  $('save-view').onclick=()=>{if(!media)return toast('원본 파일을 먼저 연결해 주세요.',true);s.view={...renderer.view};s.startTime=media instanceof HTMLVideoElement?media.currentTime:0;s.thumbnail=renderer.snapshot();changed();renderInspector();toast('현재 재생 시각과 방향을 시작 시점으로 저장했습니다.');};
  $('add-link').onclick=()=>hotspotDialog('link');$('add-info').onclick=()=>hotspotDialog('info');
  el.querySelectorAll('[data-hotspot]').forEach(b=>b.onclick=()=>{const h=s.hotspots.find(x=>x.id===b.dataset.hotspot);hotspotDialog(h.type,h);});
  $('delete-scene').onclick=()=>deleteSceneDialog(s);updateView();
}
function openDialog(title,body){const d=$('dialog');if(d.open)d.close();$('dialog-content').innerHTML=`<div class="dialog-header"><h2>${esc(title)}</h2><button class="icon-button" data-close aria-label="창 닫기">${icon('close')}</button></div><div class="dialog-body">${body}</div>`;d.querySelectorAll('[data-close]').forEach(b=>b.onclick=()=>d.close());d.showModal();}
function hotspotDialog(type,existing){
  const s=current();if(!s)return;if(!existing&&s.hotspots.length>=100)return toast('한 장면에는 최대 100개의 지점을 추가할 수 있습니다.',true);
  const targets=project.scenes.filter(x=>x.id!==s.id);if(type==='link'&&!targets.length)return toast('이동할 장면이 필요합니다. ‘내 영상’에서 다른 영상을 추가해 주세요.',true);
  const position=existing||renderer.view;
  openDialog(existing?'지점 편집':type==='link'?'이동 지점 추가':'설명 지점 추가',`<form id="hotspot-form"><p class="hint" style="margin-bottom:18px">${existing?'등록된 지점을 수정합니다.':'현재 화면의 가운데에 지점을 추가합니다.'} (${Math.round(position.yaw)}°, ${Math.round(position.pitch)}°)</p><label class="field">지점 이름<input id="hotspot-title" maxlength="80" required value="${esc(existing?.title||'')}" placeholder="${type==='link'?'예: 다음 공간으로':'예: 공간 안내'}"></label>${type==='link'?`<label class="field">이동할 장면<select id="hotspot-target">${targets.map(x=>`<option value="${esc(x.id)}" ${existing?.targetId===x.id?'selected':''}>${esc(x.title)}</option>`).join('')}</select></label>`:`<label class="field">설명<textarea id="hotspot-body" maxlength="3000" rows="5" placeholder="이 공간에 대한 설명을 적어 주세요.">${esc(existing?.body||'')}</textarea></label>`}${existing?'<label class="hint"><input id="move-hotspot" type="checkbox"> 위치도 현재 화면 가운데로 옮기기</label>':''}<div class="dialog-actions">${existing?'<button type="button" id="remove-hotspot" class="button danger">삭제</button>':''}<button type="button" class="button subtle" data-close>취소</button><button type="submit" class="button primary">${existing?'변경 저장':'지점 추가'}</button></div></form>`);
  $('hotspot-form').onsubmit=e=>{e.preventDefault();const title=$('hotspot-title').value.trim();if(!title)return;const h=existing||{id:uid(),type,yaw:renderer.view.yaw,pitch:renderer.view.pitch};h.title=title;h.body=$('hotspot-body')?.value||'';h.targetId=$('hotspot-target')?.value||'';if($('move-hotspot')?.checked){h.yaw=renderer.view.yaw;h.pitch=renderer.view.pitch;}if(!existing)s.hotspots.push(h);$('dialog').close();changed();renderInspector();renderHotspots();toast('지점을 저장했습니다. 둘러보기 모드에서 확인할 수 있습니다.');};
  if(existing)$('remove-hotspot').onclick=()=>{s.hotspots=s.hotspots.filter(h=>h.id!==existing.id);$('dialog').close();changed();renderInspector();renderHotspots();};
}
function deleteSceneDialog(s){
  openDialog('장면을 삭제할까요?',`<p>‘${esc(s.title)}’ 장면과 이 장면으로 연결된 이동 지점을 삭제합니다. 원본 영상 파일은 유지됩니다.</p><div class="dialog-actions"><button class="button subtle" data-close>취소</button><button id="confirm-delete" class="button danger">장면 삭제</button></div>`);
  $('confirm-delete').onclick=()=>{removeScene(project,s.id);$('dialog').close();changed();loadScene(project.scenes[0]?.id);};
}
function chooseMedia(relinkId=null){if(adding)return toast('현재 영상 입력을 마친 뒤 추가해 주세요.');relinkAssetId=relinkId;$('media-input').multiple=!relinkId;$('media-input').value='';$('media-input').click();}
async function addFiles(files,relinkId=null){
  if(adding)return toast('현재 영상 입력을 마친 뒤 추가해 주세요.');
  if(!config.trajectory?.available)return toast('분석 도구 설치가 필요합니다. 시작 화면의 설치 안내를 확인해 주세요.',true);
  adding=true;uploadController=new AbortController();
  try{
    for(const file of Array.from(files)){
      if(uploadController.signal.aborted)break;
      $('upload-panel').hidden=false;$('upload-name').textContent=file.name;$('upload-label').textContent='360도 영상 확인 중';$('upload-progress').value=0;
      await inspectVideoFile(file);
      if(uploadController.signal.aborted)break;
      $('upload-label').textContent='이 PC에 영상 보관 중';
      const item=await uploadVideo(file,fraction=>{$('upload-progress').value=fraction*100;$('upload-label').textContent=fraction===1?'영상을 확인하고 자동 분석을 시작합니다':`영상 보관 중 · ${Math.round(fraction*100)}%`;},uploadController.signal);
      const index=config.media.findIndex(m=>m.trajectoryKey===item.trajectoryKey);if(index<0)config.media.push(item);else config.media[index]=item;
      let asset;
      if(relinkId){asset=descriptor(relinkId);if(!asset)continue;disposeMedia();const sameSource=asset.trajectory?.source.sha256===item.trajectoryKey;Object.assign(asset,{name:item.name,size:item.size,type:'video',lastModified:0,trajectoryKey:item.trajectoryKey});if(sameSource)asset.trajectory.source.name=item.name;else{delete asset.trajectory;delete asset.routeEdits;}for(const scene of project.scenes.filter(s=>s.assetId===asset.id)){scene.thumbnail='';scene.startTime=Math.min(scene.startTime,item.duration-.05);}}
      else asset=ensureLibraryAsset(item);
      sources.set(asset.id,{url:item.url,trajectoryKey:item.trajectoryKey});changed();renderLibrary();
      await loadScene(project.scenes.find(s=>s.assetId===asset.id)?.id);await refreshLibrary();
      toast('영상이 연결되었습니다. 경로 분석이 자동으로 진행됩니다.');
      if(relinkId)break;
    }
  }catch(error){toast(error.name==='AbortError'?'영상 입력을 취소했습니다.':error.message,error.name!=='AbortError');}
  finally{adding=false;uploadController=null;$('upload-panel').hidden=true;renderLibrary();}
}
async function exportProject(){
  saveNow();
  try{
    const response=await fetch('/api/tours',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(project,null,2)}),result=await response.json();
    if(!response.ok)throw new Error(result.error||'투어를 저장하지 못했습니다.');
    const a=document.createElement('a');a.href=result.url;a.download=result.fileName;a.click();
    toast('투어 파일을 저장했습니다. 원본 영상도 함께 보관하세요.');
  }catch(error){toast(error.message,true);}
}
async function importProject(file){
  try{
    if(file.size>20*1024*1024)throw new Error('투어 JSON 파일은 최대 20MB까지 열 수 있습니다.');
    const next=validateProject(JSON.parse(await file.text()));
    if(next.assets.some(a=>a.type!=='video'))throw new Error('이 프로그램은 360도 영상 투어를 지원합니다. 사진이 포함된 투어는 기존 뷰어에서 열어 주세요.');
    openDialog('저장된 투어 열기',`<p>‘${esc(next.title)}’의 장면 ${next.scenes.length}개를 불러옵니다. 현재 작업을 보관하려면 먼저 저장해 주세요.</p><div class="dialog-actions"><button id="save-before-open" class="button subtle">현재 투어 저장</button><button id="confirm-import" class="button primary">투어 열기</button></div>`);
    $('save-before-open').onclick=exportProject;
    $('confirm-import').onclick=async()=>{++loadVersion;loadController?.abort();disposeMedia();project=next;removePresetNavigation(project);await resolveSources();$('dialog').close();saveNow();updateHeader();await loadScene(project.scenes[0]?.id);toast('투어를 불러왔습니다. 연결이 없는 장면은 원본 파일을 선택해 주세요.');};
  }catch(error){toast(error instanceof SyntaxError?'올바른 JSON 투어 파일이 아닙니다.':error.message,true);}
}
function showOpenDialog(){openDialog('영상과 투어 가져오기',`<p>영상을 넣으면 촬영 경로를 자동으로 만듭니다.</p><button id="dialog-add-media" class="dialog-choice"><span>${icon('panorama')}</span><div><strong>360° 영상 추가 → 자동 투어 생성</strong><small>MP4 · MOV · WebM / 2:1 파노라마 / 최대 2GB · 10분</small></div></button><button id="dialog-open-project" class="dialog-choice"><span>${icon('file')}</span><div><strong>저장한 투어 불러오기</strong><small>.tour.json · 경로, 장면, 설명을 이어서 사용합니다.</small></div></button><p class="hint" style="margin-top:20px">영상과 분석 결과는 프로그램을 실행한 PC에 보관됩니다. 외부 서비스에 전송하지 않습니다.</p>`);$('dialog-add-media').onclick=()=>{$('dialog').close();chooseMedia();};$('dialog-open-project').onclick=()=>{$('dialog').close();$('project-input').value='';$('project-input').click();};}
function showHelp(){openDialog('공간을 둘러보는 방법',`<div class="help-grid"><span>마우스 드래그 / 터치</span><span>원하는 방향으로 둘러보기</span><span>휠 / 두 손가락</span><span>확대 · 축소</span><span><kbd>←</kbd> <kbd>↑</kbd> <kbd>↓</kbd> <kbd>→</kbd></span><span>동선 이동 · Q/E로 좌우 둘러보기</span><span><kbd>Space</kbd> / <kbd>M</kbd></span><span>재생·정지 / 음소거</span><span><kbd>R</kbd> / <kbd>F</kbd></span><span>시점 초기화 / 전체 화면</span></div><p class="dialog-note">① <strong>영상 선택</strong>만 하면 촬영 경로 분석이 자동으로 시작됩니다.<br>② 바닥 화살표로 이동하고, 드래그로 둘러보세요. 미니맵에서도 이동할 수 있습니다.<br>③ <strong>촬영 경로</strong>에서 표시 설정을 바꾼 뒤 <strong>투어 저장</strong>으로 보관하세요.</p><p>편집 설정은 이 브라우저에 자동 저장됩니다. 투어 파일에는 영상이 포함되지 않으므로 원본도 함께 보관하세요. 360°로 이어 붙인 2:1 영상을 지원합니다. 분석된 동선은 실제 촬영 경로의 추정값이며, 위치가 확인되지 않은 구간은 동선 지도에서 연결하지 않습니다.</p><div class="dialog-actions"><button class="button primary" data-close>둘러보기 시작</button></div>`);}
async function fullscreen(){try{if(document.fullscreenElement)await document.exitFullscreen();else if($('viewer-shell').requestFullscreen)await $('viewer-shell').requestFullscreen();else toast('이 브라우저에서는 전체 화면 API를 지원하지 않습니다.',true);}catch{toast('브라우저에서 전체 화면을 허용하지 않았습니다.',true);}}
function bindEvents(){
  $('browse-mode').onclick=()=>setEditing(false);$('edit-mode').onclick=()=>setEditing(true);$('close-inspector').onclick=()=>setEditing(false);
  $('open-project').onclick=showOpenDialog;$('export-project').onclick=exportProject;
  $('media-input').onchange=()=>{const id=relinkAssetId;relinkAssetId=null;addFiles($('media-input').files,id);};$('project-input').onchange=()=>{if($('project-input').files[0])importProject($('project-input').files[0]);};
  $('message-action').onclick=()=>chooseMedia(current()?.assetId||null);
  $('play').onclick=togglePlay;$('mute').onclick=toggleMute;
  $('volume').oninput=e=>{volume=Number(e.target.value);muted=volume===0;if(media instanceof HTMLVideoElement){media.volume=volume;media.muted=muted;}updatePlayer();};
  $('speed').onchange=e=>{rate=Number(e.target.value);if(media instanceof HTMLVideoElement)media.playbackRate=rate;};
  $('timeline').oninput=e=>seekVideo(Number(e.target.value));
  $('zoom-in').onclick=()=>renderer.zoom(-8);$('zoom-out').onclick=()=>renderer.zoom(8);$('reset-view').onclick=()=>{renderer.autoRotate=false;renderer.setView(current()?.view||{yaw:0,pitch:0,fov:75});};
  $('fullscreen').onclick=fullscreen;$('auto-rotate').onclick=()=>{renderer.autoRotate=!renderer.autoRotate;updateView();};
  $('help').onclick=$('keyboard-help').onclick=showHelp;
  $('rename-project').onclick=()=>{openDialog('투어 이름 변경',`<form id="rename-form"><label class="field">투어 이름<input id="new-title" value="${esc(project.title)}" maxlength="100" required></label><div class="dialog-actions"><button type="button" class="button subtle" data-close>취소</button><button class="button primary" type="submit">저장</button></div></form>`);$('rename-form').onsubmit=e=>{e.preventDefault();const title=$('new-title').value.trim();if(!title)return;project.title=title;changed();$('dialog').close();};};
  document.addEventListener('keydown',e=>{
    if($('dialog').open||e.ctrlKey||e.metaKey||e.altKey||/INPUT|TEXTAREA|SELECT|BUTTON/.test(e.target.tagName)||e.target.isContentEditable)return;
    const key=e.key.toLowerCase(),v={...renderer.view};
    if(['arrowleft','arrowright','arrowup','arrowdown','q','e',' ','+','=','-','r','f','m','?'].includes(key))e.preventDefault();
    if(key.startsWith('arrow')){
      if(!editing&&trajectory.data&&trajectory.enabled){const direction={arrowup:'forward',arrowright:'right',arrowdown:'back',arrowleft:'left'}[key];if(!trajectory.navigateDirection(direction))toast('이 방향에 연결된 이동 지점이 없습니다.');}
      else{renderer.autoRotate=false;if(key==='arrowleft')v.yaw-=5;if(key==='arrowright')v.yaw+=5;if(key==='arrowup')v.pitch+=5;if(key==='arrowdown')v.pitch-=5;renderer.setView(v);}
    }
    if(key==='q'||key==='e'){renderer.autoRotate=false;v.yaw+=key==='q'?-15:15;renderer.setView(v);}
    if(key===' ')togglePlay();if(key==='m')toggleMute();if(key==='r')$('reset-view').click();if(key==='f')fullscreen();if(key==='+'||key==='=')renderer.zoom(-5);if(key==='-')renderer.zoom(5);if(key==='?')showHelp();
  });
  let dragDepth=0;document.addEventListener('dragenter',e=>{if(e.dataTransfer.types.includes('Files')){e.preventDefault();dragDepth++;$('drop-overlay').hidden=false;}});document.addEventListener('dragover',e=>{if(e.dataTransfer.types.includes('Files'))e.preventDefault();});
  document.addEventListener('dragleave',()=>{if(--dragDepth<=0){dragDepth=0;$('drop-overlay').hidden=true;}});
  document.addEventListener('drop',e=>{e.preventDefault();dragDepth=0;$('drop-overlay').hidden=true;const files=[...e.dataTransfer.files];if(files.length===1&&files[0].name.endsWith('.json'))importProject(files[0]);else addFiles(files);});
}
function showWelcome(value){
  document.body.classList.toggle('welcome-mode',value);$('welcome').hidden=!value;
  if(value){if(media instanceof HTMLVideoElement)media.pause();renderLibrary();}
}
function ensureLibraryAsset(item){
  let asset=project.assets.find(a=>a.trajectoryKey===item.trajectoryKey||(a.name===item.name&&a.size===item.size));
  if(!asset){
    if(project.assets.length>=200||project.scenes.length>=200)throw new Error('한 투어에 등록할 수 있는 영상 또는 장면 수를 초과했습니다.');
    asset={id:uid(),name:item.name,size:item.size,lastModified:0,type:'video',trajectoryKey:item.trajectoryKey};project.assets.push(asset);
    const scene=newScene(asset.id,item.name.replace(/\.[^.]+$/,'')+' · 시작');scene.description='바닥 화살표를 눌러 동선을 이동하세요.';project.scenes.push(scene);
  }
  asset.trajectoryKey=item.trajectoryKey;sources.set(asset.id,{url:item.url,trajectoryKey:item.trajectoryKey});return asset;
}
const stateLabel=state=>({complete:'투어 준비 완료',running:'경로 분석 중',queued:'분석 대기 중',cancelling:'분석 취소 중',cancelled:'분석 취소됨',failed:'분석 실패',interrupted:'분석 중단됨',idle:'분석 대기'}[state]||'확인 중');
function renderLibrary(){
  const list=$('library-list');list.replaceChildren();$('library-count').textContent=String(config.media.length);$('library-empty').hidden=!!config.media.length;
  for(const item of config.media){
    const b=document.createElement('button');b.className='library-card';b.dataset.key=item.trajectoryKey;
    const state=item.status?.state||'idle';b.innerHTML=`<div class="library-art" aria-hidden="true">${icon('panorama')}<span>360°</span></div><div class="library-info"><strong>${esc(item.name)}</strong><span>${formatTime(item.duration)} · ${(item.size/1024**2).toFixed(0)} MB</span><small class="state-pill ${state}">${stateLabel(state)}</small></div><span class="library-arrow">↗</span>`;
    b.onclick=async()=>{try{const asset=ensureLibraryAsset(item);changed();await loadScene(project.scenes.find(s=>s.assetId===asset.id)?.id);await refreshLibrary();}catch(error){toast(error.message,true);}};list.append(b);
  }
  const available=config.trajectory?.available;$('engine-badge').textContent=available?'자동 분석 준비됨':'분석 도구 설치 필요';$('setup-note').hidden=!!available;
}
function renderAnalysis(){
  const asset=descriptor(current()?.assetId),item=config.media.find(m=>m.trajectoryKey===asset?.trajectoryKey);const status=item?.status||trajectory?.status||{};
  const working=['queued','running','cancelling'].includes(status.state),ready=!!asset?.trajectory;
  $('analysis-strip').hidden=!asset;$('analysis-strip').dataset.state=working?'running':ready?'complete':status.state;
  $('analysis-title').textContent=working?stateLabel(status.state):ready?'자동 투어 준비 완료':stateLabel(status.state);
  $('analysis-detail').textContent=working?(status.message||'영상을 분석하고 있습니다.'):ready?`촬영 구간 ${(asset.trajectory.quality.coverage*100).toFixed(1)}% 연결 · 바닥 화살표로 이동하세요`:(status.message||'경로를 준비하고 있습니다.');
  $('analysis-progress').hidden=!working;$('analysis-progress').value=status.progress||0;
  $('analysis-cancel').hidden=!working;$('analysis-cancel').disabled=status.state==='cancelling';
  $('analysis-retry').hidden=working||ready;$('analysis-retry').disabled=!config.trajectory?.available||!item;
}
async function refreshLibrary(){
  if(libraryBusy)return;libraryBusy=true;
  try{
    const response=await fetch('/api/library');if(!response.ok)throw new Error('서버 연결 확인이 필요합니다.');config=await response.json();
    for(const item of config.media){
      const asset=project.assets.find(a=>a.trajectoryKey===item.trajectoryKey);if(!asset)continue;
      if(['queued','running','cancelling'].includes(item.status.state)){activeJobs.add(item.trajectoryKey);continue;}
      if(item.status.state==='complete'&&(!asset.trajectory||activeJobs.has(item.trajectoryKey))){
        const response=await fetch(`/api/trajectory/${item.trajectoryKey}/result`);if(!response.ok)continue;
        const data=validateTrajectory(await response.json());if(data.source.sha256!==item.trajectoryKey||data.source.size!==asset.size)continue;
        data.source.name=asset.name;asset.trajectory=data;activeJobs.delete(item.trajectoryKey);changed();
        if(asset.id===current()?.assetId)await trajectory.syncAsset(true);
      }
    }
    renderLibrary();renderAnalysis();
  }catch(error){$('engine-badge').textContent='서버 연결 확인 필요';if(!document.body.classList.contains('welcome-mode'))$('analysis-detail').textContent='서버에 연결할 수 없습니다. 프로그램 실행 창을 확인해 주세요.';}
  finally{libraryBusy=false;}
}
async function init(){
  try{renderer=new Panorama($('panorama'),updateView,message=>toast(message,true));}catch(error){showMessage('360° 화면을 시작할 수 없습니다',error.message);return;}
  trajectory=new TrajectoryController({getState:()=>({asset:descriptor(current()?.assetId),source:sources.get(current()?.assetId),sceneId:current()?.id,media,view:renderer.view,loading:sceneLoading,editing}),onChange:changed,onSeek:seekVideo,onNavigate:(time,view)=>{media.pause();renderer.autoRotate=false;renderer.setView(view);seekVideo(time);},notify:toast,openDialog,getConfig:()=>config});
  renderer.onFrame=()=>trajectory.draw();
  bindEvents();
  try{const r=await fetch('/api/config');if(r.ok)config=await r.json();}catch{}
  let restored=false;try{const saved=localStorage.getItem(KEY);if(saved){project=validateProject(JSON.parse(saved));restored=true;}}catch{toast('이전 자동 저장을 읽지 못했습니다. 저장한 투어 파일을 불러와 주세요.',true);}
  if(!restored)for(const item of config.media)ensureLibraryAsset(item);
  const removedNavigation=removePresetNavigation(project);
  if(!restored||removedNavigation)saveNow();
  await resolveSources();updateHeader();renderLibrary();
  showWelcome(true);
  $('welcome-add').onclick=()=>chooseMedia();$('welcome-import').onclick=()=>{$('project-input').value='';$('project-input').click();};
  $('library-home').onclick=()=>showWelcome(true);document.querySelector('.brand').onclick=e=>{e.preventDefault();showWelcome(true);};
  $('upload-cancel').onclick=()=>uploadController?.abort();
  $('analysis-cancel').onclick=async()=>{await trajectory.cancel();await refreshLibrary();};
  $('analysis-retry').onclick=async()=>{await trajectory.start(true);await refreshLibrary();};
  await refreshLibrary();libraryTimer=setInterval(refreshLibrary,1500);
}
void init();

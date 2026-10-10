import {formatTime,projectPoint} from './core.js';
import {validateTrajectory,poseAt,pickPath,dot,sub,cross,unit,rotate,lerp,directionalSteps,viewAfterStep,editedRouteSteps} from './trajectory-math.js';
import {keyframeRouteSteps} from './tour-keyframes.js';

const $=id=>document.getElementById(id);
const COLORS={future:'#65d9ff',past:'#efb875'};

export class TrajectoryController {
  constructor({getState,onChange,onSeek,onNavigate,notify,openDialog,getConfig}) {
    Object.assign(this,{getState,onChange,onSeek,onNavigate,notify,openDialog,getConfig});
    this.enabled=true;this.showMap=true;this.windowSeconds=12;this.heightFactor=1;this.data=null;
    this.asset=null;this.key=null;this.epoch=0;this.pollTimer=null;this.status={state:'idle'};this.lastDraw='';
    this.canvas=$('panorama');this.map=$('route-map');this.mapContext=this.map.getContext('2d');
    this.mapWrap=$('route-map-wrap');this.viewport=$('viewport');this.mapPosition=null;this.mapDrag=null;
    this.pathSegments=[];this.mapSegments=[];this.gestures=new Map();this.hover=null;
    this.steps=[];this.directions={};this.stepButtons=new Map();
    this.bindSeeking(this.map,true);
    this.bindMapDragging();
    this.map.addEventListener('keydown',e=>{
      const {media,loading}=this.getState();if(loading||!this.data||!(media instanceof HTMLVideoElement))return;
      const times={ArrowLeft:media.currentTime-5,ArrowDown:media.currentTime-5,ArrowRight:media.currentTime+5,ArrowUp:media.currentTime+5,Home:0,End:media.duration};
      if(!(e.key in times)||e.altKey||e.ctrlKey||e.metaKey)return;
      e.preventDefault();e.stopPropagation();this.onSeek(times[e.key]);this.clearHover();
    });
    $('trajectory-open').onclick=()=>this.openSettings();
    new ResizeObserver(()=>{this.positionMap();this.lastDraw='';this.draw();}).observe(this.canvas);
  }
  mapBounds(){
    const viewport=this.viewport.getBoundingClientRect(),panel=this.mapWrap.getBoundingClientRect();
    return {left:panel.left-viewport.left,top:panel.top-viewport.top,min:8,maxX:Math.max(8,viewport.width-panel.width-8),maxY:Math.max(8,viewport.height-panel.height-8)};
  }
  positionMap(){
    if(this.mapWrap.hidden)return;
    const style=this.mapWrap.style;
    if(!this.mapPosition){
      for(const property of ['left','top','right','bottom'])style.removeProperty(property);
      const bounds=this.mapBounds();
      // Keep the original default placement unless a smaller viewport clips it.
      if(bounds.left<bounds.min||bounds.left>bounds.maxX){style.left=`${Math.max(bounds.min,Math.min(bounds.maxX,bounds.left))}px`;style.right='auto';}
      if(bounds.top<bounds.min||bounds.top>bounds.maxY){style.top=`${Math.max(bounds.min,Math.min(bounds.maxY,bounds.top))}px`;style.bottom='auto';}
      return;
    }
    const bounds=this.mapBounds();
    style.left=`${bounds.min+this.mapPosition.x*(bounds.maxX-bounds.min)}px`;
    style.top=`${bounds.min+this.mapPosition.y*(bounds.maxY-bounds.min)}px`;style.right='auto';style.bottom='auto';
  }
  moveMap(left,top){
    const bounds=this.mapBounds();
    const x=Math.max(bounds.min,Math.min(bounds.maxX,left)),y=Math.max(bounds.min,Math.min(bounds.maxY,top));
    this.mapPosition={x:bounds.maxX>bounds.min?(x-bounds.min)/(bounds.maxX-bounds.min):0,y:bounds.maxY>bounds.min?(y-bounds.min)/(bounds.maxY-bounds.min):0};
    this.positionMap();
  }
  finishMapDrag(cancel=false){
    const gesture=this.mapDrag;if(!gesture)return;
    this.mapDrag=null;this.mapWrap.classList.remove('is-dragging');
    if(cancel){this.mapPosition=gesture.original;this.positionMap();}
    else if(gesture.moved)this.saveSettings();
    if(this.mapWrap.hasPointerCapture(gesture.id))this.mapWrap.releasePointerCapture(gesture.id);
  }
  bindMapDragging(){
    const handle=$('route-map-handle');
    this.mapWrap.addEventListener('pointerdown',e=>{
      if(e.button!==0)return;
      if(this.mapDrag){this.finishMapDrag(true);return;}
      if(!e.isPrimary)return;
      const bounds=this.mapBounds();
      this.mapDrag={id:e.pointerId,x:e.clientX,y:e.clientY,left:bounds.left,top:bounds.top,original:this.mapPosition?{...this.mapPosition}:null,moved:false};
      // The map canvas keeps its capture so a short click can still seek.
      if(e.target!==this.map){e.preventDefault();handle.focus({preventScroll:true});this.mapWrap.setPointerCapture(e.pointerId);}
      this.clearHover();e.stopPropagation();
    });
    this.mapWrap.addEventListener('pointermove',e=>{
      const gesture=this.mapDrag;if(!gesture||gesture.id!==e.pointerId)return;
      const dx=e.clientX-gesture.x,dy=e.clientY-gesture.y;
      if(!gesture.moved&&Math.hypot(dx,dy)<=(e.pointerType==='touch'?10:6))return;
      gesture.moved=true;const seekGesture=this.gestures.get(e.pointerId);if(seekGesture)seekGesture.cancelled=true;
      this.mapWrap.classList.add('is-dragging');this.clearHover();this.moveMap(gesture.left+dx,gesture.top+dy);e.preventDefault();
    });
    this.mapWrap.addEventListener('pointerup',e=>{if(this.mapDrag?.id===e.pointerId)this.finishMapDrag();});
    for(const event of ['pointercancel','lostpointercapture'])this.mapWrap.addEventListener(event,e=>{if(this.mapDrag?.id===e.pointerId)this.finishMapDrag(true);});
    const reset=()=>{this.finishMapDrag(true);this.mapPosition=null;this.positionMap();this.saveSettings();};
    handle.addEventListener('dblclick',reset);
    handle.addEventListener('keydown',e=>{
      if(e.altKey||e.ctrlKey||e.metaKey)return;
      if(e.key==='Home'){e.preventDefault();e.stopPropagation();reset();return;}
      const offset={ArrowLeft:[-12,0],ArrowRight:[12,0],ArrowUp:[0,-12],ArrowDown:[0,12]}[e.key];if(!offset)return;
      e.preventDefault();e.stopPropagation();const bounds=this.mapBounds();this.moveMap(bounds.left+offset[0],bounds.top+offset[1]);this.saveSettings();
    });
  }
  navigate(target) {
    const {media,view,loading,editing}=this.getState();
    if(!target||!this.data||!this.enabled||loading||editing||!(media instanceof HTMLVideoElement)||media.seeking||media.readyState<2)return false;
    const pose=poseAt(this.data,media.currentTime);if(!pose)return false;
    this.onNavigate(target.t,viewAfterStep(pose,target,view));this.clearHover();return true;
  }
  navigateDirection(direction) {
    this.draw();return this.navigate(this.directions[direction]);
  }
  drawNavigation(pose,time,view,ready) {
    const selection=this.asset?.tourKeyframes;
    this.steps=ready&&pose?(selection?.enabled?keyframeRouteSteps(this.data,time,selection):editedRouteSteps(this.data,time,this.asset?.routeEdits).filter(step=>Math.abs(step.t-time)<=this.windowSeconds)):[];this.directions=directionalSteps(this.steps,view?.yaw||0);
    const container=$('route-navigation'),width=this.canvas.clientWidth,height=this.canvas.clientHeight;
    container.hidden=!this.steps.length;
    const times=new Set(this.steps.map(step=>step.t));
    for(const [time,button] of this.stepButtons)if(!times.has(time)){button.remove();this.stepButtons.delete(time);}
    for(const target of this.steps){
      let button=this.stepButtons.get(target.t);
      if(!button){
        button=document.createElement('button');button.className='roadview-arrow';
        button.innerHTML='<span class="roadview-arrow-disc"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 20V5M5 12l7-7 7 7"/></svg></span><span class="roadview-arrow-label"></span>';
        const time=target.t;button.onclick=()=>this.navigate(this.steps.find(step=>step.t===time));
        container.append(button);this.stepButtons.set(time,button);
      }
      const pitch=-Math.atan(Math.tan(24*Math.PI/180)*this.heightFactor)*180/Math.PI;
      const point=projectPoint(target.yaw,pitch,view,width,height),index=this.data.samples.findIndex(sample=>sample.t===target.t);
      button.hidden=!point||point.x<48||point.x>width-48||point.y<90||point.y>height-42;
      const label=`${this.nodeName(index)} · ${formatTime(target.t)}`;
      button.dataset.time=target.t.toFixed(3);button.setAttribute('aria-label',label+'로 이동');
      button.querySelector('.roadview-arrow-label').textContent=label;
      if(point){button.style.left=`${point.x}px`;button.style.top=`${point.y}px`;}
    }
  }
  pick(event,map) {
    const {media,loading}=this.getState();
    if(!this.data||!this.enabled||loading||!(media instanceof HTMLVideoElement)||media.seeking||media.readyState<2||(map&&!this.showMap))return null;
    const rect=(map?this.map:this.canvas).getBoundingClientRect();
    return pickPath(map?this.mapSegments:this.pathSegments,event.clientX-rect.left,event.clientY-rect.top,event.pointerType==='touch'?16:9,media.currentTime);
  }
  clearHover() {
    this.hover=null;$('route-seek-hint').hidden=true;
    $('panorama').classList.remove('route-hit');this.map.classList.remove('route-hit');
  }
  updateHover() {
    if(!this.hover||this.gestures.size)return;
    const {event,map}=this.hover,hit=this.pick(event,map),target=map?this.map:$('panorama');
    target.classList.toggle('route-hit',!!hit);const hint=$('route-seek-hint');hint.hidden=!hit;
    if(!hit)return;
    hint.textContent=`${formatTime(hit.time)}로 이동 · 클릭`;
    const rect=$('viewport').getBoundingClientRect();
    hint.style.left=`${Math.max(8,Math.min(rect.width-hint.offsetWidth-8,event.clientX-rect.left+12))}px`;
    hint.style.top=`${Math.max(8,Math.min(rect.height-hint.offsetHeight-8,event.clientY-rect.top-36))}px`;
  }
  bindSeeking(target,map) {
    target.addEventListener('pointerdown',e=>{
      if(e.button!==0)return;
      this.draw();
      const state=this.getState();
      this.gestures.set(e.pointerId,{x:e.clientX,y:e.clientY,time:e.timeStamp,hit:this.pick(e,map),cancelled:false,epoch:this.epoch,sceneId:state.sceneId,media:state.media});
      if(this.gestures.size>1)for(const gesture of this.gestures.values())gesture.cancelled=true;
      this.clearHover();
      if(map){target.focus({preventScroll:true});target.setPointerCapture(e.pointerId);}
    });
    target.addEventListener('pointermove',e=>{
      const gesture=this.gestures.get(e.pointerId);
      if(gesture){if(Math.hypot(e.clientX-gesture.x,e.clientY-gesture.y)>(e.pointerType==='touch'?10:6))gesture.cancelled=true;return;}
      if(e.pointerType!=='touch'){this.hover={event:{clientX:e.clientX,clientY:e.clientY,pointerType:e.pointerType},map};this.updateHover();}
    });
    target.addEventListener('pointerup',e=>{
      const gesture=this.gestures.get(e.pointerId);this.gestures.delete(e.pointerId);this.clearHover();
      const state=this.getState();
      if(!gesture||gesture.cancelled||!gesture.hit||gesture.epoch!==this.epoch||gesture.sceneId!==state.sceneId||gesture.media!==state.media||e.timeStamp-gesture.time>700||Math.hypot(e.clientX-gesture.x,e.clientY-gesture.y)>(e.pointerType==='touch'?10:6))return;
      if(!this.enabled||state.loading||state.media?.seeking)return;
      this.onSeek(gesture.hit.time);
    });
    for(const name of ['pointercancel','lostpointercapture'])target.addEventListener(name,e=>{this.gestures.delete(e.pointerId);this.clearHover();});
    target.addEventListener('pointerleave',()=>this.clearHover());
  }
  async request(url,options={}) {
    const response=await fetch(url,options);
    const data=await response.json();
    if(!response.ok)throw new Error(data.error||'경로 분석 서버에 연결할 수 없습니다.');
    return data;
  }
  async syncAsset(force=false) {
    const {asset,source}=this.getState();
    if(!force&&this.asset===asset&&this.sourceURL===source?.url)return;
    this.finishMapDrag(true);
    const epoch=++this.epoch;clearTimeout(this.pollTimer);this.asset=asset;this.sourceURL=source?.url;
    this.pathSegments=[];this.mapSegments=[];this.steps=[];this.directions={};this.gestures.clear();this.clearHover();
    this.key=source?.trajectoryKey||null;this.data=null;this.loadedKey=null;this.heightFactor=1;this.status={state:'idle'};
    const settings=asset?.trajectorySettings||{};this.enabled=settings.enabled!==false;this.showMap=settings.showMap!==false;this.windowSeconds=settings.windowSeconds||12;this.heightFactor=settings.heightFactor||1;this.mapPosition=settings.mapPosition||null;
    if(asset?.type==='video'&&asset.trajectory){
      try{this.data=validateTrajectory(asset.trajectory);this.key ||= this.data.source.sha256;}catch(error){this.notify(error.message,true);}
    }
    this.lastDraw='';this.updateUI();this.draw();
    if(asset?.type!=='video'||!this.key)return;
    try{await this.poll(epoch);}catch(error){if(epoch===this.epoch){this.status={state:'unavailable',message:this.data?'저장된 경로를 표시합니다.':error.message};this.updateUI();}}
  }
  async poll(epoch=this.epoch) {
    const key=this.key;if(!key)return;
    const status=await this.request(`/api/trajectory/${key}`);
    if(epoch!==this.epoch)return;
    this.status=status;
    const source=this.getState().source;if(source)source.trajectoryKey=key;
    if(status.state==='complete'&&!this.loadedKey){await this.loadResult(epoch);}
    else if(status.state==='complete'&&this.loadedKey!==key){await this.loadResult(epoch);}
    else if(status.hasResult&&!this.data&&!['queued','running','cancelling'].includes(status.state)){await this.loadResult(epoch);}
    this.updateUI();
    if(['queued','running','cancelling'].includes(status.state))this.pollTimer=setTimeout(()=>this.poll(epoch).catch(error=>{if(epoch===this.epoch){this.status={state:'failed',message:error.message};this.updateUI();}}),1000);
  }
  async loadResult(epoch) {
    const result=validateTrajectory(await this.request(`/api/trajectory/${this.key}/result`));
    if(epoch!==this.epoch)return;
    if(result.source.size!==this.asset.size||result.source.sha256!==this.key)throw new Error('분석 결과와 원본 영상이 일치하지 않습니다.');
    result.source.name=this.asset.name;
    this.data=result;this.loadedKey=this.key;this.asset.trajectory=result;this.onChange();this.lastDraw='';this.draw();
  }
  async start(force=false) {
    const state=this.getState(),epoch=this.epoch;
    if(state.asset?.type!=='video')return;
    if(!state.source)return this.notify('원본 영상 파일을 먼저 연결해 주세요.',true);
    this.status={state:'running',message:'분석할 영상 준비 중',progress:0};this.updateUI();
    try{
      if(!state.source?.trajectoryKey){
        const blob=state.source?.file||await fetch(state.source.url).then(r=>r.blob());
        if(epoch!==this.epoch)return;
        this.status={state:'running',message:'뷰어 실행 PC에 분석용 영상 전달 중',progress:0};this.updateUI();
        const imported=await this.request(`/api/trajectory/import?name=${encodeURIComponent(state.asset.name)}`,{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:blob});
        if(epoch!==this.epoch)return;
        this.key=imported.key;state.source.trajectoryKey=imported.key;
      }else this.key=state.source.trajectoryKey;
      this.loadedKey=null;
      const status=await this.request(`/api/trajectory/${this.key}/start`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({force})});
      if(epoch!==this.epoch)return;
      this.status=status;
      this.updateUI();await this.poll(epoch);
    }catch(error){if(epoch===this.epoch){this.status={state:'failed',message:error.message};this.updateUI();this.notify(error.message,true);}}
  }
  async cancel() {
    if(!this.key)return;
    const epoch=this.epoch;
    try{const status=await this.request(`/api/trajectory/${this.key}/cancel`,{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});if(epoch!==this.epoch)return;this.status=status;this.updateUI();}catch(error){if(epoch===this.epoch)this.notify(error.message,true);}
  }
  saveSettings(){if(this.asset){this.asset.trajectorySettings={enabled:this.enabled,showMap:this.showMap,windowSeconds:this.windowSeconds,heightFactor:this.heightFactor,...(this.mapPosition?{mapPosition:{...this.mapPosition}}:{})};this.onChange();}this.lastDraw='';this.draw();}
  nodeName(index) {
    const number=this.asset?.tourKeyframes?.enabled?this.asset.tourKeyframes.frames.findIndex(frame=>frame.index===index):-1;
    if(number>=0)return `키프레임 ${number+1}`;
    return this.asset?.routeEdits?.nodes?.[index]?.name||`지점 ${index+1}`;
  }
  updateUI() {
    const available=this.asset?.type==='video';$('trajectory-open').disabled=!available;
    const running=['queued','running','cancelling'].includes(this.status.state);
    $('trajectory-open').classList.toggle('active',!!this.data&&this.enabled);
    $('trajectory-open').classList.toggle('analyzing',running);
    $('trajectory-button-label').textContent=running?'경로 분석 중':this.data?'촬영 경로':'자동 경로';
    $('route-map-wrap').hidden=!this.data||!this.enabled||!this.showMap;
    this.positionMap();
    if(!this.data||!this.enabled)$('route-gap').hidden=true;
    if($('trajectory-status')){
      $('trajectory-status').textContent=this.status.message||(this.data?'촬영 경로를 표시할 수 있습니다.':'영상에서 촬영 경로를 자동으로 추정합니다.');
      $('trajectory-progress-wrap').hidden=!running;
      $('trajectory-progress').value=this.status.progress||0;
      $('trajectory-start').hidden=running;$('trajectory-start').textContent=this.data?'다시 분석':'자동 추정 시작';
      $('trajectory-start').disabled=!available||!this.getConfig().trajectory?.available;
      $('trajectory-cancel').hidden=!running||!this.key||this.status.state==='cancelling';
      $('trajectory-display-settings').hidden=!this.data;
      $('route-height').disabled=!this.data?.floor;
      if(this.data){const q=this.data.quality;$('trajectory-quality').textContent=`촬영 구간 ${(q.coverage*100).toFixed(1)}% 추정 · ${q.registeredFrames}/${q.inputFrames} 프레임 연결`;$('trajectory-floor-note').textContent=this.data.floor?'바닥 높이는 영상의 특징점에서 추정했습니다. 선이 뜨거나 잠기면 아래에서 보정하세요.':'바닥 위치를 충분히 확인하지 못해 미니맵에만 경로를 표시합니다.';}
    }
    this.lastDraw='';
  }
  openSettings() {
    const ready=this.getConfig().trajectory?.available;
    this.openDialog('자동 촬영 경로',`<p>영상 속 카메라의 이동을 분석해, 지나온 길과 앞으로 이어질 길을 표시합니다.</p><div class="trajectory-status-card"><strong id="trajectory-status"></strong><div id="trajectory-progress-wrap" hidden><progress id="trajectory-progress" max="100" value="0"></progress><span>영상 길이와 PC 성능에 따라 시간이 걸립니다.</span></div><p id="trajectory-quality"></p></div><div id="trajectory-display-settings" hidden><div class="trajectory-checks"><label><input id="route-enabled" type="checkbox" ${this.enabled?'checked':''}> 영상에 경로 표시</label><label><input id="route-map-enabled" type="checkbox" ${this.showMap?'checked':''}> 미니맵 표시</label></div><label class="field">앞뒤 경로 표시 범위<select id="route-window"><option value="6">각각 6초</option><option value="12">각각 12초</option><option value="24">각각 24초</option><option value="60">각각 60초</option></select></label><p id="trajectory-floor-note" class="hint"></p><label class="field" style="margin-top:14px">바닥 높이 보정 <span id="height-factor-label">100%</span><input id="route-height" type="range" min="0.5" max="1.5" step="0.02" value="${this.heightFactor}" ${this.data?.floor?'':'disabled'}></label><p class="hint">파란색: 앞으로 이어질 경로 · 주황색: 지나온 경로<br>영상이나 미니맵의 경로를 클릭하면 해당 시각으로 이동합니다.<br>연결되지 않은 구간은 비워 둡니다. 벽 뒤 경로의 가림 처리는 적용되지 않습니다.</p></div><p class="hint" style="margin-top:16px">분석은 뷰어를 실행한 PC에서 처리합니다. 파일 선택으로 연 영상은 분석용으로 해당 PC에 복사됩니다. 외부 서비스로 전송하지 않습니다.</p>${ready?'':'<p class="dialog-note" style="margin-top:12px">새 영상 분석에는 분석 도구 설치가 필요합니다. 제공된 경로 데이터는 그대로 볼 수 있습니다.</p>'}<div class="dialog-actions"><button id="trajectory-cancel" class="button subtle" hidden>분석 취소</button><button id="trajectory-start" class="button primary">자동 추정 시작</button><button class="button subtle" data-close>닫기</button></div>`);
    $('route-window').value=String(this.windowSeconds);$('height-factor-label').textContent=`${Math.round(this.heightFactor*100)}%`;
    $('route-enabled').onchange=e=>{this.enabled=e.target.checked;this.updateUI();this.saveSettings();};
    $('route-map-enabled').onchange=e=>{this.showMap=e.target.checked;this.updateUI();this.saveSettings();};
    $('route-window').onchange=e=>{this.windowSeconds=Number(e.target.value);this.saveSettings();};
    $('route-height').oninput=e=>{this.heightFactor=Number(e.target.value);$('height-factor-label').textContent=`${Math.round(this.heightFactor*100)}%`;this.saveSettings();};
    $('trajectory-start').onclick=()=>this.start(!!this.data);$('trajectory-cancel').onclick=()=>this.cancel();this.updateUI();
  }
  draw() {
    const {media,view,loading,editing}=this.getState(),time=media instanceof HTMLVideoElement?media.currentTime:0,seeking=media instanceof HTMLVideoElement&&media.seeking;
    const width=this.canvas.clientWidth,height=this.canvas.clientHeight;
    const signature=`${time.toFixed(3)}:${view?.yaw}:${view?.pitch}:${view?.fov}:${width}:${height}:${this.enabled}:${this.showMap}:${!!this.data}:${loading}:${seeking}:${editing}`;
    if(this.lastDraw===signature)return;this.lastDraw=signature;
    this.pathSegments=[];this.mapSegments=[];
    if(!this.data||!this.enabled||!view||loading||seeking||!(media instanceof HTMLVideoElement)){this.drawNavigation(null,time,view,false);$('route-gap').hidden=true;this.clearHover();return;}
    const pose=poseAt(this.data,time);
    this.drawNavigation(pose,time,view,!editing&&this.enabled);
    $('route-gap').hidden=!!pose;$('route-gap').textContent='이 구간은 촬영 위치를 확인하지 못했습니다.';
    this.drawMap(pose,time,view);this.updateHover();
  }
  drawMap(pose,time,view) {
    if(!this.showMap)return;
    const w=this.map.clientWidth,h=this.map.clientHeight,ratio=Math.min(devicePixelRatio||1,2);
    if(this.map.width!==Math.round(w*ratio)||this.map.height!==Math.round(h*ratio)){this.map.width=Math.round(w*ratio);this.map.height=Math.round(h*ratio);}
    const ctx=this.mapContext;ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);
    const data=this.data,up=data.floor?.normal||data.up,first=data.samples[0];
    let right=rotate(first.q,[1,0,0]);right=unit(sub(right,up.map(v=>v*dot(right,up))));const forward=unit(cross(up,right));
    const coordinates=data.samples.map(s=>[dot(s.p,right),-dot(s.p,forward)]);
    const xs=coordinates.map(p=>p[0]),ys=coordinates.map(p=>p[1]),minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
    const scale=Math.min((w-30)/Math.max(maxX-minX,.01),(h-26)/Math.max(maxY-minY,.01));
    const point=position=>[w/2+(dot(position,right)-(maxX+minX)/2)*scale,h/2+(-dot(position,forward)-(maxY+minY)/2)*scale];
    ctx.lineCap='round';ctx.lineJoin='round';ctx.lineWidth=2;
    for(const [start,end] of data.segments)for(let i=start;i<end;i++){
      const a=data.samples[i],b=data.samples[i+1];if(b.t-a.t>data.maxGap)continue;
      const cuts=[a.t];if(time>a.t&&time<b.t)cuts.push(time);cuts.push(b.t);
      for(let j=0;j<cuts.length-1;j++){const p=point(lerp(a.p,b.p,(cuts[j]-a.t)/(b.t-a.t))),q=point(lerp(a.p,b.p,(cuts[j+1]-a.t)/(b.t-a.t)));ctx.beginPath();ctx.moveTo(...p);ctx.lineTo(...q);ctx.strokeStyle=cuts[j+1]<=time?COLORS.past:COLORS.future;ctx.stroke();this.mapSegments.push({p:{x:p[0],y:p[1]},q:{x:q[0],y:q[1]},t0:cuts[j],t1:cuts[j+1]});}
    }
    const begin=point(first.p),end=point(data.samples.at(-1).p);ctx.fillStyle='#b7d1c3';for(const p of [begin,end]){ctx.beginPath();ctx.arc(...p,2.5,0,Math.PI*2);ctx.fill();}
    if(this.asset?.tourKeyframes?.enabled)for(const frame of this.asset.tourKeyframes.frames){const p=point(data.samples[frame.index].p);ctx.beginPath();ctx.arc(...p,2.6,0,Math.PI*2);ctx.fillStyle='#cfb6ff';ctx.fill();ctx.strokeStyle='#142238';ctx.lineWidth=1;ctx.stroke();}
    if(pose){
      const p=point(pose.p),yaw=view.yaw*Math.PI/180,direction=rotate(pose.q,[Math.sin(yaw),0,Math.cos(yaw)]),angle=Math.atan2(-dot(direction,forward),dot(direction,right));
      ctx.save();ctx.translate(...p);ctx.rotate(angle);ctx.beginPath();ctx.moveTo(12,0);ctx.lineTo(-4,-5);ctx.lineTo(-1,0);ctx.lineTo(-4,5);ctx.closePath();ctx.fillStyle='#f1fff3';ctx.shadowColor='#000';ctx.shadowBlur=4;ctx.fill();ctx.restore();
    }
    $('route-map-time').textContent=formatTime(time);
    this.map.setAttribute('aria-valuemax',String(data.duration));this.map.setAttribute('aria-valuenow',time.toFixed(2));this.map.setAttribute('aria-valuetext',formatTime(time));
  }
}

import {escapeHTML as esc,formatTime} from './core.js';
import {poseAt,viewAfterStep} from './trajectory-math.js';
import {DENSITIES,connectedChunks,frameQuality,routeSignature,selectTourKeyframes,validateTourKeyframes} from './tour-keyframes.js';

const $=id=>document.getElementById(id);
function waitForMedia(video,event,signal){
  return new Promise((resolve,reject)=>{
    const cleanup=()=>{clearTimeout(timer);video.removeEventListener(event,done);video.removeEventListener('error',error);signal.removeEventListener('abort',abort);};
    const done=()=>{cleanup();resolve();},error=()=>{cleanup();reject(new Error('원본 영상 프레임을 읽지 못했습니다. 파일 연결을 확인해 주세요.'));},abort=()=>{cleanup();reject(new DOMException('Cancelled','AbortError'));};
    const timer=setTimeout(()=>{cleanup();reject(new Error('프레임 읽기가 지연됐습니다. 다시 선별해 주세요.'));},20000);
    video.addEventListener(event,done,{once:true});video.addEventListener('error',error,{once:true});signal.addEventListener('abort',abort,{once:true});if(signal.aborted)abort();
  });
}

export class KeyframeController{
  constructor({getState,onChange,onNavigate,openDialog,notify,invalidate}){
    Object.assign(this,{getState,onChange,onNavigate,openDialog,notify,invalidate});
    this.asset=null;this.data=null;this.sourceURL=null;this.running=false;this.status='';this.cards=new Map();this.thumbnails=new Map();this.rendered=null;
    $('keyframes-open').onclick=()=>this.openSettings();
    $('keyframes-settings').onclick=()=>this.openSettings();
    $('dialog').addEventListener('close',()=>$('dialog').classList.remove('keyframe-dialog'));
  }
  update(){
    const {asset,data,source,media,loading}=this.getState();
    if(this.asset!==asset||this.data!==data||this.sourceURL!==source?.url){
      this.abortController?.abort();this.running=false;this.status='';this.thumbnails.clear();
      this.asset=asset;this.data=data;this.sourceURL=source?.url;this.rendered=null;
      if(asset?.tourKeyframes&&data){try{asset.tourKeyframes=validateTourKeyframes(asset.tourKeyframes,data);}catch{delete asset.tourKeyframes;this.onChange();this.notify('촬영 경로가 바뀌어 키프레임을 다시 선별해야 합니다.');}}
      this.render();
    }
    $('keyframes-open').disabled=!data||!source||asset?.type!=='video';
    const selection=asset?.tourKeyframes;
    $('keyframes-open').classList.toggle('active',!!selection?.enabled);
    $('keyframes-button-label').textContent=this.running?'키프레임 선별 중':'키프레임 선별';
    if(selection){
      const time=media instanceof HTMLVideoElement?media.currentTime:0;
      let nearest=selection.frames[0];for(const frame of selection.frames)if(Math.abs(frame.t-time)<Math.abs(nearest.t-time))nearest=frame;
      for(const [index,button] of this.cards){button.setAttribute('aria-current',String(index===nearest.index));button.disabled=loading||!(media instanceof HTMLVideoElement)||media.seeking||media.readyState<2;}
    }
    this.updateDialog();
  }
  render(){
    const selection=this.data?this.asset?.tourKeyframes:null;
    $('keyframe-panel').hidden=!selection&&!this.running;
    $('keyframe-summary').textContent=this.running?this.status:selection?`촬영 지점 ${this.data.samples.length}개 → 키프레임 ${selection.frames.length}개 · ${selection.enabled?'선별 지점으로 이동':'기존 이동 방식'}`:'';
    $('keyframe-progress').hidden=!this.running;
    if(this.rendered===selection)return;
    this.rendered=selection;this.cards.clear();$('keyframe-list').replaceChildren();
    for(const [number,frame] of (selection?.frames||[]).entries()){
      const button=document.createElement('button');button.className='keyframe-card';button.dataset.index=frame.index;
      button.setAttribute('aria-label',`키프레임 ${number+1} · ${formatTime(frame.t)}로 이동`);button.title=`${frame.t.toFixed(3)}초 · ${frame.reason}`;
      if(frame.thumbnail){const image=document.createElement('img');image.src=frame.thumbnail;image.alt='';button.append(image);}
      else{const placeholder=document.createElement('span');placeholder.className='keyframe-placeholder';placeholder.textContent='360°';button.append(placeholder);}
      const label=document.createElement('span');label.textContent=`${String(number+1).padStart(2,'0')} · ${formatTime(frame.t)}`;button.append(label);
      button.onclick=()=>this.jump(frame);this.cards.set(frame.index,button);$('keyframe-list').append(button);
    }
  }
  jump(frame){
    const {media,view,loading}=this.getState();if(loading||!(media instanceof HTMLVideoElement)||media.seeking||media.readyState<2||!this.data)return;
    const pose=poseAt(this.data,media.currentTime),target=this.data.samples[frame.index];
    this.onNavigate(frame.t,pose?viewAfterStep(pose,target,view):view);
  }
  updateDialog(){
    if(!$('keyframe-start'))return;
    const selection=this.data?this.asset?.tourKeyframes:null;
    const signature=JSON.stringify([this.running,this.status,selection?.density,selection?.frames.length,selection?.enabled,!!this.data,!!this.sourceURL]);
    if(signature===this.dialogSignature)return;this.dialogSignature=signature;
    $('keyframe-dialog-status').textContent=this.running||this.status?this.status:selection?`${this.data.samples.length}개 촬영 지점에서 ${selection.frames.length}개를 선택했습니다.`:'촬영 경로의 시작·끝과 꺾임을 유지하고, 중복되는 이동 지점을 줄입니다.';
    $('keyframe-start').disabled=this.running||!this.data||!this.sourceURL;
    $('keyframe-density').disabled=this.running;
    $('keyframe-start').textContent=selection?'다시 자동 선별':'자동 선별 시작';
    $('keyframe-cancel').hidden=!this.running;
    $('keyframe-use').disabled=this.running||!selection;
    $('keyframe-use').checked=!!selection?.enabled;
    $('keyframe-dialog-results').innerHTML=selection?`<span>입력 지점 <strong>${this.data.samples.length}</strong></span><span>선택 지점 <strong>${selection.frames.length}</strong></span><span>동선 꺾임 <strong>${selection.frames.filter(f=>f.reason==='동선 꺾임').length}</strong></span>`:'';
  }
  openSettings(){
    if(!this.data||!this.sourceURL)return;
    this.dialogSignature=null;
    this.openDialog('투어 키프레임 선별',`<p>바닥 화살표로 이동할 대표 촬영 지점을 고릅니다. 선별한 화면을 눌러 원본 영상의 해당 시각을 확인할 수 있습니다.</p><div class="keyframe-results" id="keyframe-dialog-results"></div><label class="field">이동 지점 수<select id="keyframe-density" aria-label="이동 지점 수">${Object.entries(DENSITIES).map(([key,value])=>`<option value="${key}">${esc(value.label)}</option>`).join('')}</select></label><div class="keyframe-method"><span>01</span><p><strong>동선 유지</strong>구간 시작·끝과 주요 꺾임을 보존합니다.</p><span>02</span><p><strong>중복 줄이기</strong>이동 거리로 지점을 배치하고 정지 구간의 중복을 줄입니다.</p><span>03</span><p><strong>주변 선명도 비교</strong>가까운 후보에서 영상의 디테일과 밝기를 비교합니다.</p></div><label class="keyframe-use"><input id="keyframe-use" type="checkbox"> 선별 지점으로 바닥 화살표 이동</label><p id="keyframe-dialog-status" class="hint" role="status"></p><p class="hint">원본 영상과 촬영 경로는 유지됩니다. 선별 결과와 미리보기는 투어 저장 파일에 포함됩니다.</p><div class="dialog-actions"><button id="keyframe-cancel" class="button subtle" hidden>선별 취소</button><button id="keyframe-start" class="button primary">자동 선별 시작</button><button class="button subtle" data-close>닫기</button></div>`);
    $('dialog').classList.add('keyframe-dialog');$('keyframe-density').value=this.asset?.tourKeyframes?.density||'standard';
    $('keyframe-start').onclick=()=>this.start($('keyframe-density').value);
    $('keyframe-cancel').onclick=()=>this.abortController?.abort();
    $('keyframe-use').onchange=e=>{if(this.asset?.tourKeyframes){this.asset.tourKeyframes.enabled=e.target.checked;this.onChange();this.invalidate();this.render();}};
    this.updateDialog();
  }
  async start(density){
    if(this.running||!this.data||!this.sourceURL)return;
    const data=this.data,asset=this.asset,sourceURL=this.sourceURL,abortController=new AbortController(),signal=abortController.signal;
    this.abortController=abortController;this.running=true;this.status='원본 영상 프레임을 준비합니다.';this.render();this.updateDialog();
    const video=document.createElement('video');video.preload='auto';video.muted=true;video.playsInline=true;
    const canvas=document.createElement('canvas');canvas.width=384;canvas.height=192;const context=canvas.getContext('2d',{willReadFrequently:true});
    const seek=async time=>{
      signal.throwIfAborted();const target=Math.max(0,Math.min(time,video.duration-.01));
      if(Math.abs(video.currentTime-target)>.0001){const sought=waitForMedia(video,'seeked',signal);video.currentTime=target;await sought;}
      if(video.readyState<2)await waitForMedia(video,'loadeddata',signal);
      signal.throwIfAborted();context.drawImage(video,0,0,canvas.width,canvas.height);
    };
    try{
      const loaded=waitForMedia(video,'loadeddata',signal);video.src=sourceURL;video.load();await loaded;
      let metrics=asset.tourKeyframes?.metrics||[];
      const candidates=connectedChunks(data).flatMap(([a,b])=>Array.from({length:b-a+1},(_,i)=>a+i));
      if(metrics.length!==candidates.length||metrics.some((m,i)=>m.index!==candidates[i])){
        const measured=[];
        for(const [number,index] of candidates.entries()){
          await seek(data.samples[index].t);
          const quality=frameQuality(context.getImageData(0,0,canvas.width,canvas.height).data,canvas.width,canvas.height);
          measured.push({index,t:data.samples[index].t,...quality});this.thumbnails.set(index,canvas.toDataURL('image/jpeg',.55));
          this.status=`주변 프레임 비교 중 · ${number+1} / ${candidates.length}`;$('keyframe-progress').value=(number+1)/candidates.length*100;this.render();this.updateDialog();
        }
        metrics=measured;
      }
      const frames=selectTourKeyframes(data,metrics,density);
      for(const frame of frames){
        frame.thumbnail=this.thumbnails.get(frame.index)||asset.tourKeyframes?.frames.find(f=>f.index===frame.index)?.thumbnail||'';
        if(!frame.thumbnail){await seek(frame.t);frame.thumbnail=canvas.toDataURL('image/jpeg',.55);this.thumbnails.set(frame.index,frame.thumbnail);}
      }
      signal.throwIfAborted();
      if(this.asset!==asset||this.data!==data||this.sourceURL!==sourceURL)return;
      asset.tourKeyframes=validateTourKeyframes({version:1,sourceSha256:data.source.sha256,routeSignature:routeSignature(data),density,enabled:true,frames,metrics},data);
      this.status='';
      this.onChange();this.invalidate();this.notify(`${candidates.length}개 촬영 지점에서 ${frames.length}개 키프레임을 선별했습니다.`);
      // Leave the result visible in the panorama rather than behind the dialog.
      if($('keyframe-start'))$('dialog').close();
    }catch(error){if(this.asset===asset&&this.data===data){this.status=error.name==='AbortError'?'선별을 취소했습니다.':error.message;this.notify(this.status,error.name!=='AbortError');}}
    finally{
      video.pause();video.removeAttribute('src');video.load();
      if(this.abortController===abortController){this.running=false;this.render();this.update();}
    }
  }
}

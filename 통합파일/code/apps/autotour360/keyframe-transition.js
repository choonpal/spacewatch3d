import {clamp,radians,wrapYaw} from './core.js';

// Freeze the outgoing view before seeking. Only the two paused endpoint frames
// are shown; the original video never plays through the intervening timestamps.
export class KeyframeTransition {
  constructor({renderer,getMedia,isLoading,onStateChange,onError,duration=420,seekTimeout=10000}) {
    Object.assign(this,{renderer,getMedia,isLoading,onStateChange,onError,duration,seekTimeout});
    this.current=null;
    const cancel=()=>this.cancel();
    renderer.canvas.addEventListener('pointerdown',cancel);
    renderer.canvas.addEventListener('wheel',cancel);
    renderer.canvas.addEventListener('webglcontextlost',cancel);
    document.addEventListener('visibilitychange',()=>{if(document.hidden)this.cancel();});
  }
  get active(){return this.current!==null;}
  cancel(){if(this.current){this.current.controller.abort();this.release(this.current);}}
  release(run){
    for(const animation of run.animations)animation.cancel();
    run.overlay?.remove();
    if(this.current!==run)return;
    this.current=null;this.renderer.canvas.parentElement.classList.remove('keyframe-moving');this.onStateChange();
  }
  animate(run,target,frames,options){
    const animation=target.animate(frames,options);run.animations.push(animation);
    // Cancellation can happen while decoding, before a finished promise is awaited.
    animation.finished.catch(()=>{});return animation;
  }
  capture(yaw){
    const {renderer}=this,canvas=renderer.canvas,view=renderer.view;
    renderer.dirty=true;renderer.render();
    const overlay=document.createElement('canvas');overlay.width=canvas.width;overlay.height=canvas.height;
    overlay.getContext('2d').drawImage(canvas,0,0);overlay.className='keyframe-transition-overlay';overlay.setAttribute('aria-hidden','true');
    const delta=wrapYaw((yaw??view.yaw)-view.yaw),projected=renderer.project(yaw??view.yaw,view.pitch);
    const x=projected?projected.x/canvas.clientWidth*100:50+Math.sin(radians(delta))*40;
    overlay.style.transformOrigin=`${clamp(x,8,92)}% 55%`;
    canvas.parentElement.append(overlay);return overlay;
  }
  seek(media,time,signal){
    return new Promise((resolve,reject)=>{
      const cleanup=()=>{clearTimeout(timer);for(const name of ['seeked','loadeddata','canplay'])media.removeEventListener(name,ready);media.removeEventListener('error',error);signal.removeEventListener('abort',abort);};
      const ready=()=>{if(!media.seeking&&media.readyState>=2){cleanup();resolve();}};
      const error=()=>{cleanup();reject(new Error('이동할 키프레임을 읽지 못했습니다. 원본 영상 연결을 확인해 주세요.'));};
      const abort=()=>{cleanup();reject(new DOMException('Cancelled','AbortError'));};
      const timer=setTimeout(()=>{cleanup();reject(new Error('키프레임 이동이 지연되었습니다. 다시 시도해 주세요.'));},this.seekTimeout);
      for(const name of ['seeked','loadeddata','canplay'])media.addEventListener(name,ready);
      media.addEventListener('error',error);signal.addEventListener('abort',abort,{once:true});
      try{media.currentTime=time;ready();}catch(cause){cleanup();reject(cause);}
    });
  }
  async navigate(time,view,{yaw}={}){
    const media=this.getMedia();
    if(this.active||this.isLoading()||!(media instanceof HTMLVideoElement)||media.seeking||media.readyState<2||!Number.isFinite(time)||!Number.isFinite(media.duration))return false;
    time=clamp(time,0,media.duration);media.pause();this.renderer.autoRotate=false;
    if(Math.abs(media.currentTime-time)<.025){this.renderer.setView(view);return true;}
    const run={controller:new AbortController(),animations:[],overlay:null};this.current=run;
    try{
      const reduced=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      if(!reduced&&typeof this.renderer.canvas.animate==='function'&&!this.renderer.lost){
        run.overlay=this.capture(yaw);
        this.animate(run,run.overlay,[{transform:'scale(1)'},{transform:'scale(1.06)'}],{duration:160,easing:'ease-out',fill:'forwards'});
      }
      this.renderer.canvas.parentElement.classList.add('keyframe-moving');this.onStateChange();
      this.renderer.setView(view);
      await this.seek(media,time,run.controller.signal);
      if(this.current!==run||this.getMedia()!==media)return false;
      media.pause();this.renderer.frameDirty=true;this.renderer.dirty=true;this.renderer.render();
      if(run.overlay){
        const start=getComputedStyle(run.overlay).transform;
        for(const animation of run.animations)animation.cancel();
        const options={duration:this.duration,easing:'cubic-bezier(.22,.7,.25,1)',fill:'both'};
        const outgoing=this.animate(run,run.overlay,[{opacity:1,transform:start},{opacity:0,transform:'scale(1.075)'}],options);
        const incoming=this.animate(run,this.renderer.canvas,[{transform:'scale(1.025)',transformOrigin:run.overlay.style.transformOrigin},{transform:'scale(1)',transformOrigin:run.overlay.style.transformOrigin}],options);
        await Promise.all([outgoing.finished,incoming.finished]);
      }
      return this.current===run;
    }catch(error){if(!run.controller.signal.aborted)this.onError(error.message);return false;}
    finally{if(this.current===run){if(this.getMedia()===media)media.pause();this.release(run);}}
  }
}

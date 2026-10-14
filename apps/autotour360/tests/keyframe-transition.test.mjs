import test from 'node:test';
import assert from 'node:assert/strict';
import {KeyframeTransition} from '../keyframe-transition.js';

class Video extends EventTarget {
  constructor(){super();this.time=1;this.duration=60;this.readyState=2;this.seeking=false;this.paused=false;this.pauseCount=0;}
  get currentTime(){return this.time;}
  set currentTime(time){this.time=time;this.seeking=true;this.readyState=1;}
  pause(){this.paused=true;this.pauseCount++;}
  decode(){this.seeking=false;this.readyState=2;this.dispatchEvent(new Event('seeked'));}
}

function fixture({reduced=false,seekTimeout=1000}={}){
  const animations=[],captures=[],overlays=[],errors=[],states=[];
  const classes=new Set();
  const viewport={classList:{add:name=>classes.add(name),remove:name=>classes.delete(name)},append:node=>overlays.push(node)};
  const makeCanvas=()=>Object.assign(new EventTarget(),{
    width:800,height:400,clientWidth:800,clientHeight:400,parentElement:viewport,style:{},
    setAttribute(){},getContext:()=>({drawImage:()=>captures.push(media.currentTime)}),remove(){this.removed=true;},
    animate(frames,options){
      let resolve,reject;const finished=new Promise((yes,no)=>{resolve=yes;reject=no;});
      const animation={finished,frames,options,cancel(){this.cancelled=true;reject(new DOMException('Cancelled','AbortError'));},finish:resolve};
      animations.push(animation);return animation;
    },
  });
  const media=new Video(),canvas=makeCanvas();
  globalThis.HTMLVideoElement=Video;
  globalThis.document=Object.assign(new EventTarget(),{hidden:false,createElement:makeCanvas});
  globalThis.window={matchMedia:()=>({matches:reduced})};
  globalThis.getComputedStyle=()=>({transform:'matrix(1.04,0,0,1.04,0,0)'});
  const renderer={canvas,view:{yaw:0,pitch:0,fov:75},autoRotate:true,render(){},setView(view){this.view={...view};},project:()=>({x:700,y:200})};
  let activeMedia=media,loading=false;
  const transition=new KeyframeTransition({renderer,getMedia:()=>activeMedia,isLoading:()=>loading,onStateChange:()=>states.push(transition.active),onError:message=>errors.push(message),seekTimeout});
  return {transition,media,renderer,animations,captures,overlays,errors,states,classes,setMedia:value=>{activeMedia=value;},setLoading:value=>{loading=value;}};
}
const nextView={yaw:32,pitch:-4,fov:75};
const flush=async()=>{await Promise.resolve();await Promise.resolve();};

test('only paused endpoint frames are shown and motion starts after the target frame decodes',async()=>{
  const f=fixture();const moved=f.transition.navigate(4,nextView,{yaw:50});
  assert.equal(f.media.paused,true);assert.equal(f.renderer.autoRotate,false);
  assert.deepEqual(f.captures,[1]);assert.equal(f.media.currentTime,4);
  assert.equal(f.transition.active,true);assert.equal(f.animations.length,1);
  assert.ok(f.classes.has('keyframe-moving'));
  f.media.dispatchEvent(new Event('canplay'));assert.equal(f.animations.length,1);
  f.media.decode();await flush();
  assert.equal(f.animations.length,3);assert.equal(f.animations[1].options.duration,420);
  assert.equal(f.animations[1].frames[1].opacity,0);
  f.animations[1].finish();f.animations[2].finish();assert.equal(await moved,true);
  assert.equal(f.transition.active,false);assert.equal(f.media.paused,true);assert.equal(f.media.currentTime,4);
  assert.deepEqual(f.renderer.view,nextView);assert.deepEqual(f.states,[true,false]);
  assert.ok(f.overlays[0].removed);assert.equal(f.classes.size,0);assert.deepEqual(f.errors,[]);
});

test('an overlapping navigation cannot replace the pending destination',async()=>{
  const f=fixture();const moved=f.transition.navigate(4,nextView);
  assert.equal(await f.transition.navigate(8,nextView),false);assert.equal(f.media.currentTime,4);
  f.transition.cancel();assert.equal(await moved,false);assert.equal(f.transition.active,false);
});

test('cancelling during decoding removes the old frame and cannot launch a late animation',async()=>{
  const f=fixture();const moved=f.transition.navigate(4,nextView);
  f.renderer.canvas.dispatchEvent(new Event('pointerdown'));assert.equal(await moved,false);
  f.media.decode();await flush();assert.equal(f.animations.length,1);
  assert.ok(f.overlays[0].removed);assert.equal(f.classes.size,0);assert.deepEqual(f.errors,[]);
});

test('cancelling an animated arrival restores the untransformed canvas and keeps its video paused',async()=>{
  const f=fixture();const moved=f.transition.navigate(4,nextView);f.media.decode();await flush();
  f.renderer.canvas.dispatchEvent(new Event('wheel'));assert.equal(await moved,false);
  assert.ok(f.animations.every(animation=>animation.cancelled));assert.equal(f.media.paused,true);
  assert.equal(f.media.currentTime,4);assert.equal(f.classes.size,0);assert.deepEqual(f.errors,[]);
});

test('reduced motion seeks to the selected frame without a visual effect or playback',async()=>{
  const f=fixture({reduced:true});const moved=f.transition.navigate(4,nextView);f.media.decode();
  assert.equal(await moved,true);assert.equal(f.animations.length,0);assert.equal(f.captures.length,0);
  assert.equal(f.media.paused,true);assert.deepEqual(f.renderer.view,nextView);
});

test('seeking to the current frame pauses playback without creating an effect',async()=>{
  const f=fixture();assert.equal(await f.transition.navigate(1,nextView),true);
  assert.equal(f.media.paused,true);assert.equal(f.transition.active,false);assert.equal(f.animations.length,0);
});

test('media errors and decode timeouts leave no overlay or locked movement state',async()=>{
  for(const timeout of [false,true]){
    const f=fixture({seekTimeout:10});const moved=f.transition.navigate(4,nextView);
    if(!timeout)f.media.dispatchEvent(new Event('error'));
    assert.equal(await moved,false);assert.equal(f.errors.length,1);assert.equal(f.transition.active,false);
    assert.ok(f.overlays[0].removed);assert.equal(f.media.paused,true);assert.equal(f.classes.size,0);
  }
});

test('a scene change cancels stale work without pausing the replacement video',async()=>{
  const f=fixture();const moved=f.transition.navigate(4,nextView);
  f.transition.cancel();const replacement=new Video();f.setMedia(replacement);
  f.media.decode();assert.equal(await moved,false);assert.equal(replacement.paused,false);
});

test('loading, missing media and invalid timestamps do not start movement',async()=>{
  const f=fixture();f.setLoading(true);assert.equal(await f.transition.navigate(4,nextView),false);
  f.setLoading(false);assert.equal(await f.transition.navigate(NaN,nextView),false);
  f.setMedia(null);assert.equal(await f.transition.navigate(4,nextView),false);
  assert.equal(f.animations.length,0);assert.equal(f.states.length,0);
});

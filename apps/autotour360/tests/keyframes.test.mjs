import test from 'node:test';
import assert from 'node:assert/strict';
import {validateProject} from '../core.js';
import {poseAt,validateTrajectory} from '../trajectory-math.js';
import {connectedChunks,frameQuality,keyframeRouteSteps,routeSignature,selectTourKeyframes,validateTourKeyframes} from '../tour-keyframes.js';

function corridor(){
  return validateTrajectory({version:1,source:{name:'walk.mp4',size:100,sha256:'a'.repeat(64)},duration:20,up:[0,-1,0],floor:{normal:[0,-1,0],offset:1,cameraHeight:1},samples:Array.from({length:41},(_,index)=>({t:index*.5,p:[0,0,index*.1],q:[0,0,0,1]})),segments:[[0,40]],maxGap:1});
}
function selection(data,metrics=[],density='standard'){
  return {version:1,sourceSha256:data.source.sha256,routeSignature:routeSignature(data),density,enabled:true,frames:selectTourKeyframes(data,metrics,density),metrics};
}

test('tour selection reduces redundant observations, preserves endpoints and offers more points for detailed mode',()=>{
  const data=corridor(),frames=selectTourKeyframes(data);
  assert.equal(frames[0].index,0);assert.equal(frames.at(-1).index,40);assert.ok(frames.length<data.samples.length/2);
  assert.ok(selectTourKeyframes(data,[],'detailed').length>frames.length);
  assert.ok(selectTourKeyframes(data,[],'compact').length<frames.length);
  for(const [i,frame] of frames.entries())if(i)assert.ok(frame.index>frames[i-1].index);
});
test('stationary recording does not create a row of redundant tour stops',()=>{
  const data=corridor();data.samples.forEach(s=>s.p=[0,0,0]);
  assert.deepEqual(selectTourKeyframes(data).map(f=>f.index),[0,40]);
  assert.deepEqual(keyframeRouteSteps(data,0,selection(data)),[]);
});
test('bends survive simplification and route scale does not change the selected observations',()=>{
  const data=corridor();data.samples.forEach((s,i)=>s.p=i<=20?[0,0,i*.1]:[(i-20)*.1,0,2]);
  const frames=selectTourKeyframes(data);assert.equal(frames.find(f=>f.index===20).reason,'동선 꺾임');
  const scaled=structuredClone(data);scaled.samples.forEach(s=>s.p=s.p.map(v=>v*100));scaled.floor.cameraHeight*=100;
  assert.deepEqual(selectTourKeyframes(scaled).map(f=>f.index),frames.map(f=>f.index));
});
test('nearby image detail can improve a distance stop without shifting endpoints or bends',()=>{
  const data=corridor(),metrics=data.samples.map((s,index)=>({index,t:s.t,sharpness:index===9?100:1,exposure:.8}));
  const frames=selectTourKeyframes(data,metrics);assert.ok(frames.some(f=>f.index===9&&f.reason.includes('선명도')));
  assert.equal(frames[0].index,0);assert.equal(frames.at(-1).index,40);
});
test('selection and floor-arrow navigation never bridge an internal gap, even if the outer segment spans it',()=>{
  const data=corridor();data.samples.slice(21).forEach(s=>s.t+=3);data.duration+=3;
  assert.deepEqual(connectedChunks(data),[[0,20],[21,40]]);
  const result=selection(data);assert.ok(result.frames.some(f=>f.index===20));assert.ok(result.frames.some(f=>f.index===21));
  assert.deepEqual(keyframeRouteSteps(data,11,result),[]);
  assert.equal(poseAt(data,10).t,10);assert.ok(keyframeRouteSteps(data,10,result).every(step=>step.t<10));
  assert.ok(keyframeRouteSteps(data,9.5,result).every(step=>step.t<=10));
  assert.ok(keyframeRouteSteps(data,14,result).every(step=>step.t>=13.5));
  assert.equal(keyframeRouteSteps(data,0,result)[0].direction,1);
  assert.equal(keyframeRouteSteps(data,23,result)[0].direction,-1);
});
test('tour JSON round trip retains selected timestamps, metrics and navigation choice',()=>{
  const data=corridor(),result=selection(data,[{index:0,t:0,sharpness:42,exposure:.8}]);result.enabled=false;result.frames[0].thumbnail='data:image/jpeg;base64,YQ==';
  const project={version:1,id:'tour',assets:[{id:'v',name:'walk.mp4',size:100,type:'video',trajectory:data,tourKeyframes:result}],scenes:[]};
  const saved=validateProject(JSON.parse(JSON.stringify(project))).assets[0].tourKeyframes;
  assert.equal(saved.enabled,false);assert.equal(saved.frames[0].thumbnail,result.frames[0].thumbnail);assert.deepEqual(saved.metrics,result.metrics);
  assert.deepEqual(validateProject(JSON.parse(JSON.stringify({...project,assets:[{...project.assets[0],tourKeyframes:saved}]}))).assets[0].tourKeyframes,saved);
});
test('reject changed video or route, fabricated time, reordered indices, duplicates and invalid quality',()=>{
  const data=corridor(),result=selection(data,[{index:0,t:0,sharpness:42,exposure:.8}]);
  const corrupt=change=>{const value=structuredClone(result);change(value);assert.throws(()=>validateTourKeyframes(value,data));};
  corrupt(v=>v.sourceSha256='b'.repeat(64));corrupt(v=>v.frames[1].t+=.1);corrupt(v=>v.frames.reverse());corrupt(v=>v.frames.push({...v.frames.at(-1)}));corrupt(v=>v.metrics[0].sharpness=Infinity);
  const changed=structuredClone(data);changed.samples[0].p[0]=1;assert.throws(()=>validateTourKeyframes(result,changed));
  result.frames[0].thumbnail='data:image/svg+xml,<svg onload="evil()"/>';assert.equal(validateTourKeyframes(result,data).frames[0].thumbnail,'');
});
test('equatorial detail score distinguishes sharp edges from a flat image',()=>{
  const pixels=new Uint8ClampedArray(32*16*4),flat=new Uint8ClampedArray(pixels.length);
  for(let y=0;y<16;y++)for(let x=0;x<32;x++){const index=(y*32+x)*4,v=(x+y)%2?210:40;pixels.set([v,v,v,255],index);flat.set([125,125,125,255],index);}
  assert.ok(frameQuality(pixels,32,16).sharpness>frameQuality(flat,32,16).sharpness);assert.equal(frameQuality(flat,32,16).exposure,1);
});

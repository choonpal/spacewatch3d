import test from 'node:test';
import assert from 'node:assert/strict';
import {validateTrajectory,poseAt,floorPoint,projectSegment,pickPath,rotate,slerp} from '../trajectory-math.js';
import {validateProject} from '../core.js';

const fixture=()=>({version:1,source:{name:'video.mp4',size:10,sha256:'a'.repeat(64)},duration:6,up:[0,-1,0],floor:{normal:[0,-1,0],offset:1.7,cameraHeight:1.7,support:100},samples:[0,1,2,4,5].map(t=>({t,p:[0,0,t],q:[0,0,0,1]})),segments:[[0,2],[3,4]],maxGap:1.5,quality:{coverage:.5}});
const view={yaw:0,pitch:0,fov:90};
test('camera coordinates project floor below horizon and respect viewing direction',()=>{
  const pose={p:[0,0,0],q:[0,0,0,1]};
  const floor=floorPoint([0,0,4],fixture().floor);assert.deepEqual(floor,[0,1.7,4]);
  const line=projectSegment(floor,[1,1.7,4],pose,view,1000,500);assert.ok(Math.abs(line[0].x-500)<1e-8);assert.ok(line[0].y>250);assert.ok(line[1].x>500);
  assert.equal(projectSegment([0,1,-3],[1,1,-2],pose,view,1000,500),null);
  const turned={p:[0,0,0],q:[0,Math.SQRT1_2,0,Math.SQRT1_2]};
  const rotated=projectSegment([4,1.7,0],[4,1.7,-1],turned,view,1000,500);assert.ok(Math.abs(rotated[0].x-500)<1e-7);
});
test('near-plane clipping remains finite when a path crosses behind camera',()=>{
  const line=projectSegment([-1,1,-1],[1,1,2],{p:[0,0,0],q:[0,0,0,1]},view,1000,500,.1);
  assert.ok(line.flatMap(p=>Object.values(p)).every(Number.isFinite));assert.ok(line[0].z>=.09999);
});
test('path clicks use perspective-correct time rather than screen-space interpolation',()=>{
  const pose={p:[0,0,0],q:[0,0,0,1]},[p,q]=projectSegment([0,1,2],[4,1,8],pose,view,1000,500);
  const hit=pickPath([{p,q,t0:10,t1:20,perspective:true}],(p.x+q.x)/2,(p.y+q.y)/2);
  assert.ok(Math.abs(hit.time-12)<1e-8);
  assert.equal(pickPath([{p,q,t0:10,t1:20,perspective:true}],20,20),null);
});
test('near-plane clipping preserves the clicked timestamp of the original segment',()=>{
  const pose={p:[0,0,0],q:[0,0,0,1]},a=[-1,.1,-1],b=[1,.1,2];
  for(const [first,last,expected] of [[a,b,6],[b,a,4]]){
    const [p,q]=projectSegment(first,last,pose,view,1000,500,.1);
    // The world point [.2,.1,.8] projects to [562.5,281.25] and is 60% from a to b.
    const hit=pickPath([{p,q,t0:10*p.fraction,t1:10*q.fraction,perspective:true}],562.5,281.25);
    assert.ok(Math.abs(hit.time-expected)<1e-8);
    assert.ok(Math.min(p.z,q.z)>=.09999);
  }
});
test('minimap picking is linear, avoids empty gaps and resolves overlapping times consistently',()=>{
  const segments=[{p:{x:0,y:0},q:{x:40,y:0},t0:0,t1:4},{p:{x:60,y:0},q:{x:100,y:0},t0:6,t1:10}];
  assert.equal(pickPath(segments,20,0).time,2);
  assert.equal(pickPath(segments,50,0,5),null);
  const overlap=[segments[0],{...segments[0],t0:20,t1:24}];
  assert.equal(pickPath(overlap,20,0,9,23).time,22);
  assert.equal(pickPath(overlap,20,0,9,1).time,2);
});
test('interpolates position and shortest quaternion arc; never bridges missing interval',()=>{
  const data=validateTrajectory(fixture());assert.deepEqual(poseAt(data,.5).p,[0,0,.5]);assert.equal(poseAt(data,3),null);assert.equal(poseAt(data,5.5),null);
  assert.deepEqual(slerp([0,0,0,1],[0,0,0,-1],.5),[0,0,0,1]);
  const q=slerp([0,0,0,1],[0,1,0,0],.5);assert.ok(Math.abs(rotate(q,[0,0,1])[0]-1)<1e-7);
});
test('rejects corrupt quaternions, invalid time order, and overlapping trajectory segments',()=>{
  let data=fixture();data.samples[0].q=[0,0,0,0];assert.throws(()=>validateTrajectory(data));
  data=fixture();data.samples[1].t=0;assert.throws(()=>validateTrajectory(data));
  data=fixture();data.segments=[[0,2],[2,4]];assert.throws(()=>validateTrajectory(data));
  data=fixture();data.samples[0].p=[NaN,0,0];assert.throws(()=>validateTrajectory(data));
});
test('tour round trip retains trajectory and calibration, rejects wrong video association',()=>{
  const input={version:1,id:'tour',assets:[{id:'v',name:'video.mp4',size:10,type:'video',trajectory:fixture(),trajectorySettings:{enabled:true,showMap:true,heightFactor:1.2,windowSeconds:6}}],scenes:[]};
  const project=validateProject(JSON.parse(JSON.stringify(input)));assert.equal(project.assets[0].trajectory.samples.length,5);assert.equal(project.assets[0].trajectorySettings.heightFactor,1.2);
  input.assets[0].name='other.mp4';assert.throws(()=>validateProject(input));
});

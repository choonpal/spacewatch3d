import test from 'node:test';
import assert from 'node:assert/strict';
import {validateTrajectory,poseAt,floorPoint,projectSegment,pickPath,rotate,slerp,routeSteps,directionalSteps,viewAfterStep,routeNodeAt,defaultRouteLinks,validateRouteEdits,connectRouteNodes,editedRouteSteps} from '../trajectory-math.js';
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

test('roadview arrows never cross a missing interval or offer movement beyond a route endpoint',()=>{
  const data=validateTrajectory(fixture());
  assert.deepEqual(routeSteps(data,0).map(s=>[s.direction,s.t]),[[1,2]]);
  assert.deepEqual(routeSteps(data,2).map(s=>[s.direction,s.t]),[[-1,0]]);
  assert.deepEqual(routeSteps(data,3),[]);
  assert.deepEqual(routeSteps(data,5).map(s=>[s.direction,s.t]),[[-1,4]]);
  const gap=fixture();gap.segments=[[0,4]];
  assert.deepEqual(routeSteps(validateTrajectory(gap),1.5).map(s=>[s.direction,s.t]),[[-1,0],[1,2]]);
});

test('roadview directions follow the view and require a real horizontal displacement',()=>{
  const steps=routeSteps(validateTrajectory(fixture()),1);
  assert.equal(directionalSteps(steps,0).forward.t,2);
  assert.equal(directionalSteps(steps,0).back.t,0);
  assert.equal(directionalSteps(steps,90).left.t,2);
  assert.equal(directionalSteps(steps,90).forward,null);
  const stationary=fixture();stationary.samples.forEach(s=>s.p=[0,0,0]);
  assert.deepEqual(routeSteps(validateTrajectory(stationary),1),[]);
});

test('roadview movement preserves world view direction across camera rotation',()=>{
  const origin={q:[0,0,0,1]},destination={q:[0,Math.SQRT1_2,0,Math.SQRT1_2]};
  const shifted=viewAfterStep(origin,destination,{yaw:0,pitch:12,fov:75});
  assert.ok(Math.abs(shifted.yaw+90)<1e-8);assert.ok(Math.abs(shifted.pitch-12)<1e-8);
  assert.equal(shifted.fov,75);
});

test('editing a point name retains automatic movement and survives a tour JSON round trip',()=>{
  const data=validateTrajectory(fixture());
  const edits=validateRouteEdits({version:1,sourceSha256:data.source.sha256,nodes:{0:{t:0,name:'복도 입구'}}},data);
  assert.equal(routeNodeAt(data,.2),0);assert.equal(routeNodeAt(data,3),null);
  assert.deepEqual(editedRouteSteps(data,0,edits),routeSteps(data,0));
  const input={version:1,id:'tour',assets:[{id:'v',name:'video.mp4',size:10,type:'video',trajectory:data,routeEdits:edits}],scenes:[]};
  const saved=validateProject(JSON.parse(JSON.stringify(input))).assets[0].routeEdits;
  assert.deepEqual(saved,edits);
});

test('custom connection bearings and reverse connections account for camera rotation',()=>{
  const input=fixture();input.samples[2].q=[0,Math.SQRT1_2,0,Math.SQRT1_2];
  const data=validateTrajectory(input),edits=connectRouteNodes(data,null,0,2,90,true);
  assert.equal(directionalSteps(editedRouteSteps(data,0,edits),0).right.t,2);
  assert.ok(Math.abs(edits.nodes[2].links.find(link=>link.target===0).yaw-180)<1e-8);
  assert.equal(directionalSteps(editedRouteSteps(data,2,edits),0).back.t,0);
  assert.deepEqual(defaultRouteLinks(data,0).map(link=>link.target),[2]);
  const modified=connectRouteNodes(data,edits,0,2,-90);
  assert.equal(modified.nodes[0].links.length,1);
  assert.equal(directionalSteps(editedRouteSteps(data,0,modified),0).left.t,2);
  assert.equal(edits.nodes[0].links[0].yaw,90);
});

test('removing all connections stays disconnected after saving and follows the current camera frame',()=>{
  const input=fixture();input.samples[1].q=[0,Math.SQRT1_2,0,Math.SQRT1_2];
  const data=validateTrajectory(input),edits=connectRouteNodes(data,null,0,2,0);
  const steps=editedRouteSteps(data,.5,edits);
  assert.ok(Math.abs(steps[0].yaw+45)<1e-8);
  edits.nodes[0].links=[];
  assert.deepEqual(editedRouteSteps(data,0,validateRouteEdits(JSON.parse(JSON.stringify(edits)),data)),[]);
});

test('route edits reject source mismatches, changed timestamps, duplicate targets and missing intervals',()=>{
  const data=validateTrajectory(fixture()),edits=connectRouteNodes(data,null,0,2,0);
  const corrupt=change=>{const copy=structuredClone(edits);change(copy);assert.throws(()=>validateRouteEdits(copy,data));};
  corrupt(copy=>copy.sourceSha256='b'.repeat(64));
  corrupt(copy=>copy.nodes[0].t=.1);
  corrupt(copy=>copy.nodes[0].links[0].t=1);
  corrupt(copy=>copy.nodes[0].links[0].yaw=181);
  corrupt(copy=>copy.nodes[0].links.push({...copy.nodes[0].links[0]}));
  assert.throws(()=>connectRouteNodes(data,edits,0,3,0));
  assert.throws(()=>connectRouteNodes(data,edits,0,0,0));
  assert.throws(()=>connectRouteNodes(data,edits,0,99,0));
  const gap=fixture();gap.segments=[[0,4]];
  assert.throws(()=>connectRouteNodes(validateTrajectory(gap),null,1,3,0));
});

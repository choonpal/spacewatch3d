import test from 'node:test';
import assert from 'node:assert/strict';
import {linkedDestinations,projectKeyframe,targetVisibility,headingAfterMove,worldDirection} from '../pipeline-arrows.js';

const identity=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
function fixture(target=[1,2,10],pose=identity){
 const source={id:'source',position:[0,0,0],pose},destination={id:'destination',position:target,pose:identity};
 return {views:[source,destination],capture:{links:[{from_viewpoint:'source',to_viewpoint:'destination',connection_method:'captured_video_path',polyline_m:[source.position,target]}]}};
}
const close=(actual,expected)=>assert.ok(Math.abs(actual-expected)<1e-8,`${actual} != ${expected}`);

test('camera position projects at its exact perspective pixel, with camera Y down',()=>{
 const f=fixture(),a=linkedDestinations(f.capture,f.views,0)[0];
 const p=projectKeyframe(a,{yaw:0,pitch:0,fov:90},800,400);
 close(p.x,420);close(p.y,240);assert.deepEqual(a.position,[1,2,10]);
 close(a.pitch,Math.atan2(-2,Math.hypot(1,10))*180/Math.PI);
});
test('a rotated source camera maps its actual world target to the image centre',()=>{
 const pose=[[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]],f=fixture([10,0,0],pose);
 const a=linkedDestinations(f.capture,f.views,0)[0],p=projectKeyframe(a,{yaw:0,pitch:0,fov:80},900,600);
 close(p.x,450);close(p.y,300);
});
test('off-screen and rear targets never become artificial edge pins',()=>{
 const view={yaw:0,pitch:0,fov:80};
 assert.equal(projectKeyframe({yaw:100,pitch:0},view,800,400),null);
 assert.equal(projectKeyframe({yaw:70,pitch:0},view,400,800),null);
 assert.equal(projectKeyframe({yaw:0,pitch:80},view,800,400),null);
 const edge=projectKeyframe({yaw:0,pitch:0},view,40,40);close(edge.x,20);close(edge.y,20);
});
test('near targets keep their true height and do not receive a floor offset',()=>{
 const f=fixture([0,0,1]),a=linkedDestinations(f.capture,f.views,0)[0];
 close(a.pitch,0);close(projectKeyframe(a,{yaw:0,pitch:0,fov:90},800,400).y,200);
});
test('occlusion tests both triangle windings and only the source-to-target segment',()=>{
 const wall=z=>new Float32Array([-10,-10,z,10,-10,z,0,10,z]);
 assert.equal(targetVisibility(wall(2),[0,0,0],[0,0,10]),'occluded');
 assert.equal(targetVisibility(new Float32Array([0,10,2,10,-10,2,-10,-10,2]),[0,0,0],[0,0,10]),'occluded');
 assert.equal(targetVisibility(wall(12),[0,0,0],[0,0,10]),'visible');
 assert.equal(targetVisibility(wall(-2),[0,0,0],[0,0,10]),'visible');
 assert.equal(targetVisibility(wall(.01),[0,0,0],[0,0,10]),'visible');
 assert.equal(targetVisibility(wall(9.9),[0,0,0],[0,0,10]),'visible');
 assert.equal(targetVisibility(null,[0,0,0],[0,0,10]),'unknown');
});
test('exact target pins preserve explicit links and recorded routes',()=>{
 const f=fixture(),original=JSON.stringify(f),a=linkedDestinations(f.capture,f.views,0)[0];
 assert.equal(a.index,1);assert.equal(a.id,'destination');assert.deepEqual(a.path,[[0,0,0],[1,2,10]]);
 assert.equal(JSON.stringify(f),original);
 f.capture.links[0].polyline_m[1]=[99,99,99];assert.deepEqual(linkedDestinations(f.capture,f.views,0),[]);
});
test('navigation retains the same world viewing direction after a camera rotation',()=>{
 const rotated=[[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]],view={yaw:22,pitch:-8,fov:80};
 const next=headingAfterMove(identity,rotated,view),before=worldDirection(identity,view),after=worldDirection(rotated,next);
 before.forEach((v,i)=>close(v,after[i]));assert.equal(next.fov,80);
});

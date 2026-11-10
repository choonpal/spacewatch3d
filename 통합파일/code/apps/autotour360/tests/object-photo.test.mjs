import test from 'node:test';
import assert from 'node:assert/strict';
import {objectPhotoVisibility,reprojectObjectPhoto} from '../object-photo.js';

const camera=(x=0)=>({width:24,height:24,focal_px:16,camera_to_world:[[1,0,0,x],[0,1,0,0],[0,0,1,0],[0,0,0,1]]});
const object=[[-.6,-.6,2],[.6,-.6,2],[.6,.6,2],[-.6,-.6,2],[.6,.6,2],[-.6,.6,2]].flat();
const mesh={positions:new Float64Array(object),regionIds:new Uint8Array([1,1])};
const photo=()=>{const pixels=new Uint8ClampedArray(24*24*4);for(let y=0;y<24;y++)for(let x=0;x<24;x++){const n=(y*24+x)*4;pixels.set([x*10,y*10,63,255],n);}return pixels;};
const alpha=(rgba,x,y)=>rgba[(y*24+x)*4+3];

test('keeps actual source RGB and makes the photo background transparent',()=>{
  const rgba=reprojectObjectPhoto(mesh,1,camera(),[{frame:camera(),pixels:photo()}]);
  assert.deepEqual([...rgba.slice((11*24+11)*4,(11*24+11)*4+4)],[110,110,63,255]);
  assert.equal(alpha(rgba,1,1),0);assert.equal(alpha(rgba,6,11),0);
});
test('reprojects source pixels with the camera pose instead of moving a rectangular crop',()=>{
  const rgba=reprojectObjectPhoto(mesh,1,camera(.5),[{frame:camera(),pixels:photo()}]);
  assert.deepEqual([...rgba.slice((11*24+7)*4,(11*24+7)*4+4)],[110,110,63,255]);
  assert.equal(alpha(rgba,15,11),0);
});
test('does not copy an occluding object or unlabelled foreground into the selected object',()=>{
  for(const foregroundId of [2,0,255]){
    const foreground=object.map((v,i)=>i%3===2?1:v);
    const occluded={positions:new Float64Array([...object,...foreground]),regionIds:new Uint8Array([1,1,foregroundId,foregroundId])};
    const visibility=objectPhotoVisibility(occluded,camera());
    assert.equal(visibility.ids[11*24+11],foregroundId);
    const rgba=reprojectObjectPhoto(occluded,1,camera(),[{frame:camera(),pixels:photo(),visibility}]);
    assert.equal(alpha(rgba,11,11),0);
  }
});
test('leaves unobserved surfaces transparent instead of synthesizing an object',()=>{
  const rgba=reprojectObjectPhoto(mesh,1,camera(),[]);
  assert.ok(rgba.every(v=>v===0));
});
test('rejects hidden back surfaces even when the occluding surface belongs to the same object',()=>{
  const foreground=object.map((v,i)=>i%3===2?1:v/6);
  const selfOccluded={positions:new Float64Array([...object,...foreground]),regionIds:new Uint8Array([1,1,1,1])};
  const rgba=reprojectObjectPhoto(selfOccluded,1,camera(1),[{frame:camera(),pixels:photo()}],{inset:0});
  assert.equal(alpha(rgba,3,11),0);
});
test('rejects source surfaces behind the camera and clips triangles crossing the near plane',()=>{
  const reversed={...camera(),camera_to_world:[[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]]};
  const rgba=reprojectObjectPhoto(mesh,1,camera(),[{frame:reversed,pixels:photo()}]);
  assert.equal(alpha(rgba,11,11),0);
  const crossing={positions:new Float64Array([-.1,-.1,.05,.2,-.1,1,0,.2,1]),regionIds:new Uint8Array([1])};
  const visibility=objectPhotoVisibility(crossing,camera());
  assert.ok([...visibility.depth].some(Number.isFinite));
  assert.ok([...visibility.depth].every(v=>v===Infinity||v>=.1-1e-6));
});

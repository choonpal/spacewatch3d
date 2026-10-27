import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const {photoLayers,photoCaption}=await import('data:text/javascript;base64,'+Buffer.from(await readFile(new URL('../tools/mesh_simsac/photo-presentation.js',import.meta.url))).toString('base64'));

test('each original photo highlights only its own selected object',()=>{
  const event={type:'moved',before:'E0-I2',after:'E1-I7',object_ids:['E0-I2','E1-I7']};
  const before={key:'E0-I2',epoch:0},after={key:'E1-I7',epoch:1},unrelated={key:'E1-I9',epoch:1};
  const record={panorama:{ghosts:[before,after,unrelated]},ghosts:[before,after,unrelated]};
  for(const pano of [false,true]){
    assert.deepEqual(photoLayers(event,record,0,pano),{highlights:[before],bounds:[before]});
    assert.deepEqual(photoLayers(event,record,1,pano),{highlights:[after],bounds:[after]});
  }
  assert.equal(photoCaption(event,0),'선택 객체 · 전');
  assert.equal(photoCaption(event,1),'선택 객체 · 후');
});

test('an added object is never painted into the before RGB',()=>{
  const event={type:'added',before:null,after:'E1-I7',object_ids:['E1-I7']};
  const after={key:'E1-I7',epoch:1};
  assert.deepEqual(photoLayers(event,{panorama:{ghosts:[after]}},0,true),{highlights:[],bounds:[after]});
  assert.equal(photoCaption(event,0),'비교 위치');
});

test('a missing uncertain counterpart remains an expected location',()=>{
  const event={type:'unresolved',before:'E0-I3',after:null,object_ids:['E0-I3']};
  const before={epoch:0,image:'legacy-selected-object.png'};
  assert.deepEqual(photoLayers(event,{ghosts:[before]},1),{highlights:[],bounds:[before]});
  assert.equal(photoCaption(event,1),'비교 위치');
});

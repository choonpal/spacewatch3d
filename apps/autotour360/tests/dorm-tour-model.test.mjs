import test from 'node:test';
import assert from 'node:assert/strict';
import {decisionLabel,itemsForFilter,panoramaView,decodeIndexedMesh} from '../dorm-tour-model.js';

test('ambiguous occupancy candidates remain in both possible-type filters without becoming confirmed moves',()=>{
  const event={id:'E001',decision:'candidate',alternatives:['possible_added','possible_relocation'],confirmed:false};
  const deferred={id:'C092',decision:'deferred',alternatives:[],confirmed:false};
  const data={events:[event],records:[deferred]};
  assert.equal(decisionLabel(event),'이동·추가 가능');
  assert.deepEqual(itemsForFilter(data,'moved'),[event]);
  assert.deepEqual(itemsForFilter(data,'added'),[event]);
  assert.deepEqual(itemsForFilter(data,'removed'),[]);
  assert.deepEqual(itemsForFilter(data,'deferred'),[deferred]);
  assert.equal(decisionLabel(deferred),'판단 보류');
  assert.equal(event.confirmed,false);
});

test('a source box crossing the panorama seam keeps its narrow angular extent',()=>{
  const view=panoramaView([1015,240,1055,280],1036,518,1.5);
  assert.ok(view.yaw>170&&view.yaw<190);
  assert.equal(view.fov,35);
  assert.ok(Math.abs(view.pitch)<2);
});

test('mesh decoding rejects a truncated asset and out-of-range triangle indices',()=>{
  const bytes=new ArrayBuffer(8+3*24+12);new Uint32Array(bytes,0,2).set([3,1]);
  const indices=new Uint32Array(bytes,8+3*24,3);indices.set([0,1,2]);
  assert.equal(decodeIndexedMesh(bytes).nf,1);
  assert.throws(()=>decodeIndexedMesh(bytes.slice(0,-1)));
  indices[2]=3;assert.throws(()=>decodeIndexedMesh(bytes),/정점 연결/);
});

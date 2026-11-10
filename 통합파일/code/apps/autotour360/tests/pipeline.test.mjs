import test from 'node:test';
import assert from 'node:assert/strict';
import {decodeGeometry, orderedViews, routePolylines, mapProjection} from '../pipeline-model.js';
test('binary geometry rejects truncation and reads separate RGB and semantic buffers',()=>{
  const b=new ArrayBuffer(106),h=new DataView(b);[0x53575031,3,1,0].forEach((n,i)=>h.setUint32(4*i,n,true));
  new Float32Array(b,16,9).set([1,2,3,4,5,6,7,8,9]);new Uint8Array(b,88,9).fill(37);new Uint8Array(b,97,9).fill(129);
  const d=decodeGeometry(b);assert.deepEqual([...d.positions],[1,2,3,4,5,6,7,8,9]);assert.equal(d.rgb[0],37);assert.equal(d.semantic[0],129);
  assert.throws(()=>decodeGeometry(b.slice(0,-1)));h.setUint32(8,2,true);assert.throws(()=>decodeGeometry(b));
});
test('point previews carry the original RGB and no triangle normals',()=>{
  const b=new ArrayBuffer(46),h=new DataView(b);[0x53575031,2,2,0].forEach((n,i)=>h.setUint32(4*i,n,true));new Uint8Array(b,40,6).set([1,2,3,4,5,6]);const d=decodeGeometry(b);assert.equal(d.normals,null);assert.deepEqual([...d.rgb],[1,2,3,4,5,6]);
});
test('visit ordering preserves every image and never manufactures component bridges',()=>{
  const c={views:[{id:'a',position:[0,0,0]},{id:'b',position:[0,20,0]},{id:'c',position:[1,30,0]}],tour_order:['b','a','b','missing'],links:[{connection_method:'disconnected',polyline_m:[[0,0,0],[0,20,0]]},{connection_method:'shortest_safe_path',polyline_m:[[0,20,0],[1,30,0]]}]};
  assert.deepEqual(orderedViews(c).map(v=>v.id),['b','a','c']);assert.equal(routePolylines(c).length,1);const p=mapProjection(c.views);for(const v of c.views){const [x,y]=p(v.position);assert.ok(x>=20&&x<=620&&y>=20&&y<=180);}
});

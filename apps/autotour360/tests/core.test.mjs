import test from 'node:test';
import assert from 'node:assert/strict';
import {cameraBasis,projectPoint,removeScene,validateProject,formatTime} from '../core.js';
const view={yaw:0,pitch:0,fov:75};
const fixture=()=>({version:1,id:'tour',title:'투어',assets:[{id:'video',name:'video.mp4',type:'video',size:1}],scenes:[{id:'a',assetId:'video',title:'A',startTime:0,view,hotspots:[{id:'h',type:'link',title:'B',targetId:'b',yaw:0,pitch:0}]},{id:'b',assetId:'video',title:'B',startTime:25,view,hotspots:[]}]});
test('spherical hotspot projection: center, behind, tilted camera, 360 seam',()=>{
  for(const v of [view,{yaw:179,pitch:45,fov:55},{yaw:-70,pitch:-40,fov:90}]){
    const p=projectPoint(v.yaw,v.pitch,v,1200,600);assert.ok(Math.abs(p.x-600)<1e-7);assert.ok(Math.abs(p.y-300)<1e-7);
    assert.equal(projectPoint(v.yaw+180,-v.pitch,v,1200,600),null);
    assert.ok(Math.abs(cameraBasis(v).forward.reduce((s,x)=>s+x*x,0)-1)<1e-10);
  }
  const seam=projectPoint(-179,0,{yaw:179,pitch:0,fov:75},1200,600);assert.ok(seam.x>600&&seam.x<630);
});
test('remove scene removes inbound links and only unreferenced assets',()=>{const p=fixture();removeScene(p,'b');assert.equal(p.scenes[0].hotspots.length,0);assert.equal(p.assets.length,1);removeScene(p,'a');assert.equal(p.assets.length,0);});
test('project export/import preserves time, orientation and graph',()=>{const p=fixture();const parsed=validateProject(JSON.parse(JSON.stringify(p)));assert.equal(parsed.scenes[1].startTime,25);assert.deepEqual(parsed.scenes[0].view,view);assert.equal(parsed.scenes[0].hotspots[0].targetId,'b');});
test('reject dangling references, duplicates, unknown media and excessive objects',()=>{
  let p=fixture();p.scenes[0].hotspots[0].targetId='unknown';assert.throws(()=>validateProject(p));
  p=fixture();p.scenes[1].id='a';assert.throws(()=>validateProject(p));
  p=fixture();p.assets[0].type='html';assert.throws(()=>validateProject(p));
  p=fixture();p.scenes=new Array(201).fill(p.scenes[0]);assert.throws(()=>validateProject(p));
});
test('import strips external paths and executable thumbnail data; clamps viewpoint',()=>{
  const p=fixture();p.assets[0].url='https://external.test/private';p.assets[0].path='/etc/passwd';p.scenes[0].thumbnail='data:image/svg+xml,<svg onload="alert(1)"/>';p.scenes[0].view={yaw:999,pitch:999,fov:1};
  const clean=validateProject(p);assert.equal(clean.assets[0].url,undefined);assert.equal(clean.assets[0].path,undefined);assert.equal(clean.scenes[0].thumbnail,'');assert.equal(clean.scenes[0].view.pitch,85);assert.equal(clean.scenes[0].view.fov,35);
});
test('duration formatting handles unknown and negative values',()=>{assert.equal(formatTime(68.549),'01:08');assert.equal(formatTime(NaN),'00:00');assert.equal(formatTime(-1),'00:00');});

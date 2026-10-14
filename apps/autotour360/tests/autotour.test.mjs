import test from 'node:test';
import assert from 'node:assert/strict';
import {validateProject} from '../core.js';
test('saved tours retain stable video identity',()=>{
  const asset={id:'a',name:'video.mp4',size:123,type:'video',trajectoryKey:'a'.repeat(64)};
  const restored=validateProject({version:1,id:'p',assets:[asset],scenes:[]});
  assert.equal(restored.assets[0].trajectoryKey,asset.trajectoryKey);
});

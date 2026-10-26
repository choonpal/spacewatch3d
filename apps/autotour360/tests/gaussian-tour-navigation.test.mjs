import test from 'node:test';
import assert from 'node:assert/strict';
import {capturePhase,routeBetween,neighbor} from '../assets/gaussian-change/gaussian-dual-path-tour-20261026/tour_navigation.js';

test('separate capture paths select their own map and legacy after-anchored tours keep the before map',()=>{
 assert.equal(capturePhase({anchor_epoch:0},true),'before');
 assert.equal(capturePhase({anchor_epoch:1},true),'after');
 assert.equal(capturePhase({anchor_epoch:1},false),'before');
});

test('distinct capture ids and disconnected before segments never imply an animated walking path',()=>{
 const group={nodes:[{id:'D2-K01'},{id:'D2-K02'},{id:'D2-K03'}],
  links:[{from:'D2-K01',to:'D2-K02',polyline:[[0,0,0],[0,0,1]]}],
  segments:[{nodes:['D2-K01','D2-K02']},{nodes:['D2-K03']}]};
 assert.equal(routeBetween(group,'D2-K02','D2-K03'),null);
 assert.equal(routeBetween(group,'D2-K01','D3-K01'),null);
 assert.equal(neighbor(group,'D2-K02',1),null);
 assert.equal(neighbor(group,'D2-K03',-1),null);
});

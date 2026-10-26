import test from 'node:test';
import assert from 'node:assert/strict';
import {routeBetween,neighbor,samplePath,pathLength} from '../../../tools/gaussian_change/tour_navigation.js';

const group={nodes:[{id:'a',position:[0,0,0]},{id:'c',position:[2,0,1]},{id:'b',position:[2,0,2]},{id:'d',position:[99,0,0]}],
 segments:[{nodes:['a','c','b']},{nodes:['d']}],links:[{from:'a',to:'c',polyline:[[0,0,0],[0,0,1],[2,0,1]]},{from:'c',to:'b',polyline:[[2,0,1],[2,0,2]]}]};
test('follow stored path around the corner and retain intermediate keyframes',()=>{
 const route=routeBetween(group,'a','b');assert.deepEqual(route.ids,['a','c','b']);
 assert.deepEqual(route.points,[[0,0,0],[0,0,1],[2,0,1],[2,0,2]]);
 assert.equal(neighbor(group,'a',1),'c');assert.equal(neighbor(group,'c',1),'b');
 assert.equal(pathLength(route.points),4);assert.deepEqual(samplePath(route.points,.125),[0,0,.5]);
});
test('reverse navigation reverses the actual polyline',()=>{
 const route=routeBetween(group,'b','a');assert.deepEqual(route.ids,['b','c','a']);
 assert.deepEqual(route.points,routeBetween(group,'a','b').points.toReversed());
});
test('stop at a disconnected segment and never invent a walking connection',()=>{
 assert.equal(routeBetween(group,'b','d'),null);assert.equal(neighbor(group,'b',1),null);
 assert.equal(neighbor(group,'d',-1),null);assert.equal(routeBetween(group,'missing','a'),null);
});
test('distance sampling handles duplicate points and clamps endpoints',()=>{
 const points=[[0,0,0],[0,0,0],[0,0,3]];assert.deepEqual(samplePath(points,-1),[0,0,0]);
 assert.deepEqual(samplePath(points,.5),[0,0,1.5]);assert.deepEqual(samplePath(points,4),[0,0,3]);
});

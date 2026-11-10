import test from 'node:test';
import assert from 'node:assert/strict';
import {RouteSurface,prepareRoute,routeTriangles,sampleRecordedPath,routeArrivalDirection,simplifyGuide,chevronPlacements,routeOccluders,routeTrackTriangles} from '../pipeline-route-arrows.js';

const floorColor=[80,124,230],wallColor=[230,80,80],classes=[{name:'floor',rgb:floorColor},{name:'wall',rgb:wallColor}];
function surface(wall=false,scale=1,table=false){
 const triangles=[[-20,-20,0,20,-20,0,20,20,0],[-20,-20,0,20,20,0,-20,20,0]];
 if(wall)triangles.push([-3,3,0,3,3,0,3,3,4],[-3,3,0,3,3,4,-3,3,4]);
 if(table)triangles.push([-1,-1,1,1,-1,1,1,1,1],[-1,-1,1,1,1,1,-1,1,1]);
 return new RouteSurface({positions:new Float32Array(triangles.flat().map(v=>v*scale)),semantic:new Uint8Array(triangles.flatMap((_,i)=>Array(3).fill(i<2?floorColor:wallColor).flat()))},classes);
}
function route(path,s=surface()) {return prepareRoute(s,{position:path[0]},{id:'next',index:1,label:'K2',path,position:path.at(-1)});}

test('a visible right angle retains the recorded corner rather than drawing a diagonal',()=>{
 const path=[[0,0,2],[0,5,2],[4,5,2]],before=JSON.stringify(path),r=route(path);
 assert.equal(r.status,'ready');assert.equal(JSON.stringify(path),before);
 assert.ok(r.points.some(p=>p.point[0]===0&&p.point[1]===5));
 assert.ok(r.points.some(p=>p.point[0]>0&&p.point[1]===5));
 for(const p of r.points)assert.ok(Math.abs(p.point[0])<1e-8||Math.abs(p.point[1]-5)<1e-8);
 assert.ok(routeTriangles(r).length>0);
});
test('a blocking wall terminates the displayed route before the hidden continuation',()=>{
 const s=surface(true),r=route([[0,0,2],[0,8,2]],s);
 assert.equal(r.status,'ready');assert.ok(r.points.every(p=>p.point[1]<3));
 assert.ok(r.points.every(p=>s.visible([0,0,2],p.point)));
});
test('floor grounding ignores a non-floor horizontal tabletop',()=>{
 const s=surface(false,1,true),r=route([[0,0,2],[0,8,2]],s);
 assert.equal(r.height,2);for(const p of r.points)assert.ok(Math.abs(p.point[2]-.036)<1e-8);
});
test('display distances follow camera height and shorten for a nearby destination',()=>{
 const a=route([[0,0,2],[0,10,2]]),b=route([[0,0,4],[0,20,4]],surface(false,2));
 assert.ok(Math.abs(b.nearArc/a.nearArc-2)<1e-8);assert.ok(Math.abs(b.tipArc/a.tipArc-2)<1e-8);
 const near=route([[0,0,2],[0,1.8,2]]);assert.equal(near.status,'ready');assert.ok(near.tipArc<=1.8);assert.ok(near.nearArc>=.8);
});
test('missing floor evidence yields no fabricated ground path',()=>{
 const s=new RouteSurface({positions:new Float32Array([0,0,0,10,0,0,0,10,0]),semantic:new Uint8Array(Array(3).fill(wallColor).flat())},classes);
 const r=route([[0,0,2],[0,8,2]],s);assert.equal(r.status,'no-floor');assert.equal(routeTriangles(r).length,0);
});
test('sampling preserves vertices, zero-length input points and the arc limit without inventing an edge',()=>{
 const samples=sampleRecordedPath([[0,0,2],[0,0,2],[0,1,2],[1,1,2]],.3,1.5);
 assert.ok(samples.some(s=>s.point[0]===0&&s.point[1]===1));assert.deepEqual(samples.at(-1).point,[.5,1,2]);assert.equal(samples.at(-1).arc,1.5);
});
test('arrival orientation follows the final recorded segment, including reverse travel',()=>{
 const path=[[0,0,2],[0,5,2],[4,5,2],[4,5,2]];
 assert.deepEqual(routeArrivalDirection(path),[4,0,0]);
 assert.deepEqual(routeArrivalDirection([...path].reverse()),[0,-5,0]);
});
test('guide cleanup removes tiny jitter while retaining a real right angle',()=>{
 const points=[[0,0,0],[.01,1,0],[0,2,0],[0,3,0],[1,3,0],[2,3,0]].map((point,arc)=>({point,arc}));
 const simplified=simplifyGuide(points,.03);assert.deepEqual(simplified.map(p=>p.point),[[0,0,0],[0,3,0],[2,3,0]]);
 assert.equal(points.length,6);
});
test('short and long guides use the same chevron stroke rather than an oversized triangular head',()=>{
 const short=route([[0,0,2],[0,1.8,2]]),long=route([[0,0,2],[0,9,2]]);
 assert.ok(chevronPlacements(short).length>=2);assert.ok(chevronPlacements(long).length>=5);
 for(const r of [short,long]){const cs=chevronPlacements(r),c=cs[0];assert.ok(Math.hypot(c.points[0][0]-c.points[2][0],c.points[0][1]-c.points[2][1])<.3*r.height);assert.ok(cs.every(c=>c.arc>=r.nearArc&&c.arc<=r.tipArc));for(let i=1;i<cs.length;i++)assert.ok(cs[i].arc-cs[i-1].arc<.4*r.height);assert.ok(routeTrackTriangles(r).length>0);}
});
test('only classified horizontal floor faces leave the display depth pass; walls remain',()=>{
 const s=surface(true),occluders=routeOccluders(s.positions,s.floorFaces);
 assert.equal(s.floorFaces.size,2);assert.equal(occluders.length,18);assert.deepEqual([...occluders],[...s.positions.slice(18)]);
 assert.equal(s.visible([0,0,2],[0,5,.04]),false);
});
test('a long unobstructed corridor continues to the linked destination without a lookahead cap',()=>{
 const r=route([[0,0,2],[0,18,2]]);assert.equal(r.status,'ready');assert.equal(r.tipArc,18);assert.equal(r.totalArc,18);assert.deepEqual(r.points.at(-1).point,[0,18,.036]);
 assert.ok(chevronPlacements(r).some(c=>c.arc>15));
});
test('a later visible corridor resumes after an occluded interval, without bridging the gap',()=>{
 const triangles=[[-20,-20,0,20,-20,0,20,20,0],[-20,-20,0,20,20,0,-20,20,0],[-1,3,0,1,3,0,1,3,4],[-1,3,0,1,3,4,-1,3,4]];
 const s=new RouteSurface({positions:new Float32Array(triangles.flat()),semantic:new Uint8Array(triangles.flatMap((_,i)=>Array(3).fill(i<2?floorColor:wallColor).flat()))},classes);
 const r=route([[0,0,2],[0,5,2],[6,5,2]],s);assert.equal(r.status,'ready');assert.ok(r.runs.length>=2);assert.ok(r.points.some(p=>p.point[0]>3));
 for(let i=1;i<r.runs.length;i++)assert.ok(r.runs[i][0].arc-r.runs[i-1].at(-1).arc>.1);
 assert.ok(r.points.every(p=>s.visible([0,0,2],p.point)));
 const meshes=r.runs.map(points=>routeTrackTriangles({...r,runs:null,points}));assert.equal(routeTrackTriangles(r).length,meshes.reduce((n,m)=>n+m.length,0));
});
test('destination labels remain bound to their actual camera coordinates',()=>{
 const path=[[0,0,2],[0,5,2],[6,5,2]],anchor={id:'target',index:1,label:'K2',path,position:path.at(-1)},r=prepareRoute(surface(),{position:path[0]},anchor);
 assert.deepEqual(r.labelPoint,path.at(-1));assert.equal(r.destinationVisibility,'visible');
});

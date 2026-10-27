import test from 'node:test';
import assert from 'node:assert/strict';
import {displayEvent,changeStops,routeStops,neighborStop,focusBounds} from '../change-navigation.js';

const bounds=[[0,0,4],[1,1,5]],view=(epoch,index,x)=>({source_epoch:epoch,source_index:index,camera_to_common_cv:[[1,0,0,x],[0,1,0,1.5],[0,0,1,0],[0,0,0,1]]});
test('one representative origin is displayed without changing algorithm evidence',()=>{
 const original={type:'moved',correspondence:{before:'B'},exact_origin_confirmed:false,origin_candidates:[{object_id:'A',appearance:.95,bounds,center:[1,0,4],points:[]},{object_id:'B',appearance:.9,bounds,center:[2,0,4],points:[]}]};
 const visible=displayEvent(original);
 assert.equal(visible.origin_candidates.length,1);assert.equal(visible.origin_candidates[0].object_id,'B');
 assert.equal(visible.origin_alternative_count,2);assert.equal(visible.exact_origin_confirmed,false);
 assert.equal(original.origin_candidates.length,2);assert.equal(original.before_center,undefined);
});
test('change navigation skips empty cameras and merges events at the same keypoint',()=>{
 const views=[view(0,0,0),view(0,1,0),view(0,2,0),view(1,0,0)];
 const events=[{id:'a',type:'unresolved',bounds,observations:[1,1]},{id:'b',type:'unresolved',bounds,observations:[1]},{id:'c',type:'unresolved',bounds,observations:[3]}];
 assert.deepEqual(changeStops(events,views),[{viewIndex:1,eventIds:['a','b']},{viewIndex:3,eventIds:['c']}]);
 assert.deepEqual(routeStops(views,0).map(s=>s.viewIndex),[0,1,2]);
 assert.equal(neighborStop(changeStops(events,views),0,-1),0);assert.equal(neighborStop(changeStops(events,views),0,1),1);
});
test('unclassified change framing uses detected region instead of an entire semantic component',()=>{
 assert.deepEqual(focusBounds({type:'unresolved',bounds,before_bounds:[[-9,-9,-9],[9,9,9]]}),bounds);
 assert.deepEqual(focusBounds({type:'added',bounds,after_bounds:bounds}),bounds);
});

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {uniqueForView,visibleRegions} from '../../../tools/gaussian_change/change_regions.js';
const folder=new URL('../../../src/spacewatch3d/tasks/task_5a_change_detection/assets/panorama-change-20261025/gaussian-uncapped-tour-20261026/',import.meta.url);
const read=name=>JSON.parse(fs.readFileSync(new URL(name,folder),'utf8'));
const registry=read('change-regions.json'),fixed=read('tour-candidates.json');
test('the lounge is one card with two events regardless of its five observation views',()=>{
 const groups=visibleRegions(registry.modes.filtered,fixed.boxes.filtered);
 assert.equal(groups.length,3);assert.equal(groups[0].events.length,2);
 assert.deepEqual(groups[0].observing_keyframes,['K1','K2','K3','K5','K6']);
 for(const view of groups[0].observing_keyframes){
  const boxes=uniqueForView(fixed.boxes.filtered,registry.modes.filtered,fixed.visibility[view].filtered);
  assert.ok(boxes.every(b=>groups[0].events.some(e=>e.candidate_ids.includes(b.id))));
 }
 assert.equal(groups.reduce((n,g)=>n+g.events.length,0),5);
});
test('multiple raw boxes for one matched object create only one overlay per view',()=>{
 for(const [view,v] of Object.entries(fixed.visibility)){
  const boxes=uniqueForView(fixed.boxes.raw,registry.modes.raw,v.raw),ids=boxes.map(b=>registry.modes.raw.candidate_to_event[b.id]);
  assert.equal(ids.length,new Set(ids).size,view);
 }
});
test('filtering changes visibility without moving events to a new area or counting views',()=>{
 const only=fixed.boxes.filtered.filter(b=>b.id==='SimSaC-R12-001');
 const groups=visibleRegions(registry.modes.filtered,only);
 assert.equal(groups.length,1);assert.equal(groups[0].id,registry.modes.filtered.regions[0].id);
 assert.equal(groups[0].events.length,1);assert.deepEqual(groups[0].observing_keyframes,['K1','K3','K5']);
});
test('an event visible in one epoch keeps its strongest source overlay',()=>{
 const boxes=[{id:'weak'},{id:'strong'}],index={candidate_to_event:{weak:'same',strong:'same'}};
 const visibility={weak:{visible:true,support_rays:{before:0,after:3}},strong:{visible:true,support_rays:{before:0,after:30}}};
 assert.deepEqual(uniqueForView(boxes,index,visibility),[{id:'strong'}]);
});

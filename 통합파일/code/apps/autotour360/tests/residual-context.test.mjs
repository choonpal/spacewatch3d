import test from 'node:test';
import assert from 'node:assert/strict';
import {validate,linkedPhotos,filtered,OUTCOMES,REFERENCE_OUTCOMES,regionLabel,resultRoot,exceedCount} from '../residual-context.js';

test('residual experiments require the unchanged aligned input and meter coordinates',()=>{
  const data={kind:'reviewed_roi_residual_comparison',new_registration:false,same_input_meshes_and_poses:true,units:'meters',source_hashes:['a','b'],displays:[{},{}]};
  assert.equal(validate(data),data);
  assert.throws(()=>validate({...data,new_registration:true}));
  assert.throws(()=>validate({...data,units:'centimeters'}));
});
test('experimental original frames stay in their epoch and do not load external sources',()=>{
  const data={keyframes:[{id:'a',epoch:0,url:'/residual-comparison/frame-0-000001.jpg'},
    {id:'b',epoch:1,url:'/residual-comparison/frame-1-000002.jpg'},
    {id:'x',epoch:0,url:'https://example.com/frame-0-000001.jpg'}]};
  const region={keyframes:[[{frame_id:'a'},{frame_id:'a'},{frame_id:'b'},{frame_id:'x'}],[{frame_id:'b'}]]};
  assert.deepEqual(linkedPhotos(data,region,0).map(p=>p.frame.id),['a']);
  assert.deepEqual(linkedPhotos(data,region,1).map(p=>p.frame.id),['b']);
});
test('unassessable patches never appear as removal or as a passing change',()=>{
  const regions=[{kind:'fit',statistics:[{}, {status:'unassessable'}]},
    {kind:'local_patch',statistics:[{}, {status:'unassessable'}]},
    {kind:'local_patch',statistics:[{}, {status:'change_candidate'}]}];
  assert.equal(filtered(regions,'unassessable').length,1);
  assert.equal(filtered(regions,'change_candidate').length,1);
  assert.equal(filtered(regions,'fit').length,1);
  assert.equal(OUTCOMES.unassessable,'판단 불가');
  assert.equal(OUTCOMES.removed,undefined);
});
test('no approved reference is shown as deferred instead of a misleading zero-change count',()=>{
  assert.equal(exceedCount({valid_cores:0,exceeding_cores:0}),'보류');
  assert.equal(exceedCount({valid_cores:50,exceeding_cores:0}),'0');
  assert.equal(exceedCount({valid_cores:50,exceeding_cores:12}),'12');
});

test('common structural surfaces are a separate excluded category, never a change patch',()=>{
  const regions=[{kind:'background',statistics:[{}, {status:'background'}]},
    {kind:'local_patch',statistics:[{}, {status:'change_candidate'}]}];
  assert.equal(filtered(regions,'background').length,1);
  assert.equal(filtered(regions,'change_candidate').length,1);
  assert.equal(OUTCOMES.background,'공통 기준면 · 후보 제외');
  const data={kind:'structural_plane_residual_comparison',new_registration:false,same_input_meshes_and_poses:true,units:'meters',source_hashes:['a','b'],displays:[{},{}]};
  assert.equal(validate(data),data);
});

test('reference evaluation rows do not inflate independently found candidate patch counts',()=>{
  const regions=[{kind:'reference',title:'의자',evaluation:{reference_type:'moved',outcome:'captured'},statistics:[{}, {status:'change_candidate'}]},
    {kind:'reference',evaluation:{outcome:'missed'},statistics:[{}, {status:'unassessable'}]},
    {kind:'local_patch',statistics:[{}, {status:'change_candidate'}]}];
  assert.equal(filtered(regions,'reference').length,2);
  assert.equal(filtered(regions,'captured').length,1);
  assert.equal(filtered(regions,'missed').length,1);
  assert.equal(filtered(regions,'partial').length,0);
  assert.equal(filtered(regions,'change_candidate').length,1);
  assert.equal(regionLabel(regions[0]),'이동 · 의자 · 위치 포착');
  assert.equal(REFERENCE_OUTCOMES.missed,'미포착');
});

test('all-direction results retain the same-map and same-pose input checks',()=>{
  const data={kind:'all_surface_residual_comparison',new_registration:false,same_input_meshes_and_poses:true,units:'meters',source_hashes:['a','b'],displays:[{},{}]};
  assert.equal(validate(data),data);
  assert.throws(()=>validate({...data,same_input_meshes_and_poses:false}));
});

test('transfer photos cannot leak across dataset roots or capture epochs',()=>{
  const data={dataset_id:'bonn-placement',keyframes:[{id:'new',epoch:0,url:'/residual-comparison/bonn-placement/frame-0-000001.jpg'},
    {id:'old',epoch:0,url:'/residual-comparison/frame-0-000001.jpg'},
    {id:'after',epoch:1,url:'/residual-comparison/bonn-placement/frame-1-000002.jpg'}]};
  const region={keyframes:[[{frame_id:'new'},{frame_id:'old'},{frame_id:'after'}],[]]};
  assert.deepEqual(linkedPhotos(data,region,0).map(p=>p.frame.id),['new']);
  assert.equal(resultRoot('?dataset=bonn-placement'),'/residual-comparison/bonn-placement/');
  assert.equal(resultRoot(''),'/residual-comparison/');
  assert.throws(()=>resultRoot('?dataset=../private'));
});

test('a transfer with labels or retuning cannot be presented as frozen',()=>{
  const data={kind:'frozen_indoor_transfer',dataset_id:'bonn-placement',new_registration:false,
    same_input_meshes_and_poses:true,units:'meters',source_hashes:['a','b'],displays:[{},{}],
    transfer:{parameters_retuned:false,frozen_source_verified:true,labels_used_for_detection:false}};
  assert.equal(validate(data),data);
  assert.throws(()=>validate({...data,transfer:{...data.transfer,parameters_retuned:true}}));
  assert.throws(()=>validate({...data,transfer:{...data.transfer,labels_used_for_detection:true}}));
});

test('Dorm relative units and a new similarity cannot masquerade as metric frozen results',()=>{
  const data={kind:'aligned_mesh_pair_comparison',dataset_id:'dorm2-dorm3',new_registration:true,
    units:'reference_units',physical_scale_known:false,labels_used_for_detection:false,source_hashes:['a','b'],displays:[{},{}],
    registration:{after_to_before:[[1],[1],[1],[1]],quality:{relative_scale_after_to_before:1.1}}};
  assert.equal(validate(data),data);
  for(const mutation of [{units:'meters'},{labels_used_for_detection:true},{physical_scale_known:true}])assert.throws(()=>validate({...data,...mutation}));
  assert.equal(resultRoot('?dataset=dorm2-dorm3'),'/residual-comparison/dorm2-dorm3/');
  const frames={dataset_id:'dorm2-dorm3',keyframes:[{id:'right',epoch:0,url:'/residual-comparison/dorm2-dorm3/frame-0-000001.jpg'},
    {id:'other',epoch:0,url:'/residual-comparison/frame-0-000001.jpg'},{id:'wrongEpoch',epoch:1,url:'/residual-comparison/dorm2-dorm3/frame-1-000002.jpg'}]};
  assert.deepEqual(linkedPhotos(frames,{keyframes:[[{frame_id:'right'},{frame_id:'other'},{frame_id:'wrongEpoch'}],[]]},0).map(x=>x.frame.id),['right']);
});

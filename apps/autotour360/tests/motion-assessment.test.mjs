import test from 'node:test';
import assert from 'node:assert/strict';
import {bindMotionAssessments,assessmentFor,motionPresentation} from '../motion-assessment.js';
const data={scene:'room',source_epochs:[0,2],registration:{after_to_before:[[1]]},displays:[{mesh:{sha256:'a'}},{mesh:{sha256:'b'}}],regions:[{id:1,type:'moved'},{id:2,type:'added'}]};
const report=()=>({kind:'local_background_motion_assessment',version:1,source_scene:'room',source_epochs:[0,2],source_changes_sha256:'hash',source_registration:[[1]],source_meshes_sha256:['a','b'],assessments:[{region_id:1,status:'motion_supported',evidence:'translation',corrected_observable_displacement_m:.3,comparison_envelope_m:.04,motion_threshold_m:.08,corrected_rotation_deg:0,rotation_threshold_deg:15,local_background:{usable:true,angular_observable:true}},{region_id:2,status:'not_applicable'}]});
test('binds decisions to the exact original manifest, mesh and registration',()=>{
  const map=bindMotionAssessments(data,report(),'hash');
  assert.equal(assessmentFor(data.regions[0],map).status,'motion_supported');
  for(const mutate of [r=>r.source_changes_sha256='other',r=>r.source_meshes_sha256[0]='other',r=>r.source_epochs=[0,1],r=>r.source_registration=[[2]]]){
    const r=report();mutate(r);assert.throws(()=>bindMotionAssessments(data,r,'hash'));
  }
});
test('missing or unobservable evidence cannot silently preserve an old moved label',()=>{
  assert.equal(assessmentFor(data.regions[0],new Map()).status,'uncertain');
  assert.equal(motionPresentation(data.regions[0],{status:'uncertain',reason:'unconstrained_motion_direction'}).label,'판단 보류');
  const r=report();r.assessments[0].corrected_observable_displacement_m=.02;
  assert.throws(()=>bindMotionAssessments(data,r,'hash'));
});
test('partial directional support cannot claim that alignment explains the entire movement',()=>{
  const r=report();Object.assign(r.assessments[0],{status:'alignment_explained',corrected_displacement_m:.01,local_background:{usable:true,directions_complete:false,observable_dimensions:2}});
  assert.throws(()=>bindMotionAssessments(data,r,'hash'));
});
test('addition and removal remain presence changes rather than confirmed motion',()=>{
  assert.equal(assessmentFor(data.regions[1],new Map()).status,'not_applicable');
  assert.equal(motionPresentation(data.regions[1],{}).label,'추가');
});

export const MOTION_STATUSES={motion_supported:'이동 근거',alignment_explained:'정합 오차 범위',uncertain:'판단 보류',not_applicable:'추가·소실'};
const REASONS={
  insufficient_background:'주변에 공통으로 관측된 기준 표면이 부족합니다.',
  insufficient_spatial_holdout:'주변 오차를 별도 구역에서 확인할 표본이 부족합니다.',
  unconstrained_background:'주변 표면만으로 정합이 어긋난 방향을 알 수 없습니다.',
  unconstrained_motion_direction:'주변 기준면으로 확인하지 못한 방향이 있어 이동 여부를 확정할 수 없습니다.',
  ambiguous_object_identity:'전후 객체 대응이 모호해 이동 판정을 보류합니다.',
  weak_object_surface_fit:'전후 객체 표면의 대응이 충분하지 않습니다.',
  background_correction_too_large:'주변 정합 보정량이 적용 범위를 넘어 판단을 보류합니다.',
  motion_near_error_limit:'남은 차이가 오차 기준에 가까워 이동 판정을 보류합니다.',
  no_matching_assessment:'현재 촬영에 대응하는 주변 오차 분석이 없어 이동 판정을 보류합니다.',
};
const finite=value=>Number.isFinite(value)&&value>=0;

export function bindMotionAssessments(data,report,manifestHash){
  if(report?.kind!=='local_background_motion_assessment'||report.version!==1||report.source_changes_sha256!==manifestHash||
     report.source_scene!==data.scene||JSON.stringify(report.source_epochs)!==JSON.stringify(data.source_epochs)||
     JSON.stringify(report.source_registration)!==JSON.stringify(data.registration.after_to_before)||
     JSON.stringify(report.source_meshes_sha256)!==JSON.stringify(data.displays.map(d=>d.mesh.sha256)))throw new Error('다른 촬영의 오차 분석입니다.');
  const map=new Map(),regions=new Map(data.regions.map(region=>[region.id,region]));
  for(const assessment of report.assessments||[]){
    const region=regions.get(assessment.region_id),model=assessment.local_background;
    if(!region||map.has(region.id)||!Object.hasOwn(MOTION_STATUSES,assessment.status))throw new Error('오차 분석 항목이 올바르지 않습니다.');
    if(region.type==='moved'&&assessment.status==='not_applicable')throw new Error('객체 이동 분석이 누락됐습니다.');
    if(region.type!=='moved'&&assessment.status!=='not_applicable')throw new Error('추가·소실을 이동 오차로 재분류할 수 없습니다.');
    if(['motion_supported','alignment_explained'].includes(assessment.status)){
      if(!model?.usable||!finite(assessment.corrected_observable_displacement_m)||!finite(assessment.comparison_envelope_m)||
        !finite(assessment.motion_threshold_m)||!finite(assessment.corrected_rotation_deg)||!finite(assessment.rotation_threshold_deg)||
        region.identity_ambiguous)throw new Error('이동 판정의 근거가 부족합니다.');
      if(assessment.status==='motion_supported'&&!(assessment.corrected_observable_displacement_m>assessment.motion_threshold_m||
        (model.angular_observable&&assessment.corrected_rotation_deg>assessment.rotation_threshold_deg)))throw new Error('주변 오차를 넘는 이동 근거가 없습니다.');
      if(assessment.status==='alignment_explained'&&(!model.directions_complete||model.observable_dimensions!==3||
        !finite(assessment.corrected_displacement_m)||assessment.corrected_displacement_m>assessment.comparison_envelope_m||
        assessment.corrected_rotation_deg>model.rotation_envelope_deg))throw new Error('정합만으로 설명할 근거가 부족합니다.');
    }
    map.set(region.id,assessment);
  }
  return map;
}
export function assessmentFor(region,assessments){
  return assessments?.get(region.id)||{region_id:region.id,status:region.type==='moved'?'uncertain':'not_applicable',reason:'no_matching_assessment',local_background:{}};
}
export function motionPresentation(region,assessment){
  if(region.type!=='moved')return {label:region.type==='added'?'추가':'소실',tone:region.type,description:'기존 객체 존재 여부에 따른 분류입니다. 주변 오차를 이용한 이동 판정의 대상은 아닙니다.'};
  if(assessment.status==='motion_supported')return {label:assessment.evidence==='rotation'?'회전 근거':'이동 근거',tone:'motion-supported',
    description:assessment.evidence==='rotation'?'주변 정합 영향을 고려해도 객체의 회전이 기준을 넘습니다.':'확인 가능한 방향에서 주변 정합 영향을 고려해도 객체 이동이 기준을 넘습니다.'};
  if(assessment.status==='alignment_explained')return {label:'정합 오차 범위',tone:'alignment-explained',description:'잔여 변위가 정합·재구성 오차 범위 안에 있습니다. 실제로 움직이지 않았다는 뜻은 아니며 이동으로 확정할 근거가 부족합니다.'};
  return {label:'판단 보류',tone:'motion-uncertain',description:REASONS[assessment.reason]||'주변 오차와 실제 이동을 구분할 근거가 충분하지 않습니다.'};
}

export const OUTCOMES={change_candidate:'변화 후보',insufficient_change_evidence:'변화 근거 부족',unassessable:'판단 불가',background:'공통 기준면 · 후보 제외'};
export const REFERENCE_OUTCOMES={captured:'위치 포착',partial:'일부 표면만 포착',missed:'미포착'};
export const REFERENCE_TYPES={added:'추가',removed:'소실',moved:'이동'};
export const REASONS={all_direction_surface_difference:'여러 크기에서 노이즈 기준을 넘는 3D 표면 차이',within_local_noise_budget:'국소 노이즈 기준 이내',insufficient_source_surface_support:'원본 표면 지지 표본 부족',common_structural_surface_excluded:'전후 공통 구조면 · 후보 제외',insufficient_connected_exceedance:'연결된 변화 표면의 근거 부족',no_comparable_reference:'비슷한 조건의 기준 없음',insufficient_paired_surface_observation:'전후 유효 깊이 관측 부족',below_empirical_reference:'경험적 기준 이내',exceeds_comparable_reference:'비슷한 조건의 기준 초과',reference_does_not_transfer_to_check:'독립 확인 영역으로 기준이 전달되지 않음',insufficient_check_support:'독립 확인 표본 부족',insufficient_reference_support:'기준 추정 표본 부족',paired_original_photos_unavailable:'전후 원본 사진 검토 불가',reference_accepted_on_spatial_check:'독립 확인 통과',fit_check_surface_support_overlap:'추정·확인 영역의 실제 표면 표본이 겹침'};
export function validate(data){
  if(!['reviewed_roi_residual_comparison','structural_plane_residual_comparison','all_surface_residual_comparison','frozen_indoor_transfer'].includes(data.kind)||data.new_registration!==false||data.same_input_meshes_and_poses!==true||data.units!=='meters'||data.source_hashes?.length!==2||data.displays?.length!==2)throw new Error('정합 맵 또는 실험 결과가 다릅니다.');
  if(data.kind==='frozen_indoor_transfer'&&(data.dataset_id!=='bonn-placement'||data.transfer?.parameters_retuned!==false||data.transfer?.frozen_source_verified!==true||data.transfer?.labels_used_for_detection!==false))throw new Error('고정 설정 적용 기록을 확인할 수 없습니다.');
  return data;
}
export function linkedPhotos(data,region,epoch){
  const frames=new Map(data.keyframes.map(frame=>[frame.id,frame])),seen=new Set();
  const prefix=data.dataset_id==='bonn-placement'?'/residual-comparison/bonn-placement/':'/residual-comparison/';
  return (region.keyframes?.[epoch]||[]).flatMap(link=>{
    const frame=frames.get(link.frame_id);
    if(!frame||frame.epoch!==epoch||seen.has(frame.id)||!frame.url?.startsWith(prefix)||!/^frame-[01]-\d{6}\.jpg$/.test(frame.url.slice(prefix.length)))return [];
    seen.add(frame.id);return [{frame,link}];
  });
}
export function resultRoot(search){
  const dataset=new URLSearchParams(search).get('dataset')||'rescan';
  if(dataset==='rescan')return '/residual-comparison/';
  if(dataset==='bonn-placement')return '/residual-comparison/bonn-placement/';
  throw new Error('알 수 없는 촬영 쌍입니다.');
}
export function filtered(regions,filter){
  if(filter==='all')return regions;
  if(['fit','check','provided_object','background','reference'].includes(filter))return regions.filter(r=>r.kind===filter);
  if(Object.hasOwn(REFERENCE_OUTCOMES,filter))return regions.filter(r=>r.kind==='reference'&&r.evaluation?.outcome===filter);
  return regions.filter(r=>r.kind==='local_patch'&&r.statistics[1].status===filter);
}
export function exceedCount(summary){
  return summary.valid_cores===0?'보류':summary.exceeding_cores.toLocaleString();
}
export function regionLabel(region){
  if(region.kind==='background')return region.title;
  const e=region.evaluation;
  return e?`${REFERENCE_TYPES[e.reference_type]||'기존 항목'} · ${region.title} · ${REFERENCE_OUTCOMES[e.outcome]}`:region.title+' · '+OUTCOMES[region.statistics[1].status];
}

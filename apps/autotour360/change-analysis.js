import {createPanorama} from './change-panorama.js';
import {displayEvent,changeStops,routeStops,neighborStop} from './change-navigation.js';
const $=id=>document.getElementById(id),phases=['before','after'],names=['전','후'];
const stageGroups=[['input_validation','입력 검사'],['global_registration','전역 정합'],['regional_correction','구역 보정'],['render_and_infer','렌더링 · SimSaC'],['fusion_3d','3D 통합 · 유형 분류'],['complete','결과 보기']];
let busy=false,ready=false,job=null,result=null,cameras=null,current=0,selected=null,objectEvents=null;
let changesOnly=true,stops=[],stopIndex=0,sceneVersion=0;
const viewer=createPanorama($('panorama'),$('boxes'),id=>selectCandidate(id));
for(let i=0;i<2;i++){const section=document.createElement('fieldset'),legend=document.createElement('legend');legend.textContent=names[i]+' 촬영';section.append(legend);for(const [field,title,accept] of [['mesh','메시 맵','.npz,.ply,.obj'],['gaussian','Gaussian 맵','.ply'],['cameras','저장 카메라','.json']]){const label=document.createElement('label');label.textContent=title;const input=document.createElement('input');input.type='file';input.id=phases[i]+'-'+field;input.accept=accept;label.append(input);section.append(label);}$('file-fields').append(section);}
async function api(url,options){const response=await fetch(url,options);const data=await response.json();if(!response.ok)throw Error(data.error||'요청에 실패했습니다.');return data;}
function stages(stage){const order=stageGroups.map(x=>x[0]);let i=order.indexOf(stage);if(['input_validation','transferring','gpu_wait','mesh_sampling','gaussian_loading','model_loading'].includes(stage))i=['gaussian_loading','model_loading'].includes(stage)?3:stage==='mesh_sampling'?1:0;if(['save_3d_changes','receiving_results','object_classification'].includes(stage))i=4;$('stages').replaceChildren(...stageGroups.map(([key,title],index)=>{const el=document.createElement('span');el.className='stage'+(index<i?' done':index===i?' active':'');el.textContent=title;return el;}));}
function source(){return document.querySelector('input[name=source]:checked').value;}
function refreshSource(){const isServer=source()==='server';$('file-fields').hidden=isServer;$('coordinate-fields').hidden=isServer;$('server-note').hidden=!isServer;}
document.querySelectorAll('input[name=source]').forEach(el=>el.onchange=refreshSource);
function setBusy(value){busy=value;$('run').disabled=value||!ready;document.querySelectorAll('aside input, aside select').forEach(el=>el.disabled=value||(el.id==='server-source'&&!el.dataset.available));}
function descriptor(){if($('same-frame').checked){const identity=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],side={world_from_mesh:identity,world_from_gaussian:identity,world_from_camera:identity,camera_convention:'opencv'};return {units:'declared shared input units',before:side,after:side};}throw Error('좌표 연결 JSON을 입력하거나 동일 좌표계 조건을 확인해 주세요.');}
$('inputs').onsubmit=async event=>{event.preventDefault();if(busy)return;$('error').textContent='';try{let desc=null,files=[];if(source()==='upload'){for(const p of phases)for(const field of ['mesh','gaussian','cameras']){const id=p+'-'+field,file=$(id).files[0];if(!file)throw Error((p==='before'?'전':'후')+' 촬영의 '+field+' 파일을 선택해 주세요.');files.push([id,file]);}desc=$('descriptor').files[0]?JSON.parse(await $('descriptor').files[0].text()):descriptor();}
 setBusy(true);result=null;$('result-controls').hidden=true;$('tour-navigation').hidden=true;$('empty').hidden=false;$('state').textContent='새 작업과 입력을 준비하는 중';$('empty').querySelector('strong').textContent='새 입력을 분석하고 있습니다';$('empty').querySelector('p').textContent='정합·렌더링·변화탐지 결과가 도착하면 이 장면에 표시합니다.';stages('input_validation');job=await api('/api/change-jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source:source()})});history.replaceState(null,'','?job='+job.id);$('job-info').textContent='새 작업 '+job.id;
 for(let i=0;i<files.length;i++){const [id,file]=files[i];$('upload-status').textContent=`입력 전송 ${i+1}/${files.length} · ${file.name}`;await api(`/api/change-jobs/${job.id}/inputs/${id}?name=${encodeURIComponent(file.name)}`,{method:'PUT',headers:{'Content-Type':'application/octet-stream'},body:file});}
 $('upload-status').textContent='입력 완료 · GPU 서버에서 새로 분석합니다';await api(`/api/change-jobs/${job.id}/run`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({descriptor:{...(desc||{}),analysis_profile:$('analysis-profile').value}})});await poll();
 }catch(error){$('error').textContent=error.message;$('state').textContent='실행을 완료하지 못했습니다';setBusy(false);}};
async function poll(){while(job){const state=await api('/api/change-jobs/'+job.id);stages(state.state==='complete'?'complete':state.stage);$('elapsed').textContent=Math.round(state.elapsed_s)+'초';const titles={transferring_inputs:'맵·카메라 입력을 GPU 서버로 전송',created:'입력 준비',uploading:'입력 전송',input_validation:'입력 검사',mesh_sampling:'메시 표면 분석',global_registration:'전후 메시 전역 정합',regional_correction:'구역별 정합 보정',gaussian_loading:'보정된 Gaussian 준비',model_loading:'변화탐지 모델 준비',render_and_infer:'동일 시점 렌더링 · SimSaC 변화탐지',fusion_3d:'여러 시점의 3D 후보 통합',object_classification:'분할 객체 대응 · 이동/추가/소실 판정',receiving_results:'새 분석 결과 전송'};$('state').textContent=(titles[state.stage]||state.stage)+(Number.isInteger(state.completed)&&state.total?` · ${state.completed+1}/${state.total}`:'');if(state.state==='complete'){await loadResult();setBusy(false);return;}if(['failed','interrupted'].includes(state.state))throw Error(state.message||'변화탐지에 실패했습니다.');await new Promise(resolve=>setTimeout(resolve,1800));}}
const typeLabel=kind=>({moved:'이동 추정',added:'추가 추정',removed:'소실 추정',modified:'제자리 외관 차이',unresolved:'판정 근거 부족',unclassified:'판정 근거 부족'}[kind]||'변화후보');
const eventItems=()=>(objectEvents?.events||result.candidates).map(displayEvent);
async function loadResult(){
 const base=`/api/change-jobs/${job.id}/files/`;
 [result,cameras]=await Promise.all([api(base+'changes.json'),api(base+'cameras.json')]);
 const response=await fetch(base+'object-events.json');objectEvents=response.ok?await response.json():null;
 showComparison(result.comparison?await api(base+result.comparison):null);
 $('empty').hidden=true;$('result-controls').hidden=false;$('tour-navigation').hidden=false;$('toggle-inputs').hidden=false;document.body.classList.add('has-result');document.body.classList.remove('inputs-open');
 $('state').textContent=`분석 완료 · 고유 변화 ${eventItems().length}개`;
 const counts=objectEvents?.counts||{unresolved:result.candidates.length};
 $('result-note').textContent=`${result.render_size?`렌더 ${result.render_size.join('×')} · ${result.inference_mode==='perspective_tiles'?`시야 ${result.observations[0]?.tiles_per_phase||0}개/시점 분석`:'전체 파노라마 분석'}. `:''}자동 판정: 이동 ${counts.moved||0} · 추가 ${counts.added||0} · 소실 ${counts.removed||0} · 제자리 외관 차이 ${counts.modified||0} · 근거 부족 ${counts.unresolved||0}. 원시 후보 ${result.candidates.length}개 중 객체 근거 없는 구조 표면 ${objectEvents?.excluded_surface_candidates?.length||0}개는 객체 목록에서 제외했습니다. 구조 비교 지원 ${result.registration_supported_regions}/${result.region_count}구역. 같은 객체는 경로를 바꿔도 같은 ID입니다.`;
 changesOnly=true;current=0;$('route').value='0';rebuildNavigation();await navigate(0);$('tour-navigation').scrollIntoView({block:'start'});
}
function showComparison(report){
 $('run-comparison').hidden=!report;if(!report)return;
 const a=report.baseline,b=report.current,rate=n=>Math.round(n*100)+'%';
 $('comparison-title').textContent=`기존 → 4K 시야별 · 근거 부족 ${a.counts.unresolved}/${a.events} → ${b.counts.unresolved}/${b.events} (${rate(a.unresolved_rate)} → ${rate(b.unresolved_rate)})`;
 const counts=x=>`이동 ${x.moved||0} · 추가 ${x.added||0} · 소실 ${x.removed||0} · 외관 차이 ${x.modified||0} · 근거 부족 ${x.unresolved||0}`;
 $('comparison-summary').textContent=`기존: ${counts(a.counts)} / 새 분석: ${counts(b.counts)}. 기존 미분류 영역 중 분류된 영역 ${report.region_outcomes.classified||0}개.`;
 const outcomes={classified:'유형 분류됨',still_unresolved:'여전히 근거 부족',partly_classified:'일부만 분류됨',excluded_as_structural_surface:'구조 표면으로 제외',no_corresponding_detection:'이번 분석에 대응 검출 없음'};
 $('comparison-regions').replaceChildren(...report.baseline_unresolved_regions.map(r=>{const li=document.createElement('li');li.textContent=`기존 ${r.baseline_id}: ${outcomes[r.outcome]||r.outcome} `;for(const match of r.current_matches){const button=document.createElement('button');button.textContent=match.id+' '+typeLabel(match.type);button.onclick=()=>goToEvent(match.id).catch(showError);li.append(button);}return li;}));
 const reasons=b.unresolved_reasons||{};$('comparison-summary').textContent+=` 현재 미분류: 분할 객체 미연결 ${reasons.no_segmented_object_support||0}, 객체 대응·관측 부족 ${reasons.counterpart_or_observation_evidence_insufficient||0}.`;
 for(const motion of report.matched_motion_events||[])$('comparison-summary').textContent+=` 동일 이동 객체 외형 대응 ${motion.baseline_appearance.toFixed(3)} → ${motion.current_appearance.toFixed(3)}, 출발 대응 후보 ${motion.baseline_origin_hypotheses} → ${motion.current_origin_hypotheses}.`;
 const id=report.baseline_job_id;$('baseline-result').hidden=!/^[a-f0-9]{24}$/.test(id||'');if(!$('baseline-result').hidden)$('baseline-result').href='/change-analysis.html?job='+id;
}
async function goToEvent(id){changesOnly=true;rebuildNavigation();const index=stops.findIndex(s=>s.eventIds.includes(id));if(index<0)return;await navigate(index);viewer.focus(id);$('run-comparison').open=false;$('tour-navigation').scrollIntoView({block:'start'});}
function rebuildNavigation(){
 stops=changesOnly?changeStops(eventItems(),cameras.views):routeStops(cameras.views,Number($('route').value));
 $('mode-changes').setAttribute('aria-pressed',String(changesOnly));$('mode-route').setAttribute('aria-pressed',String(!changesOnly));$('route-filter').hidden=changesOnly;
 $('prev').textContent=changesOnly?'← 이전 변화':'← 이전 키프레임';$('next').textContent=changesOnly?'다음 변화 →':'다음 키프레임 →';
 $('navigation-note').textContent=changesOnly?'같은 변화는 대표 시점 한 번만 표시합니다. 이전·다음으로 이동하고 박스를 누르세요.':'키프레임을 눌러 경로를 둘러보세요. 주황 점은 변화가 관측된 시점입니다.';
 $('keyframe').replaceChildren(...stops.map((stop,i)=>{const view=cameras.views[stop.viewIndex],option=document.createElement('option');option.value=String(i);option.textContent=`${view.source_epoch?'후':'전'} K${view.source_index+1}${changesOnly?' · 변화 '+stop.eventIds.length+'개':''}`;return option;}));
 $('path-points').replaceChildren(...stops.map((stop,i)=>{const view=cameras.views[stop.viewIndex],events=eventItems().filter(e=>e.observations.includes(stop.viewIndex)),button=document.createElement('button');button.className='route-point';button.dataset.changed=String(events.length>0);button.textContent=`${changesOnly?(view.source_epoch?'후 ':'전 '):''}K${view.source_index+1}`;button.title=events.length?events.map(e=>typeLabel(e.type)+' '+e.id).join(' · '):'변화 관측 없음';button.onclick=()=>navigate(i).catch(showError);return button;}));
}
function showError(error){$('error').textContent=error.message;}
async function navigate(index){
 if(!stops.length){$('tour-position').textContent='관측된 변화 시점 없음';$('prev').disabled=true;$('next').disabled=true;await scene();return;}
 stopIndex=Math.max(0,Math.min(stops.length-1,index));current=stops[stopIndex].viewIndex;$('keyframe').value=String(stopIndex);
 $('tour-position').textContent=`${changesOnly?'변화 시점':'키프레임'} ${stopIndex+1} / ${stops.length}`;
 $('prev').disabled=stopIndex===0;$('next').disabled=stopIndex===stops.length-1;
 [...$('path-points').children].forEach((b,i)=>{b.classList.toggle('current',i===stopIndex);b.setAttribute('aria-current',i===stopIndex?'step':'false');});
 await scene();
}
async function scene(){
 if(!result)return;const token=++sceneVersion,view=cameras.views[current],obs=result.observations[current],base=`/api/change-jobs/${job.id}/files/`;
 const candidates=eventItems().filter(c=>c.observations.includes(current));selected=null;$('event-detail').hidden=true;$('selection').textContent=candidates.length?'표시된 박스를 누르면 후 모습이 겹쳐집니다.':'이 키프레임에는 관측된 변화가 없습니다.';
 for(const id of ['compare-before','compare-ghost','compare-after']){$(id).disabled=true;$(id).setAttribute('aria-pressed','false');}$('all-after').checked=false;
 $('view-label').textContent=`${view.source_epoch===0?'전':'후'} 경로 · K${view.source_index+1} · 변화 ${candidates.length}개 · 전 모습`;
 $('candidate-list').replaceChildren(...candidates.map(c=>{const button=document.createElement('button');button.textContent=typeLabel(c.type)+' · '+c.id;button.dataset.id=c.id;button.dataset.type=c.type;button.disabled=true;button.onclick=()=>selectCandidate(c.id);return button;}));
 await viewer.scene(base+obs.directory+'/before.png',base+obs.directory+'/after.png',view.camera_to_common_cv,candidates);
 if(token!==sceneVersion)return;
 document.querySelectorAll('#candidate-list button').forEach(b=>b.disabled=false);
 const ids=stops[stopIndex]?.eventIds||[],focus=candidates.find(c=>ids.includes(c.id))||candidates[0];if(focus)viewer.focus(focus.id);
}
const reasons={no_segmented_object_support:'이 변화 영역과 직접 연결되는 분할 객체가 없습니다.',counterpart_or_observation_evidence_insufficient:'분할 객체는 있지만 동일 객체 대응 또는 빈자리 관측이 부족합니다.',rigid_identity_and_both_rendered_vacancies_supported:'형상·외형의 1:1 대응과 출발·도착 위치의 반대 회차 빈자리를 확인했습니다.',expected_location_observed_free_in_independent_frames:'반대 회차에서 해당 객체 위치의 빈자리가 여러 시점에 관측됐습니다.',rgb_identity_weak:'RGB 외형 대응 약함',identity_frame_support:'객체 관측 시점 부족',origin_vacancy_unobserved:'출발 위치의 이후 빈자리 관측 부족',destination_vacancy_unobserved:'도착 위치의 이전 빈자리 관측 부족',repeated_object_identity_ambiguous:'비슷한 객체 사이의 대응 모호',shape_fit_weak:'형상 대응 약함',partial_shape_coverage:'대응 형상 범위 부족',displacement_within_map_error:'정합 오차보다 작은 위치 차이'};
function movementMap(event){
 const svg=$('movement-map');svg.replaceChildren();const hidden=event.type!=='moved'||!event.before_center||!event.after_center;svg.toggleAttribute('hidden',hidden);if(hidden)return;
 const origins=event.origin_candidates?.length?event.origin_candidates:[{center:event.before_center}],points=origins.map(o=>o.center).concat([event.after_center]);
 const lo=[Math.min(...points.map(p=>p[0])),Math.min(...points.map(p=>p[2]))],span=Math.max(...points.map(p=>p[0]-lo[0]),...points.map(p=>p[2]-lo[1]),.1);
 const mapped=points.map(p=>[35+(p[0]-lo[0])/span*240,120-(p[2]-lo[1])/span*80]),b=mapped.at(-1);
 const add=(tag,attrs)=>{const el=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v]of Object.entries(attrs))el.setAttribute(k,v);svg.append(el);return el;};
 mapped.slice(0,-1).forEach(a=>{add('line',{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:'#ffbd66','stroke-width':2,'stroke-dasharray':origins.length>1?'5 4':'none'});const angle=Math.atan2(b[1]-a[1],b[0]-a[0]);add('polygon',{points:`${b[0]},${b[1]} ${b[0]-10*Math.cos(angle-.5)},${b[1]-10*Math.sin(angle-.5)} ${b[0]-10*Math.cos(angle+.5)},${b[1]-10*Math.sin(angle+.5)}`,fill:'#ffbd66'});add('circle',{cx:a[0],cy:a[1],r:5,fill:'#61e7df'});});
 add('circle',{cx:b[0],cy:b[1],r:6,fill:'#ffbd66'});const originLabel=add('text',{x:140,y:55,fill:'#61e7df','font-size':12});originLabel.textContent=event.representative_origin?'청록 · 대표 전 위치':'청록 · 전 위치';const target=add('text',{x:140,y:77,fill:'#ffbd66','font-size':12});target.textContent='주황 · 후 위치';
 const label=add('text',{x:12,y:20,fill:'#a8c2d7','font-size':11});label.textContent=origins.length>1?'X–Z 평면 · 출발 객체 대응은 복수 후보':'X–Z 평면 · 객체의 출발 → 도착 위치';
}
function compare(value){if(!selected){$('selection').textContent='먼저 변화 박스를 선택해 주세요';return;}viewer.comparison(value);$('all-after').checked=false;viewer.after(false);for(const [id,weight]of [['compare-before',0],['compare-ghost',.55],['compare-after',1]])$(id).setAttribute('aria-pressed',String(value===weight));$('tour-navigation').scrollIntoView({block:'start'});}
function clearSelection(){
 selected=null;viewer.select(null);$('event-detail').hidden=true;$('selection').textContent='박스를 눌러 전후 변화를 확인하세요';
 for(const id of ['compare-before','compare-ghost','compare-after']){$(id).disabled=true;$(id).setAttribute('aria-pressed','false');}
 document.querySelectorAll('#candidate-list button').forEach(b=>b.classList.remove('selected'));
}
function selectCandidate(id){
 if(selected===id){clearSelection();return;}
 selected=id;viewer.select(id);const event=eventItems().find(e=>e.id===id);if(!event){clearSelection();return;}
 $('event-detail').hidden=false;for(const key of ['compare-before','compare-ghost','compare-after'])$(key).disabled=false;
 $('selection').textContent=`${typeLabel(event.type)} · ${event.id} · 청록은 전, 주황은 후 위치입니다.`;
 $('event-summary').textContent=event.type==='moved'?`${event.label||'객체'} · 대표 전 위치 → 후 위치 (자동 추정)`:typeLabel(event.type)+(event.label?' · '+event.label:'');
 const extra={motion_type_supported_but_origin_has_multiple_candidates:'형상·Gaussian RGB 대응과 빈자리 관측을 이용해 점수가 가장 높은 출발 위치 한 곳을 대표로 표시합니다. 출발 위치는 자동 추정입니다.',mask_instance_correspondence_shape_gain_and_destination_vacancy_supported:'분할 객체의 형상·Gaussian RGB 대응과 도착 위치의 이전 빈자리 관측으로 이동을 추정했습니다.',location_remains_occupied_with_rgb_or_surface_difference:'객체가 같은 자리에 남아 있으나 외관 또는 표면 차이가 있습니다.',same_position_object_with_surface_difference:'같은 위치의 객체에 외관 또는 표면 차이가 있습니다.',opposite_location_occluded_or_unobserved:'반대 회차에서 해당 위치가 가려졌거나 충분히 관측되지 않았습니다.'};
 $('event-reason').textContent=(extra[event.reason]||reasons[event.reason]||event.reason||'판정 근거가 없습니다.')+' '+[...new Set(event.rejection_reasons||[])].map(r=>reasons[r]||r).join(' · ');
 movementMap(event);compare(.55);document.querySelectorAll('#candidate-list button').forEach(b=>b.classList.toggle('selected',b.dataset.id===id));
}
for(const [id,weight]of [['compare-before',0],['compare-ghost',.55],['compare-after',1]])$(id).onclick=()=>compare(weight);
function switchMode(only){changesOnly=only;if(!only)$('route').value=String(cameras.views[current].source_epoch);const previous=current;rebuildNavigation();navigate(Math.max(0,stops.findIndex(s=>s.viewIndex===previous))).catch(showError);}
$('mode-changes').onclick=()=>switchMode(true);$('mode-route').onclick=()=>switchMode(false);
$('route').onchange=()=>{rebuildNavigation();navigate(0).catch(showError);};$('keyframe').onchange=()=>navigate(Number($('keyframe').value)).catch(showError);
for(const [id,delta]of [['prev',-1],['next',1]])$(id).onclick=()=>navigate(neighborStop(stops,stopIndex,delta)).catch(showError);
$('reset').onclick=()=>viewer.reset();$('all-after').onchange=()=>viewer.after($('all-after').checked);$('clear').onclick=clearSelection;
$('toggle-inputs').onclick=()=>{const open=document.body.classList.toggle('inputs-open');$('toggle-inputs').textContent=open?'투어로 돌아가기':'입력 설정';(open?document.querySelector('aside'):$('tour-navigation')).scrollIntoView({block:'start'});};
stages('input_validation');api('/api/change-jobs/config').then(async config=>{ready=config.connected;$('connection').textContent=config.connected?'GPU 서버 연결됨':'GPU 실행 설정 또는 SSH 연결 필요';$('run').disabled=!config.connected;$('server-source').disabled=!config.server_inputs_available;if(config.server_inputs_available)$('server-source').dataset.available='true';const saved=new URLSearchParams(location.search).get('job');if(saved&&/^[a-f0-9]{24}$/.test(saved)){job={id:saved};setBusy(true);$('job-info').textContent='작업 '+saved;await poll();}}).catch(error=>{$('error').textContent=error.message;setBusy(false);});

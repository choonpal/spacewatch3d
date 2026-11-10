import {Panorama} from './panorama.js';
import {PipelineMesh} from './pipeline-mesh.js';
import {orderedViews,routePolylines,mapProjection} from './pipeline-model.js';
const $=id=>document.getElementById(id);
const fail=error=>{$('error').textContent=error.message||String(error);$('error').hidden=false;};
let manifest,capture,views=[],index=0,mode='instances',imageToken=0,meshToken=0,pano,mesh;
const number=n=>Number(n).toLocaleString('ko-KR');
function element(tag,text,parent){const e=document.createElement(tag);if(text!=null)e.textContent=text;if(parent)parent.append(e);return e;}
function metric(label,value){const card=element('div',null,$('metrics'));card.className='metric';element('span',label,card);element('strong',value,card);}
function drawRoute(){
  const svg=$('route');svg.replaceChildren();if(!views.length)return;const project=mapProjection(views);
  const node=(name,attributes)=>{const e=document.createElementNS('http://www.w3.org/2000/svg',name);for(const [k,v] of Object.entries(attributes))e.setAttribute(k,v);svg.append(e);return e;};
  for(const line of routePolylines(capture))node('polyline',{points:line.map(p=>project(p).join(',')).join(' ')});
  views.forEach((view,i)=>{const [x,y]=project(view.position);const c=node('circle',{cx:x,cy:y,r:i===index?7:4,fill:i===index?'#d2f3b8':'#6aa595',tabindex:0,role:'button','aria-label':`키프레임 ${i+1}`});
    c.addEventListener('click',()=>selectView(i));c.addEventListener('keydown',e=>{if(['Enter',' '].includes(e.key)){e.preventDefault();selectView(i);}});
    if(i===index)node('text',{x:x+11,y:y+4}).textContent=String(i+1);
  });
}
async function selectView(i){
  if(!views.length){++imageToken;pano?.setMedia(null);$('view-id').textContent='생성된 키프레임 없음';$('view-index').textContent='0 / 0';$('previous').disabled=true;$('next').disabled=true;$('image-download').hidden=true;$('image-loading').hidden=false;$('image-loading').textContent='관측 위치 계획을 완료하지 못했습니다.';drawRoute();return;}
  $('image-download').hidden=false;$('image-loading').textContent='이미지 불러오는 중';
  index=Math.max(0,Math.min(views.length-1,i));const view=views[index],token=++imageToken;
  $('view-id').textContent=view.id;$('view-index').textContent=`${index+1} / ${views.length}`;
  $('previous').disabled=index===0;$('next').disabled=index===views.length-1;
  $('image-download').href=view.image;$('image-download').download=`${capture.id}-${view.id}.jpg`;
  [...$('thumbnails').children].forEach((b,j)=>b.setAttribute('aria-pressed',String(j===index)));
  drawRoute();$('image-loading').hidden=false;pano?.setMedia(null);
  const img=new Image();img.src=view.image;
  try{await img.decode();if(token!==imageToken)return;pano?.setMedia(img);$('image-loading').hidden=true;}
  catch(error){if(token===imageToken){$('image-loading').textContent='이미지를 불러오지 못했습니다.';fail(error);}}
}
async function loadMesh(){
  if(!mesh)return;const token=++meshToken;$('mesh-loading').hidden=false;
  $('mesh-loading').textContent='3D 데이터 불러오는 중';
  $('legend').hidden=mode!=='semantic';$('instance-controls').hidden=mode!=='instances';$('instance-note').hidden=mode!=='instances';
  const sample=capture.point_preview;
  $('mesh-note').textContent=mode==='points'?`표시 ${number(sample.display_points)} / 원본 ${number(sample.source_points)}점 · 휠로 확대`:'드래그 회전 · Shift + 드래그 이동 · 휠 확대';
  try{await mesh.load(capture,mode);if(token===meshToken)$('mesh-loading').hidden=true;}
  catch(error){if(token===meshToken){$('mesh-loading').textContent='3D 데이터를 불러오지 못했습니다.';fail(error);}}
}
async function selectCapture(c){
  capture=c;if(mode==='instances'&&!c.instance_mesh)mode='rgb';views=orderedViews(c);index=0;$('error').hidden=true;$('focus').disabled=!views.length;
  [...$('captures').children].forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.id===c.id)));
  document.querySelector('[data-mode=instances]').hidden=!c.instance_mesh;document.querySelectorAll('[data-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mode===mode)));
  $('instance-select').replaceChildren(new Option('전체 객체',''),...(c.objects||[]).map(o=>new Option(`${o.name} #${o.id} · ${o.faces}면`,o.id)));
  $('instance-note').textContent=c.instance_summary?(c.instance_summary.class_size_templates_used===false?`의미영역 관측 대상 ${c.instance_summary.candidate_instances}개 · 통계 제외 ${c.instance_summary.excluded_semantic_components}개. 모든 메시 면은 유지됩니다. 전체 개별 필수 관측은 해제되어 있으며 물리 인스턴스 정답은 아닙니다.`:`3D 후보 ${c.instance_summary.candidate_instances}개 · 크기 기준 ${c.instance_summary.size_references}개. 조건 밖의 면도 회색으로 남아 가림·충돌·표면 관측에 포함됩니다. 개별 필수 관측 대상은 후보뿐입니다.`):'';
  $('summary').textContent=`${c.id.toUpperCase()} · 생성된 키프레임과 공간 복원 결과를 확인하세요.`;
  const native=c.media_origin==='original_video_frame';
  $('metrics').replaceChildren();metric(native?'원본 키프레임':'3DGS 키프레임',`${views.length}장`);metric('메시 삼각형',number(c.faces));metric('원본 point cloud',`${(c.point_preview.source_points/1e6).toFixed(2)}M`);metric('변화 후보 / 판단 보류',manifest.changes?`${manifest.changes.candidates} / ${manifest.changes.deferred}`:'—');
  $('thumbnails').replaceChildren();views.forEach((v,i)=>{const b=element('button',null,$('thumbnails'));b.setAttribute('aria-label',`키프레임 ${i+1} 보기`);const img=element('img',null,b);img.src=v.thumbnail||v.image;img.alt='';img.loading='lazy';element('span',String(i+1).padStart(2,'0'),b);b.onclick=()=>selectView(i);});
  $('legend').replaceChildren();for(const row of [...c.classes.filter(r=>r.static_mesh),{name_ko:'미분류',rgb:[145,145,145]}]){const item=element('span',null,$('legend'));const swatch=element('i',null,item);swatch.style.background=`rgb(${row.rgb.join(',')})`;element('span',row.name_ko||row.name,item);}
  $('quality').textContent=native?'원본 촬영 프레임입니다. 픽셀은 보존하며, 비앵커 프레임 자세·관측률은 촬영경로 보간에 기반한 미검증 추정입니다.':views.length?'키프레임의 정확한 위치에서 3DGS로 생성한 360° 이미지입니다. 화질 검증은 아직 완료되지 않았으며 세부 흐림과 이음새가 남아 있습니다.':'선정된 관측 위치가 없어 3DGS 키프레임 이미지가 없습니다. 계획 정보를 확인해 주세요.';
  $('change-summary').textContent=manifest.changes?`변화 후보 ${manifest.changes.candidates}건 · 판단 보류 ${manifest.changes.deferred}건. 정합 상태와 구역별 근거는 상단의 변화 비교 화면에서 확인할 수 있습니다.`:'이 실행에서는 회차 간 변화 비교를 사용하지 않았습니다.';
  $('training').textContent=native?`Gaussian 학습 없음 · 원본 ${c.source_size?.join('×')} · 촬영 후보 ${number(c.capture_candidates)}장`:`학습 ${number(c.steps)}회 · 선택 모델 ${number(c.selected_step)}회 · 검증 PSNR ${c.validation_psnr.toFixed(2)} dB`;
  $('planning').textContent=`선정 기준: ${c.planning_status==='complete'?'충족':'일부 미충족'} · 전체입력 목표: ${c.input_coverage_status==='targets_met'?'충족':'미달 또는 미검증'} · 객체 품질 ${c.object_observed_count??'?'}/${c.object_input_count??'?'} · 전체 표면 독립 검증 최소 ${(100*c.whole_surface_coverage_min).toFixed(1)}% · 관측 가능 공간 균형 ${(100*c.coverage).toFixed(1)}%. ${[...c.failures,...(c.whole_input_unmet||[])].join(', ')}${c.metric_scale_verified?'':' · 실제 미터 스케일은 미검증입니다.'}`;
  $('run-id').textContent=manifest.run_id;
  $('downloads').replaceChildren();const names={all_regions:'전체 영역 목록',size_assumptions:'크기 가정',target_policy:'관측 대상 정책',all_arrays:'전체 분할 NPZ',instance_mesh:'객체 ID 메시',instance_ply:'객체 ID PLY',rgb_mesh:'RGB 메시',semantic_mesh:'분할 메시',labeled_mesh:'라벨 PLY',point_cloud:'원본 point cloud',plan:'키프레임 계획'};
  for(const [k,url] of Object.entries(c.downloads)){const a=element('a',names[k]+' ↓',$('downloads'));a.href=url;a.download='';}
  if(manifest.changes){const a=element('a','변화 결과 ↓',$('downloads'));a.href=manifest.changes.download;a.download='';}
  if(mesh){mesh.capture=c;mesh.count=0;mesh.reset();}
  pano?.setView({yaw:0,pitch:0,fov:80});await Promise.all([selectView(0),loadMesh()]);
}
async function start(){
  const response=await fetch('/api/pipeline');if(!response.ok)throw Error('이 뷰어에 연결된 파이프라인 결과가 없습니다.');manifest=await response.json();
  if(manifest.discovery){const d=manifest.discovery,provided=d.source_mode==='provided';$('discovery-info').hidden=false;$('discovery-link').hidden=provided;$('discovery-info').textContent=provided?`기존 라벨 ${d.classes.length}종 · 입력별 관측 대상 정책 적용 · 미분류·조각을 포함한 모든 메시 면을 보존합니다.`:`Florence-2 클래스 ${d.classes.length}종 · 크기는 별도 설정 파일의 가정값을 사용합니다.`;}
  if(manifest.schema_version!==1||!manifest.captures?.length)throw Error('지원하지 않는 결과 형식입니다.');
  try{pano=new Panorama($('panorama'),()=>{},fail);}catch(error){fail(error);}
  try{mesh=new PipelineMesh($('mesh'));}catch(error){fail(error);$('mesh-loading').textContent=error.message;}
  for(const c of manifest.captures){const b=element('button',c.id.toUpperCase(),$('captures'));b.dataset.id=c.id;b.onclick=()=>selectCapture(c).catch(fail);}
  if(manifest.changes?.viewer_supported){$('changes-link').hidden=false;$('changes-link').href=manifest.changes.url;}
  $('previous').onclick=()=>selectView(index-1);$('next').onclick=()=>selectView(index+1);
  $('pano-reset').onclick=()=>pano?.setView({yaw:0,pitch:0,fov:80});$('mesh-reset').onclick=()=>mesh?.reset();
  $('instance-select').onchange=()=>{const object=capture.objects.find(o=>String(o.id)===$('instance-select').value);if(object)mesh?.focusObject(object);else mesh?.reset();};
  $('focus').onclick=()=>mesh?.focus(views[index].position);$('ceiling').onchange=()=>{if(mesh){mesh.cut=$('ceiling').checked;mesh.draw();}};
  document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{mode=b.dataset.mode;document.querySelectorAll('[data-mode]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));loadMesh();});
  await selectCapture(manifest.captures[0]);
}
start().catch(fail);

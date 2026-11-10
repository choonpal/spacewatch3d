import {PipelineMesh} from './pipeline-mesh.js';
import {patchColors,boxSegments,validateComparison} from './pipeline-change-model.js';
const $=id=>document.getElementById(id), names=['before','after'], meshes=[];
const number=n=>Number(n).toLocaleString('ko-KR'), fixed=(n,d=2)=>Number.isFinite(n)?n.toFixed(d):'—';
const statusName=s=>s==='candidate'?'변화 후보':'판단 보류';
const reasons={alignment_underconstrained:'정합 신뢰 기준을 충족하지 못해 표면 차이를 변화로 확정할 수 없습니다.',coherent_surface_difference:'일관된 표면 차이와 관측 근거가 있어 변화 후보로 남았습니다.',insufficient_surface_observation:'해당 표면을 독립적인 시점에서 관측한 근거가 부족합니다.',occlusion_or_incomplete_coverage:'가림 또는 촬영 범위 부족으로 반대 메시의 같은 자리를 충분히 확인할 수 없습니다.'};
let data,selected=null;
function element(tag,text,parent){const e=document.createElement(tag);if(text!=null)e.textContent=text;if(parent)parent.append(e);return e;}
function fail(e){$('error').textContent=e.message||String(e);$('error').hidden=false;}
async function json(url){const r=await fetch(url);if(!r.ok)throw Error('결과 파일을 불러오지 못했습니다: '+r.status);return r.json();}
function metric(label,value){const box=element('div',null,$('metrics'));box.className='metric';element('span',label,box);element('strong',value,box);}
function box(epoch){
  const svg=$(names[epoch]+'-box'),mesh=meshes[epoch];svg.replaceChildren();
  if(!selected||!$('boxes').checked)return;
  const w=mesh.canvas.clientWidth,h=mesh.canvas.clientHeight;svg.setAttribute('viewBox',`0 0 ${w} ${h}`);
  for(const [a,b] of boxSegments(selected.bounds,mesh,w,h)){
    const line=document.createElementNS('http://www.w3.org/2000/svg','line');
    for(const [key,value] of Object.entries({x1:a[0],y1:a[1],x2:b[0],y2:b[1]}))line.setAttribute(key,value);svg.append(line);
  }
}
function sync(source){for(const mesh of meshes)if(mesh!==source){mesh.center=[...source.center];mesh.radius=source.radius;mesh.yaw=source.yaw;mesh.pitch=source.pitch;mesh.draw();}}
function color(){meshes.forEach((mesh,epoch)=>mesh.setColors(patchColors(mesh.data,data.events,epoch,selected,$('display').value)));}
function focus(){if(!selected)return;const b=selected.bounds;meshes[0].focusObject({bounds:b,center:b[0].map((x,i)=>(x+b[1][i])/2)});sync(meshes[0]);}
function evidence(){
  $('evidence-stats').replaceChildren();$('view-evidence').replaceChildren();
  $('selection-title').textContent=selected?selected.id:'선택된 구역 없음';$('decision').textContent=selected?statusName(selected.status):'';
  $('reason').textContent=selected?(reasons[selected.reason]||selected.reason):'현재 필터에 해당하는 구역이 없습니다.';
  $('selection-note').textContent=selected?'강조색은 이 구역이 검출된 쪽의 실제 면입니다. 반대편 점선 상자는 같은 좌표의 참고 범위이며 대응 객체·변화 판정을 뜻하지 않습니다.':'';
  $('focus-patch').disabled=!selected;
  names.forEach((name,i)=>$(name+'-note').textContent=selected?(i===selected.epoch?'선택 구역이 검출된 메시 · 원래 면 번호를 강조합니다.':'선택 구역과 같은 좌표 범위 · 대응 객체 판정 없음'):'탐지에 사용한 전체 메시 · 동일 좌표와 배율');
  if(!selected)return;
  const audit=data.audit.find(a=>a.epoch===selected.epoch);
  const own=selected.independent_own_views||[],free=selected.independent_free_views||[];
  for(const [label,value] of [['표면 거리 중앙값',fixed(selected.distance_p50_m)+' m'],['표면 거리 90% 값',fixed(selected.distance_p90_m)+' m'],['탐지 거리 기준',fixed(audit?.threshold_u,3)+' m'],['구역 표면적',fixed(selected.area_m2)+' m²'],['메시 면 수',number(selected.face_ids.length)],['독립 관측 / 비점유 근거',own.join(', ')+' / '+free.join(', ')]]){
    const card=element('div',null,$('evidence-stats'));element('small',label,card);element('strong',value,card);
  }
  for(const [key,label] of [['own_views','검출 쪽 관측 시점'],['opposite_views','반대쪽 같은 위치 관측 시점']]){
    const table=element('table',null,$('view-evidence'));element('caption',label,table);const head=element('tr',null,element('thead',null,table));
    for(const text of ['프레임','시간(s)','표면 지지','비점유','가림'])element('th',text,head);
    const body=element('tbody',null,table);for(const row of selected[key]||[]){const tr=element('tr',null,body);for(const value of [row.frame,fixed(row.timestamp_s,1),...['visible_fraction','free_fraction','occluded_fraction'].map(k=>fixed(row[k]*100,0)+'%')])element('td',value,tr);}
  }
}
function choose(event,zoom=true){selected=event;renderList();color();evidence();if(zoom)focus();else meshes.forEach(m=>m.draw());}
function filtered(){const q=$('search').value.toLowerCase().trim(),filter=$('filter').value;return data.events.filter(e=>(filter==='all'||filter===e.status)&&e.id.toLowerCase().includes(q));}
function renderList(){
  const rows=filtered(),root=$('regions');root.replaceChildren();$('list-count').textContent=`${rows.length} / ${data.events.length}구역`;
  if(!rows.length){element('p','해당하는 구역이 없습니다. 변화 없음 판정은 아닙니다.',root).className='empty';return;}
  for(const event of rows){const item=element('div',null,root);item.setAttribute('role','listitem');const button=element('button',null,item);button.className='region';button.setAttribute('aria-pressed',String(event.id===selected?.id));
    const label=element('span',event.id,button);element('small',`${data.maps[event.epoch].id.toUpperCase()} · ${number(event.face_ids.length)}면 · 거리 ${fixed(event.distance_p50_m)} m`,label);element('span',statusName(event.status),button).className='status';button.onclick=()=>choose(event);}
}
function applyFilter(){const rows=filtered();choose(rows.find(e=>e.id===selected?.id)||rows[0]||null,false);}
async function start(){
  const manifest=await json('/api/pipeline');if(!manifest.changes?.comparison)throw Error('이 실행에는 전후 표면 비교 데이터가 없습니다.');
  data=validateComparison(await json(manifest.changes.comparison));
  $('summary').textContent=`${data.baseline_capture.toUpperCase()} / ${data.current_capture.toUpperCase()} · 전체 메시와 표면 차이 구역`;
  $('source-download').href=manifest.changes.download;$('comparison-download').href=manifest.changes.comparison;$('run-id').textContent=data.run_id;
  const accepted=data.alignment.status==='accepted';
  $('alignment-status').textContent=accepted?'정합 기준 충족':data.available?'정합 불확실':'정합 불가';
  $('alignment-note').textContent=!data.available?'좌표 정합을 수행할 근거가 부족해 전후 비교를 표시할 수 없습니다.':accepted?'정합 기준을 충족한 좌표에서 표면 차이를 표시합니다. 변화 후보는 객체의 추가·삭제·이동 확정이 아닙니다.':'정합 신뢰 기준 미충족 · 계산된 정합 위치를 검토용으로 표시합니다. 주황색 구역은 판단 보류이며, 후보 0건은 변화가 없다는 뜻이 아닙니다.';
  $('alignment-note').classList.toggle('accepted',accepted);
  metric('변화 후보',data.available?number(data.counts.candidates)+'곳':'미산정');metric('판단 보류',data.available?number(data.counts.deferred)+'곳':'미산정');metric('정합 오차 기준',fixed(data.alignment.registration_error_m,3)+' m');metric('전체 메시 보존',data.available?number(data.maps.reduce((n,m)=>n+m.faces,0))+'면':'비교 불가');
  $('scale-note').textContent=data.metric_scale_verified?'입력의 미터 스케일 확인됨.':'거리(m)·면적(m²)은 재구성의 추정 스케일입니다. 실제 미터 스케일은 검증되지 않았습니다.';
  if(!data.available){document.body.classList.add('unavailable');return;}
  for(let i=0;i<2;i++){
    const map=data.maps[i],name=names[i];$(name+'-title').textContent=(i?'비교 · ':'기준 · ')+map.id.toUpperCase();$(name+'-count').textContent=number(map.faces)+'면';
    const mesh=new PipelineMesh($(name+'-mesh'));mesh.cut=false;mesh.capture={...map,bounds:data.bounds,floor_height:data.floor_height};meshes.push(mesh);
    mesh.afterDraw=()=>box(i);
    for(const event of ['pointermove','wheel'])mesh.canvas.addEventListener(event,()=>sync(mesh));
  }
  await Promise.all(meshes.map((mesh,i)=>mesh.load(mesh.capture,'rgb').then(()=>{$(names[i]+'-loading').hidden=true;mesh.reset();})));
  sync(meshes[0]);document.querySelectorAll('.comparison-toolbar :disabled,.list-controls :disabled').forEach(e=>e.disabled=false);
  $('display').onchange=color;$('filter').onchange=applyFilter;$('search').oninput=applyFilter;
  $('roof').onchange=()=>meshes.forEach(m=>{m.cut=$('roof').checked;m.draw();});$('boxes').onchange=()=>meshes.forEach(m=>m.draw());
  $('overview').onclick=()=>{meshes[0].reset();sync(meshes[0]);};$('focus-patch').onclick=focus;
  $('top').onclick=()=>{meshes[0].pitch=Math.PI/2;meshes[0].yaw=0;meshes[0].draw();sync(meshes[0]);};
  choose(data.events[0]||null);
}
start().catch(fail);

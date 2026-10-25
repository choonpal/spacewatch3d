/* Cached model masks and frozen baseline decisions only. No browser inference. */
'use strict';
const $ = s => document.querySelector(s);
const n = x => x.toLocaleString('ko-KR');
const pct = x => `${(x * 100).toFixed(2)}%`;
const escapeHTML = x => String(x).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const names = {candidate:'기존 후보',deferred:'기존 보류',group_context:'후보 주변 문맥',suppressed:'기존 후보 억제',unaudited:'기존 판정 연결 없음',no_mesh:'메쉬 hit 없음',person_excluded:'사람에 가려 비교 제외'};
let data, selected = [];
const meta = v => `${v.zone} · Dorm${v.epoch + 2} · f${v.frame} · ${v.yaw}°`;
const sourceImage = (v, suffix) => `images/${v.key}-${suffix}.png`;
const filterPerson = () => $('#exclude-person').checked;
const filtering = v => filterPerson() && v.person.available;
const baseline = v => filtering(v) ? v.baseline_filtered : v.baseline;
const currentStats = v => filtering(v) ? v.stats_filtered[$('#mode').value] : v.stats[$('#mode').value];
const visibleGroups = v => filtering(v) ? v.visible_groups_filtered : v.visible_groups;
function modelStat(v, model) { return currentStats(v)[model]; }
function matches(v) {
  if ($('#zone').value && v.zone !== $('#zone').value) return false;
  const s = modelStat(v,'simsac'), d = modelStat(v,'viewdelta');
  switch ($('#filter').value) {
    case 'candidate': return baseline(v).candidate > 0;
    case 'deferred': return s.by_baseline.deferred + d.by_baseline.deferred > 0;
    case 'outside': return s.pixels + d.pixels > s.by_baseline.candidate + d.by_baseline.candidate;
    case 'person': return ['simsac','viewdelta'].some(m => (v.stats[$('#mode').value][m].on_person_pixels || 0) > 0);
    case 'disagreement': return currentStats(v).model_disagreement > 0;
    default: return true;
  }
}
function updateMetrics() {
  const inZone = data.views.filter(v => !$('#zone').value || v.zone === $('#zone').value);
  const candidateGroups = data.groups.filter(g => data.zones.includes(g.zone) && (!$('#zone').value || g.zone === $('#zone').value));
  const basePixels = inZone.reduce((s,v) => s+baseline(v).candidate,0);
  const cards = [`<div class="metric"><b class="old">기존 정합오차 · 관측검증</b><div class="value">${candidateGroups.length} <span style="font-size:14px;font-weight:400">후보 묶음</span></div><small>선택 구역 · 보이는 트랙 표면 ${n(basePixels)} px</small></div>`];
  for (const [model,title,style] of [['simsac','SimSaC','sim'],['viewdelta','ViewDelta','vd']]) {
    const pixels = inZone.reduce((s,v) => s+modelStat(v,model).pixels,0);
    const removed=inZone.reduce((s,v)=>s+(modelStat(v,model).removed_person_pixels||0),0);
    cards.push(`<div class="metric"><b class="${style}">${title}${filterPerson()?' · 사람 제외':''}</b><div class="value">${pct(pixels/Math.max(1,inZone.length*224*224))}<span style="font-size:14px;font-weight:400"> 후보 영역</span></div><small>${n(pixels)} px · ${inZone.length}개 공통 방향 뷰${filterPerson()?`<br>사람 겹침 ${n(removed)} px 제외`:''}</small></div>`);
  }
  $('#metrics').innerHTML = cards.join('');
}
function refreshList() {
  const previous = $('#view').value;
  selected = data.views.filter(matches).sort((a,b) => baseline(b).candidate-baseline(a).candidate ||
    currentStats(b).model_disagreement-currentStats(a).model_disagreement);
  $('#view').innerHTML = selected.map(v => `<option value="${v.key}">${meta(v)}${baseline(v).candidate?' · 기존 후보 표면':''}</option>`).join('');
  if (selected.some(v => v.key === previous)) $('#view').value = previous;
  $('#empty').hidden = selected.length > 0;
  $('#case').hidden = selected.length === 0;
  updateMetrics();
  render();
}
function tile(title, style, image, count, caption) {
  return `<article class="tile"><header><span class="${style}">${title}</span><span>${count}</span></header><a href="${image}" target="_blank" rel="noopener"><img src="${image}" alt="${escapeHTML(title)}"></a><p>${caption}</p></article>`;
}
function render() {
  const index = selected.findIndex(v => v.key === $('#view').value), v = selected[index];
  $('#prev').disabled = index <= 0;
  $('#next').disabled = index < 0 || index >= selected.length-1;
  if (!v) return;
  const mode = $('#mode').value, s = modelStat(v,'simsac'), d = modelStat(v,'viewdelta'), b=baseline(v), groups=visibleGroups(v);
  const suffix=filtering(v)?'-filtered':'';
  $('#case-title').innerHTML = `${meta(v)}${groups.map(g=>`<span class="badge">${g.id} · ${n(g.pixels)} px</span>`).join('')}`;
  $('#case-count').textContent = `${index+1} / ${selected.length}`;
  const oldCaption = b.candidate ? `후보 트랙 표면 ${n(b.candidate)} px · ${groups.map(g=>g.id).join(', ')}` : '이 시점에 보이는 기존 후보 트랙 표면 없음';
  const caption = (m) => {
    const x=modelStat(v,m);
    return `기존 후보와 겹침 ${n(x.by_baseline.candidate)} px · 보류와 겹침 ${n(x.by_baseline.deferred)} px`;
  };
  $('#tiles').innerHTML = tile('기존 정합오차 · 관측검증','old',sourceImage(v,($('#review').checked?'baseline-review':'baseline')+suffix),n(b.candidate)+' px',oldCaption) +
    tile('SimSaC'+(filtering(v)?' · 사람 제외':''),'sim',sourceImage(v,mode+'-simsac'+suffix),pct(s.fraction),caption('simsac')) +
    tile('ViewDelta'+(filtering(v)?' · 사람 제외':''),'vd',sourceImage(v,mode+'-viewdelta'+suffix),pct(d.fraction),caption('viewdelta'));
  const lines = [];
  if (b.candidate) lines.push(`기존 후보 트랙 표면이 ${n(b.candidate)} px 보입니다. 주변 의자 메쉬를 포함한 그룹 전체 면적은 아닙니다.`);
  else lines.push('기존 후보 표면이 이 화면에 투영되지 않았습니다. 보류·미판정 영역을 변화 없음으로 처리하지 않았습니다.');
  if (s.by_baseline.deferred + d.by_baseline.deferred) lines.push(`기존에 보류된 면 중 SimSaC는 ${n(s.by_baseline.deferred)} px, ViewDelta는 ${n(d.by_baseline.deferred)} px를 선택했습니다.`);
  if (filtering(v)) {
    lines.push(`사람과 겹친 후보를 제외했습니다: SimSaC ${n(s.removed_person_pixels)} px, ViewDelta ${n(d.removed_person_pixels)} px. 남은 후보의 사람 마스크 겹침은 두 모델 모두 0 px입니다.`);
    const hidden=v.baseline.candidate-b.candidate;
    if(hidden)lines.push(`기존 후보 표면의 투영 ${n(hidden)} px도 RGB 사람 뒤에 가려져 비교에서 제외했습니다. 기존 객체 판정은 유지합니다.`);
  } else if (filterPerson()) lines.push('사람 마스크가 없어 이 프레임은 필터 미적용 원본을 표시합니다.');
  else if(v.person.available) lines.push(`필터 전 원본입니다. 사람 겹침: SimSaC ${n(s.on_person_pixels)} px, ViewDelta ${n(d.on_person_pixels)} px.`);
  else lines.push('이 프레임의 사람 마스크를 확인할 수 없습니다.');
  $('#observations').innerHTML = lines.map(t=>`<p>${t}</p>`).join('') + `<p class="note">기존 구역 정합 허용오차: ${v.map_error_u.toFixed(4)} 맵 단위 · 모델 후보 임계값: 평균 확률 &gt; 0.5</p>`;
  const coverage = x => b.candidate ? `${n(x.by_baseline.candidate)} / ${n(b.candidate)} px (${pct(x.by_baseline.candidate/b.candidate)})` : '— · 이 화면에 기존 후보 표면 없음';
  $('#overlap').innerHTML = `<p><b class="sim">SimSaC</b><br>보이는 기존 후보 표면 중 ${coverage(s)}</p><p><b class="vd">ViewDelta</b><br>보이는 기존 후보 표면 중 ${coverage(d)}</p><p>두 모델이 함께 선택한 영역 ${n(currentStats(v).model_intersection)} px<br>서로 다르게 선택한 영역 ${n(currentStats(v).model_disagreement)} px</p>`;
  $('#refs').innerHTML = `<div class="ref"><a href="${sourceImage(v,'rgb')}" target="_blank" rel="noopener"><img src="${sourceImage(v,'rgb')}" alt="관측 RGB ${meta(v)}"></a><small><b>현재 관측 RGB</b><br>${meta(v)}</small></div>` + v.refs.map(key=>data.views.find(x=>x.key===key)).map(ref=>`<div class="ref"><a href="${sourceImage(ref,'rgb')}" target="_blank" rel="noopener"><img src="${sourceImage(ref,'rgb')}" alt="비교 상대 ${meta(ref)}"></a><small>${meta(ref)}</small></div>`).join('');
  const oldImage = `../model-comparison-20261025/images/${v.key}`;
  $('#erps').innerHTML = [[oldImage+'-erp-rgb.jpg','현재 ERP'],[oldImage+'-erp-reference.jpg','상대 ERP 첫 예시'],...['simsac','viewdelta'].map(m=>[filtering(v)?`images/${v.key}-erp-full-${m}-filtered.jpg`:oldImage+`-erp-full-${m}.jpg`,`${m==='simsac'?'SimSaC':'ViewDelta'} · ERP 입력${filtering(v)?' · 사람 제외':''}`])].map(([path,title])=>`<div class="ref" style="width:240px"><a href="${path}" target="_blank" rel="noopener"><img style="width:240px" src="${path}" alt="${title}"></a><small>${title}</small></div>`).join('');
  $('#person-note').textContent=v.person.available ? `같은 프레임의 기존 사람 분할과 10% 여유를 둔 검출 박스의 합집합 ${n(v.person.pixels)} px를 제외합니다. 분할 ${n(v.person.segmentation_pixels)} px에 외곽·옷을 위한 박스 영역 ${n(v.person.box_margin_pixels)} px가 더해집니다. 박스 안의 주변 표면도 제외되며 모델 추론 후 적용합니다.` : '이 프레임의 기존 사람 마스크를 확인할 수 없습니다.';
  $('#person-evidence').innerHTML=v.person.available ? `<div class="ref"><a href="${sourceImage(v,'person')}" target="_blank" rel="noopener"><img src="${sourceImage(v,'person')}" alt="기존 사람 마스크"></a><small>기존 사람 마스크 · 참고용</small></div>` : '';
  const order = {candidate:0,deferred:1,group_context:2,suppressed:3};
  const candidateRows=groups.map(visible=>{
    const g=data.groups.find(x=>x.id===visible.id);
    return `<tr><td>${g.id}<br><small>${g.entries.join(' / ')}</small></td><td>${escapeHTML(g.class_name)}</td><td class="old">기존 후보<br>${n(visible.pixels)} px</td><td>${escapeHTML(g.reason)}</td></tr>`;
  }).join('');
  const recordPixels=r=>filtering(v)?r.visible_pixels_filtered:r.visible_pixels;
  $('#records').innerHTML = candidateRows + v.records.filter(r=>recordPixels(r)>0).sort((a,b)=>order[a.decision]-order[b.decision]||recordPixels(b)-recordPixels(a)).map(r=>`<tr><td>${r.id}<br><small>${r.ids.join(' / ')}</small></td><td>${escapeHTML(r.class_name)}</td><td>${names[r.decision]}<br>${n(recordPixels(r))} px</td><td>${escapeHTML(r.reason)}</td></tr>`).join('') || '<tr><td colspan="4">연결된 기존 판정 없음</td></tr>';
  $('#breakdown').innerHTML = Object.entries(names).map(([key,title])=>`<tr><td>${title}</td><td class="num">${n(s.by_baseline[key]||0)} px</td><td class="num">${n(d.by_baseline[key]||0)} px</td></tr>`).join('');
}
async function init() {
  const response = await fetch('comparison.json');
  if (!response.ok) throw new Error(`비교 자료를 읽지 못했습니다: HTTP ${response.status}`);
  data = await response.json();
  const relevant=data.groups.filter(g=>data.zones.includes(g.zone));
  $('#groups').innerHTML = `<p class="note">기존 전체 13구역에는 ${data.groups.length}개 후보 묶음이 있습니다. 현재 두 모델의 R01 · R07 · R08 범위에는 ${relevant.length}개가 포함됩니다.</p>` + relevant.map(g=>`<div class="group"><b>${escapeHTML(g.id+' · '+g.title)}</b><p>${escapeHTML(g.reason)}</p>${Object.entries(g.evidence).map(([key,e])=>`<p>실제 후보 트랙 ${key} · Dorm${e.epoch+2} · 원본 ${e.faces}면<br>기존 자체 관측 프레임: ${e.own_frames.join(', ')} · 반대 촬영 비점유 근거: ${e.vacancy_frames.join(', ')}</p>`).join('')}<p>이동·추가의 유형과 물리적 동일성은 미확정입니다. 기존 판단에 쓴 일부 프레임은 현재 파일럿 모델 입력 밖에 있습니다.</p></div>`).join('');
  for (const id of ['mode','zone','filter','exclude-person']) $('#'+id).addEventListener('change',refreshList);
  $('#view').addEventListener('change',render);
  $('#review').addEventListener('change',render);
  for (const [id,delta] of [['prev',-1],['next',1]]) $('#'+id).addEventListener('click',()=>{
    const i=selected.findIndex(v=>v.key===$('#view').value)+delta;
    if(selected[i]) {$('#view').value=selected[i].key;render();}
  });
  refreshList();
}
init().catch(error=>{$('#error').textContent=error.stack;});

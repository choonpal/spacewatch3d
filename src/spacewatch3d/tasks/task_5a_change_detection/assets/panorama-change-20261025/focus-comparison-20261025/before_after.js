'use strict';
const $=s=>document.querySelector(s), n=x=>x.toLocaleString('ko-KR'), pct=x=>`${(x*100).toFixed(2)}%`;
const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const decisions={candidate:'기존 후보',deferred:'보류',group_context:'주변 문맥',suppressed:'후보 억제'};
const sides=['before','after'], phase={before:'전',after:'후'};
let data,selected=[];
const humanOn=()=>$('#exclude-person').checked, mode=()=>$('#mode').value;
const view=(pair,side)=>data.views[pair[side]];
const filtered=v=>humanOn()&&v.person.available;
const baseline=v=>filtered(v)?v.baseline_filtered:v.baseline;
const stat=(pair,side)=>{const p=pair.modes[mode()][side];return humanOn()&&p.filtered?p.filtered:p.raw;};
const title=v=>`Dorm${v.epoch+2} · f${v.frame} · ${v.yaw}°`;
const image=(v,suffix)=>`images/${v.key}-${suffix}.png`;
function matches(pair){
 if($('#zone').value&&pair.zone!==$('#zone').value)return false;
 const stats=sides.map(s=>stat(pair,s)), bs=sides.map(s=>baseline(view(pair,s)));
 switch($('#filter').value){
  case 'person':return sides.some(s=>(pair.modes[mode()][s].raw.on_person_pixels||0)>0);
  case 'candidate':return bs.some(b=>b.candidate>0);
  case 'deferred':return stats.some(s=>s.by_baseline.deferred>0);
  case 'outside':return stats.some(s=>s.pixels>s.by_baseline.candidate);
  case 'disagreement':return stats.some(s=>s.disagreement_with_baseline_pixels>0);
  default:return true;
 }
}
function refresh(){
 const previous=$('#pair').value;
 selected=data.pairs.filter(matches).sort((a,b)=>{
  const base=p=>sides.reduce((s,k)=>s+baseline(view(p,k)).candidate,0);
  return base(b)-base(a)||sides.reduce((s,k)=>s+stat(b,k).pixels-stat(a,k).pixels,0);
 });
 $('#pair').innerHTML=selected.map(p=>`<option value="${p.id}">${p.zone} · 전 f${view(p,'before').frame} ↔ 후 f${view(p,'after').frame} · ${view(p,'before').yaw}° / ${view(p,'after').yaw}°</option>`).join('');
 if(selected.some(p=>p.id===previous))$('#pair').value=previous;
 $('#empty').hidden=selected.length>0;$('#case').hidden=selected.length===0;render();
}
function tile(side,v,path,count,caption){return `<article class="tile"><header><strong>${phase[side]} · ${title(v)}</strong><small>${count}</small></header><a href="${path}" target="_blank" rel="noopener"><img src="${path}" alt="${phase[side]} ${esc(title(v))} ${esc(caption)}"></a><p>${caption}</p></article>`;}
function render(){
 const index=selected.findIndex(p=>p.id===$('#pair').value),pair=selected[index];
 $('#prev').disabled=index<=0;$('#next').disabled=index<0||index>=selected.length-1;if(!pair)return;
 $('#case-title').textContent=`${pair.zone} · 전후 객체 변화 후보`;
 $('#case-count').textContent=`${index+1} / ${selected.length}쌍`;
 $('#phase-headings').innerHTML=sides.map(side=>`<div class="phase"><b>${phase[side]}</b>${title(view(pair,side))}</div>`).join('');
 $('#baseline-pair').innerHTML=sides.map(side=>{
  const v=view(pair,side),b=baseline(v),suffix=(filtered(v)?'-filtered':'');
  const groups=filtered(v)?v.visible_groups_filtered:v.visible_groups;
  return tile(side,v,image(v,($('#review').checked?'baseline-review':'baseline')+suffix),`${n(b.candidate)} px`,b.candidate?`기존 후보 표면 ${n(b.candidate)} px · ${groups.map(g=>g.id).join(', ')}`:'이 관측에 보이는 기존 후보 표면 없음');
 }).join('');
 $('#model-heading-note').textContent=humanOn()?'선택한 전후 쌍 · 사람 제외 적용':'선택한 전후 쌍 · 필터 전 원본';
 $('#simsac-pair').innerHTML=sides.map(side=>{
  const v=view(pair,side),s=stat(pair,side),state=filtered(v)?'filtered':'raw';
  const path=`pairs/${pair.id}-${mode()}-${side}-${state}.png`;
  return tile(side,v,path,`${pct(s.fraction)} · ${n(s.pixels)} px`,`기존 후보와 겹침 ${n(s.by_baseline.candidate)} px · 보류와 겹침 ${n(s.by_baseline.deferred)} px`);
 }).join('');
 $('#observations').innerHTML=sides.map(side=>{
  const v=view(pair,side),s=stat(pair,side),b=baseline(v),raw=pair.modes[mode()][side].raw;
  const removed=filtered(v)?s.removed_person_pixels:0;
  const filterText=filtered(v)?`사람 검출 영역의 후보 ${n(removed)} px 제외`:(humanOn()?'사람 마스크 없음 · 필터 미적용':'필터 전 원본');
  const coverage=b.candidate?`${n(s.by_baseline.candidate)} / ${n(b.candidate)} px (${pct(s.by_baseline.candidate/b.candidate)})`:'보이는 기존 후보 표면 없음';
  return `<div><b>${phase[side]} · ${title(v)}</b><p>기존 후보 ${n(b.candidate)} px · SimSaC 후보 ${n(s.pixels)} px<br>기존 후보와 겹침: ${coverage}</p><p>${filterText}<br>SimSaC 원본 ${n(raw.pixels)} px → 표시 ${n(s.pixels)} px</p></div>`;
 }).join('');
 $('#raw-pair').innerHTML=sides.map(s=>tile(s,view(pair,s),image(view(pair,s),'rgb'),'원본 RGB','선택한 전후 비교 사진')).join('');
 $('#erp-pair').innerHTML=sides.map(s=>{const v=view(pair,s);return tile(s,v,`../model-comparison-20261025/images/${v.key}-erp-rgb.jpg`,'전체 ERP','같은 촬영 프레임의 파노라마 원본');}).join('');
 $('#person-pair').innerHTML=sides.map(s=>{const v=view(pair,s);return tile(s,v,image(v,'person'),`${n(v.person.pixels)} px`,v.person.available?'사람 제외 영역 · 분할 + 여유를 둔 검출 박스':'사람 마스크 없음');}).join('');
 $('#records').innerHTML=sides.map(side=>{
  const v=view(pair,side),groups=filtered(v)?v.visible_groups_filtered:v.visible_groups;
  const groupRows=groups.map(g=>{const source=data.groups.find(x=>x.id===g.id);return `<tr><td>${phase[side]}</td><td>${g.id} / ${source.entries.join(', ')}</td><td class="old">기존 후보 · ${n(g.pixels)} px</td><td>${esc(source.reason)}</td></tr>`;}).join('');
  const pixels=r=>filtered(v)?r.visible_pixels_filtered:r.visible_pixels;
  return groupRows+v.records.filter(r=>pixels(r)>0).sort((a,b)=>pixels(b)-pixels(a)).map(r=>`<tr><td>${phase[side]}</td><td>${r.id} / ${r.ids.join(', ')}</td><td>${decisions[r.decision]} · ${n(pixels(r))} px</td><td>${esc(r.reason)}</td></tr>`).join('');
 }).join('')||'<tr><td colspan="4">이 관측에 연결된 기존 판정 없음</td></tr>';
}
async function init(){
 const response=await fetch('before-after.json');if(!response.ok)throw new Error(`전후 비교 자료 HTTP ${response.status}`);data=await response.json();
 for(const id of ['zone','filter','mode','exclude-person'])$('#'+id).addEventListener('change',refresh);
 $('#pair').addEventListener('change',render);$('#review').addEventListener('change',render);
 for(const [id,delta] of [['prev',-1],['next',1]])$('#'+id).addEventListener('click',()=>{const i=selected.findIndex(p=>p.id===$('#pair').value)+delta;if(selected[i]){$('#pair').value=selected[i].id;render();}});
 refresh();
}
init().catch(error=>{$('#error').textContent=error.stack;});

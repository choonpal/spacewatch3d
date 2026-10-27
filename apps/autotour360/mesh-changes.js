const $ = id => document.getElementById(id);
const labels = {queued:'분석 차례를 기다리고 있습니다',validating:'입력과 저장된 결과를 확인하고 있습니다',running:'변화탐지를 실행하고 있습니다',complete:'변화 후보 비교가 완료됐습니다',failed:'분석을 완료하지 못했습니다',interrupted:'분석이 중단됐습니다'};
let timer, busy = false, configured = false, connected = false, currentJob;
let captures = [];
function describeCaptures() {
  const selected=captures.filter(c=>[$('before-capture').value,$('after-capture').value].includes(c.id));
  $('capture-description').textContent=selected.map(c=>`${c.label}: 메쉬 ${c.mesh_files}개 파일 · 객체 후보 ${c.candidate_regions??'확인 중'}개 · ${c.segmentation}`).join(' / ');
}
function controls() {
  $('run').disabled = busy || !configured || !connected; $('bundle').disabled = busy || !connected;
  for(const id of ['before-capture','after-capture'])$(id).disabled=busy||!configured||!connected;
  $('run-captures').disabled=busy||!configured||!connected||!$('before-capture').value||!$('after-capture').value;
}
async function checkConnection() {
  const config=await api('/api/mesh-change-jobs/config');configured=config.configured;connected=config.connected;
  $('connection').textContent=connected?'분석 서버 연결됨':configured?'서버 SSH 재로그인 필요':'분석 환경 설정 필요';controls();
}
async function api(url) { const response = await fetch(url); const data = await response.json(); if (!response.ok) throw Error(data.error || '요청에 실패했습니다'); return data; }
function show(job) {
  currentJob = job.id;
  let resume = $('resume');
  if (!resume) { resume = document.createElement('button'); resume.id = 'resume'; resume.textContent = '진행 중인 GPU 작업에 다시 연결'; $('error').after(resume); }
  resume.hidden = !job.can_resume;
  resume.onclick = async () => {
    resume.disabled = true;
    try { const response = await fetch('/api/mesh-change-jobs/'+currentJob+'/retry', {method:'POST'});
      const value = await response.json(); if (!response.ok) throw Error(value.error); show(value); await poll(currentJob);
    } catch(error) { $('error').textContent = error.message; } finally { resume.disabled = false; }
  };
  $('state').textContent = job.stage==='waiting_for_connection' ? '서버 연결 대기 · 연결되면 기존 계산에 자동으로 다시 연결합니다' : job.state==='validating'&&job.stage==='preparing_inputs' ? '선택한 영상·메쉬·점군 입력을 준비하고 있습니다' : labels[job.state] || job.state;
  if(job.source_selection){for(const phase of ['before','after'])$(phase+'-capture').value=job.source_selection[phase];describeCaptures();}
  $('error').textContent = job.message || '';
  $('cache-state').textContent = job.cached ? '동일 입력 · 저장된 분석 재사용' : job.state === 'complete' ? '이 입력으로 새로 연산한 결과' : '';
  const stage = job.stage==='preparing_inputs' ? 'input_validation' : ['transferring','starting'].includes(job.stage) ? 'registration' : job.stage === 'receiving_results' ? 'viewer' : job.stage;
  const items = [...document.querySelectorAll('#stages li')], index = items.findIndex(el => el.dataset.stage === stage);
  items.forEach((el,i) => { el.classList.toggle('done', job.state === 'complete' || i < index); el.classList.toggle('active', i === index && job.state !== 'complete'); });
  busy = !['complete','failed','interrupted'].includes(job.state);
  controls();
  if (job.result_url) {
    if ($('result').getAttribute('src') !== job.result_url) $('result').src = job.result_url;
    $('result').hidden = false; $('empty').hidden = true;
    $('result-link').href = job.result_url; $('result-link').hidden = false;
  }
}
async function poll(id) {
  clearTimeout(timer);
  try { const job = await api('/api/mesh-change-jobs/'+id); show(job); if (busy) timer = setTimeout(() => poll(id), 2000); }
  catch(error) { $('error').textContent = error.message; timer = setTimeout(() => poll(id), 5000); }
}
function upload(file) { return new Promise((resolve,reject) => {
  const xhr = new XMLHttpRequest(); xhr.open('POST','/api/mesh-change-jobs'); xhr.setRequestHeader('Content-Type','application/zip');
  xhr.upload.onprogress = event => { if(event.lengthComputable) $('upload-progress').textContent = `입력 전송 ${Math.round(event.loaded/event.total*100)}%`; };
  xhr.onload = () => { try { const data = JSON.parse(xhr.responseText); if(xhr.status>=400) throw Error(data.error); resolve(data); } catch(error) { reject(error); } };
  xhr.onerror = () => reject(Error('입력을 전송하지 못했습니다. 서버 연결을 확인해 주세요.')); xhr.send(file);
}); }
$('upload-form').onsubmit = async event => {
  event.preventDefault(); if(busy) return; const file = $('bundle').files[0]; if(!file) return;
  begin();
  $('state').textContent = '입력을 전송하고 있습니다'; $('cache-state').textContent = '';
  try { const job = await upload(file); history.replaceState(null,'','?job='+job.id); $('upload-progress').textContent = '입력 전송 완료'; show(job); await poll(job.id); }
  catch(error) { busy = false; controls(); $('error').textContent = error.message; }
};
$('bundle').onchange=()=>{if($('bundle').files.length&&configured&&!busy)$('upload-form').requestSubmit();};
function begin() {
  clearTimeout(timer); busy=true; controls(); $('error').textContent=''; $('cache-state').textContent='';
  $('result').hidden=true; $('result').removeAttribute('src'); $('result-link').hidden=true; $('empty').hidden=false;
  if(innerWidth<900)$('stages').scrollIntoView({behavior:'smooth',block:'start'});
}
$('capture-form').onsubmit=async event=>{
  event.preventDefault(); if(busy||!configured)return;
  const selection={before:$('before-capture').value,after:$('after-capture').value};
  if(!selection.before||!selection.after)return;
  if(selection.before===selection.after){$('error').textContent='서로 다른 전후 촬영을 선택해 주세요.';return;}
  begin(); $('state').textContent='선택한 촬영 입력을 연결하고 있습니다';
  try {
    const response=await fetch('/api/mesh-change-jobs/from-captures',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(selection)});
    const job=await response.json();if(!response.ok)throw Error(job.error);
    history.replaceState(null,'','?job='+job.id);show(job);await poll(job.id);
  }catch(error){busy=false;controls();$('error').textContent=error.message;}
};
for(const phase of ['before','after'])$(phase+'-capture').onchange=()=>{
  controls();describeCaptures();
  if($('before-capture').value&&$('after-capture').value&&!busy)$('capture-form').requestSubmit();
};
try { await checkConnection();
  setInterval(()=>checkConnection().catch(()=>{$('connection').textContent='뷰어 서버 연결 확인 필요';connected=false;controls();}),15000);
  try {
    captures=(await api('/api/mesh-change-jobs/captures')).captures;
    if(captures.length){$('capture-form').hidden=false;for(const c of captures)for(const phase of ['before','after'])$(phase+'-capture').add(new Option(c.label,c.id));}
  }catch(error){$('error').textContent=error.message;}
  controls();
  const id = new URLSearchParams(location.search).get('job'); if(id) await poll(id);
} catch(error) { $('error').textContent = error.message; }

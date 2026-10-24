import {CHANGE_TYPES, CHANGE_SOURCES, CHANGE_FILTERS, regionName, evidenceSummary, contextPhotoCandidates, displayBounds, photoCandidates, photoTitle, validateReference, surfaceAssessment, mergeIndependent, filteredEvents} from './mesh-change-context.js?v=20261015-7';
const $ = id => document.getElementById(id);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const sub = (a, b) => a.map((v, i) => v-b[i]);
const dot = (a, b) => a.reduce((sum, v, i) => sum+v*b[i], 0);
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const unit = a => a.map(v => v/Math.max(Math.hypot(...a), 1e-10));
const camera = {target:[0, 0, 0], radius:10, theta:-1.12, phi:.9};
let data, reference, independent, events=[], epoch=1, selected=null, renderers=[], ready=false, geometries=[], filter='all', source='m3c2_independent', regionIds=[];
const photoLists=[[],[]], photoIndices=[0,0], photoTokens=[0,0], assessmentCache=new Map();
const dataset=typeof location!=='undefined'&&new URLSearchParams(location.search).get('dataset')==='bonn-removal'?'bonn-removal':'rescan';
const endpoints=dataset==='bonn-removal'?['/datasets/bonn-removal/comparison.json','/datasets/bonn-removal/reference.json','/datasets/bonn-removal/changes.json']:['/mesh-comparison/comparison.json','/rescan/changes.json','/m3c2-independent/changes.json'];

function multiply(a, b) {
  const out = new Float32Array(16);
  for (let j=0; j<4; j++) for (let i=0; i<4; i++) for (let k=0; k<4; k++) out[j*4+i]+=a[k*4+i]*b[j*4+k];
  return out;
}
function matrix(aspect) {
  const {radius:r, phi:p, theta:t, target} = camera;
  const eye=target.map((v,i)=>v+[r*Math.cos(p)*Math.cos(t),r*Math.cos(p)*Math.sin(t),r*Math.sin(p)][i]);
  const z=unit(sub(eye,target)), x=unit(cross([0,0,1],z)), y=cross(z,x);
  const view=new Float32Array([x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1]);
  const f=1/Math.tan(.72/2), near=.01, far=120;
  const projection=new Float32Array([f/aspect,0,0,0,0,f,0,0,0,0,(far+near)/(near-far),-1,0,0,2*far*near/(near-far),0]);
  return multiply(projection,view);
}
const vertex=`
attribute vec3 a_position,a_normal,a_rgb;attribute vec4 a_field;attribute float a_candidate;
uniform mat4 u_matrix;uniform float u_mode,u_method,u_candidateId;uniform vec3 u_focusMin,u_focusMax,u_tint;uniform mediump float u_box;varying mediump vec3 v_color;varying highp float v_height;
void main(){gl_Position=u_matrix*vec4(a_position,1.);v_height=a_position.z;
vec3 base=max(a_rgb,vec3(.15));float c=a_field.z;
float inside=step(u_focusMin.x,a_position.x)*step(a_position.x,u_focusMax.x)*step(u_focusMin.y,a_position.y)*step(a_position.y,u_focusMax.y)*step(u_focusMin.z,a_position.z)*step(a_position.z,u_focusMax.z);
if(u_candidateId>.5)inside=1.-step(.5,abs(a_candidate-u_candidateId));
vec3 color=base*.43;
if(inside>.5){color=base*.70;if(c>.5&&c<1.5)color=vec3(.96,.74,.27);if(c>1.5&&c<2.5)color=u_tint;}
if(u_mode>.5&&u_mode<1.5){
float score=clamp(a_field.x/(u_method<.5?3.:.25),0.,1.);
color=mix(vec3(.14,.29,.36),vec3(.97,.77,.29),min(score*2.,1.));
if(score>.5)color=mix(vec3(.97,.77,.29),vec3(1.,.28,.21),(score-.5)*2.);
if(c>2.5)color=vec3(.20,.26,.35);}
if(u_mode>1.5)color=base;
float light=.66+.34*abs(dot(normalize(a_normal+vec3(.0001)),normalize(vec3(.4,-.6,1.))));
v_color=color*light;if(u_box>.5)v_color=u_tint;}`;
const fragment=`precision mediump float;varying mediump vec3 v_color;varying highp float v_height;uniform highp float u_cut;uniform mediump float u_box;void main(){if(u_box<.5&&v_height>u_cut)discard;gl_FragColor=vec4(v_color,1.);}`;

function makeRenderer(canvas, geometries, method) {
  const gl=canvas.getContext('webgl',{antialias:true,alpha:false,preserveDrawingBuffer:true});
  if(!gl)throw new Error('WebGL을 사용할 수 없습니다.');
  function shader(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;}
  const program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,vertex));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,fragment));gl.linkProgram(program);
  if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(program));
  const attrs=Object.fromEntries(['position','normal','rgb','field','candidate'].map(n=>[n,gl.getAttribLocation(program,'a_'+n)]));
  const u=Object.fromEntries(['matrix','mode','method','cut','box','focusMin','focusMax','tint','candidateId'].map(n=>[n,gl.getUniformLocation(program,'u_'+n)]));
  const buffers=geometries.map((bytes,i)=>{
    const meta=data.displays[i];
    if(bytes.byteLength!==meta.count*60||meta.stride!==60||meta.count%3!==0)throw new Error('메쉬 결과 파일 형식이 올바르지 않습니다.');
    const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,bytes,gl.STATIC_DRAW);return buffer;
  });
  const candidateBuffers=regionIds.map(ids=>{const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,ids,gl.STATIC_DRAW);return b;});
  const box=gl.createBuffer();
  function draw(){
    const ratio=Math.min(devicePixelRatio||1,2),w=Math.max(1,Math.round(canvas.clientWidth*ratio)),h=Math.max(1,Math.round(canvas.clientHeight*ratio));
    if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
    gl.viewport(0,0,w,h);gl.enable(gl.DEPTH_TEST);gl.clearColor(.035,.065,.092,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.useProgram(program);
    gl.uniformMatrix4fv(u.matrix,false,matrix(w/h));gl.uniform1f(u.mode,Number($('mode').value));gl.uniform1f(u.method,method);gl.uniform1f(u.box,0);
    gl.uniform1f(u.cut,$('cut').checked?data.bounds.max[2]-.28:1000);
    const focus=selected?displayBounds(selected,epoch,data.display_origin,.10):data.bounds;
    gl.uniform3fv(u.focusMin,focus.min);gl.uniform3fv(u.focusMax,focus.max);
    gl.uniform3fv(u.tint,selected?CHANGE_TYPES[selected.type].color:[.6,.8,.9]);
    gl.uniform1f(u.candidateId,selected?.source==='m3c2_independent'?selected.candidate_id:0);
    gl.bindBuffer(gl.ARRAY_BUFFER,buffers[epoch]);
    for(const a of Object.values(attrs))gl.enableVertexAttribArray(a);
    gl.vertexAttribPointer(attrs.position,3,gl.FLOAT,false,60,0);gl.vertexAttribPointer(attrs.normal,3,gl.FLOAT,false,60,12);gl.vertexAttribPointer(attrs.rgb,3,gl.UNSIGNED_BYTE,true,60,24);gl.vertexAttribPointer(attrs.field,4,gl.FLOAT,false,60,method===0?28:44);
    gl.bindBuffer(gl.ARRAY_BUFFER,candidateBuffers[epoch]);gl.vertexAttribPointer(attrs.candidate,1,gl.UNSIGNED_SHORT,false,2,0);
    gl.drawArrays(gl.TRIANGLES,0,data.displays[epoch].count);
    if(selected){
      const {min,max}=displayBounds(selected,epoch,data.display_origin,.05),corners=[],positions=[];
      for(let i=0;i<8;i++)corners.push([i&1?max[0]:min[0],i&2?max[1]:min[1],i&4?max[2]:min[2]]);
      for(let i=0;i<8;i++)for(const bit of [1,2,4])if(!(i&bit))positions.push(...corners[i],...corners[i|bit]);
      gl.bindBuffer(gl.ARRAY_BUFFER,box);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(positions),gl.DYNAMIC_DRAW);
      gl.vertexAttribPointer(attrs.position,3,gl.FLOAT,false,12,0);
      for(const n of ['normal','rgb','field','candidate'])gl.disableVertexAttribArray(attrs[n]);
      gl.vertexAttrib1f(attrs.candidate,0);
      gl.vertexAttrib3f(attrs.normal,0,0,1);gl.vertexAttrib3f(attrs.rgb,1,1,1);gl.vertexAttrib4f(attrs.field,0,0,0,0);
      gl.uniform1f(u.box,1);gl.drawArrays(gl.LINES,0,24);
    }
  }
  let drag=null;
  canvas.addEventListener('pointerdown',e=>{if(e.button!==0)return;drag={x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId);canvas.focus();});
  canvas.addEventListener('pointermove',e=>{
    if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;
    if(e.shiftKey){const right=[-Math.sin(camera.theta),Math.cos(camera.theta),0];const up=[-Math.cos(camera.theta)*Math.sin(camera.phi),-Math.sin(camera.theta)*Math.sin(camera.phi),Math.cos(camera.phi)];camera.target=camera.target.map((v,i)=>v-right[i]*dx*camera.radius*.0015+up[i]*dy*camera.radius*.0015);}
    else{camera.theta-=dx*.006;camera.phi=clamp(camera.phi+dy*.006,-1.35,1.54);}
    drag={x:e.clientX,y:e.clientY};drawAll();
  });
  for(const event of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(event,()=>drag=null);
  canvas.addEventListener('wheel',e=>{e.preventDefault();camera.radius=clamp(camera.radius*Math.exp(e.deltaY*.001),.25,45);drawAll();},{passive:false});
  canvas.addEventListener('keydown',e=>{
    if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','=','-','r','R'].includes(e.key))return;e.preventDefault();
    if(e.key==='ArrowLeft')camera.theta+=.12;if(e.key==='ArrowRight')camera.theta-=.12;if(e.key==='ArrowUp')camera.phi=clamp(camera.phi+.1,-1.35,1.54);if(e.key==='ArrowDown')camera.phi=clamp(camera.phi-.1,-1.35,1.54);
    if(['+','='].includes(e.key))camera.radius=clamp(camera.radius*.85,.25,45);if(e.key==='-')camera.radius=clamp(camera.radius/.85,.25,45);
    if(e.key.toLowerCase()==='r')overview();else drawAll();
  });
  canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();$('status').textContent='3D 화면 연결이 끊겼습니다. 페이지를 새로 고쳐주세요.';});
  return {draw};
}
function drawAll(){for(const renderer of renderers)renderer.draw();}
function updateStats(){
  for(const [i,method]of ['m3c2','gls'].entries()){
    const container=$('stats-'+i);container.replaceChildren();
    if(!selected){
      for(const [title,value]of [['전후 사진 연결 구역','0곳'],['노이즈·관측 부족 제외',(independent.noise_filter?.suppressed_patch_count||0)+'개']]){
        const part=document.createElement('div'),small=document.createElement('small'),strong=document.createElement('strong');small.textContent=title;strong.textContent=value;part.append(small,strong);container.append(part);
      }
      $('capture-'+i).textContent=epoch===0?'이전 메쉬':'이후 메쉬';
      $('view-tag-'+i).textContent='전체 공간 · 연결된 검출 영역 없음';
      $('method-context-'+i).textContent=i===0?'노이즈 억제 후 전후 사진 연결 구역 없음':'전체 공간 참고';
      continue;
    }
    const key=selected.id+':'+epoch+':'+method;
    const own=selected.source==='m3c2_independent';
    if(!assessmentCache.has(key))assessmentCache.set(key,surfaceAssessment(geometries[epoch],displayBounds(selected,epoch,data.display_origin,.10),method,own?regionIds[epoch]:null,own?selected.candidate_id:0));
    const assessment=assessmentCache.get(key),type=CHANGE_TYPES[selected.type];
    const rows=own?[['변화 유형',type.label,selected.type],[method==='m3c2'?'표면 차이 · 변화 확률 아님':'현재 메쉬의 형상 차이',method==='m3c2'?((selected.classification_evidence.noise_filter?.median_distance_m??selected.classification_evidence.m3c2_median_distance_m)*100).toFixed(1)+'cm':assessment.median==null?'관측 부족':assessment.median.toFixed(3),'']]:[['선택한 변화',type.label+' · '+regionName(selected),selected.type],['이 위치의 표면 차이',assessment.status,assessment.status==='변화 의심'?'yellow':'']];
    for(const [title,text,style]of rows){
      const part=document.createElement('div');part.className=style;const small=document.createElement('small'),strong=document.createElement('strong');small.textContent=title;strong.textContent=text;part.append(small,strong);container.append(part);
    }
    $('capture-'+i).textContent=epoch===0?'이전 메쉬':'이후 메쉬';
    $('view-tag-'+i).textContent=type.label+' · '+regionName(selected)+' · '+(epoch===0?'이전 위치':'이후 위치');
    $('method-context-'+i).textContent=own?(i===0?'표면 차이 → 깊이 관측·형상 대응':'같은 의심 구역의 GLS 참고 비교'):type.label+' 위치의 표면 차이 · 기존 추정';
  }
}
function visibleEvents(){return filteredEvents(events,source,filter);}
function renderSources(){
  const container=$('source-filters');container.replaceChildren();
  for(const [key,title]of Object.entries(CHANGE_SOURCES)){
    if(key==='existing'&&reference.kind==='rgbd_reconstruction_pair')continue;
    const b=document.createElement('button');b.dataset.source=key;b.setAttribute('aria-pressed',String(source===key));
    b.textContent=title+' '+events.filter(event=>event.source===key).length;
    b.addEventListener('click',()=>{source=key;filter='all';renderSources();renderFilters();renderEventSelect();const next=visibleEvents()[0];if(next)choose(next);else clearSelection();});container.append(b);
  }
  const pending=filteredEvents(events,source,'suspected').length;
  $('source-note').textContent=source==='m3c2_independent'?'M3C2 + 요철·정합 오차·다중 크기·원본 깊이 지지 검사'+(pending?' · 유형 미정 '+pending+'곳은 전체 목록에 표시':''):'기존 Rescan 객체 연결을 사용한 비교';
}
function renderFilters(){
  const container=$('type-filters');container.replaceChildren();
  for(const type of CHANGE_FILTERS){
    const button=document.createElement('button');button.dataset.type=type;
    const count=filteredEvents(events,source,type).length;
    button.textContent=(type==='all'?'전체':CHANGE_TYPES[type].label)+' '+count;
    button.setAttribute('aria-pressed',String(filter===type));button.disabled=type!=='all'&&!count;
    button.addEventListener('click',()=>{filter=type;renderFilters();renderEventSelect();const next=visibleEvents()[0];if(next&&!visibleEvents().includes(selected))choose(next);else if(!next)clearSelection();});
    container.append(button);
  }
}
function renderEventSelect(){
  const select=$('event-select');select.replaceChildren();
  select.disabled=!visibleEvents().length;
  if(select.disabled){const option=document.createElement('option');option.textContent='전후 사진 연결 구역 없음';select.append(option);return;}
  for(const event of visibleEvents()){
    const option=document.createElement('option');option.value=event.id;
    option.textContent=CHANGE_TYPES[event.type].label+' · '+regionName(event)+(event.motion?' · 약 '+event.motion.displacement_m.toFixed(2)+'m':'');select.append(option);
  }
  if(selected)select.value=selected.id;
}
function overview(){
  camera.target=data.bounds.min.map((v,i)=>(v+data.bounds.max[i])/2);camera.radius=Math.max(...data.bounds.max.map((v,i)=>v-data.bounds.min[i]))*1.04;camera.phi=.94;camera.theta=-1.12;drawAll();
}
function clearSelection(){
  selected=null;
  $('event-type').textContent='0곳';$('event-type').dataset.type='';
  $('event-name').textContent='전후 사진까지 연결된 새 구역이 없습니다';$('event-status').textContent='재계산 완료';
  document.querySelector('main').style.setProperty('--event-color','#91a7b5');
  $('event-source').textContent='M3C2 · 재구성 오차 억제 적용';
  const unlinked=independent.unlinked_candidates.length;
  $('event-description').textContent='표면 요철·정합 오차와 여러 크기의 일관성, 원본 깊이 지지를 반영했습니다. '+(unlinked?'남은 후보 '+unlinked+'곳은 전후 원본 사진 연결이 부족합니다. ':'')+'이 결과가 공간 전체의 무변화를 뜻하지는 않습니다.'+(dataset==='rescan'?' 기존 Rescan 비교는 별도로 볼 수 있습니다.':'');
  $('event-evidence').hidden=false;
  $('event-evidence').textContent='노이즈·깊이 지지 부족 후보 '+(independent.noise_filter?.suppressed_patch_count||0)+'개 제외 · 사진 미연결 '+unlinked+'곳 · 아래 원본은 공간 참고용';
  $('evidence-details').hidden=true;
  for(let e=0;e<2;e++){photoLists[e]=contextPhotoCandidates(reference,e);photoIndices[e]=0;showPhoto(e,false);}
  $('mode').value='2';updateLegend();updateStats();overview();
}
function choose(event){
  if(!selected){$('mode').value='0';updateLegend();}
  selected=event;const type=CHANGE_TYPES[event.type];
  if(dataset==='bonn-removal'){
    epoch=event.type==='removed'?0:1;
    $('before').setAttribute('aria-pressed',String(epoch===0));$('after').setAttribute('aria-pressed',String(epoch===1));
  }
  const bounds={min:event.bounds.min.map((v,i)=>v-data.display_origin[i]),max:event.bounds.max.map((v,i)=>v-data.display_origin[i])};
  camera.target=bounds.min.map((v,i)=>(v+bounds.max[i])/2);camera.phi=.85;
  camera.radius=clamp(Math.max(...bounds.max.map((v,i)=>v-bounds.min[i]))*2.25,1.6,18);
  $('event-select').value=event.id;$('event-type').textContent=type.label;$('event-type').dataset.type=event.type;
  $('event-name').textContent=regionName(event);document.querySelector('main').style.setProperty('--event-color',type.css);
  const own=event.source==='m3c2_independent';
  $('event-status').textContent=event.type==='suspected'?'변화 근거 부족':'추정';
  $('event-source').textContent=own?'의심 구역 · 객체 라벨 미사용':'기존 Rescan 비교 · 객체 연결 기반';
  $('event-description').textContent=own?event.reason:event.type==='moved'?event.name+'의 위치가 약 '+event.motion.displacement_m.toFixed(2)+'m 달라진 것으로 추정했습니다. 전후 배치와 주변 사물을 비교해 보세요.':event.type==='added'?'이후 촬영에서 추가된 것으로 추정한 대상입니다. 이전의 같은 위치와 이후 모습을 비교해 보세요.':'이전 촬영에 있던 대상이 소실된 것으로 추정한 위치입니다. 이후의 같은 위치를 확인해 보세요.';
  $('event-evidence').hidden=!own;
  $('evidence-details').hidden=!own;
  if(own){
    $('event-evidence').textContent=evidenceSummary(event);
    const evidence=event.classification_evidence;
    const free=Math.max(0,...evidence.epoch_observations.map(p=>p.opposite.free));
    if(evidence.noise_filter){
      $('score-details').textContent='각 표면의 법선 방향 거리를 반경 8·16·28cm에서 계산합니다. 주변 표면의 요철(MAD와 10~90% 폭)과 두 맵의 정합 오차 가정(각 2cm)을 결합하고, 메쉬 표본 수로 오차를 줄이지 않습니다. 거리 6cm 이상이면서 이 오차 범위를 같은 방향으로 2개 이상 크기에서 넘어야 합니다. 그런 표본이 원래 패치의 55% 이상이고 자기 촬영 원본 깊이가 40% 이상을 지지해야 후보로 남습니다. 반대 시점 빈 공간 지지 비율은 '+(free*100).toFixed(1)+'%입니다. 점수는 변화 확률이 아니며, 작은 변화나 관측이 부족한 실제 변화는 놓칠 수 있습니다.';
    }else{
    $('score-details').textContent='각 표본의 점수는 |법선 방향 평균 표면 거리| ÷ 주변 표면의 흔들림·표본 수·가정 정합 오차(2cm)로 계산한 기준값(LoD95)입니다. 이 구역의 비율 중앙값은 '+evidence.m3c2_median_ratio.toFixed(2)+'입니다. 표시한 거리값과 이 비율은 변화 확률이 아니며, 실제 메쉬 오차로 보정한 신뢰도도 아닙니다. 변화 표본은 거리 6cm 이상·비율 1 이상·국소 평면성 0.25 이상을 모두 만족해야 하며, 종류 판단에는 그 표본이 구역의 55% 이상 필요합니다. 반대 시점에서 빈 공간을 관측한 표본은 '+(free*100).toFixed(1)+'%입니다(필요 55% 및 서로 떨어진 2개 이상 시점). 가림은 빈 공간으로 세지 않습니다. 바닥처럼 변화가 없는 표면도 재구성·정합 오차로 거리 차이가 생길 수 있습니다.';
    }
  }
  for(let e=0;e<2;e++){photoLists[e]=photoCandidates(reference,event,e);photoIndices[e]=0;showPhoto(e,false);}
  updateStats();drawAll();
}
function setEpoch(next){epoch=next;$('before').setAttribute('aria-pressed',String(epoch===0));$('after').setAttribute('aria-pressed',String(epoch===1));updateStats();drawAll();}
async function showPhoto(e,animate){
  $('expand-photos').disabled=true;
  const token=++photoTokens[e],item=photoLists[e][photoIndices[e]],img=$('photo-'+e),overlay=$('photo-overlay-'+e),notice=$('no-photo-'+e);
  img.hidden=true;overlay.replaceChildren();notice.hidden=false;notice.textContent=item?'원본 키프레임을 불러오는 중…':'이 위치에 연결된 원본 키프레임이 없습니다.';
  $('photo-title-'+e).textContent=photoTitle(selected,e);$('photo-check-'+e).textContent='';
  $('photo-index-'+e).textContent=item?(photoIndices[e]+1)+' / '+photoLists[e].length:'0 / 0';
  for(const b of document.querySelectorAll('[data-photo-epoch="'+e+'"]'))b.disabled=!item||(Number(b.dataset.step)<0?photoIndices[e]===0:photoIndices[e]===photoLists[e].length-1);
  if(!item){$('photo-caption-'+e).textContent='연결 사진 없음';return;}
  const loaded=new Image();loaded.src=item.frame.url;
  try{await loaded.decode();}catch{if(token===photoTokens[e])notice.textContent='원본 사진을 불러오지 못했습니다.';return;}
  if(token!==photoTokens[e])return;
  img.src=loaded.src;img.hidden=false;notice.hidden=true;
  img.alt=photoTitle(selected,e)+' · '+(selected?regionName(selected):'검출 구역 없음')+' · 원본 프레임 '+item.frame.source_frame_index;
  $('photo-caption-'+e).textContent='원본 프레임 '+item.frame.source_frame_index+' · '+item.frame.time_s.toFixed(2)+'초';
  const checked=['estimated_rgb_checked','estimated_rgb_tracked','dataset_pose'].includes(item.frame.status)&&item.link.bbox_px;
  $('photo-check-'+e).textContent=!selected?'공간 참고 · 검출 구역 없음':checked?(item.frame.status==='dataset_pose'?'제공 포즈 · 보정 적용':selected.source==='m3c2_independent'?'추정 위치 표시':'선택 위치 표시'):'연결 사진 · 위치 추정';
  if(checked){
    const [x0,y0,x1,y1]=item.link.bbox_px,rect=document.createElementNS('http://www.w3.org/2000/svg','rect');
    overlay.setAttribute('viewBox','0 0 '+item.frame.width+' '+item.frame.height);
    for(const [key,value]of Object.entries({x:x0,y:y0,width:x1-x0,height:y1-y0}))rect.setAttribute(key,value);
    overlay.append(rect);
    const text=document.createElementNS('http://www.w3.org/2000/svg','text');text.setAttribute('x',x0+5);text.setAttribute('y',Math.max(18,y0-6));text.textContent=selected.source==='m3c2_independent'?'선택 위치 · 변화 여부 확인':selected.type==='moved'?regionName(selected):'선택 위치 · '+CHANGE_TYPES[selected.type].label+' 추정';overlay.append(text);
  }
  if(animate&&!matchMedia('(prefers-reduced-motion: reduce)').matches)img.animate([{opacity:.4},{opacity:1}],{duration:180,easing:'ease-out'});
  $('expand-photos').disabled=[0,1].some(index=>$('photo-'+index).hidden||!$('photo-'+index).naturalWidth);
}
function movePhoto(e,step){const next=clamp(photoIndices[e]+step,0,photoLists[e].length-1);if(next!==photoIndices[e]){photoIndices[e]=next;showPhoto(e,true);}}
async function load(){
  const controls=[...document.querySelectorAll('.toolbar button,.toolbar select,.toolbar input,#event-select')];
  for(const control of controls)control.disabled=true;
  $('status').dataset.state='loading';
  try{
    const responses=await Promise.all(endpoints.map(url=>fetch(url)));
    if(responses.some(r=>!r.ok))throw new Error('메쉬와 원본 사진 연결 결과를 불러오지 못했습니다.');
    [data,reference,independent]=await Promise.all(responses.map(r=>r.json()));
    $('dataset-select').value=dataset;
    $('dataset-description').textContent=dataset==='bonn-removal'?'Bonn 실제 상자 제거 영상의 두 구간 · 전 1–4.5초 / 후 11.5–16.3초 · RGB-D + 제공 포즈로 각각 복원 · 객체/변화 라벨 미사용':'Rescan 실제 재촬영 쌍 · 기존 라벨 비교와 라벨 없는 의심 구역';
    $('dataset-download').href=dataset==='bonn-removal'?'/datasets/bonn-removal/changes.json':'/mesh-comparison/comparison.json';
    $('dataset-footer').textContent=dataset==='bonn-removal'?'Bonn RGB-D Dynamic Dataset · Palazzolo et al., ReFusion 2019 · 실제 시간 구간 비교':'실제 Rescan 전후 메쉬와 원본 정지 키프레임';
    if(data.kind!=='geometry_method_comparison'||data.displays?.length!==2)throw new Error('계산 결과 형식이 올바르지 않습니다.');
    $('pair').textContent=data.scene;$('status').textContent='전후 메쉬를 불러와 두 3D 지도를 준비하는 중…';
    validateReference(data,reference);const merged=mergeIndependent(data,reference,independent);events=merged.events;reference=merged.reference;
    geometries=await Promise.all(data.displays.map(async display=>{const r=await fetch(display.url);if(!r.ok)throw new Error('메쉬 결과를 불러오지 못했습니다.');return r.arrayBuffer();}));
    regionIds=await Promise.all(independent.region_masks.map(async (meta,e)=>{const r=await fetch(meta.url);if(!r.ok)throw new Error('독립 구역 표면을 불러오지 못했습니다.');const bytes=await r.arrayBuffer();if(meta.count!==data.displays[e].count||bytes.byteLength!==meta.count*2)throw new Error('독립 구역 표면 크기가 다릅니다.');return new Uint16Array(bytes);}));
    renderers=[0,1].map(method=>makeRenderer($('map-'+method),geometries,method));
    renderSources();renderFilters();renderEventSelect();const first=visibleEvents()[0];if(first)choose(first);else clearSelection();
    ready=true;for(const control of controls)control.disabled=false;
    $('event-select').disabled=!visibleEvents().length;
    $('status').dataset.state='ready';
    $('status').textContent='전후 사진을 연결한 의심 구역 '+independent.regions.length+'곳'+(dataset==='rescan'?' · 기존 비교 '+events.filter(e=>e.source==='existing').length+'개':' · 보정된 원본 RGB-D 포즈 사용');
    window.addEventListener('resize',drawAll);
  }catch(error){
    ready=false;$('status').dataset.state='error';
    $('status').textContent='3D 비교 화면을 불러오지 못했습니다. 아래 버튼으로 다시 불러와 주세요.';
    $('status').classList.add('error');$('retry').hidden=false;console.error('Mesh comparison initialization failed:',error);
  }
}
$('before').addEventListener('click',()=>{if(ready)setEpoch(0);});$('after').addEventListener('click',()=>{if(ready)setEpoch(1);});
function updateLegend(){
  const mode=$('mode').value;
  for(const item of document.querySelectorAll('.legend > span:not(#score-hint)'))item.hidden=mode!=='0';
  $('score-hint').hidden=mode==='0';
  $('score-hint').textContent=mode==='1'?'원시 표면 차이 · 재구성 노이즈 포함 · M3C2 ratio 0–3 / GLS 차이 0–0.25':'메쉬 원래 색상 · 원본 사진의 대상과 위치를 비교하세요.';
}
$('mode').addEventListener('change',()=>{
  updateLegend();
  drawAll();
});$('cut').addEventListener('change',drawAll);
$('overview').addEventListener('click',()=>{if(ready)overview();});$('top').addEventListener('click',()=>{if(ready){camera.phi=1.52;drawAll();}});
$('retry').addEventListener('click',()=>location.reload());
$('event-select').addEventListener('change',e=>{const event=events.find(item=>String(item.id)===e.target.value);if(ready&&event)choose(event);});
for(const b of document.querySelectorAll('[data-photo-epoch]'))b.addEventListener('click',()=>movePhoto(Number(b.dataset.photoEpoch),Number(b.dataset.step)));
$('expand-photos').addEventListener('click',()=>{
  $('dialog-title').textContent=selected?CHANGE_TYPES[selected.type].label+' · '+regionName(selected)+' · 전후 원본':'검출 구역 없음 · 촬영 공간 참고 원본';
  for(let e=0;e<2;e++){$('large-photo-'+e).src=$('photo-'+e).getAttribute('src')||'';$('large-photo-'+e).hidden=$('photo-'+e).hidden;$('large-caption-'+e).textContent=photoTitle(selected,e);}
  $('photo-dialog').showModal();
});
$('close-photos').addEventListener('click',()=>$('photo-dialog').close());
$('dataset-select').addEventListener('change',e=>{location.search=e.target.value==='bonn-removal'?'?dataset=bonn-removal':'';});
load();

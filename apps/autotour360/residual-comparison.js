import {OUTCOMES,REASONS,REFERENCE_OUTCOMES,REFERENCE_TYPES,regionLabel,resultRoot,validate,linkedPhotos,filtered,exceedCount} from './residual-context.js?v=6';
const $=id=>document.getElementById(id),clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const cm=x=>x==null?'—':(x*100).toFixed(2)+'cm',pct=x=>(x*100).toFixed(1)+'%';
const camera={target:[0,0,0],radius:8,theta:-1.12,phi:.85};
const sub=(a,b)=>a.map((v,i)=>v-b[i]),dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],unit=a=>a.map(v=>v/Math.max(Math.hypot(...a),1e-10));
let data,selected=null,epoch=1,ready=false,renderers=[];
const photos=[[],[]],photoIndices=[0,0],photoTokens=[0,0];
const dataRoot=resultRoot(location.search);
$('dataset').value=dataRoot.includes('bonn-placement')?'bonn-placement':'rescan';
function element(tag,text,cls){const e=document.createElement(tag);if(text!=null)e.textContent=text;if(cls)e.className=cls;return e;}
function matrix(aspect){
  const {radius:r,phi:p,theta:t,target}=camera;
  const eye=target.map((v,i)=>v+[r*Math.cos(p)*Math.cos(t),r*Math.cos(p)*Math.sin(t),r*Math.sin(p)][i]);
  const z=unit(sub(eye,target)),x=unit(cross([0,0,1],z)),y=cross(z,x);
  const view=[x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1];
  const f=1/Math.tan(.72/2),n=.01,far=120,pj=[f/aspect,0,0,0,0,f,0,0,0,0,(far+n)/(n-far),-1,0,0,2*far*n/(n-far),0],out=new Float32Array(16);
  for(let j=0;j<4;j++)for(let i=0;i<4;i++)for(let k=0;k<4;k++)out[j*4+i]+=pj[k*4+i]*view[j*4+k];
  return out;
}
const vertex=`attribute vec3 a_position,a_normal,a_rgb;attribute vec4 a_field;
uniform mat4 u_matrix;uniform float u_mode;uniform mediump float u_box;uniform vec3 u_min,u_max;
varying mediump vec3 v_color;varying highp float v_height;
void main(){gl_Position=u_matrix*vec4(a_position,1.);v_height=a_position.z;
float inside=step(u_min.x,a_position.x)*step(a_position.x,u_max.x)*step(u_min.y,a_position.y)*step(a_position.y,u_max.y)*step(u_min.z,a_position.z)*step(a_position.z,u_max.z);
vec3 base=max(a_rgb,vec3(.14)),color=base*.30;float c=a_field.z;
if(inside>.5){color=base*.8;if(c>.5&&c<1.5)color=vec3(1.,.58,.27);if(c>1.5)color=mix(base*.35,vec3(.37,.47,.55),.65);
if(u_mode>.5&&u_mode<1.5){float s=clamp(a_field.w/2.,0.,1.);color=mix(vec3(.12,.31,.40),vec3(1.,.64,.25),s);if(a_field.y<.0000001)color=vec3(.23,.30,.37);}}
if(c>2.5&&u_mode<1.5)color=mix(base*.3,vec3(.16,.77,.77),.85);
if(u_mode>1.5)color=base*.85;
float light=.65+.35*abs(dot(normalize(a_normal+vec3(.0001)),normalize(vec3(.4,-.6,1.))));v_color=color*light;
if(u_box>.5)v_color=vec3(.35,.90,.65);}`;
const fragment=`precision mediump float;varying mediump vec3 v_color;varying highp float v_height;uniform highp float u_cut;uniform mediump float u_box;void main(){if(u_box<.5&&v_height>u_cut)discard;gl_FragColor=vec4(v_color,1.);}`;
function renderer(canvas,geometry,fields,original=false){
  const gl=canvas.getContext('webgl',{antialias:true,alpha:false,preserveDrawingBuffer:true});if(!gl)throw new Error('WebGL을 사용할 수 없습니다.');
  function shader(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;}
  const program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,vertex));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,fragment));gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(program));
  const attrs=Object.fromEntries(['position','normal','rgb','field'].map(k=>[k,gl.getAttribLocation(program,'a_'+k)]));
  const u=Object.fromEntries(['matrix','mode','box','min','max','cut'].map(k=>[k,gl.getUniformLocation(program,'u_'+k)]));
  function buffer(bytes){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,bytes,gl.STATIC_DRAW);return b;}
  const meshes=geometry.map(buffer),scores=fields.map(buffer),box=gl.createBuffer();
  function draw(){
    const ratio=Math.min(devicePixelRatio||1,2),w=Math.max(1,Math.round(canvas.clientWidth*ratio)),h=Math.max(1,Math.round(canvas.clientHeight*ratio));
    if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
    gl.viewport(0,0,w,h);gl.clearColor(.035,.065,.092,1);gl.enable(gl.DEPTH_TEST);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.useProgram(program);
    gl.uniformMatrix4fv(u.matrix,false,matrix(w/h));gl.uniform1f(u.mode,original?2:['decision','ratio','original'].indexOf($('mode').value));gl.uniform1f(u.box,0);gl.uniform1f(u.cut,data.bounds.max[2]-.28);
    const bounds=selected?{min:selected.bounds.min.map((v,i)=>v-data.display_origin[i]-.05),max:selected.bounds.max.map((v,i)=>v-data.display_origin[i]+.05)}:data.bounds;
    gl.uniform3fv(u.min,bounds.min);gl.uniform3fv(u.max,bounds.max);gl.bindBuffer(gl.ARRAY_BUFFER,meshes[epoch]);for(const a of Object.values(attrs))gl.enableVertexAttribArray(a);
    gl.vertexAttribPointer(attrs.position,3,gl.FLOAT,false,60,0);gl.vertexAttribPointer(attrs.normal,3,gl.FLOAT,false,60,12);gl.vertexAttribPointer(attrs.rgb,3,gl.UNSIGNED_BYTE,true,60,24);gl.bindBuffer(gl.ARRAY_BUFFER,scores[epoch]);gl.vertexAttribPointer(attrs.field,4,gl.FLOAT,false,16,0);gl.drawArrays(gl.TRIANGLES,0,data.displays[epoch].count);
    if(selected){
      const corners=[],positions=[];for(let i=0;i<8;i++)corners.push([i&1?bounds.max[0]:bounds.min[0],i&2?bounds.max[1]:bounds.min[1],i&4?bounds.max[2]:bounds.min[2]]);
      for(let i=0;i<8;i++)for(const bit of [1,2,4])if(!(i&bit))positions.push(...corners[i],...corners[i|bit]);
      gl.bindBuffer(gl.ARRAY_BUFFER,box);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(positions),gl.DYNAMIC_DRAW);gl.vertexAttribPointer(attrs.position,3,gl.FLOAT,false,12,0);for(const n of ['normal','rgb','field'])gl.disableVertexAttribArray(attrs[n]);gl.vertexAttrib3f(attrs.normal,0,0,1);gl.vertexAttrib3f(attrs.rgb,1,1,1);gl.vertexAttrib4f(attrs.field,0,0,0,0);gl.uniform1f(u.box,1);gl.drawArrays(gl.LINES,0,24);
    }
  }
  let drag=null;canvas.addEventListener('pointerdown',e=>{if(e.button!==0)return;drag={x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId);canvas.focus();});
  canvas.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;
    if(e.shiftKey){const r=[-Math.sin(camera.theta),Math.cos(camera.theta),0],up=[-Math.cos(camera.theta)*Math.sin(camera.phi),-Math.sin(camera.theta)*Math.sin(camera.phi),Math.cos(camera.phi)];camera.target=camera.target.map((v,i)=>v-r[i]*dx*camera.radius*.0015+up[i]*dy*camera.radius*.0015);}
    else{camera.theta-=dx*.006;camera.phi=clamp(camera.phi+dy*.006,-1.35,1.54);}drag={x:e.clientX,y:e.clientY};drawAll();});
  for(const event of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(event,()=>drag=null);
  canvas.addEventListener('wheel',e=>{e.preventDefault();camera.radius=clamp(camera.radius*Math.exp(e.deltaY*.001),.3,45);drawAll();},{passive:false});
  canvas.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','=','-'].includes(e.key))return;e.preventDefault();if(e.key==='ArrowLeft')camera.theta+=.12;if(e.key==='ArrowRight')camera.theta-=.12;if(e.key==='ArrowUp')camera.phi=clamp(camera.phi+.1,-1.35,1.54);if(e.key==='ArrowDown')camera.phi=clamp(camera.phi-.1,-1.35,1.54);if(['+','='].includes(e.key))camera.radius=clamp(camera.radius*.85,.3,45);if(e.key==='-')camera.radius=clamp(camera.radius/.85,.3,45);drawAll();});
  return {draw};
}
function drawAll(){for(const r of renderers)r.draw();}
function overview(){camera.target=data.bounds.min.map((v,i)=>(v+data.bounds.max[i])/2);camera.radius=Math.max(...sub(data.bounds.max,data.bounds.min))*1.04;camera.phi=.94;drawAll();}
function resultCards(){
  for(let i=0;i<2;i++){
    $('capture-'+i).textContent=epoch===0?'이전 메쉬':'이후 메쉬';const box=$('result-'+i);box.replaceChildren();
    if(!selected){box.append(element('span','선택 영역 없음'));continue;}
    if(data.transfer&&i===0){box.dataset.status='original';box.append(element('strong','원본 메쉬'),element('div','오른쪽과 같은 촬영·같은 3D 위치'),element('div','주황 후보는 오른쪽 지도에서 확인합니다.'));continue;}
    if(selected.kind==='background'){
      box.dataset.status='background';box.append(element('strong',OUTCOMES.background),element('div','전후에 대응되는 구조면 '+selected.statistics[i].core_count+'점 · 가구와 분리하여 측정'),element('div','공통 구조면은 후보에서 제외합니다. 바닥 오차 모델은 별도로 유지합니다.'));
      continue;
    }
    if(selected.evaluation){
      const e=i===0?selected.evaluation.previous:selected.evaluation;
      box.dataset.status=e.outcome==='captured'?'change_candidate':'unassessable';
      box.append(element('strong',REFERENCE_OUTCOMES[e.outcome]),element('div','대상 가구 표면 '+e.candidate_cores+' / '+e.source_cores+'점 · '+(e.coverage_fraction==null?'표면 없음':pct(e.coverage_fraction))),element('div','후보 면적 '+e.candidate_area_m2.toFixed(3)+'m² · 최대 연결 '+e.largest_connected_cores+'점'));
      continue;
    }
    const s=selected.statistics[i];box.dataset.status=s.status;box.append(element('strong',OUTCOMES[s.status]),element('div','유효 '+s.valid_cores+'/'+s.core_count+' ('+pct(s.valid_fraction)+') · '+(s.valid_cores?'유효 표본 내 초과 '+s.exceeding_cores+' ('+pct(s.exceeding_valid_fraction)+')':'초과 판정 보류')),element('div',s.valid_cores?'초과 면적 전 '+s.exceeding_area_by_epoch_m2[0].toFixed(3)+' / 후 '+s.exceeding_area_by_epoch_m2[1].toFixed(3)+'m² · 최대 연결 '+s.largest_connected_exceedance_cores+'점':'초과 면적·연결성 미산정 · 승인된 비교 근거 없음'));
    if(s.status==='background'){box.replaceChildren(element('strong',OUTCOMES.background),element('div','전후에 대응되는 구조면 '+s.background_cores+'점 · 가구와 분리하여 측정'),element('div','이 면은 오차 기준에만 사용하며 변화 후보 개수에 포함하지 않습니다.'));}
    if(i===1&&$('mode').value==='ratio'&&!data.background)box.append(element('div','미승인 기준의 참고 비율도 포함 · 변화 판정으로 쓰지 않음'));
  }
}
function trace(){
  const box=$('trace');box.replaceChildren();if(!selected)return;
  box.append(element('p','좌표계: 기존 정합 맵의 회전 후 절대 좌표 · 단위 m · 범위 '+selected.bounds.min.map(v=>v.toFixed(3)).join(', ')+' → '+selected.bounds.max.map(v=>v.toFixed(3)).join(', ')));
  const table=element('table'),head=element('tr');for(const title of data.transfer?['반경','3D 표면 거리','고정한 바닥 기준','국소 요철 기준','최종 노이즈 기준']:['반경','이전 법선 방향 차이','새 3D 표면 거리','이전 판정 기준','새 노이즈 기준'])head.append(element('th',title));table.append(head);
  data.config.radii_m.forEach((r,j)=>{const row=element('tr');const values=data.transfer?[cm(r),cm(selected.distance_by_radius_m[j]?.median),cm(data.config.noise_budget_m[j]),cm(selected.conditions.roughness_m?.median==null?null:1.96*Math.sqrt(2)*selected.conditions.roughness_m.median),cm(selected.statistics[1].threshold_by_radius_m[j]?.median)]:[cm(r),cm(selected.distance_by_radius_m[j]?.median),cm(selected.corrected_distance_by_radius_m?.[j]?.median),cm(selected.statistics[0].threshold_by_radius_m[j]?.median),cm(selected.statistics[1].threshold_by_radius_m[j]?.median)];for(const value of values)row.append(element('td',value));table.append(row);});box.append(table);
  box.append(element('p','보류·판정 사유: '+Object.entries(selected.reasons).map(([r,n])=>(REASONS[r]||r)+' '+n+'점').join(' / ')));
  const c=selected.conditions;if(c)box.append(element('p','표면 요철 중앙값 '+cm(c.roughness_m?.median)+' · 지지 표본 수 중앙값 '+(c.density?.median?.toFixed(0)||'—')+' · 평면성 중앙값 '+(c.planarity?.median?.toFixed(3)||'—')));
  if(selected.reference_plane)box.append(element('p','대응 기준면: '+selected.reference_plane+' · 기준면과 가구는 별도 표면으로 측정합니다.'));
}
function histogram(values,threshold,check){
  const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 340 120');svg.setAttribute('role','img');svg.setAttribute('aria-label','기준 잔차 분포와 독립 확인 P95');
  const max=Math.max(.001,...values,threshold||0,check?.p95||0)*1.12,bins=new Array(25).fill(0);for(const v of values)bins[Math.min(24,Math.floor(v/max*25))]++;
  function line(x,color,dashed=false){const e=document.createElementNS(ns,'line');for(const [k,v]of Object.entries({x1:25+x/max*290,x2:25+x/max*290,y1:10,y2:92,stroke:color,'stroke-width':2}))e.setAttribute(k,v);if(dashed)e.setAttribute('stroke-dasharray','5 3');svg.append(e);}
  bins.forEach((v,i)=>{const e=document.createElementNS(ns,'rect');for(const [k,x]of Object.entries({x:25+i*290/25,y:92-v/Math.max(1,...bins)*70,width:290/25-1,height:v/Math.max(1,...bins)*70,fill:'#569b95'}))e.setAttribute(k,x);svg.append(e);});
  if(threshold!=null)line(threshold,'#8cf2be');if(check?.p95!=null)line(check.p95,'#f7b66f',true);
  for(const [x,text]of [[25,'0'],[220,cm(max)]]){const e=document.createElementNS(ns,'text');e.setAttribute('x',x);e.setAttribute('y',111);e.textContent=text;svg.append(e);}return svg;
}
function showModels(){
  const container=$('models');container.replaceChildren();
  if(data.transfer){container.append(element('p','기존 Rescan 바닥에서 계산한 '+data.config.noise_budget_m.map(cm).join(' / ')+'를 그대로 사용했습니다. 이번 촬영 쌍의 결과를 보고 기준을 다시 맞추지 않았습니다.'),element('p','비교 반경 8·16·28cm / 최소 차이 6cm / 두 크기 이상 초과 / 연결 반경 16cm / 최소 9점·0.04m². 정답 라벨은 읽지 않았습니다.'));return;}
  for(const model of data.models){
    const card=element('section',null,'model');card.append(element('h3',model.id+' · 기준 표본 '+model.fit_core_count+'점'),element('p',(model.enabled?'적용 승인':'적용 보류')+' · '+(REASONS[model.check_reason]||model.check_reason),'model-status'));
    const config=data.config.rois?.find(r=>r.id===model.roi_id);card.append(element('p',config?'추정 ROI '+config.bounds.min.join(', ')+' → '+config.bounds.max.join(', ')+'m':(model.kind==='floor'?'바닥':'수직 구조면')+' 공통 면적 약 '+model.common_support_grid_area_m2.toFixed(2)+'m² · 위치별 편차를 추정한 뒤 남은 잔차로 기준 계산'));
    for(const check of model.checks)card.append(element('p','분리한 구간 확인 '+check.valid_cores+'점 · 기준 초과 '+(check.exceedance_fraction==null?'—':pct(check.exceedance_fraction))+' · '+(check.failure_reasons||[check.reason]).map(r=>REASONS[r]||r).join(' / ')));
    const plots=element('div',null,'distribution');model.reference_distribution.forEach((dist,j)=>{const part=element('div');part.append(element('small','반경 '+cm(dist.radius_m)+' · 기준 P95 '+cm(model.thresholds_m[j])),histogram(dist.absolute_residuals_m,model.thresholds_m[j],model.checks[0]?.absolute_residual_summary[j]));plots.append(part);});card.append(plots,element('p','초록 실선: 공간 균형 기준 P95 / 주황 점선: 독립 확인 영역 P95. 분포 막대는 원시 표본 수입니다.'));container.append(card);
  }
}
function choose(region){
  selected=region;$('region').value=region.id;$('region-title').textContent=region.evaluation?regionLabel(region):region.title;
  $('region-note').textContent=region.evaluation?'기존 '+REFERENCE_TYPES[region.evaluation.reference_type]+' 항목에서 변화 위치를 찾았는지 비교 · 새 자동 종류 판정은 아님':region.kind==='background'?'전후 공통 구조면 · 청록 표면만 후보에서 제외 · 가구는 유지':region.kind==='fit'?'기준 추정용 ROI':region.kind==='check'?'기준 추정에 쓰지 않은 확인 ROI':'방향에 상관없이 비교한 국소 표면 · 아래 전후 사진으로 확인';
  for(const row of document.querySelectorAll('[data-reference]'))row.setAttribute('aria-current',String(row.dataset.reference===region.id));
  const bounds={min:region.bounds.min.map((v,i)=>v-data.display_origin[i]),max:region.bounds.max.map((v,i)=>v-data.display_origin[i])};camera.target=bounds.min.map((v,i)=>(v+bounds.max[i])/2);camera.radius=clamp(Math.max(...sub(bounds.max,bounds.min))*2.3,1.7,18);camera.phi=.90;
  for(let e=0;e<2;e++){photos[e]=linkedPhotos(data,region,e);photoIndices[e]=0;showPhoto(e);}resultCards();trace();drawAll();
}
function renderRegions(){
  const regions=filtered(data.regions,$('filter').value),select=$('region');select.replaceChildren();select.disabled=!regions.length;
  for(const r of regions){const option=element('option',regionLabel(r));option.value=r.id;select.append(option);}
  if(regions.length){choose(data.transfer?(regions.find(r=>r.paired_originals)||regions[0]):regions[0]);return;}
  selected=null;select.append(element('option','해당 분류의 구역 없음'));$('region-title').textContent='해당 분류의 구역이 없습니다';$('region-note').textContent='판단 불가를 변화 없음으로 해석하지 않습니다.';for(let e=0;e<2;e++){photos[e]=[];photoIndices[e]=0;showPhoto(e);}trace();resultCards();overview();
}
function showEvaluation(){
  if(!data.evaluation)return;
  $('evaluation-panel').hidden=false;const table=element('table'),head=element('tr');
  for(const title of ['항목','기존 종류','이전 방식','모든 방향','포착한 표면 비율'])head.append(element('th',title));table.append(head);
  for(const r of filtered(data.regions,'reference')){
    const e=r.evaluation,row=element('tr'),cell=element('td'),button=element('button',r.title);
    button.dataset.reference=r.id;button.addEventListener('click',()=>{$('filter').value='reference';renderRegions();choose(r);$('region-title').scrollIntoView({block:'start'});});cell.append(button);row.append(cell);
    for(const value of [REFERENCE_TYPES[e.reference_type],REFERENCE_OUTCOMES[e.previous.outcome],REFERENCE_OUTCOMES[e.outcome],e.coverage_fraction==null?'—':pct(e.coverage_fraction)])row.append(element('td',value));
    table.append(row);
  }
  $('evaluation-table').replaceChildren(table);$('evaluation-note').textContent='포착 기준: 해당 가구 표면에 9점 이상, 0.04m² 이상의 연결 후보가 한 곳 이상 존재. 기존 항목 라벨은 탐지 후 대조에만 사용했습니다. 전체 오탐률·이동 종류 정확도를 뜻하지 않습니다.';
}
async function showPhoto(e){
  const token=++photoTokens[e],item=photos[e][photoIndices[e]],img=$('photo-'+e),overlay=$('overlay-'+e),notice=$('no-photo-'+e);$('expand').disabled=true;img.hidden=true;overlay.replaceChildren();notice.hidden=false;notice.textContent=item?'원본 사진을 불러오는 중…':'해당 위치에 연결된 원본 사진 없음';$('caption-'+e).textContent='';$('photo-check-'+e).textContent='';$('photo-index-'+e).textContent=item?(photoIndices[e]+1)+' / '+photos[e].length:'0 / 0';
  for(const b of document.querySelectorAll('[data-epoch="'+e+'"]'))b.disabled=!item||(Number(b.dataset.step)<0?photoIndices[e]===0:photoIndices[e]===photos[e].length-1);if(!item)return;
  const load=new Image();load.src=item.frame.url;try{await load.decode();}catch{if(token===photoTokens[e])notice.textContent='원본 사진을 불러오지 못했습니다.';return;}if(token!==photoTokens[e])return;
  img.src=load.src;img.hidden=false;notice.hidden=true;img.alt=(e===0?'이전':'이후')+' 원본 · '+selected.title+' · 프레임 '+item.frame.source_frame_index;
  $('caption-'+e).textContent='프레임 '+item.frame.source_frame_index+' · '+item.frame.time_s.toFixed(2)+'초';$('photo-check-'+e).textContent=item.link.camera_check==='legacy_view_candidate'?'기존 원본 · 3D 연결 검토 필요':item.link.camera_check==='dataset_ground_truth'?'제공 카메라 포즈 · 같은 위치':'추정 카메라 · 같은 위치';
  if(item.link.bbox_px){const [x0,y0,x1,y1]=item.link.bbox_px,rect=document.createElementNS('http://www.w3.org/2000/svg','rect');overlay.setAttribute('viewBox','0 0 '+item.frame.width+' '+item.frame.height);for(const [k,v]of Object.entries({x:x0,y:y0,width:x1-x0,height:y1-y0}))rect.setAttribute(k,v);overlay.append(rect);}
  $('expand').disabled=[0,1].some(i=>$('photo-'+i).hidden||!$('photo-'+i).naturalWidth);
}
async function get(url,binary=false){const r=await fetch(url);if(!r.ok)throw new Error('결과 파일을 불러오지 못했습니다: '+url);return binary?r.arrayBuffer():r.json();}
async function load(){
  try{
    data=validate(await get(dataRoot+'comparison.json'));const geometry=await Promise.all(data.displays.map(d=>get(d.url,true)));
    const fields=await Promise.all(['fixed','empirical'].map(mode=>Promise.all([0,1].map(e=>get(dataRoot+e+'-'+mode+'.bin',true)))));
    for(let e=0;e<2;e++){const count=data.displays[e].count;if(geometry[e].byteLength!==count*60||fields.some(f=>f[e].byteLength!==count*16))throw new Error('메쉬와 판정 표면의 크기가 다릅니다.');}
    renderers=fields.map((f,i)=>renderer($('map-'+i),geometry,f,Boolean(data.transfer&&i===0)));$('pair').textContent=data.scene;
    $('result-download').href=dataRoot+'comparison.json';
    const enabled=data.models.filter(m=>m.enabled).length;
    for(let i=0;i<2;i++)$('method-'+i).textContent=data.variants[i].label;
    $('exceed-count').textContent=data.variants.map(v=>exceedCount(v.summary)).join(' → ');
    if(data.transfer){
      const t=data.transfer,a=data.assessable_nonbackground;
      document.title='SpaceWatch3D · 새 촬영 쌍 / 고정 설정';$('dataset-caption').textContent='Bonn · 처음 적용한 촬영 / 설정 고정';
      $('page-title').textContent='처음 적용한 전후 촬영의 변화 후보';$('method-description').textContent='왼쪽: 원본 메쉬 / 오른쪽: Rescan 설정을 그대로 적용한 변화 후보';
      $('summary-models').textContent='변화 후보 · 연결 표면';$('models-count').textContent=t.candidate_patches+'개';$('models-note').textContent='전후 사진 모두 연결 '+t.candidate_patches_with_paired_originals+'개 · 객체 개수와 다름';
      $('summary-exceed').textContent='변화 후보 표면 지점';$('exceed-count').textContent=t.candidate_cores.toLocaleString()+'점';$('candidate-note').textContent='깊이의 존재·빈 공간 지지 '+t.visibility_supported_candidates+'점';
      $('summary-valid').textContent='구조면 밖에서 비교 가능한 지점';$('valid-count').textContent=a.current.toLocaleString()+' / '+a.total.toLocaleString();$('assessable-note').textContent='바닥 기준값·검출 조건 그대로 · 라벨 사용 없음';
      $('legend').textContent='오른쪽 지도: 주황은 변화 의심 표면 / 청록은 공통 구조면 · 후보 제외 / 회색은 비교 근거 부족. 왼쪽은 원본 메쉬입니다.';
      document.querySelector('.maps').setAttribute('aria-label','같은 촬영의 원본 메쉬와 고정 설정 변화 후보 비교');document.querySelector('.photo-actions span').textContent='점선은 제공된 카메라 포즈로 연결한 같은 3D 위치입니다. 원본 정지 사진이며 자동 재생하지 않습니다.';
      $('context-note').textContent='처음 사용한 Bonn 상자 놓기 영상의 앞·뒤 구간을 각각 복원했습니다. 독립적인 두 번의 방문 촬영은 아닙니다. 기존 Bonn 상자 제거와는 같은 실험 공간입니다. 정답 라벨 없이 모든 후보를 표시하며, 복원 누락·가림에 따른 차이도 포함될 수 있습니다.';
      $('noise-details-summary').textContent='그대로 고정한 오차 기준과 검출 조건';$('result-download').textContent='고정 설정·출처·결과 JSON ↓';
      for(const option of [...$('filter').options])if(['reference','captured','partial','missed'].includes(option.value))option.remove();
      $('map-0').setAttribute('aria-label','원본 메쉬 지도. 드래그 회전, 휠 확대');$('map-1').setAttribute('aria-label','고정 설정 변화 후보 지도. 드래그 회전, 휠 확대');
      $('filter').value=t.candidate_patches?'change_candidate':'all';
    }else if(data.evaluation){
      const e=data.evaluation,a=data.assessable_nonbackground;$('models-count').textContent=e.previous_captured+' → '+e.captured+' / '+e.reference_count;
      $('valid-count').textContent=a.previous.toLocaleString()+' → '+a.current.toLocaleString();
      $('assessable-note').textContent='구조면 제외 '+a.total.toLocaleString()+'점 중 비교 가능';
      $('candidate-note').textContent=e.candidate_cores_outside_reference_items.toLocaleString()+'점은 12개 항목 밖 · 추가 후보와 오탐이 섞일 수 있음';
      $('filter').value='reference';
    }else{
      $('models-count').textContent=enabled+' / '+data.models.length+'개';$('valid-count').textContent=data.background?data.background.area_by_epoch_m2.map(a=>a.toFixed(1)).join(' / ')+'m²':pct(data.variants[1].summary.valid_fraction);
      if(data.background&&filtered(data.regions,'change_candidate').length)$('filter').value='change_candidate';else $('filter').value='all';
    }
    showEvaluation();showModels();renderRegions();ready=true;$('status').dataset.state='ready';$('status').textContent=data.transfer?'고정 설정 적용 완료 · 바닥 기준 '+data.config.noise_budget_m.map(cm).join(' / ')+' · 전 '+data.transfer.windows_s[0].map(v=>v.toFixed(2)).join('–')+'초 / 후 '+data.transfer.windows_s[1].map(v=>v.toFixed(2)).join('–')+'초':data.evaluation?'계산 완료 · 바닥 오차 기준 '+data.config.noise_budget_m.map(cm).join(' / ')+' 유지 · 가구의 옆면·등받이·윗면 모두 비교':data.background?'계산 완료 · 공통 구조면 분리 · 사진을 선택해 전후 확인':enabled?'계산 완료':'계산 완료 · 승인된 기준 없음: 경험적 변화 판정 보류';
    window.addEventListener('resize',drawAll);
  }catch(error){$('status').dataset.state='error';$('status').textContent='잔여 오차 실험 화면을 불러오지 못했습니다. 새로 고침해 주세요.';console.error(error);}
}
$('filter').addEventListener('change',()=>{if(ready)renderRegions();});$('region').addEventListener('change',e=>{if(ready)choose(data.regions.find(r=>r.id===e.target.value));});
$('dataset').addEventListener('change',()=>{location.href='residual-comparison.html'+($('dataset').value==='rescan'?'':'?dataset=bonn-placement');});
for(const [id,e]of [['before',0],['after',1]])$(id).addEventListener('click',()=>{if(!ready)return;epoch=e;$('before').setAttribute('aria-pressed',String(e===0));$('after').setAttribute('aria-pressed',String(e===1));resultCards();drawAll();});
$('mode').addEventListener('change',()=>{if(ready){resultCards();drawAll();}});$('overview').addEventListener('click',()=>{if(ready)overview();});$('top').addEventListener('click',()=>{if(ready){camera.phi=1.52;drawAll();}});
for(const b of document.querySelectorAll('[data-epoch]'))b.addEventListener('click',()=>{const e=Number(b.dataset.epoch),next=clamp(photoIndices[e]+Number(b.dataset.step),0,photos[e].length-1);if(next!==photoIndices[e]){photoIndices[e]=next;showPhoto(e);}});
$('expand').addEventListener('click',()=>{for(let e=0;e<2;e++)$('large-'+e).src=$('photo-'+e).src;$('photo-dialog').showModal();});$('close').addEventListener('click',()=>$('photo-dialog').close());
load();

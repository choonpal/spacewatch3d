const $ = id => document.getElementById(id);
const count = n => n.toLocaleString('ko-KR');
const distance = n => Number(n).toFixed(3);
const percent = n => (n * 100).toFixed(1) + '%';
let evidence, renderers = [], selected = null;
function changeInfo(region) {
  const change=region.change||{type:region.source_epoch===0?'removed':'added',status:'candidate',displacement_m:null};
  const type=['removed','added','moved','unknown'].includes(change.type)?change.type:'unknown';
  const candidate=change.status!=='confirmed';
  const movement=type==='moved'&&Number.isFinite(change.displacement_m)&&change.displacement_m>=0?change.displacement_m:null;
  const title={removed:'제거',added:'추가',moved:'이동',unknown:'변화 미분류'}[type]+(candidate&&type!=='unknown'?' 후보':'');
  return {type,title,movement,label:type==='moved'?title+' · '+(movement===null?'거리 미산정':movement.toFixed(2)+' m'):title};
}
function boundsFor(region, epoch) {
  return (epoch===0?region.change?.before_bounds:region.change?.after_bounds)||region.bounds;
}
function updateChangeLabels(epoch, matrix, canvas) {
  const layer=$(epoch===0?'before-labels':'after-labels');
  layer.replaceChildren();
  const regions=selected?[selected]:evidence.regions.filter(r=>r.source_epoch===epoch||changeInfo(r).type==='moved');
  const occupied=[];
  for(const region of regions) {
    const bounds=boundsFor(region,epoch), center=bounds.min.map((v,i)=>(v+bounds.max[i])/2-evidence.display_origin[i]);
    const clip=[0,0,0,0];
    for(let row=0;row<4;row++)clip[row]=matrix[row]*center[0]+matrix[4+row]*center[1]+matrix[8+row]*center[2]+matrix[12+row];
    if(clip[3]<=0||Math.abs(clip[0]/clip[3])>1||Math.abs(clip[1]/clip[3])>1||Math.abs(clip[2]/clip[3])>1)continue;
    const info=changeInfo(region), label=document.createElement('span');
    label.className='change-label '+info.type;
    const counterpart=selected&&region.source_epoch!==epoch&&info.type!=='moved';
    label.textContent='영역 '+region.id+' · '+info.label+(counterpart?' · 같은 위치':'');
    const x=(clip[0]/clip[3]*.5+.5)*canvas.clientWidth;
    let y=clamp((.5-clip[1]/clip[3]*.5)*canvas.clientHeight-30,18,canvas.clientHeight-42);
    // Nearby region labels stack vertically while following the shared camera.
    for(const point of occupied)if(Math.abs(point.x-x)<120&&Math.abs(point.y-y)<34)y=clamp(point.y+36,18,canvas.clientHeight-42);
    occupied.push({x,y});label.style.left=clamp(x,85,Math.max(85,canvas.clientWidth-85))+'px';label.style.top=y+'px';layer.append(label);
  }
}
const camera = { target: [0, 0, 0], radius: 5.3, theta: -1.12, phi: .72 };
const clamp = (n, a, b) => Math.max(a, Math.min(b, n));
const sub = (a, b) => a.map((v, i) => v - b[i]);
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const unit = a => a.map(v => v / Math.hypot(...a));
function multiply(a, b) {
  const out = new Float32Array(16);
  for (let j=0;j<4;j++) for (let i=0;i<4;i++) for (let k=0;k<4;k++) out[j*4+i] += a[k*4+i]*b[j*4+k];
  return out;
}
function viewMatrix() {
  const r=camera.radius, p=camera.phi, t=camera.theta;
  const eye=camera.target.map((v,i)=>v+[r*Math.cos(p)*Math.cos(t),r*Math.cos(p)*Math.sin(t),r*Math.sin(p)][i]);
  const z=unit(sub(eye,camera.target)), x=unit(cross([0,0,1],z)), y=cross(z,x);
  return new Float32Array([x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1]);
}
function projection(aspect) {
  const f=1/Math.tan(.64/2), near=.01, far=100;
  return new Float32Array([f/aspect,0,0,0,0,f,0,0,0,0,(far+near)/(near-far),-1,0,0,2*far*near/(near-far),0]);
}
const vertexSource = `
attribute vec3 a_position;
attribute vec3 a_rgb;
attribute float a_region;
uniform mat4 u_matrix;
uniform float u_size, u_selected, u_highlight;
uniform mediump float u_line;
uniform vec3 u_tint;
varying mediump vec3 v_color;
void main(){
  gl_Position=u_matrix*vec4(a_position,1.);
  gl_PointSize=u_size;
  vec3 rgb=max(a_rgb,vec3(.09));
  float changed=step(.5,a_region);
  float chosen=1.-step(.5,abs(a_region-u_selected));
  float active=mix(chosen,changed,1.-step(.5,u_selected));
  vec3 context=mix(rgb*.65,rgb*.55,step(.5,u_selected));
  v_color=mix(rgb,mix(context,u_tint,active),u_highlight);
  if(u_line>.5) v_color=vec3(.76,.91,.69);
}`;
const fragmentSource = `
precision mediump float;
uniform mediump float u_line;
varying mediump vec3 v_color;
void main(){
  if(u_line<.5 && distance(gl_PointCoord,vec2(.5))>.5) discard;
  gl_FragColor=vec4(v_color,1.);
}`;
function makeRenderer(canvas, bytes, display) {
  const gl=canvas.getContext('webgl',{antialias:true,alpha:false,preserveDrawingBuffer:true});
  if(!gl) throw new Error('이 브라우저에서 WebGL을 사용할 수 없습니다.');
  if(bytes.byteLength!==display.count*20) throw new Error('점군 파일 크기가 올바르지 않습니다.');
  function shader(type, text) {
    const s=gl.createShader(type); gl.shaderSource(s,text); gl.compileShader(s);
    if(!gl.getShaderParameter(s,gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }
  const program=gl.createProgram();
  gl.attachShader(program,shader(gl.VERTEX_SHADER,vertexSource)); gl.attachShader(program,shader(gl.FRAGMENT_SHADER,fragmentSource)); gl.linkProgram(program);
  if(!gl.getProgramParameter(program,gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  const pos=gl.getAttribLocation(program,'a_position'), rgb=gl.getAttribLocation(program,'a_rgb'), region=gl.getAttribLocation(program,'a_region');
  const uniforms=Object.fromEntries(['matrix','size','selected','highlight','line','tint'].map(n=>[n,gl.getUniformLocation(program,'u_'+n)]));
  const buffer=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,buffer); gl.bufferData(gl.ARRAY_BUFFER,bytes,gl.STATIC_DRAW);
  const box=gl.createBuffer();
  let boxCount=0;
  const renderer={
    setBox(bounds) {
      if(!bounds){boxCount=0;return;}
      const min=sub(bounds.min,evidence.display_origin), max=sub(bounds.max,evidence.display_origin), corners=[];
      for(let i=0;i<8;i++) corners.push([i&1?max[0]:min[0],i&2?max[1]:min[1],i&4?max[2]:min[2]]);
      const points=[];
      for(let i=0;i<8;i++) for(const bit of [1,2,4]) if(!(i&bit)) points.push(...corners[i],...corners[i|bit]);
      gl.bindBuffer(gl.ARRAY_BUFFER,box);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(points),gl.STATIC_DRAW);boxCount=points.length/3;
    },
    draw() {
      const ratio=Math.min(window.devicePixelRatio||1,2);
      const width=Math.max(1,Math.round(canvas.clientWidth*ratio)), height=Math.max(1,Math.round(canvas.clientHeight*ratio));
      if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}
      gl.viewport(0,0,width,height);gl.clearColor(.067,.096,.104,1);gl.enable(gl.DEPTH_TEST);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.useProgram(program);
      const matrix=multiply(projection(width/height),viewMatrix());
      gl.uniformMatrix4fv(uniforms.matrix,false,matrix);
      gl.uniform1f(uniforms.size,2.2*ratio);
      gl.uniform1f(uniforms.selected,selected?.id||0);gl.uniform1f(uniforms.highlight,$('highlight').checked?1:0);
      gl.uniform3fv(uniforms.tint,display.epoch===0?[1,.59,.31]:[.32,.73,1]);
      gl.uniform1f(uniforms.line,0);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);
      gl.enableVertexAttribArray(pos);gl.vertexAttribPointer(pos,3,gl.FLOAT,false,20,0);
      gl.enableVertexAttribArray(rgb);gl.vertexAttribPointer(rgb,3,gl.UNSIGNED_BYTE,true,20,12);
      gl.enableVertexAttribArray(region);gl.vertexAttribPointer(region,1,gl.UNSIGNED_BYTE,false,20,15);
      gl.drawArrays(gl.POINTS,0,display.count);
      if(boxCount) {
        gl.uniform1f(uniforms.line,1);gl.disableVertexAttribArray(rgb);gl.disableVertexAttribArray(region);
        gl.vertexAttrib3f(rgb,1,1,1);gl.vertexAttrib1f(region,0);
        gl.bindBuffer(gl.ARRAY_BUFFER,box);gl.vertexAttribPointer(pos,3,gl.FLOAT,false,12,0);
        gl.drawArrays(gl.LINES,0,boxCount);
      }
      updateChangeLabels(display.epoch,matrix,canvas);
    }
  };
  let dragging=null;
  canvas.addEventListener('pointerdown',e=>{if(e.button!==0)return;dragging={x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId);canvas.focus();});
  canvas.addEventListener('pointermove',e=>{
    if(!dragging)return;
    camera.theta-=(e.clientX-dragging.x)*.006;camera.phi=clamp(camera.phi+(e.clientY-dragging.y)*.006,-1.4,1.54);
    dragging={x:e.clientX,y:e.clientY};draw();
  });
  for(const event of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(event,()=>dragging=null);
  canvas.addEventListener('wheel',e=>{e.preventDefault();camera.radius=clamp(camera.radius*Math.exp(e.deltaY*.001),.2,30);draw();},{passive:false});
  canvas.addEventListener('keydown',e=>{
    const key=e.key;
    if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','=','-','r','R'].includes(key))return;
    e.preventDefault();
    if(key==='ArrowLeft')camera.theta+=.13;
    if(key==='ArrowRight')camera.theta-=.13;
    if(key==='ArrowUp')camera.phi=clamp(camera.phi+.1,-1.4,1.54);
    if(key==='ArrowDown')camera.phi=clamp(camera.phi-.1,-1.4,1.54);
    if(key==='+'||key==='=')camera.radius=clamp(camera.radius*.85,.2,30);
    if(key==='-')camera.radius=clamp(camera.radius/ .85,.2,30);
    if(key.toLowerCase()==='r'){overview();return;}
    draw();
  });
  return renderer;
}
function draw(){for(const renderer of renderers)renderer.draw();}
function focus(bounds) {
  camera.target=bounds.min.map((v,i)=>(v+bounds.max[i])/2-evidence.display_origin[i]);
  camera.radius=Math.max(.7,Math.hypot(...sub(bounds.max,bounds.min))*1.8);
}
function overview(){
  selected=null;focus(evidence.bounds);
  camera.theta=-1.12;camera.phi=.72;
  renderers.forEach(r=>r.setBox(null));
  document.querySelectorAll('.region').forEach(b=>b.setAttribute('aria-pressed','false'));
  $('evidence-detail').replaceChildren();
  const p=document.createElement('p');p.className='detail-note';p.textContent='제거·추가 후보 라벨이 붙은 영역을 선택하세요. 객체 대응 정보가 연결되면 이동 거리도 이곳에 표시합니다.';
  $('evidence-detail').append(p);draw();
}
function selectRegion(region) {
  selected=region;
  const bounds=[boundsFor(region,0),boundsFor(region,1)];
  focus({min:bounds[0].min.map((v,i)=>Math.min(v,bounds[1].min[i])),max:bounds[0].max.map((v,i)=>Math.max(v,bounds[1].max[i]))});
  renderers.forEach((r,i)=>r.setBox(bounds[i]));
  document.querySelectorAll('.region').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.region)===region.id)));
  const d=region.distances, side=region.source_epoch===0?'이전':'이후', opposite=region.source_epoch===0?'이후':'이전', info=changeInfo(region);
  $('evidence-detail').innerHTML=`
    <h3>영역 ${region.id} · ${info.label}</h3>
    <div class="event-description">${info.type==='removed'?'이전에 있던 표면의 제거 후보':info.type==='added'?'이후에 나타난 표면의 추가 후보':info.type==='moved'?'전후 위치가 연결된 이동 결과':'변화 종류 정보 없음'}</div>
    <span class="status-chip">최종 판정 · 추가 검증 필요</span>
    <div class="movement-distance"><span>객체 이동 거리</span><strong>${info.movement===null?'산정 불가':info.movement.toFixed(2)+' m'}</strong><small>${info.type==='moved'&&info.movement!==null?'전후 동일 객체 대응 결과의 미터 단위 거리':'전후 동일 객체의 대응 정보가 필요합니다.'}</small></div>
    <dl class="metrics"><div><dt>표면 간 거리 중앙값</dt><dd>${distance(d.median)}</dd></div><div><dt>임계값 초과 표본</dt><dd>${percent(d.above_threshold_fraction)}</dd></div><div><dt>표면 간 거리 90백분위</dt><dd>${distance(d.p90)}</dd></div><div><dt>측정 표본 수</dt><dd>${count(region.measured_point_count)}</dd></div></dl>
    <p class="detail-note">${side} 영역 ${count(region.point_count)}점 중 공간 표본 ${count(region.measured_point_count)}점에서 ${opposite} 전체 점군의 최근접 점까지 측정했습니다. 통계는 이 표본에 대한 값입니다. 거리 단위는 원본 좌표 단위이며 임계값은 ${evidence.settings.distance_threshold}입니다.</p>
    <ul class="checks"><li><span>기하 차이</span><b class="${region.geometric_check==='supported'?'ok':''}">${region.geometric_check==='supported'?'차이 근거 있음':'겹치는 표면 있음'}</b></li><li><span>동일 객체의 이동</span><b>${info.type==='moved'?'대응 결과 입력':'객체 대응 없음'}</b></li><li><span>가림 · 실제 빈 공간</span><b>깊이·카메라 없음</b></li><li><span>전후 키프레임</span><b>원본 영상 없음</b></li></ul>
    <details class="refs"><summary>원본 포인트 근거 확인</summary><p>PLY의 원본 레코드 순서(0부터)입니다. 상대 점은 최근접 표면점이며, 같은 객체의 대응점이라는 뜻은 아닙니다.</p><table><thead><tr><th>${side} 점 번호</th><th>${opposite} 점 번호</th><th>거리</th></tr></thead><tbody>${region.examples.map(p=>`<tr><td>${p.point_index}</td><td>${p.opposite_point_index}</td><td>${distance(p.distance)}</td></tr>`).join('')}</tbody></table></details>
    <a class="download-ref" href="${region.point_refs.url}" download="region-${region.id}-point-evidence.bin">표본 포인트·거리 근거 ↓</a><br><a class="download-ref" href="${region.all_point_indices.url}" download="region-${region.id}-indices.bin">영역 전체 원본 점 번호 ↓</a>`;
  draw();
}
async function load() {
  try {
    const response=await fetch('/evidence/evidence.json');
    if(!response.ok)throw new Error('비교 데이터가 없습니다. tools/build_change_evidence_demo.py로 근거를 먼저 생성해 주세요.');
    evidence=await response.json();
    if(evidence.kind!=='change_evidence_demo'||evidence.version!==1)throw new Error('지원하지 않는 근거 형식입니다.');
    const buffers=await Promise.all(evidence.displays.map(async d=>{
      const r=await fetch(d.url);if(!r.ok)throw new Error('점군 파일을 불러오지 못했습니다.');return r.arrayBuffer();
    }));
    renderers=evidence.displays.map((d,i)=>makeRenderer($(d.epoch===0?'before-cloud':'after-cloud'),buffers[i],d));
    evidence.sources.forEach(s=>$(s.epoch===0?'before-count':'after-count').textContent=count(s.point_count)+'점 · 원본');
    $('region-count').textContent=evidence.regions.length+'개';
    $('removed-summary').textContent='제거 후보 · '+evidence.regions.filter(r=>changeInfo(r).type==='removed').length+'개';
    $('added-summary').textContent='추가 후보 · '+evidence.regions.filter(r=>changeInfo(r).type==='added').length+'개';
    const moved=evidence.regions.filter(r=>changeInfo(r).type==='moved');
    $('moved-summary').textContent=moved.length?'이동 · '+moved.length+'개':'이동 · 대응 정보 없음';
    evidence.regions.forEach(r=>{
      const b=document.createElement('button');b.className='region';b.dataset.region=r.id;b.setAttribute('aria-pressed','false');
      const info=changeInfo(r);
      b.innerHTML=`<span class="region-line"><strong>영역 ${r.id}</strong><span class="change-badge ${info.type}">${info.label}</span></span><small>${r.source_epoch===0?'이전':'이후'} · ${count(r.point_count)}점 · 표면 간 거리 ${distance(r.distances.median)}</small>`;
      b.addEventListener('click',()=>selectRegion(r));$('regions').append(b);
    });
    const excluded=evidence.sources.reduce((n,s)=>n+s.excluded_small_region_count,0);
    $('provenance').innerHTML=`<p><a href="https://github.com/3DOM-FBK/multichange3d" target="_blank" rel="noreferrer">MultiChange3D 공식 데이터</a> · Open Space RGB-D · Epoch 0–1. 실제 공간의 재촬영이며, 저자가 제공한 공통 좌표계를 사용했습니다.</p><p>변화 입력: 공식 cd_type 라벨. 이 화면은 탐지 모델의 예측이나 정확도 평가를 보여주지 않습니다. 변화 영역은 연결된 표면을 묶은 결과이며 객체 분할 결과가 아닙니다.</p><p>거리 계산: 영역마다 ${evidence.distance_sampling.voxel} 단위 복셀의 원본 대표점, 최대 ${count(evidence.distance_sampling.maximum_points_per_region)}개를 반대 회차 전체 원본 점군과 비교. 모든 통계는 측정 표본에 한정됩니다.</p><p>화면: 원본 포인트 중 ${evidence.settings.render_voxel} 단위 복셀 대표점 ${evidence.displays.map(d=>count(d.count)).join(' / ')}점. 영역 묶음: ${evidence.settings.region_voxel} 단위 · 최소 ${count(evidence.settings.minimum_region_points)}점. 작은 영역 ${count(excluded)}점은 제외했습니다.</p><p>변화 없음으로 라벨된 표면의 거리 중앙값: ${evidence.unchanged_controls.map(c=>distance(c.distances.median)).join(' / ')}. 참고 진단값이며 정합 검증 완료를 뜻하지 않습니다.</p><p>원본 SHA-256 (파일 출처 확인)</p>${evidence.sources.map(s=>`<p>Epoch ${s.epoch}: <code>${s.sha256}</code></p>`).join('')}`;
    $('load-status').textContent='비교 준비 완료 · 원본 '+count(evidence.sources.reduce((n,s)=>n+s.point_count,0))+'점 · 공식 변화 라벨 기반 근거';
    $('highlight').addEventListener('change',draw);$('overview').addEventListener('click',overview);
    $('top-view').addEventListener('click',()=>{camera.phi=1.53;draw();});
    $('zoom-in').addEventListener('click',()=>{camera.radius=clamp(camera.radius*.8,.2,30);draw();});
    $('zoom-out').addEventListener('click',()=>{camera.radius=clamp(camera.radius/ .8,.2,30);draw();});
    new ResizeObserver(draw).observe(document.querySelector('.viewports'));
    overview();
  } catch(error) {$('load-status').classList.add('error');$('load-status').textContent=error.message;console.error(error);}
}
load();

const $=id=>document.getElementById(id);
const count=n=>Number(n).toLocaleString('ko-KR');
const changeTitle={moved:'이동',added:'추가',removed:'소실'};
const clamp=(n,a,b)=>Math.max(a,Math.min(b,n));
let data, selected=null, renderers=[], photoLists=[[],[]], photoIndices=[0,0], photoTokens=[0,0];
const camera={target:[0,0,0],radius:8,theta:-1.05,phi:.9};
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const sub=(a,b)=>a.map((v,i)=>v-b[i]);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit=a=>a.map(v=>v/Math.hypot(...a));
function multiply(a,b){const out=new Float32Array(16);for(let j=0;j<4;j++)for(let i=0;i<4;i++)for(let k=0;k<4;k++)out[j*4+i]+=a[k*4+i]*b[j*4+k];return out;}
function matrix(aspect){
  const r=camera.radius,p=camera.phi,t=camera.theta;
  const eye=camera.target.map((v,i)=>v+[r*Math.cos(p)*Math.cos(t),r*Math.cos(p)*Math.sin(t),r*Math.sin(p)][i]);
  const z=unit(sub(eye,camera.target)),x=unit(cross([0,0,1],z)),y=cross(z,x);
  const view=new Float32Array([x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1]);
  const f=1/Math.tan(.72/2),near=.01,far=100;
  const projection=new Float32Array([f/aspect,0,0,0,0,f,0,0,0,0,(far+near)/(near-far),-1,0,0,2*far*near/(near-far),0]);
  return multiply(projection,view);
}
const vertex=`
attribute vec3 a_position,a_normal,a_rgb;attribute float a_region;attribute vec4 a_instance;
uniform mat4 u_matrix;uniform float u_size,u_selected,u_mode;varying mediump vec3 v_color;
void main(){gl_Position=u_matrix*vec4(a_position,1.);gl_PointSize=u_size;
float id=a_instance.x+256.*a_instance.y;vec3 instance=.28+.65*fract(sin(vec3(id+1.,id+4.,id+9.)*vec3(12.9898,78.233,37.719))*43758.5453);
float changeType=floor(a_region/64.);float eventId=mod(a_region,64.);
float active=step(.5,a_region);if(u_selected>.5)active*=1.-step(.5,abs(eventId-u_selected));
vec3 base=max(a_rgb,vec3(.05));vec3 color=base;
vec3 tint=vec3(.75,.58,1.);if(changeType>1.5)tint=vec3(.35,.75,1.);if(changeType>2.5)tint=vec3(1.,.57,.28);
if(u_mode>.5&&u_mode<1.5)color=mix(base*.65,tint,active);
if(u_mode>1.5&&u_mode<2.5)color=instance;if(u_mode>2.5)color=vec3(.51,.66,.64);
float light=.58+.42*abs(dot(normalize(a_normal+vec3(.0001)),normalize(vec3(.4,-.6,1.))));
v_color=color*light;}`;
const fragment=`precision mediump float;uniform mediump float u_points;varying mediump vec3 v_color;void main(){if(u_points>.5&&distance(gl_PointCoord,vec2(.5))>.5)discard;gl_FragColor=vec4(v_color,1.);}`;
function makeRenderer(canvas,geometries,epoch){
  const gl=canvas.getContext('webgl',{antialias:true,alpha:false,preserveDrawingBuffer:true});
  if(!gl)throw new Error('WebGL을 사용할 수 없습니다.');
  function shader(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;}
  const program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,vertex));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,fragment));gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(program));
  const attrs=Object.fromEntries(['position','normal','rgb','region','instance'].map(n=>[n,gl.getAttribLocation(program,'a_'+n)]));
  const u=Object.fromEntries(['matrix','size','selected','mode','points'].map(n=>[n,gl.getUniformLocation(program,'u_'+n)]));
  const buffers={};
  for(const [kind,item]of Object.entries(geometries)){
    if(item.bytes.byteLength!==item.meta.count*32)throw new Error('메쉬/점군 파일 크기가 올바르지 않습니다.');
    const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,item.bytes,gl.STATIC_DRAW);buffers[kind]={buffer,count:item.meta.count};
  }
  const lines=gl.createBuffer();let lineCount=0,lineReady=false;
  const box=gl.createBuffer();
  function simplePositions(buffer,positions){gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(positions),gl.STATIC_DRAW);}
  function bindSimple(buffer){gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.enableVertexAttribArray(attrs.position);gl.vertexAttribPointer(attrs.position,3,gl.FLOAT,false,12,0);for(const n of ['normal','rgb','region','instance'])gl.disableVertexAttribArray(attrs[n]);gl.vertexAttrib3f(attrs.normal,0,0,1);gl.vertexAttrib3f(attrs.rgb,.8,1,.8);gl.vertexAttrib1f(attrs.region,0);gl.vertexAttrib4f(attrs.instance,0,0,0,0);}
  function draw(){
    const ratio=Math.min(devicePixelRatio||1,2),w=Math.max(1,Math.round(canvas.clientWidth*ratio)),h=Math.max(1,Math.round(canvas.clientHeight*ratio));
    if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
    gl.viewport(0,0,w,h);gl.enable(gl.DEPTH_TEST);gl.clearColor(.065,.092,.099,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.useProgram(program);
    gl.uniformMatrix4fv(u.matrix,false,matrix(w/h));gl.uniform1f(u.size,2.5*ratio);gl.uniform1f(u.selected,selected?.id||0);gl.uniform1f(u.mode,Number($('color').value));
    const kind=$('geometry').value,item=buffers[kind];gl.uniform1f(u.points,kind==='points'?1:0);gl.bindBuffer(gl.ARRAY_BUFFER,item.buffer);
    for(const attr of Object.values(attrs))gl.enableVertexAttribArray(attr);
    gl.vertexAttribPointer(attrs.position,3,gl.FLOAT,false,32,0);gl.vertexAttribPointer(attrs.normal,3,gl.FLOAT,false,32,12);gl.vertexAttribPointer(attrs.rgb,3,gl.UNSIGNED_BYTE,true,32,24);gl.vertexAttribPointer(attrs.region,1,gl.UNSIGNED_BYTE,false,32,27);gl.vertexAttribPointer(attrs.instance,4,gl.UNSIGNED_BYTE,false,32,28);
    gl.drawArrays(kind==='mesh'?gl.TRIANGLES:gl.POINTS,0,item.count);
    if($('wire').checked&&kind==='mesh'){
      if(!lineReady){const view=new DataView(geometries.mesh.bytes),positions=[];for(let i=0;i<item.count;i+=3)for(const [a,b]of [[0,1],[1,2],[2,0]])for(const v of [a,b])for(let k=0;k<3;k++)positions.push(view.getFloat32((i+v)*32+k*4,true));simplePositions(lines,positions);lineCount=positions.length/3;lineReady=true;}
      bindSimple(lines);gl.uniform1f(u.mode,3);gl.uniform1f(u.points,0);gl.drawArrays(gl.LINES,0,lineCount);
    }
    if(selected){
      const bounds=(epoch===0?selected.before_bounds:selected.after_bounds)||selected.bounds;
      const min=sub(bounds.min,data.display_origin),max=sub(bounds.max,data.display_origin),corners=[],positions=[];
      for(let i=0;i<8;i++)corners.push([i&1?max[0]:min[0],i&2?max[1]:min[1],i&4?max[2]:min[2]]);
      for(let i=0;i<8;i++)for(const bit of [1,2,4])if(!(i&bit))positions.push(...corners[i],...corners[i|bit]);
      simplePositions(box,positions);bindSimple(box);gl.uniform1f(u.mode,0);gl.uniform1f(u.points,0);gl.drawArrays(gl.LINES,0,24);
      if(selected.type==='moved'&&selected.motion){
        const display=p=>data.display_rotation.map(row=>dot(row,p)).map((v,i)=>v-data.display_origin[i]);
        const a=display(selected.motion.before_reference_m),b=display(selected.motion.after_reference_m),direction=unit(sub(a,b));
        const side=unit(cross(direction,[0,0,1]));const length=Math.min(.09,Math.hypot(...sub(a,b))*.3);
        const head1=b.map((v,i)=>v+direction[i]*length+side[i]*length*.45),head2=b.map((v,i)=>v+direction[i]*length-side[i]*length*.45);
        simplePositions(box,[...a,...b,...b,...head1,...b,...head2]);bindSimple(box);gl.vertexAttrib3f(attrs.rgb,.83,.65,1.);gl.disable(gl.DEPTH_TEST);gl.drawArrays(gl.LINES,0,6);gl.enable(gl.DEPTH_TEST);
      }
    }
    const meta=data.displays[epoch][kind];$('map-info-'+epoch).textContent=kind==='mesh'?count(meta.triangle_count)+' 삼각형':count(meta.count)+' 원본 점';
    const emptyLocation=selected&&((selected.type==='added'&&epoch===0)||(selected.type==='removed'&&epoch===1));
    $('map-label-'+epoch).className='map-label '+(selected?.type||'');
    $('map-label-'+epoch).textContent=selected?selected.name+' · '+(emptyLocation?'같은 위치':changeTitle[selected.type]+(selected.motion?' '+selected.motion.displacement_m.toFixed(2)+' m':'')):'이동 · 추가 · 소실';
  }
  let drag=null;
  canvas.addEventListener('pointerdown',e=>{if(e.button!==0)return;drag={x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId);canvas.focus();});
  canvas.addEventListener('pointermove',e=>{if(!drag)return;camera.theta-=(e.clientX-drag.x)*.006;camera.phi=clamp(camera.phi+(e.clientY-drag.y)*.006,-1.35,1.54);drag={x:e.clientX,y:e.clientY};drawAll();});
  for(const event of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(event,()=>drag=null);
  canvas.addEventListener('wheel',e=>{e.preventDefault();camera.radius=clamp(camera.radius*Math.exp(e.deltaY*.001),.3,30);drawAll();},{passive:false});
  canvas.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','=','-','r','R'].includes(e.key))return;e.preventDefault();if(e.key==='ArrowLeft')camera.theta+=.12;if(e.key==='ArrowRight')camera.theta-=.12;if(e.key==='ArrowUp')camera.phi=clamp(camera.phi+.1,-1.35,1.54);if(e.key==='ArrowDown')camera.phi=clamp(camera.phi-.1,-1.35,1.54);if(['+','='].includes(e.key))camera.radius=clamp(camera.radius*.85,.3,30);if(e.key==='-')camera.radius=clamp(camera.radius/.85,.3,30);if(e.key.toLowerCase()==='r')overview();else drawAll();});
  canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();$('status').textContent='3D 화면 연결이 끊겼습니다. 페이지를 새로 고쳐주세요.';});
  return{draw};
}
function drawAll(){for(const r of renderers)r.draw();}
function overview(){selected=null;camera.target=[0,0,0];camera.radius=Math.max(...data.bounds.max.map((v,i)=>v-data.bounds.min[i]))*1.12;camera.phi=.95;renderList();showDetail();showPhotos();drawAll();}
function choose(region){selected=region;camera.target=region.bounds.min.map((v,i)=>(v+region.bounds.max[i])/2-data.display_origin[i]);camera.radius=clamp(Math.max(...region.bounds.max.map((v,i)=>v-region.bounds.min[i]))*2.1,1.5,10);renderList();showDetail();showPhotos();drawAll();}
function renderList(){
  const list=$('regions');list.replaceChildren();const filter=$('change-filter').value;const visible=data.regions.filter(r=>filter==='all'||r.type===filter);$('region-count').textContent=visible.length+'개';
  if(!visible.length){const p=document.createElement('p');p.className='hint';p.textContent='이 전후 쌍에서 '+(changeTitle[filter]||'변화')+'로 분류된 객체가 없습니다.';list.append(p);}
  for(const r of visible){const b=document.createElement('button');b.className='region';b.setAttribute('aria-pressed',String(selected?.id===r.id));
    b.innerHTML='<span class="row"><strong>'+r.name+'</strong><span class="chip '+r.type+'">'+changeTitle[r.type]+'</span></span><small>'+(r.motion?'이동 '+r.motion.displacement_m.toFixed(2)+' m · 회전 '+r.motion.rotation_deg.toFixed(0)+'°':'전후 객체 유무 변화')+'</small><small>'+(r.paired_photos?'전후 사진 연결':'일부 사진 연결 없음')+'</small>';
    b.addEventListener('click',()=>choose(r));list.append(b);}
}
function showDetail(){
  const el=$('detail');if(!selected){el.innerHTML='<p class="detail-note">객체를 선택하면 전후 위치와 사진을 볼 수 있습니다.</p>';return;}
  const r=selected;
  el.innerHTML='<h3>'+r.name+' · '+changeTitle[r.type]+'</h3>'+(r.motion?'<div class="detail-metrics"><div><small>이동 거리</small><strong>'+r.motion.displacement_m.toFixed(2)+' m</strong></div><div><small>회전</small><strong>'+r.motion.rotation_deg.toFixed(0)+'°</strong></div></div>':'<p class="detail-note">'+(r.type==='added'?'이후 촬영에서 나타난 객체입니다.':'이전 촬영에만 관측된 객체입니다.')+'</p>')+'<p class="detail-note">'+(r.paired_photos?'아래 원본 사진에서 같은 객체의 전후 모습을 확인할 수 있습니다.':'연결할 수 있는 원본 사진만 표시합니다.')+'</p>';
}
function showPhotos(){
  $('photo-context').textContent=selected?selected.name+'의 전후 모습':'원본 영상에서 추출한 정지 프레임';
  for(let epoch=0;epoch<2;epoch++){
    $('photo-overlay-'+epoch).setAttribute('class',selected?.type||'');
    photoLists[epoch]=selected?selected.keyframes[epoch].map(link=>({frame:data.keyframes.find(f=>f.id===link.frame_id),link})):data.keyframes.filter(f=>f.epoch===epoch).map(frame=>({frame,link:null}));
    photoIndices[epoch]=0;showPhoto(epoch,false);
  }
}
async function showPhoto(epoch,animate){
  const token=++photoTokens[epoch],list=photoLists[epoch],item=list[photoIndices[epoch]],img=$('photo-'+epoch),overlay=$('photo-overlay-'+epoch),check=$('photo-check-'+epoch);
  overlay.replaceChildren();$('no-photo-'+epoch).hidden=!!item;img.hidden=!item;
  for(const button of document.querySelectorAll('.photo-nav button[data-epoch="'+epoch+'"]'))button.disabled=!item||(Number(button.dataset.step)<0?photoIndices[epoch]===0:photoIndices[epoch]===list.length-1);
  if(!item){check.textContent='연결 없음';check.className='';$('photo-caption-'+epoch).textContent='이 위치의 대응 관측을 확인할 수 없습니다.';$('photo-index-'+epoch).textContent='0 / 0';return;}
  const {frame,link}=item;const loaded=new Image();loaded.src=frame.url;
  try{await loaded.decode();}catch{if(token===photoTokens[epoch])$('photo-caption-'+epoch).textContent='원본 키프레임을 불러오지 못했습니다.';return;}
  if(token!==photoTokens[epoch])return;
  img.src=loaded.src;img.alt=(epoch===0?'이전':'이후')+' 원본 촬영, 프레임 '+frame.source_frame_index;
  if(animate&&!matchMedia('(prefers-reduced-motion: reduce)').matches)img.animate([{opacity:.45,transform:'scale(1.025)'},{opacity:1,transform:'scale(1)'}],{duration:240,easing:'ease-out'});
  const checked=frame.status==='estimated_rgb_checked';check.textContent=checked?'객체 위치 표시':'연결 사진 · 추정';check.className=checked?'checked':'';
  const emptyLocation=selected&&((selected.type==='added'&&epoch===0)||(selected.type==='removed'&&epoch===1));
  $('photo-caption-'+epoch).textContent=(emptyLocation?(selected.type==='added'?'추가 전 같은 위치':'소실 후 같은 위치')+' · ':'')+'원본 프레임 '+frame.source_frame_index+' · '+frame.time_s.toFixed(2)+'초';
  $('photo-index-'+epoch).textContent=(photoIndices[epoch]+1)+' / '+list.length;
  if(link?.bbox_px&&checked){const [x0,y0,x1,y1]=link.bbox_px,rect=document.createElementNS('http://www.w3.org/2000/svg','rect');for(const [key,v]of Object.entries({x:x0,y:y0,width:x1-x0,height:y1-y0}))rect.setAttribute(key,String(v));overlay.append(rect);const text=document.createElementNS('http://www.w3.org/2000/svg','text');text.setAttribute('x',String(x0+5));text.setAttribute('y',String(Math.max(16,y0-6)));text.textContent=selected.name;overlay.append(text);}
}
function movePhoto(epoch,step){const next=clamp(photoIndices[epoch]+step,0,photoLists[epoch].length-1);if(next===photoIndices[epoch])return;photoIndices[epoch]=next;showPhoto(epoch,true);}
function fillInformation(){
  $('capture-pair').textContent='REAL RECAPTURE / '+data.scene.toUpperCase().replace('_',' ')+' / CAPTURES '+data.source_epochs.join(' → ');
  for(let epoch=0;epoch<2;epoch++)$('capture-'+epoch).textContent='Capture '+data.source_epochs[epoch];
  const summary=$('change-summary');summary.replaceChildren();
  for(const type of ['moved','added','removed']){const button=document.createElement('button');button.className='change-badge '+type;button.textContent=changeTitle[type]+' '+data.change_summary[type]+'개';button.addEventListener('click',()=>{$('change-filter').value=type;overview();});summary.append(button);}
  const d=data.displays[0];$('mesh-example').innerHTML='<p>이전 지도: 원본 '+count(d.source_points)+'점 → 복원 정점 '+count(d.full_mesh_vertices)+'개 / 삼각형 '+count(d.full_mesh_triangles)+'개. 화면에는 '+count(d.mesh.triangle_count)+'개 삼각형을 표시합니다.</p><p><code>V = [x, y, z]</code>는 정점 좌표, <code>F = [i, j, k]</code>는 삼각형을 이루는 정점 번호입니다.</p>';
}
async function load(){
  try{
    const response=await fetch('/rescan/changes.json');if(!response.ok)throw new Error('Rescan 변화 결과를 불러오지 못했습니다.');data=await response.json();
    if(data.kind!=='rescan_object_changes'||!data.displays||!data.regions)throw new Error('객체 변화 결과가 아직 준비되지 않았습니다.');
    fillInformation();const all=await Promise.all(data.displays.map(async display=>{const geometries={};await Promise.all(['mesh','points'].map(async kind=>{const r=await fetch(display[kind].url);if(!r.ok)throw new Error('3D 파일을 불러오지 못했습니다.');geometries[kind]={bytes:await r.arrayBuffer(),meta:display[kind]};}));return geometries;}));
    renderers=all.map((geometry,epoch)=>makeRenderer($('map-'+epoch),geometry,epoch));overview();
    const first=data.regions.find(r=>r.type==='added'&&r.paired_photos)||[...data.regions].sort((a,b)=>(b.motion?.displacement_m||0)-(a.motion?.displacement_m||0))[0];if(first)choose(first);
    $('status').textContent='Rescan 실제 전후 한 쌍 · 이동 '+data.change_summary.moved+'개 · 추가 '+data.change_summary.added+'개 · 소실 '+data.change_summary.removed+'개';
    window.addEventListener('resize',drawAll);
  }catch(error){$('status').textContent=error.message;$('status').classList.add('error');}
}
$('geometry').addEventListener('change',()=>{$('wire').disabled=$('geometry').value==='points';drawAll();});$('color').addEventListener('change',drawAll);$('wire').addEventListener('change',drawAll);$('overview').addEventListener('click',overview);$('top').addEventListener('click',()=>{camera.phi=1.52;drawAll();});$('change-filter').addEventListener('change',overview);
for(const button of document.querySelectorAll('.photo-nav button'))button.addEventListener('click',()=>movePhoto(Number(button.dataset.epoch),Number(button.dataset.step)));
for(let epoch=0;epoch<2;epoch++)$('photo-panel-'+epoch).addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight'].includes(e.key))return;e.preventDefault();movePhoto(epoch,e.key==='ArrowLeft'?-1:1);});
load();

// Structure registration inspection. Colors describe epochs, never change labels.
const $=id=>document.getElementById(id),clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const sub=(a,b)=>a.map((x,i)=>x-b[i]),dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit=a=>a.map(x=>x/Math.max(1e-10,Math.hypot(...a)));
const columnMajor=m=>new Float32Array(m[0].map((_,i)=>m.map(row=>row[i])).flat());
const pct=x=>(100*x).toFixed(1)+'%',u=x=>x==null?'—':x.toFixed(3)+' u';
function el(tag,text,cls){const e=document.createElement(tag);if(text!=null)e.textContent=text;if(cls)e.className=cls;return e;}
async function get(url,binary=false){const r=await fetch(url);if(!r.ok)throw new Error('정합 파일을 불러오지 못했습니다: '+url);return binary?r.arrayBuffer():r.json();}

export async function mountDormAlignment(root){
  document.body.classList.add('alignment-page');document.title='Dorm2 · Dorm3 정합 겹쳐 보기';
  document.querySelector('header>span').textContent='Dorm2 / Dorm3 · 정합 확인';
  const main=document.querySelector('main');main.replaceChildren();
  const intro=el('div',null,'intro'),heading=el('div');heading.append(el('h1','두 메시를 겹쳐서 정합 확인'),el('p','주황 Dorm2 · 청록 Dorm3 · 같은 화면에서 바닥과 벽의 어긋남 확인'));intro.append(heading);main.append(intro);
  const status=el('p','정합 메시와 구조면 표본을 불러오는 중…');status.id='status';main.append(status);
  try{
    const data=await get(root+'alignment-view.json');
    if(data.kind!=='dorm_structural_alignment_view'||data.units!=='reference_units'||data.labels_used!==false)throw new Error('정합 자료 형식이 다릅니다.');
    const warning=el('p',data.identity_note,'alignment-warning');warning.id='identity-warning';main.append(warning);
    const controls=el('div',null,'toolbar alignment-controls');
    function select(id,label,options){const l=el('label',label+' '),s=el('select');s.id=id;for(const [value,text]of options){const o=el('option',text);o.value=value;s.append(o);}l.append(s);controls.append(l);return s;}
    const caseSelect=select('alignment-case','정합',[['aligned','다시 맞춘 정합'],['previous','이전 임시 정합'],['raw','좌표 보정 전']]);
    const surface=select('alignment-surface','표면',[['all','전체 메시'],['structure','바닥 + 벽'],['floor','바닥만'],['wall','벽만']]);
    const zoneSelect=select('alignment-zone','위치',data.zones.map(z=>[z.id,z.title]));
    for(const [id,text]of [['show-dorm2','Dorm2'],['show-dorm3','Dorm3'],['clip-ceiling','상부 가리기']]){const label=el('label',text),box=el('input');box.type='checkbox';box.id=id;box.checked=true;label.prepend(box);controls.append(label);}
    const sliderLabel=el('label','겹침 비율 '),slider=el('input');slider.id='alignment-blend';slider.type='range';slider.min='0';slider.max='100';slider.value='50';sliderLabel.append(slider);controls.append(sliderLabel);
    for(const [id,label]of [['alignment-overview','전체'],['alignment-top','위에서'],['alignment-oblique','비스듬히']]){const button=el('button',label);button.id=id;controls.append(button);}
    main.append(controls);
    const summary=el('section',null,'alignment-summary');summary.id='alignment-summary';main.append(summary);
    const wrap=el('div',null,'alignment-canvas-wrap'),canvas=el('canvas');canvas.id='alignment-map';canvas.tabIndex=0;canvas.setAttribute('aria-label','Dorm2 주황 메시와 Dorm3 청록 메시를 겹친 정합 지도. 드래그 회전, Shift 드래그 이동, 휠 확대.');wrap.append(canvas);
    const tag=el('span','드래그 회전 · Shift 이동 · 휠 확대','view-tag');wrap.append(tag);main.append(wrap);
    main.append(el('p','주황과 청록의 윤곽이 벌어질수록 불일치가 큽니다. 밝게 섞이는 면은 화면상 겹침입니다. 위 수치는 전체 구조면 표본 기준이며, 위치 선택은 지도와 사진의 관찰 위치를 바꿉니다.','alignment-hint'));
    const evidence=el('section',null,'alignment-evidence');evidence.append(el('h2','선택 위치의 원본 사진'));
    const photos=el('div',null,'photos');
    for(let e=0;e<2;e++){const fig=el('figure',null,'photo'),head=el('div','Dorm'+(e+2),'photo-heading'),img=el('img');img.id='alignment-photo-'+e;img.alt='Dorm'+(e+2)+' 원본 파노라마';const box=el('div',null,'photo-wrap'),svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.id='alignment-photo-box-'+e;box.append(img,svg);const caption=el('figcaption');caption.id='alignment-caption-'+e;fig.append(head,box,caption);photos.append(fig);}evidence.append(photos,el('p','사진 점선은 추정 카메라로 투영한 선택 구조 구역입니다. 사진 대응만으로 동일 장소가 확인된 것은 아닙니다.'));main.append(evidence);
    const details=el('details'),title=el('summary','정합 계산·원본 연결 기록');details.append(title,el('p',data.method_note),el('p',data.metric_note));
    const table=el('table'),thead=el('tr');for(const t of ['원본','영상·사진 연결','카메라 변환 오차','영상에서 직접 꺼낸 장면'])thead.append(el('th',t));table.append(thead);
    for(const [e,a]of data.source_audit.entries()){const row=el('tr');for(const t of [a.name,a.video_sha256_matches_input&&a.photos.every(p=>p.manifest_matches&&p.input_bundle_matches)?'기록과 일치':'확인 필요',a.source_camera_transform_max_abs_error.toExponential(2),e===0?'6초 · 안내판 5F':'6초 · 안내판 3F'])row.append(el('td',t));table.append(row);}details.append(table);main.append(details);
    const download=el('a','정합 행렬·표면별 수치 JSON ↓');download.href=root+'alignment-view.json';main.append(el('footer'));main.lastChild.append(el('span','u = Dorm2 복원 단위 · 실측 미터 축척 없음'),download);
    const buffers=await Promise.all(data.meshes.map(m=>get(root+m.file,true)));
    for(let e=0;e<2;e++)if(buffers[e].byteLength!==data.meshes[e].count*60)throw new Error('메시 자료 크기가 다릅니다.');
    const gl=canvas.getContext('webgl',{antialias:true,alpha:false,preserveDrawingBuffer:true});if(!gl)throw new Error('WebGL을 사용할 수 없습니다.');
    function program(vs,fs){const p=gl.createProgram();for(const [type,source]of [[gl.VERTEX_SHADER,vs],[gl.FRAGMENT_SHADER,fs]]){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));gl.attachShader(p,s);}gl.linkProgram(p);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p));return p;}
    const meshProgram=program(`attribute vec3 a_position,a_normal;attribute float a_kind;uniform mat4 u_camera,u_model;uniform vec3 u_origin;varying vec3 v_normal;varying float v_height,v_kind;
      void main(){vec3 p=(u_model*vec4(a_position,1.)).xyz;v_height=p.y;v_kind=a_kind;v_normal=mat3(u_model)*a_normal;gl_Position=u_camera*vec4(vec3(p.x,-p.z,p.y)-u_origin,1.);}`,
      `precision highp float;varying vec3 v_normal;varying float v_height,v_kind;uniform float u_cut,u_surface;uniform vec3 u_color;
      void main(){if(v_height>u_cut)discard;if(u_surface>.5&&v_kind<.5)discard;if(u_surface>1.5&&u_surface<2.5&&abs(v_kind-1.)>.1)discard;if(u_surface>2.5&&abs(v_kind-2.)>.1)discard;
      float light=.60+.40*abs(dot(normalize(v_normal+vec3(.00001)),normalize(vec3(.4,1.,-.5))));gl_FragColor=vec4(u_color*light,1.);}`);
    const composite=program('attribute vec2 a_xy;varying vec2 v_uv;void main(){v_uv=a_xy*.5+.5;gl_Position=vec4(a_xy,0.,1.);}',
      `precision mediump float;varying vec2 v_uv;uniform sampler2D u_first,u_second;uniform vec2 u_enabled;uniform float u_blend;
      void main(){vec4 a=texture2D(u_first,v_uv),b=texture2D(u_second,v_uv);float wa=a.a*u_enabled.x*(1.-u_blend),wb=b.a*u_enabled.y*u_blend;vec3 bg=vec3(.035,.065,.092);float sum=wa+wb;
      vec3 c=sum>.0001?(a.rgb*wa+b.rgb*wb)/sum:bg;float opacity=clamp(max(wa,wb)*1.7,0.,1.);gl_FragColor=vec4(mix(bg,c,opacity),1.);}`);
    function buffer(value){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,value,gl.STATIC_DRAW);return b;}
    const meshBuffers=buffers.map(buffer),quad=buffer(new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]));
    const attrs=Object.fromEntries(['position','normal','kind'].map(k=>[k,gl.getAttribLocation(meshProgram,'a_'+k)]));
    const uniforms=Object.fromEntries(['camera','model','origin','cut','surface','color'].map(k=>[k,gl.getUniformLocation(meshProgram,'u_'+k)]));
    const quadAttr=gl.getAttribLocation(composite,'a_xy'),cu=Object.fromEntries(['first','second','enabled','blend'].map(k=>[k,gl.getUniformLocation(composite,'u_'+k)]));
    const targets=[0,1].map(()=>({framebuffer:gl.createFramebuffer(),texture:gl.createTexture(),depth:gl.createRenderbuffer()}));
    let width=0,height=0;const camera={target:[0,0,0],radius:20,theta:-.5,phi:.80};let zone=data.zones[0];
    const active=()=>data.cases.find(c=>c.id===caseSelect.value);
    function resize(){const ratio=Math.min(devicePixelRatio||1,1.6),w=Math.max(1,Math.round(canvas.clientWidth*ratio)),h=Math.max(1,Math.round(canvas.clientHeight*ratio));if(w===width&&h===height)return;width=canvas.width=w;height=canvas.height=h;
      for(const t of targets){gl.bindTexture(gl.TEXTURE_2D,t.texture);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,w,h,0,gl.RGBA,gl.UNSIGNED_BYTE,null);gl.bindRenderbuffer(gl.RENDERBUFFER,t.depth);gl.renderbufferStorage(gl.RENDERBUFFER,gl.DEPTH_COMPONENT16,w,h);gl.bindFramebuffer(gl.FRAMEBUFFER,t.framebuffer);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,t.texture,0);gl.framebufferRenderbuffer(gl.FRAMEBUFFER,gl.DEPTH_ATTACHMENT,gl.RENDERBUFFER,t.depth);if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw new Error('겹침 렌더링 버퍼 생성 실패');}}
    function projection(){const {radius:r,phi:p,theta:t,target}=camera,eye=target.map((v,i)=>v+[r*Math.cos(p)*Math.cos(t),r*Math.cos(p)*Math.sin(t),r*Math.sin(p)][i]);
      const z=unit(sub(eye,target)),x=unit(cross([0,0,1],z)),y=cross(z,x),view=[x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1];
      const f=1/Math.tan(.70/2),n=.02,far=300,pj=[f/(width/height),0,0,0,0,f,0,0,0,0,(far+n)/(n-far),-1,0,0,2*far*n/(n-far),0],out=new Float32Array(16);
      for(let j=0;j<4;j++)for(let i=0;i<4;i++)for(let k=0;k<4;k++)out[j*4+i]+=pj[k*4+i]*view[j*4+k];return out;}
    function draw(){resize();const c=active();gl.viewport(0,0,width,height);gl.disable(gl.BLEND);gl.disable(gl.CULL_FACE);gl.enable(gl.DEPTH_TEST);
      for(let e=0;e<2;e++){gl.bindFramebuffer(gl.FRAMEBUFFER,targets[e].framebuffer);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.useProgram(meshProgram);gl.uniformMatrix4fv(uniforms.camera,false,projection());gl.uniformMatrix4fv(uniforms.model,false,columnMajor(e?c.matrix:data.identity));gl.uniform3fv(uniforms.origin,data.origin);gl.uniform1f(uniforms.cut,$('clip-ceiling').checked?data.upper_cut:1e6);gl.uniform1f(uniforms.surface,['all','structure','floor','wall'].indexOf(surface.value));gl.uniform3fv(uniforms.color,e?[.23,.87,.94]:[1.,.49,.16]);gl.bindBuffer(gl.ARRAY_BUFFER,meshBuffers[e]);
        for(const a of Object.values(attrs))gl.enableVertexAttribArray(a);gl.vertexAttribPointer(attrs.position,3,gl.FLOAT,false,60,0);gl.vertexAttribPointer(attrs.normal,3,gl.FLOAT,false,60,12);gl.vertexAttribPointer(attrs.kind,1,gl.UNSIGNED_BYTE,false,60,27);gl.drawArrays(gl.TRIANGLES,0,data.meshes[e].count);for(const a of Object.values(attrs))gl.disableVertexAttribArray(a);}
      gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.disable(gl.DEPTH_TEST);gl.useProgram(composite);gl.bindBuffer(gl.ARRAY_BUFFER,quad);gl.enableVertexAttribArray(quadAttr);gl.vertexAttribPointer(quadAttr,2,gl.FLOAT,false,8,0);
      for(let e=0;e<2;e++){gl.activeTexture(gl.TEXTURE0+e);gl.bindTexture(gl.TEXTURE_2D,targets[e].texture);}gl.uniform1i(cu.first,0);gl.uniform1i(cu.second,1);gl.uniform2f(cu.enabled,$('show-dorm2').checked?1:0,$('show-dorm3').checked?1:0);gl.uniform1f(cu.blend,Number(slider.value)/100);gl.drawArrays(gl.TRIANGLES,0,6);gl.disableVertexAttribArray(quadAttr);
      canvas.dataset.case=c.id;canvas.dataset.surface=surface.value;}
    function stats(){const c=active(),s=c.statistics;summary.replaceChildren();for(const [title,value,note]of [
      ['전체 구조면 겹침',pct(s.all.overlap),'표본 양방향 거리 ≤ '+u(data.tolerance)+' · 법선 방향 일치'],
      ['바닥의 3D 차이 중앙값',u(s.floor.distance_median),'겹침 '+pct(s.floor.overlap)+' · P95 '+u(s.floor.distance_p95)],
      ['벽의 3D 차이 중앙값',u(s.wall.distance_median),'겹침 '+pct(s.wall.overlap)+' · P95 '+u(s.wall.distance_p95)]]){const card=el('div');card.append(el('small',title),el('strong',value),el('span',note));summary.append(card);}status.dataset.state='ready';status.textContent=c.label+' · 상대 축척 '+c.scale.toFixed(4)+'배 · '+data.registration_status;}
    function focus(all=false){const bounds=all?active().bounds:zone.bounds;camera.target=bounds.min.map((v,i)=>(v+bounds.max[i])/2-data.origin[i]);const span=sub(bounds.max,bounds.min);camera.radius=Math.max(all?3:data.reference_height*5,Math.max(span[0],span[1]/Math.max(.5,width/Math.max(1,height)),span[2])*1.65);draw();}
    function showPhotos(){for(let e=0;e<2;e++){const photo=zone.photos[e],svg=$('alignment-photo-box-'+e);$('alignment-photo-'+e).src=root+photo.file;$('alignment-caption-'+e).textContent=photo.caption;svg.replaceChildren();svg.setAttribute('viewBox','0 0 1036 518');for(const b of photo.boxes||[]){const rect=document.createElementNS('http://www.w3.org/2000/svg','rect');for(const [k,v]of Object.entries({x:b[0],y:b[1],width:b[2]-b[0],height:b[3]-b[1]}))rect.setAttribute(k,v);svg.append(rect);}}}
    let drag=null;canvas.addEventListener('pointerdown',e=>{if(e.button!==0)return;drag={x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId);canvas.focus();});canvas.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;
      if(e.shiftKey){const r=[-Math.sin(camera.theta),Math.cos(camera.theta),0],up=[-Math.cos(camera.theta)*Math.sin(camera.phi),-Math.sin(camera.theta)*Math.sin(camera.phi),Math.cos(camera.phi)];camera.target=camera.target.map((v,i)=>v-r[i]*dx*camera.radius*.0015+up[i]*dy*camera.radius*.0015);}else{camera.theta-=dx*.005;camera.phi=clamp(camera.phi+dy*.005,.02,1.56);}drag={x:e.clientX,y:e.clientY};draw();});for(const name of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(name,()=>drag=null);
    canvas.addEventListener('wheel',e=>{e.preventDefault();camera.radius=clamp(camera.radius*Math.exp(e.deltaY*.001),.6,220);draw();},{passive:false});
    canvas.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','=','-'].includes(e.key))return;e.preventDefault();if(e.key==='ArrowLeft')camera.theta+=.12;if(e.key==='ArrowRight')camera.theta-=.12;if(e.key==='ArrowUp')camera.phi=clamp(camera.phi+.1,.02,1.56);if(e.key==='ArrowDown')camera.phi=clamp(camera.phi-.1,.02,1.56);if(['+','='].includes(e.key))camera.radius=Math.max(.6,camera.radius*.85);if(e.key==='-')camera.radius=Math.min(220,camera.radius/.85);draw();});
    caseSelect.addEventListener('change',()=>{stats();focus(true);evidence.hidden=caseSelect.value!=='aligned';});zoneSelect.addEventListener('change',()=>{zone=data.zones.find(z=>z.id===zoneSelect.value);showPhotos();focus(zone.id==='all');});
    for(const id of ['show-dorm2','show-dorm3','clip-ceiling'])$(id).addEventListener('change',draw);surface.addEventListener('change',draw);slider.addEventListener('input',draw);
    $('alignment-overview').addEventListener('click',()=>focus(true));$('alignment-top').addEventListener('click',()=>{camera.phi=1.555;focus(zone.id==='all');camera.radius*=1.15;draw();});$('alignment-oblique').addEventListener('click',()=>{camera.phi=.8;draw();});window.addEventListener('resize',draw);
    resize();stats();showPhotos();focus(true);
  }catch(error){status.dataset.state='error';status.textContent='정합 화면을 불러오지 못했습니다: '+error.message;console.error(error);}
}

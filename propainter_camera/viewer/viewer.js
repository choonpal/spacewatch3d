(() => {
  'use strict';
  const items=window.CAMERA_DATA, el=id=>document.getElementById(id),canvas=el('panorama');
  const state={index:0,mode:'restored',yaw:0,pitch:0,fov:90};
  let token=0,ready=false,textureSize=[1024,512];
  try {
    if (!items || !items.length) throw new Error('표시할 촬영 프레임이 없습니다.');
    const gl=canvas.getContext('webgl');
    if (!gl) throw new Error('이 브라우저에서 WebGL을 사용할 수 없습니다.');
    const shader=(type,source)=>{const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;};
    const program=gl.createProgram();
    gl.attachShader(program,shader(gl.VERTEX_SHADER,'attribute vec2 p;varying vec2 v;void main(){v=p;gl_Position=vec4(p,0.,1.);}'));
    gl.attachShader(program,shader(gl.FRAGMENT_SHADER,`precision highp float;varying vec2 v;uniform sampler2D tex;uniform float yaw,pitch,tanFov,aspect;uniform vec2 imageSize;void main(){vec3 d=normalize(vec3(v.x*aspect*tanFov,-v.y*tanFov,1.));float cp=cos(pitch),sp=sin(pitch);d=vec3(d.x,cp*d.y+sp*d.z,-sp*d.y+cp*d.z);float cy=cos(yaw),sy=sin(yaw);d=vec3(cy*d.x+sy*d.z,d.y,-sy*d.x+cy*d.z);vec2 uv=vec2(atan(d.x,d.z)/6.28318530718+.5,asin(clamp(d.y,-1.,1.))/3.14159265359+.5);uv=(uv*(imageSize-1.)+.5)/imageSize;gl_FragColor=texture2D(tex,uv);}`));
    gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(program));gl.useProgram(program);
    const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),gl.STATIC_DRAW);
    const p=gl.getAttribLocation(program,'p');gl.enableVertexAttribArray(p);gl.vertexAttribPointer(p,2,gl.FLOAT,false,0,0);
    const uniforms=Object.fromEntries(['yaw','pitch','tanFov','aspect','imageSize'].map(name=>[name,gl.getUniformLocation(program,name)]));
    const texture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,texture);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    function draw(){if(!ready)return;const ratio=Math.min(window.devicePixelRatio||1,2);canvas.width=Math.round(canvas.clientWidth*ratio);canvas.height=Math.round(canvas.clientHeight*ratio);gl.viewport(0,0,canvas.width,canvas.height);gl.uniform1f(uniforms.yaw,state.yaw);gl.uniform1f(uniforms.pitch,state.pitch);gl.uniform1f(uniforms.tanFov,Math.tan(state.fov*Math.PI/360));gl.uniform1f(uniforms.aspect,canvas.width/canvas.height);gl.uniform2f(uniforms.imageSize,...textureSize);gl.drawArrays(gl.TRIANGLES,0,6);}
    function focus(){const t=items[state.index];state.yaw=t.focus.yaw;state.pitch=t.focus.pitch;draw();}
    function select(index,refocus=true){state.index=(index+items.length)%items.length;const t=items[state.index],current=++token;ready=false;el('choice').value=String(state.index);el('error').textContent='이미지 불러오는 중…';if(refocus)focus();const file=state.mode==='restored'?t.file:t.original;el('download').href=file;el('download').download=t.id+'_'+state.mode+'.png';el('meta').textContent='프레임 '+t.camera_frame_id+' · '+t.timestamp_s.toFixed(2)+'초 · '+t.width+' × '+t.height+' · 위치 ['+t.pose.slice(0,3).map(row=>row[3].toFixed(3)).join(', ')+']';const image=new Image();image.onload=()=>{if(current!==token)return;gl.bindTexture(gl.TEXTURE_2D,texture);gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,false);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGB,gl.RGB,gl.UNSIGNED_BYTE,image);textureSize=[image.width,image.height];ready=true;el('error').textContent='';draw();};image.onerror=()=>{if(current===token)el('error').textContent='이미지를 불러오지 못했습니다: '+file;};image.src=file;}
    items.forEach((t,i)=>{const option=document.createElement('option');option.value=String(i);option.textContent='프레임 '+t.camera_frame_id+' · '+t.timestamp_s.toFixed(2)+'초';el('choice').append(option);});
    el('choice').onchange=e=>select(+e.target.value);el('prev').onclick=()=>select(state.index-1);el('next').onclick=()=>select(state.index+1);
    for(const mode of ['restored','original'])el(mode).onclick=()=>{state.mode=mode;for(const m of ['restored','original'])el(m).setAttribute('aria-pressed',String(m===mode));select(state.index,false);};
    el('front').onclick=()=>{state.yaw=0;state.pitch=0;draw();};el('focus').onclick=focus;
    function setFov(value){state.fov=Math.max(30,Math.min(110,value));el('fov').value=String(state.fov);el('fov-value').textContent=Math.round(state.fov)+'°';draw();}
    el('fov').oninput=e=>setFov(+e.target.value);canvas.addEventListener('wheel',e=>{e.preventDefault();setFov(state.fov+e.deltaY*.03);},{passive:false});
    let drag=null;canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY,state.yaw,state.pitch];canvas.setPointerCapture(e.pointerId);};canvas.onpointerup=canvas.onpointercancel=()=>drag=null;canvas.onpointermove=e=>{if(!drag)return;state.yaw=drag[2]-(e.clientX-drag[0])*.004;state.pitch=Math.max(-1.55,Math.min(1.55,drag[3]-(e.clientY-drag[1])*.004));draw();};canvas.onkeydown=e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key))return;e.preventDefault();state.yaw+=(e.key==='ArrowRight'?.08:e.key==='ArrowLeft'?-.08:0);state.pitch=Math.max(-1.55,Math.min(1.55,state.pitch+(e.key==='ArrowDown'?.08:e.key==='ArrowUp'?-.08:0)));draw();};
    window.addEventListener('resize',draw);select(0);
  }catch(error){el('error').textContent=error.message;console.error(error);}
})();

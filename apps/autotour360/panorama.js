import {cameraBasis, radians, clamp, wrapYaw, projectPoint} from './core.js';

// A single full-screen triangle projects viewing rays into an equirectangular texture.
// yaw=0 samples the center of the source image; pitch is positive toward the ceiling.
export class Panorama {
  constructor(canvas, onChange, onError) {
    this.canvas = canvas; this.onChange = onChange; this.onError = onError;
    this.view = {yaw:0,pitch:0,fov:75}; this.autoRotate = false; this.dirty = true;
    this.media = null; this.frameDirty = false; this.lastVideoTime = -1; this.frameCallback = null;
    this.pointers = new Map(); this.initialize();
    this.observer = new ResizeObserver(() => this.resize()); this.observer.observe(canvas);
    canvas.addEventListener('webglcontextlost', e => {e.preventDefault(); this.lost=true; this.onError('그래픽 연결이 잠시 끊겼습니다. 복구를 기다리거나 새로고침해 주세요.');});
    canvas.addEventListener('webglcontextrestored', () => {this.initialize(); this.lost=false; this.frameDirty=true; this.dirty=true;});
    canvas.addEventListener('pointerdown', e => {canvas.focus(); canvas.setPointerCapture(e.pointerId); this.pointers.set(e.pointerId,{x:e.clientX,y:e.clientY}); this.autoRotate=false; this.onChange();});
    canvas.addEventListener('pointermove', e => {
      const old=this.pointers.get(e.pointerId); if (!old) return;
      const other=[...this.pointers.entries()].find(([id])=>id!==e.pointerId)?.[1];
      if (other) {
        const before=Math.hypot(old.x-other.x,old.y-other.y), after=Math.hypot(e.clientX-other.x,e.clientY-other.y);
        this.view.fov=clamp(this.view.fov+(before-after)*.15,35,100);
      } else {
        const sensitivity=this.view.fov/this.canvas.clientHeight;
        this.view.yaw=wrapYaw(this.view.yaw-(e.clientX-old.x)*sensitivity);
        this.view.pitch=clamp(this.view.pitch+(e.clientY-old.y)*sensitivity,-85,85);
      }
      this.pointers.set(e.pointerId,{x:e.clientX,y:e.clientY});this.dirty=true;this.onChange();
    });
    for(const event of ['pointerup','pointercancel','lostpointercapture']) canvas.addEventListener(event,e=>this.pointers.delete(e.pointerId));
    canvas.addEventListener('wheel',e=>{e.preventDefault();this.zoom(e.deltaY*.035);},{passive:false});
    this.lastFrame=performance.now(); this.loop=this.loop.bind(this);requestAnimationFrame(this.loop);
  }
  initialize() {
    const gl=this.canvas.getContext('webgl',{alpha:false,antialias:false,preserveDrawingBuffer:false,powerPreference:'high-performance'});
    if (!gl) throw new Error('이 브라우저에서 WebGL을 사용할 수 없습니다. Chrome 또는 Edge의 하드웨어 가속을 켜 주세요.');
    this.gl=gl;
    const compile=(type,source)=>{const shader=gl.createShader(type);gl.shaderSource(shader,source);gl.compileShader(shader);if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(shader));return shader;};
    const vertex=compile(gl.VERTEX_SHADER,'attribute vec2 position; varying vec2 screen; void main(){screen=position;gl_Position=vec4(position,0.0,1.0);}');
    const fragment=compile(gl.FRAGMENT_SHADER,`
      precision highp float;
      varying vec2 screen; uniform sampler2D panorama;
      uniform vec3 forward; uniform vec3 right; uniform vec3 up;
      uniform float aspect; uniform float fovScale;
      void main(){
        vec3 ray=normalize(forward+screen.x*aspect*fovScale*right+screen.y*fovScale*up);
        vec2 uv=vec2(fract(0.5+atan(ray.x,ray.z)/6.28318530718),0.5+asin(clamp(ray.y,-1.0,1.0))/3.14159265359);
        gl_FragColor=texture2D(panorama,uv);
      }`);
    const program=gl.createProgram();gl.attachShader(program,vertex);gl.attachShader(program,fragment);gl.linkProgram(program);
    if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(program));
    gl.useProgram(program);gl.deleteShader(vertex);gl.deleteShader(fragment);
    const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,3,-1,-1,3]),gl.STATIC_DRAW);
    const position=gl.getAttribLocation(program,'position');gl.enableVertexAttribArray(position);gl.vertexAttribPointer(position,2,gl.FLOAT,false,0,0);
    this.uniforms=Object.fromEntries(['forward','right','up','aspect','fovScale'].map(k=>[k,gl.getUniformLocation(program,k)]));
    this.texture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,this.texture);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGB,1,1,0,gl.RGB,gl.UNSIGNED_BYTE,new Uint8Array([17,23,25]));
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,true);
    this.maxTexture=gl.getParameter(gl.MAX_TEXTURE_SIZE);
    this.resize();
  }
  resize(){const ratio=Math.min(window.devicePixelRatio||1,2);const w=Math.round(this.canvas.clientWidth*ratio),h=Math.round(this.canvas.clientHeight*ratio);if(w&&h){this.canvas.width=w;this.canvas.height=h;this.gl.viewport(0,0,w,h);this.dirty=true;}}
  setView(view){this.view={yaw:wrapYaw(view.yaw),pitch:clamp(view.pitch,-85,85),fov:clamp(view.fov,35,100)};this.dirty=true;this.onChange();}
  zoom(delta){this.view.fov=clamp(this.view.fov+delta,35,100);this.dirty=true;this.onChange();}
  setMedia(media){
    if(this.frameCallback !== null && this.media?.cancelVideoFrameCallback)this.media.cancelVideoFrameCallback(this.frameCallback);
    this.frameCallback=null;this.media=media;this.frameDirty=!!media;this.lastVideoTime=-1;this.dirty=true;
    if(!media){this.gl.texImage2D(this.gl.TEXTURE_2D,0,this.gl.RGB,1,1,0,this.gl.RGB,this.gl.UNSIGNED_BYTE,new Uint8Array([17,23,25]));return;}
    if(media instanceof HTMLVideoElement && media.requestVideoFrameCallback){
      const next=()=>{if(this.media!==media)return;this.frameDirty=true;this.frameCallback=media.requestVideoFrameCallback(next);};
      this.frameCallback=media.requestVideoFrameCallback(next);
    }
  }
  render(){
    if(this.lost)return;
    const gl=this.gl,media=this.media;
    if(media && (media instanceof HTMLVideoElement ? media.readyState>=2 : media.complete)){
      const shouldUpload=this.frameDirty || (media instanceof HTMLVideoElement && !media.requestVideoFrameCallback && media.currentTime!==this.lastVideoTime);
      if(shouldUpload){
        try{
          let source=media;
          const w=media.videoWidth||media.naturalWidth,h=media.videoHeight||media.naturalHeight;
          if(w>this.maxTexture || h>this.maxTexture){
            this.downscale ||= document.createElement('canvas');const factor=this.maxTexture/Math.max(w,h);
            const sw=Math.floor(w*factor),sh=Math.floor(h*factor);
            if(this.downscale.width!==sw || this.downscale.height!==sh){this.downscale.width=sw;this.downscale.height=sh;}
            this.downscale.getContext('2d').drawImage(media,0,0,sw,sh);source=this.downscale;
          }
          gl.texImage2D(gl.TEXTURE_2D,0,gl.RGB,gl.RGB,gl.UNSIGNED_BYTE,source);
          this.lastVideoTime=media.currentTime;this.frameDirty=false;this.dirty=true;
        }catch(error){this.setMedia(null);this.onError(`영상을 표시할 수 없습니다. ${error.message}`);}
      }
    }
    if(!this.dirty)return;
    const basis=cameraBasis(this.view);
    for(const k of ['forward','right','up'])gl.uniform3fv(this.uniforms[k],basis[k]);
    gl.uniform1f(this.uniforms.aspect,this.canvas.width/this.canvas.height);gl.uniform1f(this.uniforms.fovScale,Math.tan(radians(this.view.fov)/2));
    gl.drawArrays(gl.TRIANGLES,0,3);this.dirty=false;
  }
  loop(now){const dt=Math.min((now-this.lastFrame)/1000,.1);this.lastFrame=now;if(!document.hidden){if(this.autoRotate){this.view.yaw=wrapYaw(this.view.yaw+dt*5);this.dirty=true;this.onChange();}this.render();this.onFrame?.(now);}requestAnimationFrame(this.loop);}
  project(yaw,pitch){return projectPoint(yaw,pitch,this.view,this.canvas.clientWidth,this.canvas.clientHeight);}
  snapshot(){this.dirty=true;this.render();const c=document.createElement('canvas');c.width=288;c.height=180;c.getContext('2d').drawImage(this.canvas,0,0,c.width,c.height);return c.toDataURL('image/jpeg',.7);}
}

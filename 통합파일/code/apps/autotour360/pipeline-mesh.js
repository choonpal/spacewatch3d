import {decodeGeometry} from './pipeline-model.js';

export class PipelineMesh {
  constructor(canvas) {
    this.canvas=canvas; this.gl=canvas.getContext('webgl2',{antialias:true,alpha:false});
    if(!this.gl) throw new Error('메시 보기에는 WebGL 2가 필요합니다.');
    const gl=this.gl;
    const shader=(type,source)=>{const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;};
    const vertex=shader(gl.VERTEX_SHADER,`#version 300 es
      in vec3 position; in vec3 normal; in vec3 color;
      uniform vec3 center,right,up,forward; uniform float radius,aspect;
      out vec3 tint; out float height;
      void main(){vec3 p=vec3(position.x,position.z,-position.y)-center;
        gl_Position=vec4(dot(p,right)/(radius*aspect),dot(p,up)/radius,-dot(p,forward)/(radius*8.),1.);
        gl_PointSize=2.;height=position.z;tint=color*(.72+.28*abs(dot(normal,normalize(vec3(.3,.4,1.)))));}`);
    const fragment=shader(gl.FRAGMENT_SHADER,`#version 300 es
      precision highp float; in vec3 tint; in float height; uniform float cut; out vec4 pixel;
      void main(){if(height>cut)discard;pixel=vec4(tint,1.);}`);
    this.program=gl.createProgram();gl.attachShader(this.program,vertex);gl.attachShader(this.program,fragment);gl.linkProgram(this.program);
    if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(this.program));
    gl.deleteShader(vertex);gl.deleteShader(fragment);gl.useProgram(this.program);
    this.uniforms=Object.fromEntries(['center','right','up','forward','radius','aspect','cut'].map(n=>[n,gl.getUniformLocation(this.program,n)]));
    this.attributes=Object.fromEntries(['position','normal','color'].map(n=>[n,gl.getAttribLocation(this.program,n)]));
    this.buffers=[];this.mode='rgb';this.cut=true;this.pending=0;this.reset();
    this.observer=new ResizeObserver(()=>this.draw());this.observer.observe(canvas);
    canvas.addEventListener('contextmenu',e=>e.preventDefault());
    canvas.addEventListener('pointerdown',e=>{canvas.focus();canvas.setPointerCapture(e.pointerId);this.drag={x:e.clientX,y:e.clientY,pan:e.button===1||e.button===2||e.shiftKey};});
    canvas.addEventListener('pointermove',e=>{if(!this.drag)return;const dx=e.clientX-this.drag.x,dy=e.clientY-this.drag.y;
      if(e.shiftKey||this.drag.pan){const k=2*this.radius/canvas.clientHeight;const {right,up}=this.basis();this.center=this.center.map((v,i)=>v-dx*k*right[i]+dy*k*up[i]);}
      else {this.yaw+=dx*.008;this.pitch=Math.max(-1.55,Math.min(1.55,this.pitch+dy*.008));}
      this.drag={...this.drag,x:e.clientX,y:e.clientY};this.draw();});
    for(const event of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(event,()=>this.drag=null);
    canvas.addEventListener('wheel',e=>{e.preventDefault();this.radius=Math.max(.2,Math.min(500,this.radius*Math.exp(e.deltaY*.001)));this.draw();},{passive:false});
    canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();this.lost=true;});
    canvas.addEventListener('webglcontextrestored',()=>location.reload());
  }
  reset(){this.yaw=-.45;this.pitch=-.95;this.center=[0,0,0];this.radius=10;if(this.capture){const [a,b]=this.capture.bounds;this.center=[(a[0]+b[0])/2,(a[2]+b[2])/2,-(a[1]+b[1])/2];const extent=Math.hypot(...a.map((v,i)=>b[i]-v));this.radius=extent*.55;}this.draw();}
  focus(position){this.center=[position[0],position[2],-position[1]];this.radius=5;this.draw();}
  focusObject(object){this.center=[object.center[0],object.center[2],-object.center[1]];this.radius=Math.max(.3,Math.hypot(...object.bounds[0].map((v,i)=>object.bounds[1][i]-v))*.85);this.draw();}
  basis(){const c=Math.cos(this.yaw),s=Math.sin(this.yaw),cp=Math.cos(this.pitch),sp=Math.sin(this.pitch);return {right:[c,0,-s],up:[s*sp,cp,c*sp],forward:[s*cp,-sp,c*cp]};}
  async load(capture,mode){
    const token=++this.pending;this.capture=capture;this.mode=mode;
    const url=mode==='points'?capture.points:mode==='instances'?capture.instance_mesh:capture.mesh;
    const response=await fetch(url);if(!response.ok)throw Error('3D 파일을 불러오지 못했습니다.');
    const data=decodeGeometry(await response.arrayBuffer());if(token!==this.pending)return;
    this.upload(data,mode);
  }
  upload(data,mode='rgb'){
    this.data=data;
    const gl=this.gl;this.buffers.forEach(b=>gl.deleteBuffer(b));this.buffers=[];gl.useProgram(this.program);
    const attribute=(name,values,type,normalize=false)=>{const index=this.attributes[name];if(!values){gl.disableVertexAttribArray(index);gl.vertexAttrib3f(index,0,0,1);return;}
      const b=gl.createBuffer();this.buffers.push(b);if(name==='color')this.colorBuffer=b;gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,values,gl.STATIC_DRAW);gl.enableVertexAttribArray(index);gl.vertexAttribPointer(index,3,type,normalize,0,0);};
    attribute('position',data.positions,gl.FLOAT);attribute('normal',data.normals,gl.FLOAT);
    attribute('color',['semantic','instances'].includes(mode)?data.semantic:data.rgb,gl.UNSIGNED_BYTE,true);
    this.count=data.count;this.kind=data.kind;this.draw();
  }
  setColors(colors){
    if(colors.length!==this.count*3)throw Error('메시 색상 크기가 맞지 않습니다.');
    const gl=this.gl;gl.bindBuffer(gl.ARRAY_BUFFER,this.colorBuffer);gl.bufferSubData(gl.ARRAY_BUFFER,0,colors);this.draw();
  }
  draw(){if(!this.gl||this.lost)return;const gl=this.gl,c=this.canvas;
    const scale=Math.min(devicePixelRatio||1,2),w=Math.round(c.clientWidth*scale),h=Math.round(c.clientHeight*scale);if(!w||!h)return;
    if(c.width!==w||c.height!==h){c.width=w;c.height=h;}gl.viewport(0,0,w,h);gl.clearColor(.047,.067,.078,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    if(!this.count)return;gl.useProgram(this.program);gl.enable(gl.DEPTH_TEST);gl.disable(gl.CULL_FACE);
    const basis=this.basis();for(const k of ['right','up','forward'])gl.uniform3fv(this.uniforms[k],basis[k]);
    gl.uniform3fv(this.uniforms.center,this.center);gl.uniform1f(this.uniforms.radius,this.radius);gl.uniform1f(this.uniforms.aspect,w/h);
    gl.uniform1f(this.uniforms.cut,this.cut?this.capture.floor_height+2.1:1e10);gl.drawArrays(this.kind===1?gl.TRIANGLES:gl.POINTS,0,this.count);this.afterDraw?.();
  }
}

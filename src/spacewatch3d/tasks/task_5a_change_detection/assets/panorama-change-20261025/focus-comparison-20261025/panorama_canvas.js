const PI=Math.PI,rad=d=>d*PI/180,deg=r=>r*180/PI;
export const normalize=v=>{const l=Math.hypot(...v);return v.map(x=>x/Math.max(l,1e-12));};
export const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
export const sub=(a,b)=>a.map((x,i)=>x-b[i]);
export const rotate=(R,v)=>R.map(row=>row.reduce((s,x,i)=>s+x*v[i],0));
export const inverseRotate=(R,v)=>[0,1,2].map(j=>R.reduce((s,row,i)=>s+row[j]*v[i],0));
function basis(yaw,pitch){const y=rad(yaw),p=rad(pitch),f=[Math.sin(y)*Math.cos(p),Math.sin(p),Math.cos(y)*Math.cos(p)],r=normalize(cross(f,[0,1,0])),u=cross(r,f);return [r,u,f];}

/** Real whole-sphere ERP renderer. Camera rotations are camera-to-common-world. */
export class PanoramaCanvas{
 constructor(canvas,method,onInput,onFocus,onProjection){
  this.canvas=canvas;this.method=method;this.onInput=onInput;this.onFocus=onFocus;this.onProjection=onProjection;
  this.yaw=0;this.pitch=0;this.fov=65;this.full=false;this.exclude=true;this.review=false;this.overlay=false;this.pointers=new Map();
  this.ghostEnabled=false;this.ghostOpacity=.55;this.ghostBounds=[.5,.01,.4,.6];
  const gl=this.gl=canvas.getContext('webgl',{alpha:false,antialias:false,preserveDrawingBuffer:true});
  if(!gl)throw new Error('360° 화면에 필요한 WebGL을 사용할 수 없습니다.');
  const make=(type,code)=>{const s=gl.createShader(type);gl.shaderSource(s,code);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;};
  const vs=make(gl.VERTEX_SHADER,'attribute vec2 aPosition;varying vec2 vPosition;void main(){vPosition=aPosition;gl_Position=vec4(aPosition,0.,1.);}');
  const fs=make(gl.FRAGMENT_SHADER,`
   precision highp float;varying vec2 vPosition;uniform sampler2D uRgb,uMask,uGhostRgb,uGhostMask;uniform mat3 uViewToERP;
   uniform float uAspect,uTanFov,uFull,uMethod,uExclude,uReview,uOverlay,uGhostEnabled,uGhostOpacity;uniform vec4 uGhostBounds;const float PI=3.14159265359;
   void main(){
    vec3 ray;
    if(uFull>.5){float longitude=vPosition.x*PI,latitude=vPosition.y*PI*.5;ray=vec3(sin(longitude)*cos(latitude),sin(latitude),cos(longitude)*cos(latitude));}
    else ray=normalize(vec3(vPosition.x*uAspect*uTanFov,vPosition.y*uTanFov,1.));
    vec3 camera=normalize(uViewToERP*ray);vec2 uv=vec2(fract(atan(camera.x,camera.z)/(2.*PI)+.5),asin(clamp(camera.y,-1.,1.))/PI+.5);
    vec3 rgb=texture2D(uRgb,uv).rgb;vec4 mask=texture2D(uMask,uv);float alpha=0.;vec3 tint=rgb;
    if(uMethod>.5){if(mask.r>.5){tint=vec3(.796,.6,1.);alpha=.57;}}
    else{float flags=floor(mask.r*255.+.5);if(mod(floor(flags/8.),2.)>.5){tint=vec3(1.,.706,.329);alpha=.62;}
     else if(uReview>.5&&mod(floor(flags/2.),2.)>.5){tint=vec3(.592,.682,.773);alpha=.35;}
     else if(uReview>.5&&mod(floor(flags/4.),2.)>.5){tint=vec3(.753,.725,.533);alpha=.35;}}
    if(uExclude>.5&&mask.g>.5)alpha=0.;if(uOverlay<.5)alpha=0.;vec3 color=mix(rgb,tint,alpha);
    if(uGhostEnabled>.5){float dx=abs(fract(uv.x-uGhostBounds.x+.5)-.5);vec4 valid=texture2D(uGhostMask,uv);
     if(dx<=uGhostBounds.y&&uv.y>=uGhostBounds.z&&uv.y<=uGhostBounds.w&&valid.r>.5&&(uExclude<.5||valid.g<.5))
      color=mix(color,texture2D(uGhostRgb,uv).rgb,uGhostOpacity);}
    gl_FragColor=vec4(color,1.);
   }`);
  this.program=gl.createProgram();gl.attachShader(this.program,vs);gl.attachShader(this.program,fs);gl.linkProgram(this.program);
  if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(this.program));
  gl.deleteShader(vs);gl.deleteShader(fs);gl.useProgram(this.program);
  const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),gl.STATIC_DRAW);
  const location=gl.getAttribLocation(this.program,'aPosition');gl.enableVertexAttribArray(location);gl.vertexAttribPointer(location,2,gl.FLOAT,false,0,0);
  this.uniforms=Object.fromEntries(['uRgb','uMask','uGhostRgb','uGhostMask','uViewToERP','uAspect','uTanFov','uFull','uMethod','uExclude','uReview','uOverlay','uGhostEnabled','uGhostOpacity','uGhostBounds'].map(k=>[k,gl.getUniformLocation(this.program,k)]));
  this.textures=[0,1,2,3].map(()=>gl.createTexture());
  this.textures.forEach((texture,i)=>{gl.activeTexture(gl.TEXTURE0+i);gl.bindTexture(gl.TEXTURE_2D,texture);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.REPEAT);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,i%2?gl.NEAREST:gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,i%2?gl.NEAREST:gl.LINEAR);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,1,1,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array([20,30,40,255]));});
  this.resizeObserver=new ResizeObserver(()=>this.draw());this.resizeObserver.observe(canvas);this.bind();
 }
 setScene(rgb,mask,camera){
  this.camera=camera;const gl=this.gl;gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,false);gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL,gl.NONE);
  [rgb,mask].forEach((image,i)=>{gl.activeTexture(gl.TEXTURE0+i);gl.bindTexture(gl.TEXTURE_2D,this.textures[i]);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,image);});this.draw();
 }
 setGhost(rgb,mask){const gl=this.gl;[rgb,mask].forEach((image,i)=>{gl.activeTexture(gl.TEXTURE2+i);gl.bindTexture(gl.TEXTURE_2D,this.textures[2+i]);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,image);});this.draw();}
 projectDirection(direction){const [r,u,f]=basis(this.yaw,this.pitch),dot=a=>a.reduce((s,x,i)=>s+x*direction[i],0),d=[dot(r),dot(u),dot(f)];
  if(this.full){const v=normalize(d);return{x:Math.atan2(v[0],v[2])/PI,y:Math.asin(v[1])/(PI/2),visible:true};}
  if(d[2]<=.001)return{visible:false};const rect=this.canvas.getBoundingClientRect(),t=Math.tan(rad(this.fov)/2);return{x:d[0]/(d[2]*t*(rect.width/rect.height)),y:d[1]/(d[2]*t),visible:true};}
 lookAt(direction,fov=this.fov){const d=normalize(direction);this.setView(deg(Math.atan2(d[0],d[2])),deg(Math.asin(Math.max(-1,Math.min(1,d[1])))),fov);}
 setView(yaw,pitch,fov=this.fov){this.yaw=((yaw+180)%360+360)%360-180;this.pitch=Math.max(-85,Math.min(85,pitch));this.fov=Math.max(35,Math.min(105,fov));this.draw();}
 forward(){return basis(this.yaw,this.pitch)[2];}
 directionAt(x,y){const [r,u,f]=basis(this.yaw,this.pitch);let ray;if(this.full){const lo=x*PI,la=y*PI/2;ray=[Math.sin(lo)*Math.cos(la),Math.sin(la),Math.cos(lo)*Math.cos(la)];}else{const rect=this.canvas.getBoundingClientRect(),t=Math.tan(rad(this.fov)/2);ray=normalize([x*(rect.width/rect.height)*t,y*t,1]);}return normalize([0,1,2].map(i=>r[i]*ray[0]+u[i]*ray[1]+f[i]*ray[2]));}
 draw(){
  if(!this.camera)return;const {canvas,gl}=this,rect=canvas.getBoundingClientRect();if(!rect.width||!rect.height)return;
  const ratio=Math.min(devicePixelRatio||1,1.5),width=Math.round(rect.width*ratio),height=Math.round(rect.height*ratio);
  if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}gl.viewport(0,0,width,height);gl.useProgram(this.program);
  const columns=basis(this.yaw,this.pitch).map(v=>inverseRotate(this.camera.rotation,v));
  gl.uniformMatrix3fv(this.uniforms.uViewToERP,false,new Float32Array(columns.flat()));
  gl.uniform1i(this.uniforms.uRgb,0);gl.uniform1i(this.uniforms.uMask,1);gl.uniform1i(this.uniforms.uGhostRgb,2);gl.uniform1i(this.uniforms.uGhostMask,3);
  gl.uniform4fv(this.uniforms.uGhostBounds,new Float32Array(this.ghostBounds));
  for(const [key,value]of Object.entries({uAspect:width/height,uTanFov:Math.tan(rad(this.fov)/2),uFull:this.full?1:0,uMethod:this.method,uExclude:this.exclude?1:0,uReview:this.review?1:0,uOverlay:this.overlay?1:0,uGhostEnabled:this.ghostEnabled?1:0,uGhostOpacity:this.ghostOpacity}))gl.uniform1f(this.uniforms[key],value);
  gl.drawArrays(gl.TRIANGLES,0,6);
  this.onDraw?.(this);
 }
 bind(){const c=this.canvas,surface=c.parentElement;
  surface.addEventListener('pointerdown',e=>{if(e.target.closest('.stage-tools'))return;e.preventDefault();c.focus({preventScroll:true});surface.setPointerCapture(e.pointerId);this.pointers.set(e.pointerId,{xy:[e.clientX,e.clientY],distance:0,box:e.target.closest('[data-region-box]')?.dataset.regionBox});});
  surface.addEventListener('pointermove',e=>{const p=this.pointers.get(e.pointerId);if(!p)return;const dx=e.clientX-p.xy[0],dy=e.clientY-p.xy[1];p.distance+=Math.hypot(dx,dy);if(p.distance<4)return;const sensitivity=(this.full?360:this.fov*1.5)/c.clientWidth;this.setView(this.yaw+dx*sensitivity,this.pitch+dy*sensitivity);p.xy=[e.clientX,e.clientY];this.onInput(this);});
  surface.addEventListener('pointerup',e=>{const p=this.pointers.get(e.pointerId);this.pointers.delete(e.pointerId);if(p?.box&&p.distance<4)this.onBoxClick?.(p.box);});
  for(const type of ['pointercancel','lostpointercapture'])surface.addEventListener(type,e=>this.pointers.delete(e.pointerId));
  surface.addEventListener('wheel',e=>{if(document.activeElement!==c)return;e.preventDefault();if(this.full)this.onProjection('look');this.setView(this.yaw,this.pitch,this.fov+e.deltaY*.04);this.onInput(this);},{passive:false});
  surface.addEventListener('dblclick',e=>{if(e.target.closest('button'))return;const r=c.getBoundingClientRect();this.onFocus(this,this.directionAt((e.clientX-r.left)/r.width*2-1,1-(e.clientY-r.top)/r.height*2));});
  c.addEventListener('keydown',e=>{let y=this.yaw,p=this.pitch;if(e.key==='ArrowLeft')y+=8;else if(e.key==='ArrowRight')y-=8;else if(e.key==='ArrowUp')p+=8;else if(e.key==='ArrowDown')p-=8;else return;e.preventDefault();this.setView(y,p);this.onInput(this);});
 }
}

import {routeTriangles,routeTrackTriangles,routeOccluders} from './pipeline-route-arrows.js';

// A transparent world-space overlay. The full original mesh writes DEPTH ONLY;
// walls, furniture and floor clip the route at every pixel, not just its centre.
export class RouteOverlay {
 constructor(canvas){
  this.canvas=canvas;const gl=canvas.getContext('webgl',{alpha:true,antialias:true,premultipliedAlpha:false,powerPreference:'high-performance'});
  if(!gl)throw Error('경로 표시용 그래픽을 초기화하지 못했습니다.');this.gl=gl;this.routes=[];
  const compile=(type,source)=>{const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;};
  const vs=compile(gl.VERTEX_SHADER,`attribute vec3 position;varying float groundDistance;
   uniform vec3 origin,right,up,forward;uniform float fovScale,aspect;
   void main(){vec3 p=position-origin;groundDistance=length(p.xy);float z=dot(p,forward);
    gl_Position=vec4(dot(p,right)/(fovScale*aspect),dot(p,up)/fovScale,1.00060018*z-.060018,z);}`);
  const fs=compile(gl.FRAGMENT_SHADER,'precision mediump float; uniform vec4 color; uniform float fadeNear,fadeFull; varying float groundDistance; void main(){gl_FragColor=vec4(color.rgb,color.a*smoothstep(fadeNear,fadeFull,groundDistance));}');
  this.program=gl.createProgram();gl.attachShader(this.program,vs);gl.attachShader(this.program,fs);gl.linkProgram(this.program);
  if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(this.program));
  gl.deleteShader(vs);gl.deleteShader(fs);this.attribute=gl.getAttribLocation(this.program,'position');
  this.uniforms=Object.fromEntries(['origin','right','up','forward','fovScale','aspect','color','fadeNear','fadeFull'].map(k=>[k,gl.getUniformLocation(this.program,k)]));
  this.mesh=null;canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();this.lost=true;});
  canvas.addEventListener('webglcontextrestored',()=>location.reload());
 }
 buffer(positions){const gl=this.gl,b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,positions,gl.STATIC_DRAW);return {buffer:b,count:positions.length/3};}
 setScene(positions,floorFaces=null){const gl=this.gl;if(this.mesh)gl.deleteBuffer(this.mesh.buffer);this.mesh=positions?this.buffer(routeOccluders(positions,floorFaces)):null;this.setRoutes([]);}
 setRoutes(routes,currentIndex=0){const gl=this.gl;for(const r of this.routes)for(const part of ['track','outer','inner'])gl.deleteBuffer(r[part].buffer);this.routes=routes.filter(r=>r.status==='ready').map(r=>({track:this.buffer(routeTrackTriangles(r)),outer:this.buffer(routeTriangles(r)),inner:this.buffer(routeTriangles(r,.62)),height:r.height,color:r.index>currentIndex?[.12,.62,.69,.90]:[.31,.45,.54,.86]}));}
 draw(current,view){
  if(this.lost)return;const gl=this.gl,c=this.canvas,ratio=Math.min(devicePixelRatio||1,2),w=Math.round(c.clientWidth*ratio),h=Math.round(c.clientHeight*ratio);
  if(!w||!h)return;if(c.width!==w||c.height!==h){c.width=w;c.height=h;}gl.viewport(0,0,w,h);gl.colorMask(true,true,true,true);gl.depthMask(true);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  if(!this.mesh||!current)return;
  const y=view.yaw*Math.PI/180,p=view.pitch*Math.PI/180,local={forward:[Math.sin(y)*Math.cos(p),Math.sin(p),Math.cos(y)*Math.cos(p)],right:[Math.cos(y),0,-Math.sin(y)],up:[-Math.sin(y)*Math.sin(p),Math.cos(p),-Math.cos(y)*Math.sin(p)]};
  const world=v=>current.pose.slice(0,3).map(row=>row[0]*v[0]-row[1]*v[1]+row[2]*v[2]);
  gl.useProgram(this.program);gl.uniform3fv(this.uniforms.origin,current.position);for(const key of ['right','up','forward'])gl.uniform3fv(this.uniforms[key],world(local[key]));
  gl.uniform1f(this.uniforms.fovScale,Math.tan(view.fov*Math.PI/360));gl.uniform1f(this.uniforms.aspect,w/h);
  gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);gl.disable(gl.CULL_FACE);gl.disable(gl.BLEND);
  const draw=data=>{gl.bindBuffer(gl.ARRAY_BUFFER,data.buffer);gl.enableVertexAttribArray(this.attribute);gl.vertexAttribPointer(this.attribute,3,gl.FLOAT,false,0,0);gl.drawArrays(gl.TRIANGLES,0,data.count);};
  gl.uniform1f(this.uniforms.fadeNear,0);gl.uniform1f(this.uniforms.fadeFull,.01);gl.colorMask(false,false,false,false);draw(this.mesh);gl.colorMask(true,true,true,true);gl.depthMask(false);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
  for(const r of this.routes){gl.uniform1f(this.uniforms.fadeNear,.55*r.height);gl.uniform1f(this.uniforms.fadeFull,1.0*r.height);gl.uniform4fv(this.uniforms.color,[...r.color.slice(0,3),.32]);draw(r.track);gl.uniform4fv(this.uniforms.color,r.color);draw(r.outer);gl.uniform4fv(this.uniforms.color,[.98,1,1,1]);draw(r.inner);}
  gl.depthMask(true);
 }
}

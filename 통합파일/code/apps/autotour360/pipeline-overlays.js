// Display-only projection of receipted change surfaces onto the current source ERP.
// Original panorama renderer, source pixels, event decisions and keypoints are untouched.
import {cameraBasis,radians} from './core.js';
export class ChangeOverlay {
  constructor(canvas){
    this.canvas=canvas;this.token=0;this.enabled=true;
    const gl=this.gl=canvas.getContext('webgl',{alpha:true,premultipliedAlpha:false,antialias:false});
    if(!gl)throw Error('변화 표시 그래픽을 열 수 없습니다.');
    const shader=(type,source)=>{const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;};
    const v=shader(gl.VERTEX_SHADER,'attribute vec2 position; varying vec2 screen; void main(){screen=position;gl_Position=vec4(position,0.,1.);}');
    const f=shader(gl.FRAGMENT_SHADER,`precision highp float;varying vec2 screen;uniform sampler2D mask;uniform vec3 forward,right,up;uniform float aspect,fovScale;
      void main(){vec3 ray=normalize(forward+screen.x*aspect*fovScale*right+screen.y*fovScale*up);
      vec2 uv=vec2(fract(.5+atan(ray.x,ray.z)/6.28318530718),.5+asin(clamp(ray.y,-1.,1.))/3.14159265359);
      gl_FragColor=texture2D(mask,uv);}`);
    const p=this.program=gl.createProgram();gl.attachShader(p,v);gl.attachShader(p,f);gl.linkProgram(p);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(p));gl.useProgram(p);
    const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,3,-1,-1,3]),gl.STATIC_DRAW);const a=gl.getAttribLocation(p,'position');gl.enableVertexAttribArray(a);gl.vertexAttribPointer(a,2,gl.FLOAT,false,0,0);
    this.uniforms=Object.fromEntries(['forward','right','up','aspect','fovScale'].map(k=>[k,gl.getUniformLocation(p,k)]));
    this.texture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,this.texture);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,true);this.clear();
    canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();this.lost=true;});
  }
  clear(){++this.token;const gl=this.gl;gl.bindTexture(gl.TEXTURE_2D,this.texture);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,1,1,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array(4));if(this.view)this.render(this.view);}
  async load(url){this.clear();const token=this.token;if(!url)return;const img=new Image();img.src=url;await img.decode();if(token!==this.token)return;const gl=this.gl;gl.bindTexture(gl.TEXTURE_2D,this.texture);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,img);if(this.view)this.render(this.view);}
  render(view){
    this.view=view;const c=this.canvas,gl=this.gl;if(this.lost)return;const s=Math.min(devicePixelRatio||1,2),w=Math.round(c.clientWidth*s),h=Math.round(c.clientHeight*s);if(!w||!h)return;if(c.width!==w||c.height!==h){c.width=w;c.height=h;}gl.viewport(0,0,w,h);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);if(!this.enabled)return;gl.useProgram(this.program);const b=cameraBasis(view);for(const k of ['forward','right','up'])gl.uniform3fv(this.uniforms[k],b[k]);gl.uniform1f(this.uniforms.aspect,w/h);gl.uniform1f(this.uniforms.fovScale,Math.tan(radians(view.fov)/2));gl.drawArrays(gl.TRIANGLES,0,3);
  }
}

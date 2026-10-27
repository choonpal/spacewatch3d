const $ = id => document.getElementById(id);
const query = new URLSearchParams(location.search);
const job = query.get('job');
const names = {door:'문',chair:'의자',table:'테이블',cabinet:'수납장',refrigerator:'냉장고',window:'창문',
  'vacuum cleaner':'청소기','fire extinguisher':'소화기',unknown:'미분류'};
const reviews = [];
const fail = error => { $('error').hidden=false; $('error').textContent=error.message; $('status').textContent='입력을 확인해 주세요'; console.error(error); };
const fetchJson = async url => {const r=await fetch(url);const d=await r.json();if(!r.ok)throw Error(d.error||'입력을 읽지 못했습니다.');return d;};
let THREE, base;
const rowMatrix = rows => new THREE.Matrix4().set(...rows.flat());
const uvCode = 'vec2 panoUV(vec3 p){vec3 d=normalize(vec3(p.x,-p.y,-p.z));return vec2(atan(d.x,d.z)/6.28318530718+.5,.5-asin(clamp(d.y,-1.,1.))/3.14159265359);}';

function renderer(host, label) {
  const r=new THREE.WebGLRenderer({antialias:true});r.setPixelRatio(Math.min(devicePixelRatio,2));
  r.domElement.tabIndex=0;r.domElement.setAttribute('aria-label',label);host.append(r.domElement);return r;
}
function shapeMaterial(photo=false) {
  return new THREE.ShaderMaterial({side:THREE.DoubleSide,transparent:true,depthWrite:false,depthTest:true,
    uniforms:{opacity:{value:.55},selected:{value:0},person:{value:null},excludePerson:{value:0},photo:{value:photo?1:0}},
    vertexShader:'attribute vec4 shade;attribute float instance;varying vec4 colour;varying float objectId;varying vec3 ray;void main(){colour=shade;objectId=instance;ray=(modelMatrix*vec4(position,1.)).xyz;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}',
    fragmentShader:`uniform float opacity;uniform float selected;uniform sampler2D person;uniform float excludePerson;uniform float photo;varying vec4 colour;varying float objectId;varying vec3 ray;${uvCode}
      void main(){if(colour.a<.5||(selected>.5&&abs(objectId-selected)>.5))discard;
      if(photo>.5&&excludePerson>.5&&texture2D(person,panoUV(ray)).r>.5)discard;
      gl_FragColor=vec4(pow(colour.rgb,vec3(2.2)),opacity);
      #include <colorspace_fragment>
      }`});
}

class CaptureReview {
  constructor(spec, geometry, ids, card) {
    this.spec=spec;this.geometry=geometry;this.ids=ids;this.card=card;this.selected=0;this.index=0;this.token=0;
    this.panoHost=card.querySelector('.pano');this.meshHost=card.querySelector('.mesh.viewport');
    this.panoRenderer=renderer(this.panoHost,`${spec.epoch?'후':'전'} RGB와 분할 메쉬 360도 비교`);
    this.meshRenderer=renderer(this.meshHost,`${spec.epoch?'후':'전'} 분할 메쉬 3D 회전`);
    this.panoScene=new THREE.Scene();this.meshScene=new THREE.Scene();this.meshScene.background=new THREE.Color('#101b2a');
    this.panoCamera=new THREE.PerspectiveCamera(85,1,.005,2000);this.meshCamera=new THREE.PerspectiveCamera(48,1,.01,2000);
    this.yaw=0;this.pitch=0;this.azimuth=.75;this.elevation=.65;
    this.photoGroup=new THREE.Group();this.photoGroup.matrixAutoUpdate=false;this.panoScene.add(this.photoGroup);
    const depthMaterial=new THREE.MeshBasicMaterial({side:THREE.DoubleSide,colorWrite:false});
    for(const [scene,group] of [[this.panoScene,this.photoGroup],[this.meshScene,this.meshScene]]) {
      const depth=new THREE.Mesh(geometry,depthMaterial);depth.renderOrder=-3;group.add(depth);
      if(scene===this.meshScene)this.meshDepth=depth;
    }
    this.overlay=new THREE.Mesh(geometry,shapeMaterial(true));this.overlay.renderOrder=1;this.photoGroup.add(this.overlay);
    const context=new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({color:'#8ca0b6',side:THREE.DoubleSide,transparent:true,opacity:.14,depthWrite:false}));
    context.renderOrder=-2;this.meshScene.add(context);this.meshContext=context;
    this.mesh=new THREE.Mesh(geometry,shapeMaterial());this.mesh.material.uniforms.opacity.value=1;
    this.mesh.material.transparent=false;this.mesh.material.depthWrite=true;this.meshScene.add(this.mesh);
    this.uniforms={rgb:{value:null},outline:{value:null},showRgb:{value:1},showOutline:{value:1}};
    const sphere=new THREE.Mesh(new THREE.SphereGeometry(900,64,32),new THREE.ShaderMaterial({
      side:THREE.BackSide,depthWrite:false,depthTest:false,uniforms:this.uniforms,
      vertexShader:'varying vec3 ray;void main(){ray=position;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}',
      fragmentShader:`uniform sampler2D rgb;uniform sampler2D outline;uniform float showRgb;uniform float showOutline;varying vec3 ray;${uvCode}
        void main(){vec2 uv=panoUV(ray);vec4 s=texture2D(outline,uv);vec3 c=mix(vec3(.015,.025,.045),texture2D(rgb,uv).rgb,showRgb);gl_FragColor=vec4(mix(c,s.rgb,s.a*showOutline),1.);
        #include <colorspace_fragment>
        }`}));sphere.renderOrder=-10;this.panoScene.add(sphere);
    const bounds=new THREE.Box3(new THREE.Vector3(...spec.bounds[0]),new THREE.Vector3(...spec.bounds[1]));
    const labelled=new THREE.Box3(),positions=geometry.attributes.position;
    ids.forEach((id,i)=>{if(id>0)for(let j=0;j<3;j++)labelled.expandByPoint(new THREE.Vector3().fromBufferAttribute(positions,i*3+j));});
    if(!labelled.isEmpty())bounds.copy(labelled);
    this.allBounds=bounds;this.orbitTarget=bounds.getCenter(new THREE.Vector3());this.radius=Math.max(bounds.getSize(new THREE.Vector3()).length()*.7,1);
    this.photoMarker=new THREE.Mesh(new THREE.SphereGeometry(.09),new THREE.MeshBasicMaterial({color:'#82eeff',depthTest:false}));
    this.photoMarker.renderOrder=5;this.meshScene.add(this.photoMarker);
    this.bindPointer(this.panoRenderer.domElement,true);this.bindPointer(this.meshRenderer.domElement,false);
    this.meshRenderer.domElement.ondblclick=()=>this.fit(bounds);
    new ResizeObserver(()=>this.render()).observe(this.panoHost);new ResizeObserver(()=>this.render()).observe(this.meshHost);
    this.frameSelect=card.querySelector('.frame');this.objectSelect=card.querySelector('.objects');
    this.frameSelect.setAttribute('aria-label',`${spec.epoch?'후':'전'} 원본 프레임`);
    this.objectSelect.setAttribute('aria-label',`${spec.epoch?'후':'전'} 메쉬 객체`);
    spec.frames.forEach((f,i)=>{const o=new Option(`${f.id} · ${f.timestamp_s?.toFixed(2)??'—'}초`,String(i));this.frameSelect.add(o);});
    spec.objects.forEach(o=>this.objectSelect.add(new Option(`${names[o.label]||o.label} #${o.id} · ${o.faces}면`,String(o.id))));
    this.frameSelect.onchange=()=>this.showFrame(Number(this.frameSelect.value)).catch(fail);
    card.querySelector('.previous').onclick=()=>this.showFrame(Math.max(0,this.index-1)).catch(fail);
    card.querySelector('.next').onclick=()=>this.showFrame(Math.min(spec.frames.length-1,this.index+1)).catch(fail);
    this.objectSelect.onchange=()=>this.choose(Number(this.objectSelect.value));
    card.querySelector('.focus').onclick=()=>this.focusObject();
    card.querySelector('.epoch').textContent=spec.epoch?'후':'전';card.querySelector('h2').textContent=spec.name;
    card.querySelector('.counts').textContent=`분할 객체 ${spec.objects.length}개 · ${spec.labelled_faces.toLocaleString()} / ${spec.faces.toLocaleString()}면`;
  }
  bindPointer(canvas, photo) {
    let drag;
    canvas.onpointerdown=e=>{canvas.focus();canvas.setPointerCapture(e.pointerId);drag={x:e.clientX,y:e.clientY,a:photo?this.yaw:this.azimuth,b:photo?this.pitch:this.elevation,moved:false};};
    canvas.onpointermove=e=>{if(!drag)return;const x=e.clientX-drag.x,y=e.clientY-drag.y;drag.moved ||= Math.hypot(x,y)>5;
      if(photo){this.yaw=drag.a-x*.004;this.pitch=THREE.MathUtils.clamp(drag.b+y*.004,-1.5,1.5);}
      else{this.azimuth=drag.a-x*.006;this.elevation=THREE.MathUtils.clamp(drag.b+y*.006,-1.45,1.45);}this.render();};
    canvas.onpointerup=e=>{if(drag&&!drag.moved)this.pick(e,photo);drag=null;};canvas.onpointercancel=()=>{drag=null;};
    canvas.onwheel=e=>{e.preventDefault();if(photo)this.panoCamera.fov=THREE.MathUtils.clamp(this.panoCamera.fov+e.deltaY*.04,30,115);
      else this.radius=THREE.MathUtils.clamp(this.radius*Math.exp(e.deltaY*.001),.1,500);this.render();};
    canvas.onkeydown=e=>{const d={ArrowLeft:-1,ArrowRight:1,ArrowUp:1,ArrowDown:-1}[e.key];if(!d)return;e.preventDefault();
      if(photo){if(e.key==='ArrowLeft'||e.key==='ArrowRight')this.yaw+=d*.1;else this.pitch=THREE.MathUtils.clamp(this.pitch+d*.1,-1.5,1.5);}
      else this.azimuth+=d*.1;this.render();};
  }
  async showFrame(index) {
    const token=++this.token,f=this.spec.frames[index];this.index=index;this.frameSelect.value=String(index);
    this.card.querySelector('.previous').disabled=index===0;this.card.querySelector('.next').disabled=index===this.spec.frames.length-1;
    const loader=new THREE.TextureLoader();
    const textures=await Promise.all([f.image,f.outline,f.person].map(path=>loader.loadAsync(base+path)));
    if(token!==this.token){textures.forEach(t=>t.dispose());return;}
    this.textures?.forEach(t=>t.dispose());this.textures=textures;textures[0].colorSpace=THREE.SRGBColorSpace;
    textures.forEach(t=>{t.wrapS=THREE.RepeatWrapping;});
    this.uniforms.rgb.value=textures[0];this.uniforms.outline.value=textures[1];this.overlay.material.uniforms.person.value=textures[2];
    this.photoGroup.matrix.copy(rowMatrix(f.camera_from_world));this.photoGroup.updateMatrixWorld(true);
    this.photoMarker.position.setFromMatrixPosition(this.photoGroup.matrix.clone().invert());
    this.apply();this.render();
  }
  apply() {
    this.overlay.visible=$('overlay').checked;this.overlay.material.uniforms.opacity.value=Number($('opacity').value)/100;
    this.overlay.material.wireframe=$('wire').checked;
    this.overlay.material.uniforms.excludePerson.value=$('person').checked?1:0;
    this.uniforms.showRgb.value=$('rgb').checked?1:0;this.uniforms.showOutline.value=$('outline').checked?1:0;
    this.meshDepth.visible=this.meshContext.visible=$('context').checked;
    this.render();
  }
  choose(id) {
    this.selected=id;this.objectSelect.value=String(id);
    this.overlay.material.uniforms.selected.value=id;this.mesh.material.uniforms.selected.value=id;
    const o=this.spec.objects.find(o=>o.id===id);
    this.card.querySelector('.picked').textContent=o?`${names[o.label]||o.label} #${id} · ${o.faces}개 원본 삼각형`:'전체 분할 객체';this.render();
  }
  objectBounds() {
    if(!this.selected)return this.allBounds.clone();
    const box=new THREE.Box3(),p=this.geometry.attributes.position;
    for(let face=0;face<this.ids.length;face++)if(this.ids[face]===this.selected)
      for(let j=0;j<3;j++)box.expandByPoint(new THREE.Vector3().fromBufferAttribute(p,face*3+j));
    return box;
  }
  focusObject() {
    if(!this.selected){this.yaw=0;this.pitch=0;this.panoCamera.fov=85;this.fit(this.allBounds);return;}
    const bounds=this.objectBounds(),target=bounds.getCenter(new THREE.Vector3()).applyMatrix4(this.photoGroup.matrix).normalize();
    this.yaw=Math.atan2(target.x,-target.z);this.pitch=Math.asin(target.y);this.panoCamera.fov=55;this.fit(bounds);
  }
  fit(box) {this.orbitTarget.copy(box.getCenter(new THREE.Vector3()));this.radius=Math.max(box.getSize(new THREE.Vector3()).length()*1.4,.5);this.render();}
  pick(e,photo) {
    const canvas=photo?this.panoRenderer.domElement:this.meshRenderer.domElement,r=canvas.getBoundingClientRect(),cast=new THREE.Raycaster();
    cast.setFromCamera(new THREE.Vector2((e.clientX-r.left)/r.width*2-1,1-(e.clientY-r.top)/r.height*2),photo?this.panoCamera:this.meshCamera);
    const hits=cast.intersectObject(photo?this.overlay:this.mesh);
    const hit=photo||$('context').checked?hits[0]:hits.find(h=>this.ids[h.faceIndex]>0&&(!this.selected||this.ids[h.faceIndex]===this.selected));
    if(hit){const id=this.ids[hit.faceIndex];if(id>0)this.choose(id);else this.card.querySelector('.picked').textContent='미분할 표면 · 객체 ID 없음';}
  }
  render() {
    if(!this.panoCamera)return;
    const size=(r,h,c)=>{if(!h.clientWidth||!h.clientHeight)return false;r.setSize(h.clientWidth,h.clientHeight,false);c.aspect=h.clientWidth/h.clientHeight;c.updateProjectionMatrix();return true;};
    if(this.uniforms.rgb.value&&size(this.panoRenderer,this.panoHost,this.panoCamera)){
      this.panoCamera.lookAt(Math.sin(this.yaw)*Math.cos(this.pitch),Math.sin(this.pitch),-Math.cos(this.yaw)*Math.cos(this.pitch));
      this.panoCamera.updateMatrixWorld();this.panoRenderer.render(this.panoScene,this.panoCamera);
    }
    if(size(this.meshRenderer,this.meshHost,this.meshCamera)){
      this.meshCamera.position.copy(this.orbitTarget).add(new THREE.Vector3(Math.sin(this.azimuth)*Math.cos(this.elevation),Math.sin(this.elevation),Math.cos(this.azimuth)*Math.cos(this.elevation)).multiplyScalar(this.radius));
      this.meshCamera.lookAt(this.orbitTarget);this.meshCamera.updateMatrixWorld();this.meshRenderer.render(this.meshScene,this.meshCamera);
    }
  }
}

async function start() {
  if(!/^[a-f0-9]{24}$/.test(job||''))throw Error('완료된 변화탐지 작업의 job 번호가 필요합니다.');
  base=`/api/mesh-change-jobs/${job}/segmentation-review/`;
  const [data,module]=await Promise.all([fetchJson(base+'manifest.json'),import(`/api/mesh-change-jobs/${job}/files/preview/vendor/three.module.js`)]);THREE=module;
  for(const spec of data.scenes){
    const response=await fetch(base+spec.mesh);if(!response.ok)throw Error('분할 메쉬를 읽지 못했습니다.');const bytes=await response.arrayBuffer(),n=spec.vertices;
    if(bytes.byteLength!==n*16+spec.faces*4)throw Error('분할 메쉬의 배열 크기가 일치하지 않습니다.');
    const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(bytes,0,n*3),3));
    geometry.setAttribute('shade',new THREE.BufferAttribute(new Uint8Array(bytes,n*12,n*4),4,true));
    const ids=new Uint32Array(bytes,n*16,spec.faces),instances=new Float32Array(n);ids.forEach((id,i)=>instances.fill(id,i*3,i*3+3));
    geometry.setAttribute('instance',new THREE.BufferAttribute(instances,1));geometry.computeBoundingSphere();
    const card=$('capture').content.firstElementChild.cloneNode(true);$('pair').append(card);
    const review=new CaptureReview(spec,geometry,ids,card);reviews.push(review);
    const frame=Math.max(0,spec.frames.findIndex(f=>f.id===query.get('frame'+spec.epoch)));
    await review.showFrame(frame);
    const object=Number(query.get('object'+spec.epoch));if(spec.objects.some(o=>o.id===object))review.choose(object);
    for(const [key,limit] of [['yaw',Math.PI*2],['pitch',1.5]]){
      const value=Number(query.get(key+spec.epoch));if(Number.isFinite(value))review[key]=THREE.MathUtils.clamp(value,-limit,limit);
    }
    if(query.get('fov'))review.panoCamera.fov=THREE.MathUtils.clamp(Number(query.get('fov'))||85,30,115);
    review.render();
  }
  $('status').textContent='기존 분할 입력 · 탐지 재연산 없음';$('scope').textContent=data.scope;
  $('provenance').textContent=JSON.stringify({job,projection:data.projection,person_mask_source:data.person_mask_source,
    captures:data.scenes.map(s=>({capture:s.name,mesh_sha256:s.source_mesh_sha256,frames:s.frames.length,labelled_faces:s.labelled_faces}))},null,2);
  for(const id of ['overlay','opacity','outline','person','wire','rgb','context'])$(id).oninput=()=>{
    $('opacity-value').textContent=$('opacity').value+'%';reviews.forEach(r=>r.apply());};
  $('reset').onclick=()=>reviews.forEach(r=>{r.yaw=0;r.pitch=0;r.panoCamera.fov=85;r.render();});
}
start().catch(fail);

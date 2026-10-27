import * as THREE from 'three';
import { OrbitControls } from './preview/vendor/OrbitControls.js';
import { photoLayers, photoCaption } from './photo-presentation.js';

const $ = id => document.getElementById(id);
const candidateStyle={text:'변화 후보',color:'#e9bb73',order:0};
const kinds=Object.fromEntries(['added','moved','removed','modified','unresolved'].map(kind=>[kind,candidateStyle]));
const names = { chair: '의자', table: '테이블', 'dining table': '테이블', door: '문',
  refrigerator: '냉장고', 'fire extinguisher': '소화기', window: '창문', cabinet: '수납장',
  sofa: '소파', box:'박스', person:'사람', unknown: '미분류 객체', plant: '화분', monitor: '모니터', 'vacuum cleaner': '청소기' };
const reasons = {
  registered_rgb_instance_without_mesh:'원본 객체 분할 영역에서 SimSaC 차이가 여러 프레임에 반복됐습니다. 촬영자를 제외하고 메쉬·점군으로 정합한 카메라를 사용합니다. 원본 메쉬가 불완전한 객체는 RGB 마스크로 표시합니다.',
  segmented_object_appears_over_observed_background: '객체가 있는 촬영의 원본 분할과 SimSaC에서 반복 확인됐습니다. 반대 촬영은 같은 위치의 배경을 가림 없이 반복 관측했고, 같은 종류의 객체 마스크는 검출되지 않았습니다. 메쉬는 위치를 제공하며, 이 후보는 영상 분할에 의존하는 자동 추정입니다.',
  counterpart_in_rgb_segmentation_but_mesh_incomplete: '반대 촬영의 원본 세그멘테이션에도 같은 종류의 객체가 보입니다. 메쉬 분할이 누락되거나 불완전해 원본 영상을 직접 비교할 후보로 남겼습니다.',
  segmented_identity_observation_insufficient: '메쉬와 원본 세그멘테이션이 일치하는 독립 관측이 부족합니다. 전후 객체 조각의 대응을 더 확인해야 합니다.',
  one_sided_rgb_difference: '같은 위치에 대응하지만 SimSaC 변화가 양쪽 촬영에서 반복 확인되지 않았습니다. 시점·조명·분할 차이일 수 있어 원본에서 차이를 더 확인해야 합니다.',
  registration_uncertainty_unsupported: '이 구역의 별도 검증 표면에서 정합 오차를 충분히 확인하지 못했습니다. 전후 차이를 해석할 때 이 구역의 오차를 함께 확인해야 합니다.',
  rigid_identity_and_both_vacancies_supported: '전후 객체의 형상과 원본 영상 특징이 대응합니다. 정합 오차보다 큰 위치 차이가 있고, 두 위치의 반대 촬영에서도 주변 공간을 반복 관측했습니다.',
  expected_location_observed_free_in_independent_frames: '객체가 있는 촬영에서 반복 관측됐고, 반대 촬영의 같은 자리는 가림 없이 더 먼 배경으로 관측됐습니다.',
  same_location_object_with_repeatable_simsac_surface_change: '전후 객체는 같은 위치에 대응합니다. 해당 객체 표면에서 SimSaC 변화가 여러 원본 프레임에 반복돼 후보 표면으로 연결했습니다.',
  counterpart_identity_ambiguous: '비슷한 종류·형상의 전후 객체가 있어 대응이 모호합니다. 전후 원본에서 같은 객체인지 추가 확인이 필요합니다.',
  own_observation_insufficient: '객체를 독립적으로 확인할 수 있는 원본 프레임이 부족합니다.',
  surface_still_occupies_expected_location: '반대 촬영의 예상 자리에 표면이 남아 있습니다. 분할 누락·외관 변화·가림을 구분할 근거가 부족합니다.',
  opposite_location_occluded_or_unobserved: '반대 촬영의 같은 자리가 가려졌거나 충분히 관측되지 않았습니다. 객체의 같은 위치를 원본 영상에서 더 확인해야 합니다.',
};
const json = async url => {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} 읽기 실패 (${r.status})`);
  return r.json();
};
const buffers = new Map();
const loadBuffer = url => {
  if (!buffers.has(url)) buffers.set(url, fetch(url).then(r => {
    if (!r.ok) throw new Error(`3D 파일 읽기 실패: ${url}`);
    return r.arrayBuffer();
  }));
  return buffers.get(url);
};
const images = new Map();
const loadImage = url => {
  if (!images.has(url)) images.set(url, new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`원본 영상 읽기 실패: ${url}`));
    image.src = url;
  }));
  return images.get(url);
};
const host = $('viewport');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
host.append(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color('#111c29');
const camera = new THREE.PerspectiveCamera(44, 1, .01, 2000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xe8f4ff, 0x344254, 2.3));
const light = new THREE.DirectionalLight(0xffffff, 2.5);
light.position.set(3, 10, 7);
scene.add(light);
const context = [], clouds = [], selectedGroup = new THREE.Group(), markerGroup = new THREE.Group();
scene.add(selectedGroup, markerGroup);
let data, filtered = [], selected = null, photoIndex = 0, filter = 'all', phase = 'both';
let photoGhost = false, photoGeneration = 0;
let generation = 0, globalBounds = new THREE.Box3(), transition = null;
const geometryCache = new Map();
const query = new URLSearchParams(location.search);
const reportError = error => { $('error').textContent = error.message; console.error(error); };
const typeName = () => '변화 후보';
const objectName = event => names[event.label] || event.label;
const asBox = bounds => new THREE.Box3(new THREE.Vector3(...bounds[0]), new THREE.Vector3(...bounds[1]));

function flyTo(bounds, overview = false) {
  const box = asBox(bounds), target = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = Math.max(size.length() * (overview ? .8 : 1.5), .65);
  const direction = new THREE.Vector3(.8, .55, 1).normalize();
  transition = { time: performance.now(), fromPosition: camera.position.clone(),
    fromTarget: controls.target.clone(), position: target.clone().addScaledVector(direction, radius), target };
}
function overview() { flyTo([globalBounds.min.toArray(), globalBounds.max.toArray()], true); }
renderer.domElement.addEventListener('pointerdown', () => { transition = null; });
function resize() {
  if (!host.clientWidth || !host.clientHeight) return;
  camera.aspect = host.clientWidth / host.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(host.clientWidth, host.clientHeight);
  if (selected) drawPhotos().catch(reportError);
}
new ResizeObserver(resize).observe(host);
resize();
renderer.setAnimationLoop(() => {
  if (!host.clientHeight) return;
  if (transition) {
    const t = Math.min(1, (performance.now() - transition.time) / 550);
    const smooth = t * t * (3 - 2 * t);
    camera.position.lerpVectors(transition.fromPosition, transition.position, smooth);
    controls.target.lerpVectors(transition.fromTarget, transition.target, smooth);
    if (t === 1) transition = null;
  }
  controls.update();
  renderer.render(scene, camera);
});

function clearGroup(group) {
  for (const child of [...group.children]) {
    child.traverse(item => {
      if (item.geometry && !item.userData.cachedGeometry) item.geometry.dispose();
      if (item.material) {
        for (const material of Array.isArray(item.material) ? item.material : [item.material]) material.dispose();
      }
    });
    group.remove(child);
  }
}
function boxLines(bounds, color, dashed = false) {
  const b = asBox(bounds), geometry = new THREE.EdgesGeometry(new THREE.BoxGeometry(...b.getSize(new THREE.Vector3()).toArray()));
  const material = dashed ? new THREE.LineDashedMaterial({ color, dashSize: .055, gapSize: .035, depthTest: false })
    : new THREE.LineBasicMaterial({ color, transparent: true, opacity: .8, depthTest: false });
  const line = new THREE.LineSegments(geometry, material);
  if (dashed) line.computeLineDistances();
  line.position.copy(b.getCenter(new THREE.Vector3()));
  return line;
}
async function objectMesh(key, color) {
  const record = data.objects[key];
  if (!geometryCache.has(key)) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(await loadBuffer(record.mesh)), 3));
    geometry.computeVertexNormals();
    geometryCache.set(key, geometry);
  }
  const mesh = new THREE.Mesh(geometryCache.get(key), new THREE.MeshStandardMaterial({
    color, side: THREE.DoubleSide, transparent: true, opacity: Number($('ghost-opacity').value)/100, depthWrite: false, depthTest: false, roughness: .8,
  }));
  mesh.userData = { cachedGeometry: true, epoch: record.epoch };
  mesh.renderOrder = 3 + record.epoch;
  return mesh;
}
function updateVisibility() {
  for (let i = 0; i < context.length; i++) {
    const on = phase === 'both' || phase === String(i);
    context[i].visible = $('context').checked && on;
    clouds[i].visible = $('pointcloud').checked && on;
  }
  for (const item of selectedGroup.children) {
    item.visible = item.userData.epoch == null || phase === 'both' || phase === String(item.userData.epoch);
  }
  for (const button of document.querySelectorAll('[data-phase]')) button.setAttribute('aria-pressed', String(button.dataset.phase === phase));
}
async function showGeometry(event, token) {
  clearGroup(selectedGroup);
  if(event.rgb_only){$('map-legend').textContent='이 객체는 RGB 마스크로 관측됐으며 원본 메쉬가 불완전합니다.';return;}
  const beforeColor = '#57d8dd';
  const afterColor = '#ffc26a';
  const parts = await Promise.all(event.object_ids.map(key => objectMesh(key, data.objects[key].epoch ? afterColor : beforeColor)));
  if (generation !== token) { for (const mesh of parts) mesh.material.dispose(); return; }
  for (const mesh of parts) {
    selectedGroup.add(mesh);
    const key = event.object_ids.find(k => data.objects[k].epoch === mesh.userData.epoch);
    const line = boxLines(data.objects[key].bounds, mesh.userData.epoch ? afterColor : beforeColor);
    line.userData.epoch = mesh.userData.epoch;
    selectedGroup.add(line);
  }
  if (event.type === 'moved' && event.before && event.after) {
    const a = new THREE.Vector3(...data.objects[event.before].center), b = new THREE.Vector3(...data.objects[event.after].center);
    const delta = b.clone().sub(a), length = delta.length();
    if (length > 0) {
      const arrow = new THREE.ArrowHelper(delta.normalize(), a, length, afterColor, Math.min(length * .22, .25), Math.min(length * .1, .12));
      arrow.traverse(item => { if (item.material) { item.material.depthTest = false; item.renderOrder = 8; } });
      selectedGroup.add(arrow);
    }
  } else if (event.type === 'added' || event.type === 'removed') {
    const line = boxLines(event.bounds, kinds[event.type].color, true);
    line.userData.epoch = event.type === 'added' ? 0 : 1;
    selectedGroup.add(line);
  }
  $('map-legend').textContent='청록: 전 촬영의 선택 객체 · 주황: 후 촬영의 선택 객체';
  updateVisibility();
  flyTo(event.bounds);
}
const panoramas = new Map();
function syncPhotoComparison() {
  $('photo-ghost').setAttribute('aria-pressed',String(photoGhost));
  $('photo-ghost').textContent=photoGhost?'원본으로 돌아가기':'박스 클릭 · 객체 전후 비교';
  $('photo-hint').textContent=photoGhost?'선택한 객체만 강조합니다. 다시 누르면 바운딩박스만 표시합니다.':'박스를 누르면 그 객체의 전후 모습만 강조합니다.';
}
function comparePhoto(on) {
  photoGhost=on;phase='both';updateVisibility();syncPhotoComparison();
  $('photo-mode').value=on?'object':'context';
  for(const p of panoramas.values())p.focus();
  drawPhotos().catch(reportError);
}
class Panorama {
  constructor(wrapper, epoch) {
    this.wrapper=wrapper; this.epoch=epoch; this.key=null; this.token=0;
    this.renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio,2));
    this.renderer.domElement.className='panorama-canvas';
    this.renderer.domElement.setAttribute('aria-label',`${epoch?'후':'전'} 360도 원본 영상 · 드래그 회전`);
    this.renderer.domElement.tabIndex=0;
    wrapper.append(this.renderer.domElement);
    this.scene=new THREE.Scene();this.camera=new THREE.PerspectiveCamera(85,1,.001,2000);
    this.group=new THREE.Group();this.scene.add(this.group);
    const uvCode=`vec2 panoUV(vec3 p){vec3 d=normalize(vec3(p.x,-p.y,-p.z));return vec2(atan(d.x,d.z)/6.28318530718+.5,.5-asin(clamp(d.y,-1.,1.))/3.14159265359);}`;
    this.uniforms={rgb:{value:null},person:{value:null}};
    this.background=new THREE.Mesh(new THREE.SphereGeometry(900,64,32),new THREE.ShaderMaterial({
      side:THREE.BackSide,depthWrite:false,uniforms:this.uniforms,
      vertexShader:'varying vec3 ray;void main(){ray=position;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}',
      fragmentShader:`uniform sampler2D rgb;varying vec3 ray;${uvCode}void main(){gl_FragColor=texture2D(rgb,panoUV(ray));\n#include <colorspace_fragment>\n}`
    }));this.background.renderOrder=-10;this.scene.add(this.background);
    this.uvCode=uvCode;this.boxes=[];this.boxPositions=[];
    this.hint=document.createElement('span');this.hint.className='pano-hint';this.hint.textContent='360° · 드래그 회전 · 휠 확대';wrapper.append(this.hint);
    let drag=null;
    const canvas=this.renderer.domElement;
    canvas.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,yaw:this.yaw,pitch:this.pitch,moved:false};canvas.setPointerCapture(e.pointerId);};
    canvas.onpointermove=e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.moved ||= Math.hypot(dx,dy)>5;
      this.yaw=drag.yaw-dx*.004;this.pitch=THREE.MathUtils.clamp(drag.pitch+dy*.004,-1.5,1.5);this.render();};
    canvas.onpointerup=e=>{if(drag&&!drag.moved){
      const inBox=this.boxes.some(box=>{if(box.hidden)return false;const r=box.getBoundingClientRect();return e.clientX>=r.left&&e.clientX<=r.right&&e.clientY>=r.top&&e.clientY<=r.bottom;});
      if(inBox)comparePhoto(!photoGhost);}
      drag=null;};
    canvas.onpointercancel=()=>{drag=null;};
    canvas.onwheel=e=>{e.preventDefault();this.camera.fov=THREE.MathUtils.clamp(this.camera.fov+e.deltaY*.04,35,115);this.render();};
    canvas.onkeydown=e=>{const steps={ArrowLeft:[-.12,0],ArrowRight:[.12,0],ArrowUp:[0,.1],ArrowDown:[0,-.1]};
      if(e.key==='Enter'||e.key===' '){e.preventDefault();comparePhoto(!photoGhost);return;}
      if(steps[e.key]){e.preventDefault();this.yaw+=steps[e.key][0];this.pitch=THREE.MathUtils.clamp(this.pitch+steps[e.key][1],-1.5,1.5);this.render();}};
    new ResizeObserver(()=>this.render()).observe(wrapper);
  }
  hide(){this.renderer.domElement.hidden=true;this.hint.hidden=true;for(const b of this.boxes)b.hidden=true;}
  focus(){if(!this.direction)return;const d=new THREE.Vector3(...this.direction).normalize();this.yaw=Math.atan2(d.x,-d.z);this.pitch=Math.asin(d.y);this.camera.fov=$('photo-mode').value==='object'?55:85;this.render();}
  async show(record,event){
    const p=record.panorama,key=p.image+':'+event.id;
    this.renderer.domElement.hidden=false;this.hint.hidden=false;
    if(this.key!==key){this.key=key;const token=++this.token;
      this.group.visible=false;for(const box of this.boxes)box.hidden=true;
      const [rgb,person]=await Promise.all([loadImage(p.image),loadImage(p.person_mask)]);
      if(token!==this.token)return;
      for(const name of ['rgb','person'])this.uniforms[name].value?.dispose();
      const texture=new THREE.Texture(rgb);texture.colorSpace=THREE.SRGBColorSpace;texture.wrapS=THREE.RepeatWrapping;texture.needsUpdate=true;
      const mask=new THREE.Texture(person);mask.needsUpdate=true;
      this.uniforms.rgb.value=texture;this.uniforms.person.value=mask;
      clearGroup(this.group);for(const b of this.boxes)b.remove();this.boxes=[];
      const layers=photoLayers(event,record,this.epoch,true);
      this.boxPositions=new Float32Array(layers.bounds.flatMap(g=>g.positions));
      for(const g of layers.highlights){
        if(g.observed===false)continue;
        const imageMask=g.mask_image?await loadImage(g.mask_image):null;
        if(token!==this.token)return;
        const instanceMask=imageMask?new THREE.Texture(imageMask):null;
        if(instanceMask)instanceMask.needsUpdate=true;
        const geometry=imageMask?new THREE.SphereGeometry(10,64,32):new THREE.BufferGeometry();
        if(!imageMask)geometry.setAttribute('position',new THREE.Float32BufferAttribute(g.positions,3));
        const material=new THREE.ShaderMaterial({transparent:true,side:THREE.DoubleSide,depthWrite:false,depthTest:false,
          uniforms:{person:{value:mask},instanceMask:{value:instanceMask||mask},colour:{value:new THREE.Color(g.epoch?'#ffc26a':'#57d8dd')},alpha:{value:0}},
          vertexShader:'varying vec3 ray;void main(){ray=position;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}',
          fragmentShader:`uniform sampler2D person;uniform sampler2D instanceMask;uniform vec3 colour;uniform float alpha;varying vec3 ray;${this.uvCode}void main(){if(texture2D(person,panoUV(ray)).r>.5)discard;${imageMask?'if(texture2D(instanceMask,panoUV(ray)).r<.5)discard;':''}gl_FragColor=vec4(colour,alpha);\n#include <colorspace_fragment>\n}`});
        const mesh=new THREE.Mesh(geometry,material);mesh.renderOrder=g.epoch+1;mesh.userData.epoch=g.epoch;this.group.add(mesh);
      }
      const box=document.createElement('div');box.className='pano-box';box.style.borderColor='#ffc26a';
      box.style.borderStyle=record.role==='observed_object'?'solid':'dashed';
      box.setAttribute('aria-hidden','true');this.wrapper.append(box);this.boxes.push(box);
      this.direction=p.direction;this.focus();
    }
    for(const box of this.boxes){box.textContent='';box.dataset.label=photoGhost?photoCaption(event,this.epoch):'변화 후보';box.classList.toggle('comparing',photoGhost);}
    this.group.visible=photoGhost;
    for(const mesh of this.group.children)mesh.material.uniforms.alpha.value=Number($('ghost-opacity').value)/100;
    this.render();
  }
  render(){
    if(this.renderer.domElement.hidden||!this.uniforms.rgb.value)return;
    const w=this.wrapper.clientWidth,h=this.wrapper.clientHeight;if(!w||!h)return;
    this.renderer.setSize(w,h,false);this.camera.aspect=w/h;this.camera.updateProjectionMatrix();
    this.camera.lookAt(Math.sin(this.yaw)*Math.cos(this.pitch),Math.sin(this.pitch),-Math.cos(this.yaw)*Math.cos(this.pitch));this.camera.updateMatrixWorld();
    this.renderer.render(this.scene,this.camera);
    const view=this.camera.matrixWorldInverse;
    this.boxes.forEach(box=>{const pos=this.boxPositions;let x0=Infinity,y0=Infinity,x1=-Infinity,y1=-Infinity;
      for(let i=0;i<pos.length;i+=3){const p=new THREE.Vector3(pos[i],pos[i+1],pos[i+2]);if(p.clone().applyMatrix4(view).z>=-.001)continue;p.project(this.camera);x0=Math.min(x0,p.x);x1=Math.max(x1,p.x);y0=Math.min(y0,p.y);y1=Math.max(y1,p.y);}
      box.hidden=!(x1>=-1&&x0<=1&&y1>=-1&&y0<=1);if(box.hidden)return;
      const left=THREE.MathUtils.clamp((x0+1)*w/2,0,w),top=THREE.MathUtils.clamp((1-y1)*h/2,0,h);
      Object.assign(box.style,{left:left+'px',top:top+'px',width:Math.max(1,Math.min(w,(x1+1)*w/2)-left)+'px',height:Math.max(1,Math.min(h,(1-y0)*h/2)-top)+'px'});
    });
  }
}
async function drawPhoto(epoch, record, token) {
  const flat=$('photo-'+epoch), flatWrapper=flat.parentElement;
  if(record?.panorama){
    flat.hidden=true;flatWrapper.querySelector('.photo-hit')?.remove();$('photo-empty-'+epoch).hidden=true;
    $('photo-role-'+epoch).textContent=record.role==='observed_object'?'분할 객체 · 360°':'예상 자리 · 360°';
    $('photo-info-'+epoch).textContent=`원본 360° 프레임 ${record.frame_id} · ${record.timestamp_s?.toFixed(2)??'—'}초${photoGhost?' · '+photoCaption(selected,epoch):''}`;
    if(!panoramas.has(epoch))panoramas.set(epoch,new Panorama(flatWrapper,epoch));
    await panoramas.get(epoch).show(record,selected);return;
  }
  panoramas.get(epoch)?.hide();flat.hidden=false;
  const canvas = $('photo-' + epoch), ctx = canvas.getContext('2d');
  const dpr = Math.min(devicePixelRatio, 2);
  canvas.width = Math.round(canvas.clientWidth * dpr);
  canvas.height = Math.round(canvas.clientHeight * dpr);
  ctx.fillStyle = '#09131d';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  $('photo-empty-' + epoch).hidden = !!record;
  $('photo-role-' + epoch).textContent = record ? (record.role === 'observed_object' ? '관측한 객체' : '반대 촬영의 예상 자리') : '관측 미확보';
  $('photo-info-' + epoch).textContent = record ? `원본 프레임 ${record.frame_id} · ${record.timestamp_s == null ? '시각 미기록' : record.timestamp_s.toFixed(2) + '초'} · ${record.face}` : '';
  if (!record) { canvas.parentElement.querySelector('.photo-hit')?.remove(); return; }
  const image = await loadImage(record.image);
  if (token !== photoGeneration) return;
  let [x0, y0, x1, y1] = record.bbox;
  const factor = $('photo-mode').value === 'object' ? 1.55 : 2.5;
  const minimum = $('photo-mode').value === 'object' ? 190 : 440;
  const w = Math.min(image.width, Math.max(minimum, (x1 - x0) * factor));
  const h = Math.min(image.height, Math.max(minimum, (y1 - y0) * factor));
  const x = Math.max(0, Math.min(image.width - w, (x0 + x1 - w) / 2));
  const y = Math.max(0, Math.min(image.height - h, (y0 + y1 - h) / 2));
  const scale = Math.min(canvas.width / w, canvas.height / h);
  const ox = (canvas.width - w * scale) / 2, oy = (canvas.height - h * scale) / 2;
  ctx.drawImage(image, x, y, w, h, ox, oy, w * scale, h * scale);
  if (photoGhost) {
    const overlays = await Promise.all(photoLayers(selected,record,epoch).highlights.map(g => loadImage(g.image)));
    if (token !== photoGeneration) return;
    ctx.globalAlpha = Number($('ghost-opacity').value)/65;
    for (const overlay of overlays) ctx.drawImage(overlay, x, y, w, h, ox, oy, w * scale, h * scale);
    ctx.globalAlpha = 1;
  }
  const wrapper = canvas.parentElement;
  wrapper.querySelector('.photo-hit')?.remove();
  const hit = document.createElement('button'); hit.className = 'photo-hit';
  hit.setAttribute('aria-label', '변화 박스: 선택 객체 전후 비교');
  Object.assign(hit.style, {left: ((ox+(x0-x)*scale)/dpr)+'px', top: ((oy+(y0-y)*scale)/dpr)+'px',
    width: Math.max(20,(x1-x0)*scale/dpr)+'px', height: Math.max(20,(y1-y0)*scale/dpr)+'px'});
  hit.onclick = () => comparePhoto(!photoGhost);
  wrapper.append(hit);
  const color = selected.type === 'moved' && epoch === 0 ? '#57d8dd' : kinds[selected.type].color;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2 * dpr;
  ctx.setLineDash(record.role === 'expected_location' ? [7 * dpr, 5 * dpr] : []);
  ctx.strokeRect(ox + (x0 - x) * scale, oy + (y0 - y) * scale, (x1 - x0) * scale, (y1 - y0) * scale);
  ctx.setLineDash([]);
}
async function drawPhotos() {
  if (!selected) return;
  const token = ++photoGeneration;
  const sets = selected.photo_sets || [], record = sets[photoIndex] || {};
  $('photo-position').textContent = `${photoIndex + 1} / ${Math.max(sets.length, 1)}`;
  $('previous-photo').disabled = photoIndex === 0;
  $('next-photo').disabled = photoIndex >= sets.length - 1;
  await Promise.all([drawPhoto(0, record['0'], token), drawPhoto(1, record['1'], token)]);
}
function metrics(event) {
  const container = $('metrics');
  container.replaceChildren();
  const changes = Math.max(0, ...Object.values(event.source_change_frames).map(v => v.length));
  const free = Object.values(event.vacancy).map(records => records.filter(v => v.usable && v.free_fraction >= .65 && v.occluded_fraction <= .2).length);
  const match = Array.isArray(event.matching) ? event.matching[0] : event.matching;
  const semantic = event.reason_code === 'segmented_object_appears_over_observed_background';
  const absence = semantic ? Object.values(event.segmented_background_frames || {}).map(frames => frames.length) : free;
  const entries = [['정합 오차 기준', Number(event.registration_error_u || 0).toFixed(3)+' u'], ['SimSaC 반복 관측', `${changes} 프레임`], [semantic ? '배경·분할 관측' : '빈자리 관측', `${absence.join(' / ') || 0} 프레임`]];
  if (event.type === 'moved') entries.push(['위치 차이 · 맵 단위', match.displacement_u.toFixed(3)]);
  else if (match) entries.push(['형상 대응', `${Math.round(match.shape_f1 * 100)}%`]);
  for (const [label, value] of entries) {
    const div = document.createElement('div'), small = document.createElement('small'), strong = document.createElement('strong');
    small.textContent = label; strong.textContent = value; div.append(small, strong); container.append(div);
  }
}
async function selectEvent(event, writeUrl = true) {
  selected = event;
  photoGhost = false; phase = 'both'; $('photo-mode').value='context';syncPhotoComparison();
  photoIndex = 0;
  const token = ++generation;
  for (const item of document.querySelectorAll('.event-item')) item.setAttribute('aria-pressed', String(item.dataset.id === event.id));
  document.documentElement.style.setProperty('--event-color', kinds[event.type].color);
  $('event-badge').textContent = typeName(event);
  $('event-title').textContent = objectName(event);
  $('event-identities').textContent = `${event.id} · ${event.before || '전 객체 없음'} → ${event.after || '후 객체 없음'} · 자동 추정`;
  $('decision-title').textContent = '후보 관측 근거';
  $('decision-reason').textContent = reasons[event.reason_code] || event.reason_code;
  $('event-detail').textContent = JSON.stringify({ object_ids: event.object_ids, reason: event.reason_code,
    matching: event.matching, vacancy: event.vacancy, segmented_background_frames: event.segmented_background_frames,
    source_change_frames: event.source_change_frames }, null, 2);
  $('event-position').textContent = `${filtered.indexOf(event) + 1} / ${filtered.length}`;
  $('previous-event').disabled = filtered.indexOf(event) <= 0;
  $('next-event').disabled = filtered.indexOf(event) >= filtered.length - 1;
  metrics(event);
  if (writeUrl) history.replaceState(null, '', '?' + new URLSearchParams({ type: filter, event: event.id }));
  await Promise.all([showGeometry(event, token), drawPhotos()]);
}
function emptySelection() {
  selected = null; generation++;
  photoGhost=false;syncPhotoComparison();
  clearGroup(selectedGroup);
  $('event-badge').textContent = '변화 후보';
  $('event-title').textContent = '조건을 충족한 항목이 없습니다';
  $('event-identities').textContent = '다른 객체 종류를 선택하세요.';
  $('event-position').textContent = '0 / 0';
  $('decision-reason').textContent = '선택한 객체 종류의 후보가 없습니다.';
  $('event-detail').textContent = '';
  $('metrics').replaceChildren();
  $('previous-event').disabled = $('next-event').disabled = true;
  const photoToken = ++photoGeneration;
  for (const epoch of [0, 1]) drawPhoto(epoch, null, photoToken).catch(reportError);
  overview();
}
async function renderList(preferred) {
  const type = $('object-type').value;
  filtered = data.events.filter(e => (type === 'all' || e.label === type));
  const list = $('event-list'); list.replaceChildren();
  $('list-count').textContent = `${filtered.length}개 항목 · 선택하면 객체와 전후 영상을 표시합니다`;
  for (const event of filtered) {
    const button = document.createElement('button'); button.className = 'event-item'; button.dataset.id = event.id;
    button.style.setProperty('--event-color', kinds[event.type].color);
    const row = document.createElement('div'); row.className = 'row';
    const name = document.createElement('span'); name.className = 'name'; name.textContent = objectName(event);
    const kind = document.createElement('span'); kind.className = 'type-text'; kind.textContent = typeName(event);
    row.append(name, kind);
    const detail = document.createElement('small'); detail.textContent = `${event.id} · ${event.before || '—'} → ${event.after || '—'}`;
    button.append(row, detail);
    button.onclick = () => selectEvent(event).catch(reportError);
    list.append(button);
  }
  for (const button of document.querySelectorAll('[data-type]')) button.setAttribute('aria-pressed', String(button.dataset.type === filter));
  if (!filtered.length) {
    const p = document.createElement('p'); p.className = 'empty-list'; p.textContent = '선택한 객체 종류의 변화 후보가 없습니다.'; list.append(p);
    emptySelection(); return;
  }
  await selectEvent(filtered.find(e => e.id === preferred) || filtered[0]);
}
async function start() {
  data = await json('viewer-data.json');
  const align=data.alignment_summary;
  const reused=data.job_provenance?.reused_probabilities;
  if(align)$('alignment-summary').textContent=`${reused?'기존 정합·SimSaC 확률 재사용':align.new_alignment_fitted?'새 정합 계산':'입력 정합 검증'} · ${align.region_count}구역 · 실제 점군 ${align.independent_pointcloud?.filter(Boolean).length || 0}/2 · ${reused?'객체 크기·시점 검증 후 재집계':align.held_out_validation?'별도 표면으로 오차 검증':'전역 잔차 기준'}`;
  $('scope').textContent = data.scope || '원본 영상과 관측 근거를 비교한 변화 후보입니다.';
  document.querySelectorAll('.photo-heading h4').forEach((el,i) => { el.textContent = (i?'후 · ':'전 · ')+(data.labels?.[i] || (i?'비교 촬영':'기준 촬영')); });
  $('ghost-opacity').oninput = () => { selectedGroup.traverse(item => { if(item.isMesh && item.userData.cachedGeometry) item.material.opacity = Number($('ghost-opacity').value)/100; }); drawPhotos().catch(reportError); };
  $('photo-reset').onclick = () => {for(const p of panoramas.values())p.focus();};
  $('photo-ghost').onclick = () => comparePhoto(!photoGhost);
  data.events.sort((a, b) => a.id.localeCompare(b.id));
  $('pointcloud').parentElement.hidden = !data.scenes.some(s => s.cloud_points > 0);
  $('total').textContent = `${data.events.length}개`;
  for (const label of [...new Set(data.events.map(e => e.label))].sort()) {
    const option = document.createElement('option'); option.value = label; option.textContent = names[label] || label; $('object-type').append(option);
  }
  $('object-type').onchange = () => renderList().catch(reportError);
  for (const record of data.scenes) {
    const buffer = await loadBuffer(record.mesh), n = record.vertices;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(buffer, 0, n * 3), 3));
    geometry.computeVertexNormals(); geometry.computeBoundingBox(); globalBounds.union(geometry.boundingBox);
    const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: record.epoch ? '#a5a9b0' : '#8d9baa',
      side: THREE.DoubleSide, transparent: true, opacity: .105, depthWrite: false, roughness: 1, flatShading: true }));
    scene.add(mesh); context.push(mesh);
    const cloudGeometry = new THREE.BufferGeometry();
    cloudGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(await loadBuffer(record.cloud)), 3));
    const cloud = new THREE.Points(cloudGeometry, new THREE.PointsMaterial({ color: record.epoch ? '#ffc26a' : '#57d8dd',
      size: .018, transparent: true, opacity: .28, depthWrite: false }));
    scene.add(cloud); clouds.push(cloud);
  }
  for (const event of data.events) {
    if(event.rgb_only)continue;
    const b = asBox(event.bounds), size = b.getSize(new THREE.Vector3()).max(new THREE.Vector3(.05, .05, .05));
    const hit = new THREE.Mesh(new THREE.BoxGeometry(...size.toArray()), new THREE.MeshBasicMaterial({ color: kinds[event.type].color,
      transparent: true, opacity: .025, depthWrite: false }));
    const outline = boxLines(event.bounds, kinds[event.type].color); outline.material.opacity=.18; outline.userData.eventId = event.id; scene.add(outline);
    hit.position.copy(b.getCenter(new THREE.Vector3())); hit.userData.eventId = event.id; markerGroup.add(hit);
  }
  const cast = new THREE.Raycaster(); let pointer;
  renderer.domElement.addEventListener('pointerdown', e => { pointer = [e.clientX, e.clientY]; });
  renderer.domElement.addEventListener('pointerup', e => {
    if (!pointer || Math.hypot(e.clientX - pointer[0], e.clientY - pointer[1]) > 5) return;
    const rect = renderer.domElement.getBoundingClientRect();
    cast.setFromCamera(new THREE.Vector2((e.clientX - rect.left) / rect.width * 2 - 1, 1 - (e.clientY - rect.top) / rect.height * 2), camera);
    const hit = cast.intersectObjects(markerGroup.children).find(h => filtered.some(event => event.id === h.object.userData.eventId));
    if (hit) selectEvent(data.events.find(event => event.id === hit.object.userData.eventId)).catch(reportError);
  });
  for (const button of document.querySelectorAll('[data-phase]')) button.onclick = () => { phase = button.dataset.phase; updateVisibility(); };
  $('context').onchange = $('pointcloud').onchange = updateVisibility;
  $('focus-object').onclick = () => selected?.bounds && flyTo(selected.bounds);
  $('overview').onclick = overview;
  $('previous-event').onclick = () => selectEvent(filtered[Math.max(0, filtered.indexOf(selected) - 1)]).catch(reportError);
  $('next-event').onclick = () => selectEvent(filtered[Math.min(filtered.length - 1, filtered.indexOf(selected) + 1)]).catch(reportError);
  $('previous-photo').onclick = () => { photoIndex = Math.max(0, photoIndex - 1); drawPhotos().catch(reportError); };
  $('next-photo').onclick = () => { photoIndex = Math.min(selected.photo_sets.length - 1, photoIndex + 1); drawPhotos().catch(reportError); };
  $('photo-mode').onchange = () => {for(const p of panoramas.values())p.focus();drawPhotos().catch(reportError);};
  filter = 'all';
  const center = globalBounds.getCenter(new THREE.Vector3());
  camera.position.copy(center).add(new THREE.Vector3(8, 8, 8)); controls.target.copy(center); controls.update();
  updateVisibility();
  await renderList(query.get('event'));
  document.body.dataset.ready = 'true';
}
start().catch(reportError);

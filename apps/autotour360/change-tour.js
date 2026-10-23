import {decodeObjectMesh,objectPhotoVisibility,reprojectObjectPhoto} from './object-photo.js?v=1';
import {createDormChangeTour} from './dorm-change-tour.js?v=whole-object-photo-1';

const TYPES={moved:'이동',added:'추가',removed:'소실'};
const COLORS=['#ffad73','#78caff'];

// Mounted in the existing Auto Tour viewport; no page or project replacement.
export function createChangeTour({onFrame=()=>{},onError=()=>{}}={}){
  if(new URLSearchParams(location.search).get('source')!=='rescan')return createDormChangeTour({onFrame,onError});
  return createRescanChangeTour({onFrame,onError});
}

function createRescanChangeTour({onFrame=()=>{},onError=()=>{}}={}){
  const picker=document.createElement('div');picker.className='dorm-source-picker';
  picker.innerHTML='<label for="change-source-select">촬영 데이터</label><select id="change-source-select"><option value="dorm">Dorm2 / Dorm3</option><option value="rescan" selected>기존 Rescan</option></select>';
  picker.querySelector('select').onchange=e=>{const url=new URL(location.href);url.searchParams.set('source',e.target.value);url.searchParams.set('mode','changes');location.href=url;};
  document.getElementById('change-sidebar').prepend(picker);
const $=id=>document.getElementById('change-'+id);
let data, meshes, frames=[[],[]], epoch=1, index=0, selected=null, loadToken=0, busy=true;
let active=false,ready=false,loadingData=null,ghostToken=0;
const imageCache=new Map();
const sourcePhotos=new Map(),photoGhosts=new Map();
let changeFilter='all';
const matchesFilter=region=>changeFilter==='all'||region.type===changeFilter;
const distance=value=>Number.isFinite(value)?(value>=1?`${value.toFixed(2)} m`:`${(value*100).toFixed(1)} cm`):'확인 불가';

function renderFilters(){
  for(const [key,label]of [['all','전체'],...Object.entries(TYPES)]){
    const button=$('filter-'+key),count=data.regions.filter(region=>key==='all'||region.type===key).length;
    button.textContent=`${label} ${count}`;button.setAttribute('aria-pressed',String(changeFilter===key));button.disabled=!count;
    button.onclick=()=>{changeFilter=key;if(selected&&!matchesFilter(selected)){selected=null;showDetail();void drawGhost();}renderFilters();renderObjects();renderBoxes();report();};
  }
}

function resize(){
  if(!active)return;
  const frame=frames[epoch][index],layer=$('photo-layer');
  const width=frame?.width||648,height=frame?.height||484;
  const scale=Math.min(layer.clientWidth/width,layer.clientHeight/height);
  $('photo-stage').style.width=`${Math.max(0,width*scale)}px`;
  $('photo-stage').style.height=`${Math.max(0,height*scale)}px`;
}
function report(){if(active&&ready)onFrame({data,frame:frames[epoch][index],epoch,index,count:frames[epoch].length,selected});}

async function get(url,binary=false){
  const response=await fetch(url);
  if(!response.ok)throw new Error('변화 투어 자료를 불러오지 못했습니다.');
  return binary?response.arrayBuffer():response.json();
}
function image(url){
  if(!imageCache.has(url)){
    const img=new Image();img.src=url;
    imageCache.set(url,img.decode().then(()=>img).catch(error=>{imageCache.delete(url);throw error;}));
  }
  return imageCache.get(url);
}
function parseMesh(bytes,meta){
  return decodeObjectMesh(bytes,meta,data.display_origin,data.display_rotation);
}
function cameraPoint(world,frame){
  const pose=frame.camera_to_world,p=world.map((v,i)=>v-pose[i][3]);
  return [0,1,2].map(a=>p.reduce((sum,v,i)=>sum+v*pose[i][a],0));
}
function pixel(p,frame){return [frame.focal_px*p[0]/p[2]+(frame.width-1)/2,frame.focal_px*p[1]/p[2]+(frame.height-1)/2];}
async function sourcePhoto(e,frame){
  const key=`${e}:${frame.id}`;
  if(!sourcePhotos.has(key)){
    const promise=(async()=>{
      const img=await image(frame.url),canvas=document.createElement('canvas');
      canvas.width=frame.width;canvas.height=frame.height;
      const ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.drawImage(img,0,0,frame.width,frame.height);
      return {frame,pixels:ctx.getImageData(0,0,frame.width,frame.height).data,visibility:objectPhotoVisibility(meshes[e],frame)};
    })().catch(error=>{sourcePhotos.delete(key);throw error;});
    sourcePhotos.set(key,promise);
  }
  return sourcePhotos.get(key);
}
function textureFrames(region,e,target){
  const distance=frame=>{
    const a=frame.camera_to_world,b=target.camera_to_world;
    return [0,1,2].reduce((sum,k)=>sum+(a[k][3]-b[k][3])**2+2*(a[k][2]-b[k][2])**2,0);
  };
  const linked=new Set(links(region,e).map(link=>link.frame_id));
  return frames[e].filter(frame=>linked.has(frame.id)||frame.id===target.id).sort((a,b)=>distance(a)-distance(b)).slice(0,3);
}
async function photoGhost(region,e,target){
  const key=`${region.id}:${e}:${target.id}`;
  if(!photoGhosts.has(key)){
    const promise=(async()=>{
      const loaded=await Promise.allSettled(textureFrames(region,e,target).map(frame=>sourcePhoto(e,frame)));
      const sources=loaded.filter(result=>result.status==='fulfilled').map(result=>result.value);
      const rgba=reprojectObjectPhoto(meshes[e],region.id,target,sources),canvas=document.createElement('canvas');
      canvas.width=target.width;canvas.height=target.height;
      const count=rgba.reduce((sum,value,i)=>sum+(i%4===3&&value>0?1:0),0);
      // A narrow coloured edge identifies the capture while preserving the object's photograph.
      const edge=e===0?[255,173,115]:[120,202,255],width=target.width;
      for(let y=1;y<target.height-1;y++)for(let x=1;x<width-1;x++){
        const i=(y*width+x)*4;
        if(rgba[i+3]&&(!rgba[i-1]||!rgba[i+7]||!rgba[i-width*4+3]||!rgba[i+width*4+3]))rgba.set(edge,i);
      }
      canvas.getContext('2d').putImageData(new ImageData(rgba,target.width,target.height),0,0);
      return {canvas,count};
    })().catch(error=>{photoGhosts.delete(key);throw error;});
    photoGhosts.set(key,promise);
    if(photoGhosts.size>16)photoGhosts.delete(photoGhosts.keys().next().value);
  }
  return photoGhosts.get(key);
}
async function drawGhost(){
  const token=++ghostToken;
  const frame=frames[epoch][index],canvas=$('ghosts');
  canvas.hidden=true;canvas.setAttribute('aria-busy','false');
  if(!frame)return;
  canvas.width=frame.width;canvas.height=frame.height;
  const ctx=canvas.getContext('2d');
  if(!selected||busy||!active)return;
  const region=selected,epochs=region.type==='removed'?[0]:region.type==='added'?[1]:[0,1];
  canvas.setAttribute('aria-busy','true');$('scene-hint').textContent=`${region.name} · 원본 사진에서 객체를 불러오는 중`;
  try{
    const cutouts=await Promise.all(epochs.map(e=>photoGhost(region,e,frame)));
    if(token!==ghostToken||!active||busy)return;
    const count=cutouts.reduce((sum,cutout)=>sum+cutout.count,0);
    canvas.hidden=!count;canvas.setAttribute('aria-busy','false');
    ctx.globalAlpha=Number($('ghost-opacity').value)/100;
    for(const cutout of cutouts)ctx.drawImage(cutout.canvas,0,0);
    ctx.globalAlpha=1;
    $('scene-hint').textContent=count?`${region.name} · 원본 사진의 객체만 반투명 표시 · 주황 테두리: 이전 / 파랑 테두리: 이후`:`${region.name} · 이 시점에 겹칠 수 있는 객체 사진이 없습니다.`;
    drawMovement(ctx,frame);
  }catch(error){
    if(token===ghostToken&&active){canvas.setAttribute('aria-busy','false');canvas.hidden=true;$('scene-hint').textContent='객체 사진을 불러오지 못했습니다. 다시 선택해 주세요.';}
  }
}
function drawMovement(ctx,frame){
  if(selected.type==='moved'){
    const centers=[0,1].map(e=>{
      const bounds=e===0?selected.before_bounds:selected.after_bounds;
      const display=bounds.min.map((v,k)=>(v+bounds.max[k])/2);
      const world=data.display_rotation[0].map((_,a)=>data.display_rotation.reduce((s,row,b)=>s+row[a]*display[b],0));
      return cameraPoint(world,frame);
    });
    if(centers.every(p=>p[2]>.1)){
      const [a,b]=centers.map(p=>pixel(p,frame));
      if([a,b].every(p=>p[0]>0&&p[0]<frame.width&&p[1]>0&&p[1]<frame.height)){
        ctx.strokeStyle='#fff3be';ctx.lineWidth=2;ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();ctx.setLineDash([]);
        const angle=Math.atan2(b[1]-a[1],b[0]-a[0]);ctx.fillStyle='#fff3be';ctx.beginPath();ctx.moveTo(...b);for(const d of [-.5,.5])ctx.lineTo(b[0]-12*Math.cos(angle+d),b[1]-12*Math.sin(angle+d));ctx.closePath();ctx.fill();
      }
    }
  }
}
function links(region,e){
  return region.keyframes[e].filter(l=>l.bbox_px&&frames[e].some(f=>f.id===l.frame_id));
}
function bestFrame(region,e){return frames[e].find(f=>f.id===links(region,e)[0]?.frame_id);}
function renderBoxes(){
  const root=$('change-boxes');root.replaceChildren();
  const frame=frames[epoch][index];if(!frame||busy)return;
  const items=data.regions.filter(matchesFilter).flatMap(region=>{
    const link=links(region,epoch).find(l=>l.frame_id===frame.id);
    return link?[{region,box:link.bbox_px}]:[];
  }).sort((a,b)=>(b.box[2]-b.box[0])*(b.box[3]-b.box[1])-(a.box[2]-a.box[0])*(a.box[3]-a.box[1]));
  for(const {region,box}of items){
    const button=document.createElement('button');button.className='change-box';button.dataset.region=region.id;
    const color=region.type==='added'?COLORS[1]:region.type==='removed'?COLORS[0]:'#c5a7ff';
    button.classList.add(region.type);
    const [x0,y0,x1,y1]=box;
    Object.assign(button.style,{left:`${x0/frame.width*100}%`,top:`${y0/frame.height*100}%`,width:`${(x1-x0)/frame.width*100}%`,height:`${(y1-y0)/frame.height*100}%`});
    button.style.setProperty('--color',color);button.setAttribute('aria-label',`${region.name} · ${TYPES[region.type]} · 객체 사진 겹쳐보기`);button.setAttribute('aria-pressed',String(selected?.id===region.id));
    const text=document.createElement('span');text.textContent=`${TYPES[region.type]} · ${region.name}`;button.append(text);
    button.onclick=()=>select(region,false);root.append(button);
  }
  $('scene-hint').textContent=selected?`${selected.name} · 원본 객체 사진 겹쳐보기`:`이 지점의 변화 ${items.length}개 · 박스를 눌러 객체를 확인하세요.`;
}
function renderObjects(){
  $('object-list').replaceChildren();
  for(const region of data.regions.filter(matchesFilter)){
    const button=document.createElement('button');button.dataset.region=region.id;button.setAttribute('aria-pressed',String(selected?.id===region.id));
    const row=document.createElement('span');row.className='object-row';
    const name=document.createElement('strong');name.textContent=region.name;
    const chip=document.createElement('span');chip.className=`type-chip ${region.type}`;chip.textContent=TYPES[region.type];row.append(name,chip);
    const note=document.createElement('small');note.textContent=region.motion?`관측 변위 ${distance(region.motion.displacement_m)}`:(region.type==='added'?'이후에 나타난 객체':'이전에 있던 객체');
    button.append(row,note);button.onclick=()=>select(region,true);$('object-list').append(button);
  }
}
function showDetail(){
  $('selection-detail').hidden=!selected;$('clear-selection').hidden=!selected;
  if(!selected)return;
  $('selected-name').textContent=selected.name;$('change-type').textContent=TYPES[selected.type];$('change-type').className=`type-chip ${selected.type}`;
  $('change-description').textContent=selected.type==='moved'?`기존 이동 분류 · 추정 변위 ${distance(selected.motion?.displacement_m)}. 같은 사진 시점에 전후 객체 사진을 겹쳐 표시합니다.`:selected.type==='added'?'이후에 나타난 객체입니다. 파란 테두리로 이후 객체 사진을 표시합니다.':'이전에 있던 객체입니다. 주황 테두리로 이전 객체 사진을 표시합니다.';
  for(const e of [0,1]){
    const frame=bestFrame(selected,e),fallback=selected.keyframes[e].map(l=>data.keyframes.find(f=>f.id===l.frame_id)).find(Boolean);
    const photo=frame||fallback;$('paired-'+e).disabled=!frame;
    $('paired-photo-'+e).hidden=!photo;
    if(photo)$('paired-photo-'+e).src=photo.url;
    $('paired-note-'+e).textContent=frame?`프레임 ${frame.source_frame_index} · 눌러서 이 지점으로 이동`:'카메라 연결 검토 필요 · 객체 사진 투영 생략';
    $('paired-'+e).onclick=()=>{if(frame)void go(e,frames[e].indexOf(frame));};
  }
}
async function select(region,jump){
  if(!matchesFilter(region)){changeFilter='all';renderFilters();}
  selected=region;showDetail();renderObjects();
  if(jump){
    const currentFrame=frames[epoch][index];
    const inFrame=links(region,epoch).some(l=>l.frame_id===currentFrame?.id);
    if(!inFrame){
      let e=epoch,frame=bestFrame(region,e);
      if(!frame){e=1-e;frame=bestFrame(region,e);}
      if(frame){await go(e,frames[e].indexOf(frame));return;}
    }
  }
  renderBoxes();void drawGhost();report();
}
function renderFrames(){
  const list=$('frame-list');list.replaceChildren();
  for(const [i,f]of frames[epoch].entries()){
    const b=document.createElement('button');b.setAttribute('aria-label',`지점 ${i+1} · 프레임 ${f.source_frame_index}`);b.setAttribute('aria-current',String(i===index));
    const img=new Image();img.src=f.url;img.alt='';const caption=document.createElement('small');caption.textContent=`지점 ${i+1} · ${f.time_s.toFixed(1)}초`;b.append(img,caption);b.onclick=()=>void go(epoch,i);list.append(b);
  }
}
async function go(e,i){
  if(!active||!frames[e]?.[i])return;
  const token=++loadToken;++ghostToken;busy=true;$('photo-stage').setAttribute('aria-busy','true');$('loading').hidden=false;$('change-boxes').replaceChildren();$('ghosts').hidden=true;
  $('previous-frame').disabled=true;$('next-frame').disabled=true;
  try{
    const frame=frames[e][i],img=await image(frame.url);if(token!==loadToken||!active)return;
    epoch=e;index=i;
    $('tour-photo').src=img.src;$('tour-photo').alt=`${e===0?'이전':'이후'} 원본 촬영 · 프레임 ${frame.source_frame_index}`;
    $('photo-stage').style.aspectRatio=`${frame.width}/${frame.height}`;
    $('capture-label').textContent=`${e===0?'이전':'이후'} 촬영 · Rescan scene_e_${data.source_epochs[e]}`;
    $('frame-label').textContent=`원본 프레임 ${frame.source_frame_index} · ${frame.time_s.toFixed(1)}초`;
    $('frame-index').textContent=`지점 ${i+1} / ${frames[e].length} · 정지 키프레임`;
    for(const n of [0,1])$('epoch-'+n).setAttribute('aria-pressed',String(n===e));
    busy=false;$('photo-stage').setAttribute('aria-busy','false');$('loading').hidden=true;
    $('previous-frame').disabled=i===0;$('next-frame').disabled=i===frames[e].length-1;
    renderFrames();renderBoxes();void drawGhost();resize();report();
    for(const next of [frames[e][i-1],frames[e][i+1]])if(next)image(next.url).catch(()=>{});
  }catch(error){if(token===loadToken&&active){busy=false;$('photo-stage').setAttribute('aria-busy','false');$('loading').textContent=error.message;onError(error.message);}}
}
async function load(){
  try{
    const manifestBytes=await get('/rescan/changes.json',true);
    data=JSON.parse(new TextDecoder().decode(manifestBytes));
    if(data.kind!=='rescan_object_changes'||data.synthetic!==false)throw new Error('연결된 실제 전후 객체 자료가 필요합니다.');
    // The existing RGB-checked cameras are used; depth-only poses never create photo boxes.
    frames=[0,1].map(e=>data.keyframes.filter(f=>f.epoch===e&&f.status==='estimated_rgb_checked'&&f.camera_to_world?.length===4&&f.focal_px>0).sort((a,b)=>a.source_frame_index-b.source_frame_index));
    if(frames.some(list=>!list.length))throw new Error('사진과 연결된 전후 카메라 위치가 필요합니다.');
    const bytes=await Promise.all(data.displays.map(d=>get(d.mesh.url,true)));
    meshes=bytes.map((b,e)=>parseMesh(b,data.displays[e].mesh));
    $('change-count').textContent=`${data.regions.length}개`;
    renderFilters();renderObjects();
    $('previous-frame').onclick=()=>{if(!busy)void go(epoch,index-1);};$('next-frame').onclick=()=>{if(!busy)void go(epoch,index+1);};
    for(const e of [0,1])$('epoch-'+e).onclick=()=>{const f=selected&&bestFrame(selected,e);void go(e,f?frames[e].indexOf(f):0);};
    $('clear-selection').onclick=()=>{selected=null;showDetail();renderObjects();renderBoxes();void drawGhost();report();};
    $('ghost-opacity').oninput=()=>void drawGhost();
    const start=data.regions.find(r=>r.type==='removed'&&bestFrame(r,1));
    const frame=(start&&bestFrame(start,1))||frames[1][0];
    epoch=1;index=frames[1].indexOf(frame);ready=true;
    await go(epoch,index);
  }catch(error){$('loading').hidden=true;$('error').hidden=false;$('error').textContent=error.message;onError(error.message);}
}
new ResizeObserver(resize).observe($('photo-layer'));
return {
  activate(){
    active=true;$('error').hidden=true;
    if(ready)return go(epoch,index);
    if(!loadingData)loadingData=load().finally(()=>{loadingData=null;});
    return loadingData;
  },
  deactivate(){active=false;++loadToken;++ghostToken;busy=false;},
  navigate(step){if(active&&ready&&!busy)void go(epoch,index+step);},
  clear(){if(ready)$('clear-selection').click();},
  resize,
};
}

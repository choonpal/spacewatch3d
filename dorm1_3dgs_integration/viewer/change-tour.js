const TYPES={moved:'이동',added:'추가',removed:'소실'};
const COLORS=['#ffad73','#78caff'];

// Mounted in the existing Auto Tour viewport; no page or project replacement.
export function createChangeTour({onFrame=()=>{},onError=()=>{}}={}){
const $=id=>document.getElementById('change-'+id);
let data, meshes, frames=[[],[]], epoch=1, index=0, selected=null, loadToken=0, busy=true;
let active=false,ready=false,loadingData=null;
const imageCache=new Map();
const masks=new Map();

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
  if(meta.stride!==32||bytes.byteLength!==meta.count*32||meta.count%3)throw new Error('객체 메쉬 형식을 읽을 수 없습니다.');
  const view=new DataView(bytes),byRegion=new Map();
  for(let i=0;i<meta.count;i+=3){
    const ids=[0,1,2].map(k=>view.getUint8((i+k)*32+27)%64);
    // Mixed boundary triangles do not define the selected object's silhouette.
    if(!ids[0]||!ids.every(id=>id===ids[0]))continue;
    const triangle=[];
    for(let k=0;k<3;k++){
      const p=[0,1,2].map(a=>view.getFloat32((i+k)*32+a*4,true)+data.display_origin[a]);
      triangle.push(data.display_rotation[0].map((_,a)=>data.display_rotation.reduce((sum,row,b)=>sum+row[a]*p[b],0)));
    }
    if(!byRegion.has(ids[0]))byRegion.set(ids[0],[]);
    byRegion.get(ids[0]).push(triangle);
  }
  return byRegion;
}
function cameraPoint(world,frame){
  const pose=frame.camera_to_world,p=world.map((v,i)=>v-pose[i][3]);
  return [0,1,2].map(a=>p.reduce((sum,v,i)=>sum+v*pose[i][a],0));
}
function clipNear(points){
  const result=[];
  for(let i=0;i<points.length;i++){
    const a=points[i],b=points[(i+1)%points.length],ain=a[2]>=.1,bin=b[2]>=.1;
    if(ain)result.push(a);
    if(ain!==bin){const t=(.1-a[2])/(b[2]-a[2]);result.push(a.map((v,k)=>v+t*(b[k]-v)));}
  }
  return result;
}
function pixel(p,frame){return [frame.focal_px*p[0]/p[2]+(frame.width-1)/2,frame.focal_px*p[1]/p[2]+(frame.height-1)/2];}
function silhouette(region,sourceEpoch,frame){
  const key=`${region.id}:${sourceEpoch}:${frame.id}`;
  if(masks.has(key))return masks.get(key);
  const canvas=document.createElement('canvas');canvas.width=frame.width;canvas.height=frame.height;
  const ctx=canvas.getContext('2d');ctx.fillStyle=COLORS[sourceEpoch];
  for(const tri of meshes[sourceEpoch].get(region.id)||[]){
    const poly=clipNear(tri.map(p=>cameraPoint(p,frame))).map(p=>pixel(p,frame));
    if(poly.length<3)continue;
    if(poly.every(p=>p[0]<0)||poly.every(p=>p[0]>frame.width)||poly.every(p=>p[1]<0)||poly.every(p=>p[1]>frame.height))continue;
    ctx.beginPath();ctx.moveTo(...poly[0]);for(const p of poly.slice(1))ctx.lineTo(...p);ctx.closePath();ctx.fill();
  }
  masks.set(key,canvas);
  if(masks.size>24)masks.delete(masks.keys().next().value);
  return canvas;
}
function drawGhost(){
  const frame=frames[epoch][index],canvas=$('ghosts');
  canvas.hidden=!selected||busy;
  if(!frame)return;
  canvas.width=frame.width;canvas.height=frame.height;
  const ctx=canvas.getContext('2d');
  if(!selected||busy)return;
  ctx.globalAlpha=Number($('ghost-opacity').value)/100;
  for(const e of [0,1])ctx.drawImage(silhouette(selected,e,frame),0,0);
  ctx.globalAlpha=1;
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
  const items=data.regions.flatMap(region=>{
    const link=links(region,epoch).find(l=>l.frame_id===frame.id);
    return link?[{region,box:link.bbox_px}]:[];
  }).sort((a,b)=>(b.box[2]-b.box[0])*(b.box[3]-b.box[1])-(a.box[2]-a.box[0])*(a.box[3]-a.box[1]));
  for(const {region,box}of items){
    const button=document.createElement('button');button.className='change-box';button.dataset.region=region.id;
    const color=region.type==='added'?COLORS[1]:region.type==='removed'?COLORS[0]:'#c5a7ff';
    const [x0,y0,x1,y1]=box;
    Object.assign(button.style,{left:`${x0/frame.width*100}%`,top:`${y0/frame.height*100}%`,width:`${(x1-x0)/frame.width*100}%`,height:`${(y1-y0)/frame.height*100}%`});
    button.style.setProperty('--color',color);button.setAttribute('aria-label',`${region.name} · ${TYPES[region.type]} · 그림자 보기`);button.setAttribute('aria-pressed',String(selected?.id===region.id));
    const text=document.createElement('span');text.textContent=`${TYPES[region.type]} · ${region.name}`;button.append(text);
    button.onclick=()=>select(region,false);root.append(button);
  }
  $('scene-hint').textContent=selected?`${selected.name} · 이전은 주황, 이후는 파랑 그림자`:`이 지점의 변화 ${items.length}개 · 박스를 눌러 형상을 확인하세요.`;
}
function renderObjects(){
  $('object-list').replaceChildren();
  for(const region of data.regions){
    const button=document.createElement('button');button.dataset.region=region.id;button.setAttribute('aria-pressed',String(selected?.id===region.id));
    const row=document.createElement('span');row.className='object-row';
    const name=document.createElement('strong');name.textContent=region.name;
    const chip=document.createElement('span');chip.className=`type-chip ${region.type}`;chip.textContent=TYPES[region.type];row.append(name,chip);
    const note=document.createElement('small');note.textContent=region.motion?`${region.motion.displacement_m.toFixed(2)} m 이동`:(region.type==='added'?'이후에 나타난 객체':'이전에 있던 객체');
    button.append(row,note);button.onclick=()=>select(region,true);$('object-list').append(button);
  }
}
function showDetail(){
  $('selection-detail').hidden=!selected;$('clear-selection').hidden=!selected;
  if(!selected)return;
  $('selected-name').textContent=selected.name;$('change-type').textContent=TYPES[selected.type];$('change-type').className=`type-chip ${selected.type}`;
  $('change-description').textContent=selected.type==='moved'?`이전 위치에서 ${selected.motion.displacement_m.toFixed(2)} m 이동했습니다. 같은 사진 시점에 전후 형상을 겹쳐 표시합니다.`:selected.type==='added'?'이후에 나타난 객체입니다. 파란 그림자로 추가된 표면을 표시합니다.':'이전에 있던 객체입니다. 주황 그림자로 사라진 위치의 표면을 표시합니다.';
  for(const e of [0,1]){
    const frame=bestFrame(selected,e),fallback=selected.keyframes[e].map(l=>data.keyframes.find(f=>f.id===l.frame_id)).find(Boolean);
    const photo=frame||fallback;$('paired-'+e).disabled=!frame;
    $('paired-photo-'+e).hidden=!photo;
    if(photo)$('paired-photo-'+e).src=photo.url;
    $('paired-note-'+e).textContent=frame?`프레임 ${frame.source_frame_index} · 눌러서 이 지점으로 이동`:'카메라 연결 검토 필요 · 그림자 투영 생략';
    $('paired-'+e).onclick=()=>{if(frame)void go(e,frames[e].indexOf(frame));};
  }
}
async function select(region,jump){
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
  renderBoxes();drawGhost();report();
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
  const token=++loadToken;busy=true;$('photo-stage').setAttribute('aria-busy','true');$('loading').hidden=false;$('change-boxes').replaceChildren();$('ghosts').hidden=true;
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
    renderFrames();renderBoxes();drawGhost();resize();report();
    for(const next of [frames[e][i-1],frames[e][i+1]])if(next)image(next.url).catch(()=>{});
  }catch(error){if(token===loadToken&&active){busy=false;$('photo-stage').setAttribute('aria-busy','false');$('loading').textContent=error.message;onError(error.message);}}
}
async function load(){
  try{
    data=await get('/rescan/changes.json');
    if(data.kind!=='rescan_object_changes'||data.synthetic!==false)throw new Error('연결된 실제 전후 객체 자료가 필요합니다.');
    // The existing RGB-checked cameras are used; depth-only poses never create photo boxes.
    frames=[0,1].map(e=>data.keyframes.filter(f=>f.epoch===e&&f.status==='estimated_rgb_checked'&&f.camera_to_world?.length===4&&f.focal_px>0).sort((a,b)=>a.source_frame_index-b.source_frame_index));
    if(frames.some(list=>!list.length))throw new Error('사진과 연결된 전후 카메라 위치가 필요합니다.');
    const bytes=await Promise.all(data.displays.map(d=>get(d.mesh.url,true)));
    meshes=bytes.map((b,e)=>parseMesh(b,data.displays[e].mesh));
    $('change-count').textContent=`${data.regions.length}개`;
    renderObjects();
    $('previous-frame').onclick=()=>{if(!busy)void go(epoch,index-1);};$('next-frame').onclick=()=>{if(!busy)void go(epoch,index+1);};
    for(const e of [0,1])$('epoch-'+e).onclick=()=>{const f=selected&&bestFrame(selected,e);void go(e,f?frames[e].indexOf(f):0);};
    $('clear-selection').onclick=()=>{selected=null;showDetail();renderObjects();renderBoxes();drawGhost();report();};
    $('ghost-opacity').oninput=drawGhost;
    const start=data.regions.find(r=>r.type==='removed'&&bestFrame(r,1));
    const frame=start?bestFrame(start,1):frames[1][0];
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
  deactivate(){active=false;++loadToken;busy=false;},
  navigate(step){if(active&&ready&&!busy)void go(epoch,index+step);},
  clear(){if(ready)$('clear-selection').click();},
  resize,
};
}

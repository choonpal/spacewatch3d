import * as THREE from 'three';
import {routeBetween,neighbor,samplePath,pathLength} from './tour_navigation.js';

const $=q=>document.querySelector(q),svgNS='http://www.w3.org/2000/svg';
const v=a=>new THREE.Vector3().fromArray(a);
const pose=view=>new THREE.Matrix4().fromArray(view.camera_to_common_cv.flat()).transpose().multiply(new THREE.Matrix4().makeScale(1,-1,-1));
const quaternion=view=>new THREE.Quaternion().setFromRotationMatrix(pose(view));

export function createTour({routes,data,camera,candidatesAt,ready,depart,arrive}){
 const views=new Map(data.zones.map(z=>[z.id,z]));
 let group,current,segment,motion=null,playing=false,queue=[],nextAt=Infinity,lastReady=false,mapProject;
 const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
 const markers=new Map(),mapNodes=new Map();
 const count=id=>candidatesAt(id).length;
 const node=id=>group.nodes.find(n=>n.id===id);
 const label=id=>`${id}로 이동 · 후보 ${count(id)}영역`;
 const status=text=>{$('#tour-status').textContent=text;};
 function controls(){
  const unavailable=!ready()||!!motion;
  $('#tour-prev').disabled=unavailable||!neighbor(group,current,-1);
  $('#tour-next').disabled=unavailable||!neighbor(group,current,1);
  $('#tour-play').disabled=!ready()||(!motion&&!playing&&!neighbor(group,current,1));
  $('#tour-play').textContent=playing?'자동 이동 중지':'자동 둘러보기';
  $('#tour-play').setAttribute('aria-pressed',String(playing));
  for(const id of ['zone','plan-group','tour-segment','map-before','map-after','reset'])$('#'+id).disabled=unavailable;
  for(const b of $('#tour-strip').children)b.disabled=unavailable;
  for(const {button} of markers.values())button.disabled=unavailable;
  $('#stage').dataset.tourMoving=String(!!motion);
  $('#stage').dataset.keyframe=current;
  $('#stage').dataset.tourTarget=motion?.target||'';
 }
 function selection(){
  $('#zone').value=current;
  for(const b of $('#tour-strip').children){const chosen=b.dataset.view===current;b.setAttribute('aria-current',chosen?'step':'false');}
  for(const[id,circle]of mapNodes){circle.setAttribute('fill',id===current?'#e4f6ff':count(id)?'#cb99ff':'#47718e');circle.setAttribute('r',id===current?'6':'4');circle.parentNode.setAttribute('aria-current',id===current?'step':'false');}
  $('#tour-current').textContent=`${current} · ${segment.nodes.indexOf(current)+1} / ${segment.nodes.length}`;
  buildMarkers();controls();
 }
 function buildMarkers(){
  $('#tour-hotspots').replaceChildren();markers.clear();
  for(const direction of [-1,1]){
   const id=neighbor(group,current,direction);if(!id)continue;
   const button=document.createElement('button');button.type='button';button.className='tour-hotspot';
   button.setAttribute('aria-label',label(id));button.dataset.view=id;
   button.innerHTML=`<span class="tour-chevron">${direction>0?'↑':'↶'}</span><span>${id}${count(id)?' · '+count(id)+'영역':''}</span>`;
   button.addEventListener('click',()=>go(id));$('#tour-hotspots').append(button);
   const position=v(node(id).position);position.y-=data.height_u*.5;
   markers.set(id,{button,position});
  }
 }
 function buildMap(){
  const svg=$('#tour-map');svg.replaceChildren();mapNodes.clear();
  const ids=new Set(segment.nodes),edges=group.links.filter(e=>ids.has(e.from)&&ids.has(e.to));
  const points=[...segment.nodes.map(id=>node(id).position),...edges.flatMap(e=>e.polyline)];
  const xs=points.map(p=>p[0]),zs=points.map(p=>p[2]);const minX=Math.min(...xs),maxX=Math.max(...xs),minZ=Math.min(...zs),maxZ=Math.max(...zs);
  const scale=Math.min(200/Math.max(1,maxX-minX),105/Math.max(1,maxZ-minZ));
  mapProject=p=>[120+(p[0]-(minX+maxX)/2)*scale,70+(p[2]-(minZ+maxZ)/2)*scale];
  for(const edge of edges){const line=document.createElementNS(svgNS,'polyline');line.setAttribute('points',edge.polyline.map(p=>mapProject(p).join(',')).join(' '));line.setAttribute('class','tour-map-path');svg.append(line);}
  for(const id of segment.nodes){
   const p=mapProject(node(id).position),g=document.createElementNS(svgNS,'g');g.setAttribute('role','button');g.setAttribute('tabindex','0');g.setAttribute('aria-label',label(id));g.setAttribute('class','tour-map-node');
   const c=document.createElementNS(svgNS,'circle');c.setAttribute('cx',p[0]);c.setAttribute('cy',p[1]);g.append(c);
   const text=document.createElementNS(svgNS,'text');text.setAttribute('x',p[0]+8);text.setAttribute('y',p[1]+4);text.textContent=id.slice(-3);g.append(text);
   const activate=()=>{if(!motion)go(id);};g.addEventListener('click',activate);g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();activate();}});
   svg.append(g);mapNodes.set(id,c);
  }
  const indicator=document.createElementNS(svgNS,'path');indicator.id='tour-map-camera';indicator.setAttribute('d','M -4 5 L 0 -7 L 4 5 Z');svg.append(indicator);
 }
 function buildOptions(){
  $('#tour-segment').innerHTML=group.segments.map((s,i)=>`<option value="${s.id}">구간 ${i+1} · ${s.nodes.length}시점${s.nodes.length===1?' (단독)':''}</option>`).join('');
  $('#tour-segment').value=segment.id;
  $('#zone').innerHTML=segment.nodes.map(id=>`<option value="${id}">${id} · ${views.get(id).registration_zone} · 후보 ${count(id)}</option>`).join('');
  $('#tour-strip').replaceChildren();
  for(const id of segment.nodes){const button=document.createElement('button');button.type='button';button.dataset.view=id;button.setAttribute('aria-label',label(id));
   const name=document.createElement('span');name.textContent=id;const badge=document.createElement('small');badge.textContent=count(id)?`${count(id)}영역`:'후보 없음';button.classList.toggle('has-candidates',count(id)>0);button.append(name,badge);button.addEventListener('click',()=>go(id));$('#tour-strip').append(button);}
  buildMap();selection();
 }
 function anchor(id,message){
  current=id;arrive(id,false);selection();
  status(message||`${id} 도착 · 후보 ${count(id)}영역. 박스를 누르면 후 맵이 반투명하게 겹칩니다.`);
 }
 function selectSegment(id,initial=false){
  if(motion)return;playing=false;queue=[];nextAt=Infinity;
  segment=group.segments.find(s=>s.id===id);buildOptions();
  anchor(segment.nodes[0],initial?`${group.id} · 연결된 구간에서 키프레임을 눌러 둘러보세요.`:'다른 연결 구간으로 전환했습니다. 구간 사이의 이동 경로는 연결하지 않았습니다.');
 }
 function selectGroup(id,initial=false){
  if(motion)return;group=routes.groups.find(g=>g.id===id);if(!group)throw new Error('Unknown tour plan');
  $('#plan-group').value=group.id;
  // Choose the longest stored connected segment, independent of changes.
  const longest=group.segments.reduce((a,b)=>b.nodes.length>a.nodes.length?b:a);
  selectSegment(longest.id,initial);
 }
 function startStep(target,now){
  const route=routeBetween(group,current,target);if(!route||route.ids.length!==2)throw new Error('Missing saved adjacent path');
  depart();
  motion={target,points:route.points,start:now,duration:reduced?0:Math.max(900,Math.min(5500,pathLength(route.points)/data.height_u*650)),
          from:camera.quaternion.clone(),to:quaternion(views.get(target))};
  $('#tour-current').textContent=`${current} → ${target}`;
  status(`${current} → ${target} 이동 중 · 도착하면 해당 시점의 후보 박스를 표시합니다.`);controls();
 }
 function go(target){
  if(motion||!ready())return;
  playing=false;nextAt=Infinity;
  if(!segment.nodes.includes(target))return;
  if(target===current){anchor(target);return;}
  const route=routeBetween(group,current,target);if(!route){status('저장된 연결 경로가 없습니다.');return;}
  queue=route.ids.slice(1);startStep(queue.shift(),performance.now());
 }
 function tick(now){
  if(ready()!==lastReady){lastReady=ready();controls();}
  if(motion){
   const f=motion.duration?Math.min(1,(now-motion.start)/motion.duration):1,t=f*f*(3-2*f);
   camera.position.fromArray(samplePath(motion.points,t));camera.quaternion.copy(motion.from).slerp(motion.to,t);camera.updateMatrixWorld(true);
   if(f===1){const target=motion.target;motion=null;anchor(target);nextAt=now+1600;
    if(!queue.length&&!neighbor(group,current,1)&&playing){playing=false;status('이 연결 구간의 끝입니다. 다른 구간은 구간 선택에서 전환하세요.');controls();}
   }
  }else if(now>=nextAt){
   if(queue.length)startStep(queue.shift(),now);
   else if(playing){const next=neighbor(group,current,1);if(next)startStep(next,now);else{playing=false;controls();}}
   else nextAt=Infinity;
  }
  const forward=camera.getWorldDirection(new THREE.Vector3());
  for(const {button,position} of markers.values()){
   const p=position.clone().project(camera),behind=position.clone().sub(camera.position).dot(forward)<=.01;
   button.hidden=!!motion||behind||Math.abs(p.x)>.94||p.y<-.8||p.y>.85;
   if(!button.hidden){button.style.left=(p.x+1)*50+'%';button.style.top=(1-p.y)*50+'%';}
  }
  if(mapProject){const p=mapProject(camera.position.toArray()),angle=Math.atan2(forward.x,-forward.z)*180/Math.PI;$('#tour-map-camera').setAttribute('transform',`translate(${p[0]} ${p[1]}) rotate(${angle})`);}
 }
 $('#tour-segment').addEventListener('change',()=>selectSegment($('#tour-segment').value));
 $('#tour-prev').addEventListener('click',()=>go(neighbor(group,current,-1)));
 $('#tour-next').addEventListener('click',()=>go(neighbor(group,current,1)));
 $('#tour-play').addEventListener('click',()=>{playing=!playing;if(!playing){queue=[];status(motion?'현재 이동을 마치고 자동 이동을 중지합니다.':'자동 이동을 중지했습니다.');}else nextAt=performance.now();controls();});
 selectGroup($('#plan-group').value,true);
 return {tick,go,selectGroup,get moving(){return!!motion;},
  refresh(){buildOptions();if(!motion)arrive(current,true);},
  step(direction){const target=neighbor(group,current,direction);if(target)go(target);}};
}

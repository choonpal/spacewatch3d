import * as THREE from 'three';
import {routeBetween,neighbor,samplePath,pathLength} from './tour_navigation.js';

const $=q=>document.querySelector(q),svgNS='http://www.w3.org/2000/svg';
const v=a=>new THREE.Vector3().fromArray(a);
const pose=view=>new THREE.Matrix4().fromArray(view.camera_to_common_cv.flat()).transpose().multiply(new THREE.Matrix4().makeScale(1,-1,-1));
const quaternion=view=>new THREE.Quaternion().setFromRotationMatrix(pose(view));

export function createTour({routes,data,camera,candidatesAt,regionsFor=null,ready,depart,arrive}){
 const views=new Map(data.zones.map(z=>[z.id,z]));
 let group,current,segment,motion=null,playing=false,nextAt=Infinity,lastReady=false,mapProject,afterArrival=null;
 const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
 const markers=new Map(),mapNodes=new Map();
 const count=id=>candidatesAt(id).length;
 const node=id=>group.nodes.find(n=>n.id===id);
 const areasAt=id=>regionsFor?.().filter(r=>r.observing_keyframes.includes(id))||[];
 const observationLabel=id=>areasAt(id).map(r=>r.label+' 관측').join(' / ')||'경유 시점';
 const label=id=>regionsFor?`${id}로 이동 · ${observationLabel(id)}`:`${id}로 이동 · 후보 ${count(id)}영역`;
 const palette=['#cb99ff','#71cddd','#ffc181','#a0dc9d'];
 const areaColor=id=>{const region=regionsFor?.().find(r=>r.observing_keyframes.includes(id));return region?palette[region.color_index%palette.length]:'#47718e';};
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
  if(regionsFor)for(const b of $('#candidates').querySelectorAll('button'))b.disabled=unavailable||b.hasAttribute('data-region-view')&&!b.dataset.regionView;
  for(const {button} of markers.values())button.disabled=unavailable;
  for(const circle of mapNodes.values())circle.parentNode.setAttribute('aria-disabled',String(unavailable));
  $('#stage').dataset.tourMoving=String(!!motion);
  $('#stage').dataset.keyframe=current;
  $('#stage').dataset.tourTarget=motion?.target||'';
 }
 function selection(){
  $('#zone').value=current;
  for(const b of $('#tour-strip').children){const chosen=b.dataset.view===current;b.setAttribute('aria-current',chosen?'step':'false');}
  for(const[id,circle]of mapNodes){circle.setAttribute('fill',id===current?'#e4f6ff':regionsFor?areaColor(id):count(id)?'#cb99ff':'#47718e');circle.setAttribute('r',id===current?'6':'4');circle.parentNode.setAttribute('aria-current',id===current?'step':'false');}
  $('#tour-current').textContent=`${current} · ${segment.nodes.indexOf(current)+1} / ${segment.nodes.length}`;
  buildMarkers();controls();
 }
 function buildMarkers(){
  $('#tour-hotspots').replaceChildren();markers.clear();
  for(const direction of [-1,1]){
   const id=neighbor(group,current,direction);if(!id)continue;
   const button=document.createElement('button');button.type='button';button.className='tour-hotspot';
   button.setAttribute('aria-label',label(id));button.dataset.view=id;
   button.innerHTML=`<span class="tour-chevron">${direction>0?'↑':'↶'}</span><span>${id}${regionsFor?' · '+observationLabel(id):count(id)?' · '+count(id)+'영역':''}</span>`;
   button.addEventListener('click',()=>go(id));$('#tour-hotspots').append(button);
   const position=v(node(id).position);position.y-=data.height_u*.5;
   markers.set(id,{button,position});
  }
 }
 function buildMap(){
  const svg=$('#tour-map');svg.replaceChildren();mapNodes.clear();const mapHeight=segment.nodes.length>10?360:140;svg.setAttribute('viewBox',`0 0 240 ${mapHeight}`);
  const ids=new Set(segment.nodes),edges=group.links.filter(e=>ids.has(e.from)&&ids.has(e.to));
  const points=[...segment.nodes.map(id=>node(id).position),...edges.flatMap(e=>e.polyline)];
  const xs=points.map(p=>p[0]),zs=points.map(p=>p[2]);const minX=Math.min(...xs),maxX=Math.max(...xs),minZ=Math.min(...zs),maxZ=Math.max(...zs);
  const scale=Math.min(190/Math.max(1,maxX-minX),(mapHeight-35)/Math.max(1,maxZ-minZ));
  mapProject=p=>[120+(p[0]-(minX+maxX)/2)*scale,mapHeight/2+(p[2]-(minZ+maxZ)/2)*scale];
  for(const edge of edges){const line=document.createElementNS(svgNS,'polyline');line.setAttribute('points',edge.polyline.map(p=>mapProject(p).join(',')).join(' '));line.setAttribute('class','tour-map-path');svg.append(line);}
  for(const id of segment.nodes){
   const p=mapProject(node(id).position),g=document.createElementNS(svgNS,'g');g.setAttribute('role','button');g.setAttribute('tabindex','0');g.setAttribute('aria-label',label(id));g.setAttribute('class','tour-map-node');
   const left=/^K\d+$/.test(id)&&Number(id.slice(1))%2===0;
   const hit=document.createElementNS(svgNS,'rect');hit.setAttribute('x',p[0]+(left?-35:-7));hit.setAttribute('y',p[1]-7);hit.setAttribute('width','42');hit.setAttribute('height','14');hit.setAttribute('fill','transparent');hit.setAttribute('stroke','none');hit.setAttribute('pointer-events','all');g.append(hit);
   const c=document.createElementNS(svgNS,'circle');c.setAttribute('cx',p[0]);c.setAttribute('cy',p[1]);g.append(c);
   const text=document.createElementNS(svgNS,'text');text.setAttribute('x',p[0]+(left?-8:8));text.setAttribute('text-anchor',left?'end':'start');text.setAttribute('y',p[1]+4);text.textContent=id.slice(-3);g.append(text);
   const activate=()=>{if(!motion)go(id);};g.addEventListener('click',activate);g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();activate();}});
   svg.append(g);mapNodes.set(id,c);
  }
  const indicator=document.createElementNS(svgNS,'path');indicator.id='tour-map-camera';indicator.setAttribute('d','M -4 5 L 0 -7 L 4 5 Z');svg.append(indicator);
 }
 function buildOptions(){
  $('#tour-segment').innerHTML=group.segments.map((s,i)=>`<option value="${s.id}">구간 ${i+1} · ${s.nodes.length}시점${s.nodes.length===1?' (단독)':''}</option>`).join('');
  $('#tour-segment').value=segment.id;
  $('#zone').innerHTML=segment.nodes.map(id=>`<option value="${id}">${id} · ${regionsFor?observationLabel(id):views.get(id).registration_zone+' · 후보 '+count(id)}</option>`).join('');
  $('#tour-strip').replaceChildren();
  for(const id of segment.nodes){const button=document.createElement('button');button.type='button';button.dataset.view=id;button.setAttribute('aria-label',label(id));
   const name=document.createElement('span');name.textContent=id;const badge=document.createElement('small');badge.textContent=regionsFor?observationLabel(id):count(id)?`${count(id)}영역`:'후보 없음';button.classList.toggle('has-candidates',count(id)>0);button.append(name,badge);button.addEventListener('click',()=>go(id));$('#tour-strip').append(button);}
  buildMap();selection();
 }
 function anchor(id,message){
  current=id;arrive(id,false);selection();
  status(message||(regionsFor?`${id} · ${observationLabel(id)} · 등록된 같은 후보 ${count(id)}개를 이 시점에 연결합니다.`:`${id} 도착 · 후보 ${count(id)}영역. 이동 박스를 누르면 출발→도착 색 그림자를 표시합니다.`));
  const action=afterArrival;afterArrival=null;action?.();
 }
 function selectSegment(id,initial=false){
  if(motion)return;playing=false;nextAt=Infinity;
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
 function startStep(target,now,savedRoute=null){
  const route=savedRoute||routeBetween(group,current,target);if(!route)throw new Error('Missing saved path');
  depart();
  motion={target,points:route.points,start:now,duration:reduced?0:Math.max(900,Math.min(5500,pathLength(route.points)/data.height_u*650)),
          from:camera.quaternion.clone(),to:quaternion(views.get(target))};
  $('#tour-current').textContent=`${current} → ${target}`;
  status(regionsFor?`${current} → ${target} 이동 중 · 도착하면 같은 후보를 이 시점에 연결합니다.`:`${current} → ${target} 이동 중 · 도착하면 해당 시점의 후보 박스를 표시합니다.`);controls();
 }
 function go(target,callback=null){
  if(motion||!ready())return false;
  playing=false;nextAt=Infinity;
  if(!segment.nodes.includes(target))return false;
  afterArrival=callback;
  if(target===current){anchor(target);return true;}
  const route=routeBetween(group,current,target);if(!route){status('저장된 연결 경로가 없습니다.');return;}
  startStep(target,performance.now(),route);return true;
 }
 function tick(now){
  if(ready()!==lastReady){lastReady=ready();controls();}
  if(motion){
   const f=motion.duration?Math.min(1,(now-motion.start)/motion.duration):1,t=f*f*(3-2*f);
   camera.position.fromArray(samplePath(motion.points,t));camera.quaternion.copy(motion.from).slerp(motion.to,t);camera.updateMatrixWorld(true);
   if(f===1){const target=motion.target;motion=null;anchor(target);nextAt=now+1600;
    if(!neighbor(group,current,1)&&playing){playing=false;status(group.segments.length>1?'이 연결 구간의 끝입니다. 다른 구간은 구간 선택에서 전환하세요.':'K1–K21 경로의 끝입니다. 지도나 키프레임을 눌러 다시 둘러볼 수 있습니다.');controls();}
   }
  }else if(now>=nextAt){
   if(playing){const next=neighbor(group,current,1);if(next)startStep(next,now);else{playing=false;controls();}}
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
 $('#tour-play').addEventListener('click',()=>{playing=!playing;if(!playing){status(motion?'현재 이동을 마치고 자동 이동을 중지합니다.':'자동 이동을 중지했습니다.');}else nextAt=performance.now();controls();});
 selectGroup($('#plan-group').value,true);
 return {tick,go,selectGroup,get moving(){return!!motion;},
  pause(){playing=false;nextAt=Infinity;controls();},
  refresh(){buildOptions();if(!motion){arrive(current,true);status(regionsFor?`${current} · ${observationLabel(current)} · 같은 후보 ${count(current)}개 관측`:`${current} · 현재 표시 조건의 후보 ${count(current)}영역. 박스를 눌러 근거를 확인하세요.`);}},
  step(direction){const target=neighbor(group,current,direction);if(target)go(target);}};
}

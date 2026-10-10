// Trajectory coordinates follow COLMAP: camera x right, y down, z forward.
// Quaternions are camera-to-world in [x, y, z, w] order. Scale is arbitrary.
export const dot = (a,b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
export const sub = (a,b) => a.map((x,i)=>x-b[i]);
export const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
// Preserve already-unit values so repeated JSON imports do not accumulate rounding.
export const unit = a => {const length=Math.hypot(...a);return Math.abs(length-1)<1e-12?[...a]:a.map(x=>x/length);};
export const lerp = (a,b,t) => a.map((x,i)=>x+(b[i]-x)*t);
export function rotate(q,v) {
  const [x,y,z,w]=q;
  const tx=2*(y*v[2]-z*v[1]),ty=2*(z*v[0]-x*v[2]),tz=2*(x*v[1]-y*v[0]);
  return [v[0]+w*tx+y*tz-z*ty,v[1]+w*ty+z*tx-x*tz,v[2]+w*tz+x*ty-y*tx];
}
export function slerp(a,b,t) {
  let cosine=a.reduce((sum,x,i)=>sum+x*b[i],0);
  if(cosine<0){b=b.map(x=>-x);cosine=-cosine;}
  if(cosine>.9995)return unit(lerp(a,b,t));
  const angle=Math.acos(Math.min(1,cosine)),s=Math.sin(angle);
  return a.map((x,i)=>(Math.sin((1-t)*angle)*x+Math.sin(t*angle)*b[i])/s);
}
export function validateTrajectory(value) {
  const fail=()=>{throw new Error('촬영 경로 데이터가 올바르지 않습니다. 다시 분석해 주세요.');};
  const vector=(v,n)=>Array.isArray(v)&&v.length===n&&v.every(x=>Number.isFinite(x)&&Math.abs(x)<1e9);
  if(!value||value.version!==1||!Array.isArray(value.samples)||value.samples.length<2||value.samples.length>5000||!vector(value.up,3)||Math.hypot(...value.up)<.1)fail();
  if(!value.source||typeof value.source.name!=='string'||!Number.isFinite(value.source.size)||value.source.size<=0||!/^[a-f0-9]{64}$/.test(value.source.sha256))fail();
  if(!Number.isFinite(value.duration)||value.duration<=0||value.duration>864000)fail();
  let previous=-1;
  const samples=value.samples.map(s=>{
    if(!s||!Number.isFinite(s.t)||s.t<0||s.t<=previous||s.t>value.duration+.1||!vector(s.p,3)||!vector(s.q,4)||Math.abs(Math.hypot(...s.q)-1)>.1)fail();
    previous=s.t;
    return {t:s.t,p:[...s.p],q:unit(s.q),observations:Math.max(0,Math.min(1e6,Math.floor(s.observations)||0))};
  });
  if(!Array.isArray(value.segments)||value.segments.length>samples.length)fail();
  let last=-1;
  const segments=value.segments.map(segment=>{
    if(!Array.isArray(segment)||segment.length!==2||!segment.every(Number.isInteger)||segment[0]<=last||segment[0]<0||segment[1]<=segment[0]||segment[1]>=samples.length)fail();
    last=segment[1];return [...segment];
  });
  let floor=null;
  if(value.floor){
    const f=value.floor;
    if(!vector(f.normal,3)||Math.abs(Math.hypot(...f.normal)-1)>.1||!Number.isFinite(f.offset)||Math.abs(f.offset)>1e9||!Number.isFinite(f.cameraHeight)||f.cameraHeight<=0)fail();
    const norm=Math.hypot(...f.normal),length=Math.abs(norm-1)<1e-12?1:norm;
    floor={normal:unit(f.normal),offset:f.offset/length,cameraHeight:f.cameraHeight,support:Math.max(0,Math.floor(f.support)||0),method:'sparse-point-ransac',estimated:true};
  }
  const quality={};
  for(const key of ['inputFrames','registeredFrames','points','unregisteredFrames','medianReprojectionError'])quality[key]=Number.isFinite(value.quality?.[key])?Math.max(0,value.quality[key]):0;
  quality.coverage=Number.isFinite(value.quality?.coverage)?Math.max(0,Math.min(1,value.quality.coverage)):0;
  return {version:1,source:{name:value.source.name.slice(0,255),size:value.source.size,sha256:value.source.sha256},duration:value.duration,engine:String(value.engine||'SfM').slice(0,120),engineVersion:String(value.engineVersion||'').slice(0,60),scale:'arbitrary',coordinates:'camera-to-world; camera axes right/down/forward',samples,segments,up:unit(value.up),floor,quality,maxGap:Number.isFinite(value.maxGap)?Math.max(.1,Math.min(10,value.maxGap)):1.5,processingSeconds:Number.isFinite(value.processingSeconds)?value.processingSeconds:0,warnings:Array.isArray(value.warnings)?value.warnings.filter(s=>typeof s==='string').map(s=>s.slice(0,80)).slice(0,10):[]};
}
export function poseAt(trajectory,time) {
  const samples=trajectory.samples;
  const segment=trajectory.segments.find(([a,b])=>time>=samples[a].t&&time<=samples[b].t);
  if(!segment)return null;
  let [low,high]=segment;
  while(high-low>1){const middle=(low+high)>>1;if(samples[middle].t<=time)low=middle;else high=middle;}
  const a=samples[low],b=samples[high],alpha=Math.max(0,Math.min(1,(time-a.t)/(b.t-a.t)));
  if(time===a.t||time===b.t){const sample=time===a.t?a:b;return {t:time,p:[...sample.p],q:[...sample.q],segment};}
  if(b.t-a.t>trajectory.maxGap)return null;
  return {t:time,p:lerp(a.p,b.p,alpha),q:slerp(a.q,b.q,alpha),segment};
}
export function floorPoint(position,floor,heightFactor=1) {
  const distance=dot(position,floor.normal)+floor.offset+floor.cameraHeight*(heightFactor-1);
  return position.map((x,i)=>x-floor.normal[i]*distance);
}
export function cameraVector(point,pose,view) {
  const inverse=[-pose.q[0],-pose.q[1],-pose.q[2],pose.q[3]];
  const cv=rotate(inverse,sub(point,pose.p)),v=[cv[0],-cv[1],cv[2]];
  const y=view.yaw*Math.PI/180,p=view.pitch*Math.PI/180;
  return [dot(v,[Math.cos(y),0,-Math.sin(y)]),dot(v,[-Math.sin(y)*Math.sin(p),Math.cos(p),-Math.cos(y)*Math.sin(p)]),dot(v,[Math.sin(y)*Math.cos(p),Math.sin(p),Math.cos(y)*Math.cos(p)])];
}
// Clip in camera space before dividing by depth, so lines never jump across the
// screen when the viewer rotates past a segment or looks behind the camera.
export function projectSegment(a,b,pose,view,width,height,near=.01) {
  let x=cameraVector(a,pose,view),y=cameraVector(b,pose,view);
  let start=0,end=1;
  if(x[2]<near&&y[2]<near)return null;
  if(x[2]<near){start=(near-x[2])/(y[2]-x[2]);x=lerp(x,y,start);}
  if(y[2]<near){const fraction=(near-y[2])/(x[2]-y[2]);end=1+(start-1)*fraction;y=lerp(y,x,fraction);}
  const focal=height/(2*Math.tan(view.fov*Math.PI/360));
  const project=(v,fraction)=>({x:width/2+v[0]/v[2]*focal,y:height/2-v[1]/v[2]*focal,z:v[2],fraction});
  return [project(x,start),project(y,end)];
}

// Pick only drawn segments. Perspective correction maps screen distance back to
// the original 3D segment's time; clipped endpoints carry their own timestamps.
export function pickPath(segments,x,y,tolerance=9,currentTime=0) {
  let best=null;
  for(const segment of segments){
    const {p,q,t0,t1}=segment,dx=q.x-p.x,dy=q.y-p.y,length=dx*dx+dy*dy;
    if(length<1e-8)continue;
    const fraction=Math.max(0,Math.min(1,((x-p.x)*dx+(y-p.y)*dy)/length));
    const distance=Math.hypot(x-p.x-fraction*dx,y-p.y-fraction*dy);
    if(distance>Math.max(tolerance,(segment.stroke||2)/2+4))continue;
    const alpha=segment.perspective?(fraction/q.z)/((1-fraction)/p.z+fraction/q.z):fraction;
    const time=t0+(t1-t0)*alpha;
    if(!Number.isFinite(time))continue;
    if(!best||distance<best.distance-.5||(Math.abs(distance-best.distance)<=.5&&Math.abs(time-currentTime)<Math.abs(best.time-currentTime)))best={time,distance,x:p.x+fraction*dx,y:p.y+fraction*dy};
  }
  return best;
}

// Roadview steps stay on the connected, observed part of the current route.
export function routeSteps(trajectory,time,{stepSeconds=2.5,maxSeconds=6}={}) {
  const pose=poseAt(trajectory,time);
  if(!pose)return [];
  const [start,end]=pose.segment,samples=trajectory.samples,up=trajectory.floor?.normal||trajectory.up;
  const minDistance=(trajectory.floor?.cameraHeight||0)*.2,steps=[];
  for(const direction of [-1,1]){
    const indices=[];
    for(let i=start;i<=end;i++)if((samples[i].t-time)*direction>.001)indices.push(i);
    if(direction<0)indices.reverse();
    let previousTime=time,target=null;
    for(const i of indices){
      const sample=samples[i],elapsed=Math.abs(sample.t-time);
      if(Math.abs(sample.t-previousTime)>trajectory.maxGap||elapsed>maxSeconds)break;
      previousTime=sample.t;
      const delta=sub(sample.p,pose.p),horizontal=sub(delta,up.map(x=>x*dot(delta,up))),distance=Math.hypot(...horizontal);
      if(distance<1e-6)continue;
      const inverse=[-pose.q[0],-pose.q[1],-pose.q[2],pose.q[3]],camera=rotate(inverse,horizontal);
      target={...sample,direction,yaw:Math.atan2(camera[0],camera[2])*180/Math.PI};
      if(elapsed>=stepSeconds&&distance>=minDistance)break;
    }
    if(target)steps.push(target);
  }
  return steps;
}

export function directionalSteps(steps,viewYaw) {
  const result={forward:null,right:null,back:null,left:null};
  for(const [key,offset] of Object.entries({forward:0,right:90,back:180,left:-90})){
    let nearest=45;
    for(const step of steps){
      const difference=Math.abs(((step.yaw-viewYaw-offset+180)%360+360)%360-180);
      if(difference<nearest){nearest=difference;result[key]=step;}
    }
  }
  return result;
}

// Keep the same world viewing direction when the recorded camera has rotated.
export function viewAfterStep(pose,target,view) {
  const yaw=view.yaw*Math.PI/180,pitch=view.pitch*Math.PI/180;
  const world=rotate(pose.q,[Math.sin(yaw)*Math.cos(pitch),-Math.sin(pitch),Math.cos(yaw)*Math.cos(pitch)]);
  const inverse=[-target.q[0],-target.q[1],-target.q[2],target.q[3]],camera=rotate(inverse,world);
  return {yaw:Math.atan2(camera[0],camera[2])*180/Math.PI,pitch:Math.asin(Math.max(-1,Math.min(1,-camera[1])))*180/Math.PI,fov:view.fov};
}

export function connectedRouteNodes(data,from,to) {
  if(!Number.isInteger(from)||!Number.isInteger(to)||from===to||from<0||to<0||from>=data.samples.length||to>=data.samples.length)return false;
  const low=Math.min(from,to),high=Math.max(from,to);
  if(!data.segments.some(([a,b])=>low>=a&&high<=b))return false;
  for(let i=low;i<high;i++)if(data.samples[i+1].t-data.samples[i].t>data.maxGap)return false;
  return true;
}

export function routeNodeAt(data,time) {
  const pose=poseAt(data,time);if(!pose)return null;
  const [start,end]=pose.segment;
  let index=start;
  for(let i=start+1;i<=end;i++)if(Math.abs(data.samples[i].t-time)<Math.abs(data.samples[index].t-time))index=i;
  return index;
}

export function defaultRouteLinks(data,index) {
  return routeSteps(data,data.samples[index].t).map(step=>({target:data.samples.findIndex(sample=>sample.t===step.t),t:step.t,yaw:step.yaw}));
}

export function validateRouteEdits(input,data) {
  const fail=()=>{throw new Error('경로 편집 정보가 원본 촬영 지점과 일치하지 않습니다.');};
  if(!data||!input||input.version!==1||input.sourceSha256!==data.source.sha256||!input.nodes||Array.isArray(input.nodes)||typeof input.nodes!=='object'||Object.keys(input.nodes).length>data.samples.length)fail();
  const nodes={};
  for(const [key,node] of Object.entries(input.nodes)){
    const index=Number(key);
    if(!/^(0|[1-9]\d*)$/.test(key)||!data.samples[index]||node?.t!==data.samples[index].t)fail();
    const saved={t:node.t};
    if(node.name!==undefined){if(typeof node.name!=='string'||!node.name.trim()||node.name.length>60)fail();saved.name=node.name.trim();}
    if(node.links!==undefined){
      if(!Array.isArray(node.links)||node.links.length>20)fail();
      const targets=new Set();
      saved.links=node.links.map(link=>{
        if(!link||!connectedRouteNodes(data,index,link.target)||targets.has(link.target)||link.t!==data.samples[link.target].t||!Number.isFinite(link.yaw)||link.yaw < -180||link.yaw>180)fail();
        targets.add(link.target);return {target:link.target,t:link.t,yaw:link.yaw};
      });
    }
    nodes[key]=saved;
  }
  return {version:1,sourceSha256:data.source.sha256,nodes};
}

export function connectRouteNodes(data,input,from,to,yaw,bidirectional=false) {
  if(!connectedRouteNodes(data,from,to)||!Number.isFinite(yaw)||yaw < -180||yaw>180)throw new Error('같은 촬영 구간의 다른 지점과 올바른 방향을 선택해 주세요.');
  const edits=validateRouteEdits(input||{version:1,sourceSha256:data.source.sha256,nodes:{}},data);
  const connect=(source,target,bearing)=>{
    const node=edits.nodes[source]||{t:data.samples[source].t};node.links ||= defaultRouteLinks(data,source);
    const existing=node.links.findIndex(link=>link.target===target),link={target,t:data.samples[target].t,yaw:bearing};
    if(existing>=0)node.links[existing]=link;
    else{if(node.links.length>=20)throw new Error('이동 연결은 지점당 최대 20개입니다.');node.links.push(link);}
    edits.nodes[source]=node;
  };
  connect(from,to,yaw);
  if(bidirectional){
    const angle=yaw*Math.PI/180,world=rotate(data.samples[from].q,[Math.sin(angle),0,Math.cos(angle)]);
    const q=data.samples[to].q,reverse=rotate([-q[0],-q[1],-q[2],q[3]],world.map(value=>-value));
    connect(to,from,Math.atan2(reverse[0],reverse[2])*180/Math.PI);
  }
  return validateRouteEdits(edits,data);
}

export function editedRouteSteps(data,time,edits) {
  const index=routeNodeAt(data,time),node=edits?.nodes?.[index];
  if(!Array.isArray(node?.links))return routeSteps(data,time);
  const pose=poseAt(data,time),source=data.samples[index],inverse=[-pose.q[0],-pose.q[1],-pose.q[2],pose.q[3]];
  return node.links.filter(link=>connectedRouteNodes(data,index,link.target)&&Math.abs(link.t-time)>.001).map(link=>{
    const yaw=link.yaw*Math.PI/180,world=rotate(source.q,[Math.sin(yaw),0,Math.cos(yaw)]),camera=rotate(inverse,world);
    return {...data.samples[link.target],yaw:Math.atan2(camera[0],camera[2])*180/Math.PI};
  });
}

import {distance} from './pipeline-tour.js';

const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const sub=(a,b)=>a.map((v,i)=>v-b[i]);
const mix=(a,b,t)=>a.map((v,i)=>v+(b[i]-v)*t);

// A shared spatial index for original mesh surfaces. This never edits geometry,
// moves the recorded route, or treats the recorded route as certified walkable.
export class RouteSurface {
 constructor(data,classes,cell=.75){
  this.positions=data.positions;this.cell=cell;this.bins=new Map();this.floorBins=new Map();this.large=[];this.floorLarge=[];this.floorFaces=new Set();
  const floor=classes.find(c=>c.name.toLowerCase()==='floor')?.rgb;
  for(let face=0;face<this.positions.length/9;face++){
   const t=this.triangle(face),normal=cross(sub(t[1],t[0]),sub(t[2],t[0]));
   const s=data.semantic,isFloor=floor&&s&&Math.abs(normal[2])>.65*Math.hypot(...normal)&&[0,1,2].filter(k=>floor.every((v,j)=>s[face*9+k*3+j]===v)).length>=2;
   if(isFloor)this.floorFaces.add(face);
   const xs=t.map(p=>p[0]),ys=t.map(p=>p[1]);
   const a=Math.floor(Math.min(...xs)/cell),b=Math.floor(Math.max(...xs)/cell),c=Math.floor(Math.min(...ys)/cell),d=Math.floor(Math.max(...ys)/cell);
   if((b-a+1)*(d-c+1)>1024){this.large.push(face);if(isFloor)this.floorLarge.push(face);continue;}
   for(let x=a;x<=b;x++)for(let y=c;y<=d;y++){const key=x+','+y;if(!this.bins.has(key))this.bins.set(key,[]);this.bins.get(key).push(face);if(isFloor){if(!this.floorBins.has(key))this.floorBins.set(key,[]);this.floorBins.get(key).push(face);}}
  }
 }
 triangle(face){const i=face*9,p=this.positions;return [[p[i],p[i+1],p[i+2]],[p[i+3],p[i+4],p[i+5]],[p[i+6],p[i+7],p[i+8]]];}
 candidates(a,b,floor=false){
  const bins=floor?this.floorBins:this.bins,found=new Set(floor?this.floorLarge:this.large),cell=this.cell;
  for(let x=Math.floor(Math.min(a[0],b[0])/cell);x<=Math.floor(Math.max(a[0],b[0])/cell);x++)for(let y=Math.floor(Math.min(a[1],b[1])/cell);y<=Math.floor(Math.max(a[1],b[1])/cell);y++)for(const face of bins.get(x+','+y)||[])found.add(face);
  return found;
 }
 hit(face,origin,direction,min,max){
  const [a,b,c]=this.triangle(face),e1=sub(b,a),e2=sub(c,a),h=cross(direction,e2),det=dot(e1,h);
  if(Math.abs(det)<1e-10)return null;
  const inv=1/det,s=sub(origin,a),u=inv*dot(s,h);if(u< -1e-9||u>1+1e-9)return null;
  const q=cross(s,e1),v=inv*dot(direction,q);if(v< -1e-9||u+v>1+1e-9)return null;
  const t=inv*dot(e2,q);return t>=min&&t<=max?t:null;
 }
 ground(camera){
  let closest=4.5,faceId=null;
  for(const face of this.candidates(camera,camera,true)){const t=this.hit(face,camera,[0,0,-1],.4,closest);if(t!==null){closest=t;faceId=face;}}
  return faceId===null?null:{point:[camera[0],camera[1],camera[2]-closest],face:faceId,height:closest};
 }
 visible(origin,target){
  const delta=sub(target,origin),length=Math.hypot(...delta);if(length<.1)return false;
  const direction=delta.map(v=>v/length);
  for(const face of this.candidates(origin,target)){
   const hit=this.hit(face,origin,direction,.04,length+.005);if(hit===null)continue;
   // Ignore only tiny local floor self-intersections near the painted endpoint.
   // Floors farther along the ray, walls, steps and furniture still block it.
   const height=Math.abs(origin[2]-target[2]);
   if(this.floorFaces.has(face)&&length-hit<.35*height&&Math.abs(origin[2]+hit*direction[2]-target[2])<.10*height)continue;
   return false;
  }
  return true;
 }
}

export function sampleRecordedPath(path,step,maxArc=Infinity){
 const result=[{point:[...path[0]],arc:0}];let arc=0,next=step;
 for(let i=1;i<path.length;i++){
  const a=path[i-1],b=path[i],length=distance(a,b);if(length<1e-8)continue;
  const end=Math.min(arc+length,maxArc);
  while(next<end-1e-8){result.push({point:mix(a,b,(next-arc)/length),arc:next});next+=step;}
  if(end>result.at(-1).arc+1e-8)result.push({point:mix(a,b,(end-arc)/length),arc:end});
  arc+=length;if(arc>=maxArc)break;
  while(next<=arc+1e-8)next+=step;
 }
 return result;
}
export function atArc(points,arc){
 if(arc<=points[0].arc)return {...points[0],point:[...points[0].point]};
 for(let i=1;i<points.length;i++)if(arc<=points[i].arc){const a=points[i-1],b=points[i];return {point:mix(a.point,b.point,(arc-a.arc)/(b.arc-a.arc)),arc};}
 return {...points.at(-1),point:[...points.at(-1).point]};
}
function section(points,start,end){return [atArc(points,start),...points.filter(p=>p.arc>start&&p.arc<end),atArc(points,end)];}

export function prepareRoute(surface,current,anchor){
 if(!surface)return {...anchor,status:'loading',points:[]};
 const destinationVisibility=surface.visible(current.position,anchor.position)?'visible':'occluded';
 let floor=surface.ground(current.position);
 if(!floor){for(const s of sampleRecordedPath(anchor.path,.15,.6)){floor=surface.ground(s.point);if(floor)break;}}
 if(!floor)return {...anchor,destinationVisibility,status:'no-floor',points:[],runs:[]};
 const total=anchor.path.slice(1).reduce((s,p,i)=>s+distance(anchor.path[i],p),0);
 const height=floor.height,near=Math.min(.8*height,.55*total),lift=.018*height;
 // Inspect the entire explicit connection. Distance/first-corner caps used to
 // make an intermediate bend look like the destination. Visibility is tested
 // independently along the whole route; view clipping is done by the renderer.
 const samples=sampleRecordedPath(anchor.path,.1*height),runs=[];let grounded=[],reason='short';
 const finish=()=>{if(grounded.length>1&&grounded.at(-1).arc-grounded[0].arc>.08*height)runs.push(grounded);grounded=[];};
 for(const s of samples){
  if(s.arc<near-1e-8)continue;
  const g=surface.ground(s.point);if(!g){reason='no-floor';finish();continue;}
  const point=[g.point[0],g.point[1],g.point[2]+lift];
  if(!surface.visible(current.position,point)){reason='occluded';finish();continue;}
  grounded.push({point,arc:s.arc,floorFace:g.face});
 }
 finish();const points=runs.flat();
 if(!points.length)return {...anchor,height,destinationVisibility,status:reason,points:[],runs:[]};
 return {...anchor,status:'ready',height,lift,points,runs,origin:[...current.position],destinationVisibility,
  label:anchor.label,labelPoint:anchor.position,nearArc:points[0].arc,tipArc:points.at(-1).arc,totalArc:total};
}

function pointSegmentDistance(p,a,b){const d=sub(b,a),den=dot(d,d),t=den?Math.max(0,Math.min(1,dot(sub(p,a),d)/den)):0;return distance(p,mix(a,b,t));}
export function simplifyGuide(points,tolerance){
 if(points.length<3)return points;
 let far=0,index=0;for(let i=1;i<points.length-1;i++){const d=pointSegmentDistance(points[i].point,points[0].point,points.at(-1).point);if(d>far){far=d;index=i;}}
 if(far<=tolerance)return [points[0],points.at(-1)];
 return [...simplifyGuide(points.slice(0,index+1),tolerance).slice(0,-1),...simplifyGuide(points.slice(index),tolerance)];
}
function roundStroke(points,half){
 const out=[],tri=(a,b,c)=>out.push(...a,...b,...c);
 for(let i=1;i<points.length;i++){
  const a=points[i-1],b=points[i],d=sub(b,a),n=Math.hypot(d[0],d[1]);if(n<1e-8)continue;
  const off=[-d[1]/n*half,d[0]/n*half,0],left=p=>p.map((v,k)=>v+off[k]),right=p=>p.map((v,k)=>v-off[k]);
  tri(left(a),right(a),left(b));tri(right(a),right(b),left(b));
 }
 for(const p of points)for(let i=0;i<12;i++){const a=i*Math.PI/6,b=(i+1)*Math.PI/6;tri(p,[p[0]+half*Math.cos(a),p[1]+half*Math.sin(a),p[2]],[p[0]+half*Math.cos(b),p[1]+half*Math.sin(b),p[2]]);}
 return out;
}
export function chevronPlacements(route){
 if(route.status!=='ready')return [];
 if(route.runs)return route.runs.flatMap(points=>chevronPlacements({...route,runs:null,points}));
 const p=route.points,h=route.height,start=p[0].arc,end=p.at(-1).arc,span=end-start;
 const arcs=[];
 const spacing=Math.min(.32*h,span*.35);
 for(let arc=start+Math.min(.20*h,span*.33);arc<=end-.07*h+1e-8;arc+=spacing)arcs.push(arc);
 if(!arcs.length)arcs.push(start+span*.55);
 return arcs.map(arc=>{
  const tip=atArc(p,arc).point,a=atArc(p,Math.max(start,arc-.08*h)).point,b=atArc(p,Math.min(end,arc+.08*h)).point,d=sub(b,a),n=Math.hypot(d[0],d[1]);
  if(n<1e-8)return null;
  const length=Math.min(.16*h,(arc-start)*.85),width=.13*h*(length/(.16*h)),back=[tip[0]-d[0]/n*length,tip[1]-d[1]/n*length,tip[2]];
  return {arc,points:[[back[0]-d[1]/n*width,back[1]+d[0]/n*width,back[2]],tip,[back[0]+d[1]/n*width,back[1]-d[0]/n*width,back[2]]]};
 }).filter(Boolean);
}
export function routeTriangles(route,widthScale=1){
 return new Float32Array(chevronPlacements(route).flatMap(c=>roundStroke(c.points,.017*route.height*widthScale)));
}
export function routeTrackTriangles(route){
 if(route.status!=='ready')return new Float32Array();
 if(route.runs){const pieces=route.runs.map(points=>routeTrackTriangles({...route,runs:null,points})),out=new Float32Array(pieces.reduce((n,p)=>n+p.length,0));let at=0;for(const p of pieces){out.set(p,at);at+=p.length;}return out;}
 const guide=simplifyGuide(route.points,.015*route.height);
 return new Float32Array(roundStroke(guide.map(p=>p.point),.014*route.height));
}
export function routeOccluders(positions,floorFaces){
 if(!floorFaces?.size)return positions;
 const result=new Float32Array(positions.length-floorFaces.size*9);let at=0;
 for(let face=0;face<positions.length/9;face++)if(!floorFaces.has(face)){result.set(positions.subarray(face*9,face*9+9),at);at+=9;}
 return result;
}

export function routeArrivalDirection(path){
 const end=path.at(-1);let arc=0;
 for(let i=path.length-2;i>=0;i--){arc+=distance(path[i],path[i+1]);if(arc>=.5||i===0){const d=sub(end,path[i]);if(Math.hypot(...d)>1e-8)return d;}}
 return null;
}

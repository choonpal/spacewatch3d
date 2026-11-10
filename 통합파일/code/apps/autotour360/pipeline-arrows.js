import {distance} from './pipeline-tour.js';

const rad=x=>x*Math.PI/180,deg=x=>x*180/Math.PI;
const wrap=x=>((x+180)%360+360)%360-180;
const validPose=pose=>Array.isArray(pose)&&pose.length===4&&pose.every(row=>Array.isArray(row)&&row.length===4&&row.every(Number.isFinite));
function basis(view){const y=rad(view.yaw),p=rad(view.pitch);return {forward:[Math.sin(y)*Math.cos(p),Math.sin(p),Math.cos(y)*Math.cos(p)],right:[Math.cos(y),0,-Math.sin(y)],up:[-Math.sin(y)*Math.sin(p),Math.cos(p),-Math.cos(y)*Math.sin(p)]};}
export function directionAngles(pose,world){
  if(!validPose(pose)||!Array.isArray(world)||world.length!==3||!world.every(Number.isFinite)||Math.hypot(...world)<1e-9)return null;
  // PanoVGGT camera Y points DOWN. The existing panorama display's Y points UP.
  const camera=[0,1,2].map(j=>world.reduce((sum,v,i)=>sum+pose[i][j]*v,0));
  return {yaw:wrap(deg(Math.atan2(camera[0],camera[2]))),pitch:deg(Math.atan2(-camera[1],Math.hypot(camera[0],camera[2])))};
}
export function worldDirection(pose,view){
  if(!validPose(pose)||!view)return null;
  const f=basis(view).forward,camera=[f[0],-f[1],f[2]];
  return [0,1,2].map(i=>camera.reduce((sum,v,j)=>sum+pose[i][j]*v,0));
}
export function headingAfterMove(from,to,view,direction=null){
  if(!view)return null;const world=direction||worldDirection(from,view),angles=directionAngles(to,world);
  return angles?{...angles,fov:view.fov}:{...view};
}
export function linkedDestinations(capture,views,index){
  const current=views[index];if(!current||!validPose(current.pose))return [];
  const byId=new Map(views.map((v,i)=>[v.id,{view:v,index:i}])),seen=new Set(),output=[];
  for(const link of capture.links||[]){
    if(!['captured_video_path','shortest_safe_path','spatial_proximity_jump'].includes(link.connection_method))continue;
    const forward=link.from_viewpoint===current.id,reverse=link.to_viewpoint===current.id;
    if(!forward&&!reverse)continue;const target=byId.get(forward?link.to_viewpoint:link.from_viewpoint);
    if(!target||seen.has(target.view.id)||!link.polyline_m?.length)continue;
    const path=forward?link.polyline_m:[...link.polyline_m].reverse();
    if(distance(path[0],current.position)>1e-7||distance(path.at(-1),target.view.position)>1e-7)continue;
    const length=path.slice(1).reduce((sum,p,i)=>sum+distance(path[i],p),0);if(length<1e-8)continue;
    // A photo-jump hotspot identifies the destination camera, not the first
    // recorded walking segment (which can start by retracing in the opposite
    // direction). The grey route is unchanged and is NOT a straight shortcut.
    const direction=target.view.position.map((v,i)=>v-current.position[i]);
    if(Math.hypot(...direction)<1e-8)continue;
    const actual=directionAngles(current.pose,direction);if(!actual)continue;
    // The marker is the saved destination CAMERA position. Do not lower it to
    // an invented floor point, clamp it to an edge, or offset overlapping pins.
    output.push({index:target.index,id:target.view.id,label:target.view.label||'K'+(target.index+1),position:target.view.position,path,direction,yaw:actual.yaw,pitch:actual.pitch});seen.add(target.view.id);
  }
  return output.sort((a,b)=>a.index-b.index);
}
export function projectKeyframe(anchor,view,width,height){
  if(!(width>0&&height>0))return null;
  const ray=basis(anchor).forward,b=basis(view),dot=a=>a.reduce((sum,v,i)=>sum+v*ray[i],0),z=dot(b.forward);
  if(z<=1e-8)return null;
  const scale=Math.tan(rad(view.fov)/2);
  const x=width*(1+dot(b.right)/(z*scale*width/height))/2,y=height*(1-dot(b.up)/(z*scale))/2;
  return x>=0&&x<=width&&y>=0&&y<=height?{x,y}:null;
}
export function targetVisibility(positions,origin,target){
  if(!positions?.length||positions.length%9)return 'unknown';
  const delta=target.map((v,i)=>v-origin[i]),length=Math.hypot(...delta);
  if(!Number.isFinite(length)||length<1e-8)return 'unknown';
  const [dx,dy,dz]=delta.map(v=>v/length),[ox,oy,oz]=origin;
  // Double-sided first-surface segment test on the unchanged full mesh.
  // Numerical endpoint margins do not move the projected camera position.
  for(let i=0;i<positions.length;i+=9){
    const ax=positions[i],ay=positions[i+1],az=positions[i+2];
    const e1x=positions[i+3]-ax,e1y=positions[i+4]-ay,e1z=positions[i+5]-az;
    const e2x=positions[i+6]-ax,e2y=positions[i+7]-ay,e2z=positions[i+8]-az;
    const hx=dy*e2z-dz*e2y,hy=dz*e2x-dx*e2z,hz=dx*e2y-dy*e2x;
    const det=e1x*hx+e1y*hy+e1z*hz;if(Math.abs(det)<1e-10)continue;
    const inv=1/det,sx=ox-ax,sy=oy-ay,sz=oz-az;
    const u=inv*(sx*hx+sy*hy+sz*hz);if(u<0||u>1)continue;
    const qx=sy*e1z-sz*e1y,qy=sz*e1x-sx*e1z,qz=sx*e1y-sy*e1x;
    const v=inv*(dx*qx+dy*qy+dz*qz);if(v<0||u+v>1)continue;
    const t=inv*(e2x*qx+e2y*qy+e2z*qz);
    if(t>.05&&t<length-.2)return 'occluded';
  }
  return 'visible';
}

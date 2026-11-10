// Display-only motion. Never creates a viewpoint, image or shortcut in the plan.
export const distance=(a,b)=>Math.hypot(...a.map((v,i)=>v-b[i]));
const same=(a,b)=>distance(a,b)<1e-8;
export function movementPath(capture,views,from,to){
  if(from===to)return [views[from].position];
  const step=to>from?1:-1,path=[views[from].position];
  for(let i=from;i!==to;i+=step){
    const a=views[i],b=views[i+step];
    const forward=capture.links.find(l=>l.from_viewpoint===a.id&&l.to_viewpoint===b.id);
    const reverse=capture.links.find(l=>l.from_viewpoint===b.id&&l.to_viewpoint===a.id);
    const link=forward||reverse;
    if(!link||!['captured_video_path','shortest_safe_path','spatial_proximity_jump'].includes(link.connection_method)||!link.polyline_m?.length)return null;
    const points=forward?link.polyline_m:[...link.polyline_m].reverse();
    if(!same(points[0],a.position)||!same(points.at(-1),b.position))return null;
    for(const p of points)if(!same(path.at(-1),p))path.push(p);
  }
  return path;
}
export function samplePath(path,t){
  if(!path?.length)return null;
  const lengths=path.slice(1).map((p,i)=>distance(path[i],p)),total=lengths.reduce((a,b)=>a+b,0);
  if(!total)return [...path[0]];
  let target=Math.max(0,Math.min(1,t))*total;
  for(let i=0;i<lengths.length;i++){
    if(target<=lengths[i]&&lengths[i]>0){const a=target/lengths[i];return path[i].map((v,k)=>v+(path[i+1][k]-v)*a);}
    target-=lengths[i];
  }
  return [...path.at(-1)];
}
export function projectMap(position,camera,width,height){
  const p=[position[0]-camera.center[0],position[2]-camera.center[1],-position[1]-camera.center[2]];
  const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),aspect=width/height;
  return [width*(1+dot(p,camera.basis.right)/(camera.radius*aspect))/2,height*(1-dot(p,camera.basis.up)/camera.radius)/2];
}
export function fitMapRadius(bounds,basis,aspect){
  const [a,b]=bounds,center=a.map((v,i)=>(v+b[i])/2),points=[];
  for(const x of [a[0],b[0]])for(const y of [a[1],b[1]])for(const z of [a[2],b[2]])points.push([x-center[0],z-center[2],-(y-center[1])]);
  const extent=axis=>Math.max(...points.map(p=>Math.abs(p.reduce((s,v,i)=>s+v*basis[axis][i],0))));
  return Math.max(.2,1.17*Math.max(extent('up'),extent('right')/Math.max(aspect,.1)));
}
export function animateTransition(duration,render,clock={request:cb=>requestAnimationFrame(cb),cancel:id=>cancelAnimationFrame(id),now:()=>performance.now()}){
  let id,finished=false,resolve;const started=clock.now();
  const done=new Promise(r=>resolve=r);
  const finish=ok=>{if(finished)return;finished=true;if(id!==undefined)clock.cancel(id);resolve(ok);};
  const tick=now=>{if(finished)return;const t=duration>0?Math.max(0,Math.min(1,(now-started)/duration)):1;render(t*t*(3-2*t));if(t===1)finish(true);else id=clock.request(tick);};
  if(duration<=0)tick(started);else id=clock.request(tick);
  return {done,cancel:()=>finish(false)};
}

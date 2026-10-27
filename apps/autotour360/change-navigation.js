// Presentation only: original detector events and correspondence uncertainty stay intact.
export function displayEvent(event){
 if(event.representative_origin||event.type!=='moved'||!event.origin_candidates?.length)return event;
 const origins=[...event.origin_candidates].sort((a,b)=>(b.appearance||0)-(a.appearance||0)||(b.shape_f1||0)-(a.shape_f1||0)||String(a.object_id).localeCompare(String(b.object_id)));
 const origin=origins.find(o=>o.object_id===event.correspondence?.before)||origins[0];
 return {...event,before_bounds:origin.bounds,before_center:origin.center,before_points:origin.points,
  origin_candidates:[origin],representative_origin:true,origin_alternative_count:origins.length};
}

export function focusBounds(event){
 if(event.type==='unresolved'||!event.type)return event.bounds;
 return event.before_bounds||event.after_bounds||event.bounds;
}

function cameraScore(event,view){
 const position=view.camera_to_common_cv.map(row=>row[3]).slice(0,3);
 const focus=focusBounds(event),center=focus[0].map((x,k)=>(x+focus[1][k])/2);
 const centers=event.type==='moved'?[event.before_center,event.after_center].filter(Boolean):[center];
 const rays=centers.map(c=>c.map((x,k)=>x-position[k])),distances=rays.map(r=>Math.hypot(...r));
 const bounds=focusBounds(event),radius=Math.max(.01,Math.hypot(...bounds[1].map((x,k)=>x-bounds[0][k]))/2);
 const angle=rays.length===2?Math.acos(Math.max(-1,Math.min(1,rays[0].reduce((s,x,k)=>s+x*rays[1][k],0)/Math.max(1e-8,distances[0]*distances[1])))):0;
 return angle+Math.max(0,radius*1.2-Math.min(...distances))*10/radius+.12*Math.abs(Math.log(Math.max(...distances,.001)/(4*radius)));
}

export function changeStops(events,views){
 const stops=new Map();
 for(const source of events){
  const event=displayEvent(source),visible=[...new Set(event.observations||[])].filter(i=>views[i]);
  visible.sort((a,b)=>cameraScore(event,views[a])-cameraScore(event,views[b])||a-b);
  if(!visible.length)continue;
  const viewIndex=visible[0];
  if(!stops.has(viewIndex))stops.set(viewIndex,{viewIndex,eventIds:[]});
  stops.get(viewIndex).eventIds.push(event.id);
 }
 return [...stops.values()].sort((a,b)=>views[a.viewIndex].source_epoch-views[b.viewIndex].source_epoch||views[a.viewIndex].source_index-views[b.viewIndex].source_index||a.viewIndex-b.viewIndex);
}

export function routeStops(views,epoch){
 return views.flatMap((v,i)=>v.source_epoch===epoch?[{viewIndex:i,eventIds:[]}]:[]);
}

export function neighborStop(stops,index,delta){
 return Math.max(0,Math.min(stops.length-1,index+delta));
}

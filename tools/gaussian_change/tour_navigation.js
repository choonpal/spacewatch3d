// Saved graph navigation is independent of candidate counts and display filters.
export function routeBetween(group,from,to){
 if(from===to)return group.nodes.some(n=>n.id===from)?{ids:[from],points:[group.nodes.find(n=>n.id===from).position]}:null;
 const adjacency=new Map(group.nodes.map(n=>[n.id,[]]));
 for(const edge of group.links){
  adjacency.get(edge.from)?.push({to:edge.to,path:edge.polyline});
  adjacency.get(edge.to)?.push({to:edge.from,path:[...edge.polyline].reverse()});
 }
 if(!adjacency.has(from)||!adjacency.has(to))return null;
 const queue=[from],parents=new Map([[from,null]]);
 for(let i=0;i<queue.length&&!parents.has(to);i++)for(const edge of adjacency.get(queue[i])){
  if(!parents.has(edge.to)){parents.set(edge.to,{from:queue[i],path:edge.path});queue.push(edge.to);}
 }
 if(!parents.has(to))return null;
 const steps=[];let node=to;while(node!==from){const edge=parents.get(node);steps.unshift({id:node,...edge});node=edge.from;}
 return {ids:[from,...steps.map(s=>s.id)],points:steps.flatMap((s,i)=>i?s.path.slice(1):s.path)};
}
export function neighbor(group,id,direction){
 const segment=group.segments.find(s=>s.nodes.includes(id));if(!segment)return null;
 return segment.nodes[segment.nodes.indexOf(id)+direction]||null;
}
export function samplePath(points,fraction){
 if(!points.length)throw new Error('Empty saved path');
 const lengths=points.slice(1).map((p,i)=>Math.hypot(...p.map((x,k)=>x-points[i][k])));
 let remaining=Math.max(0,Math.min(1,fraction))*lengths.reduce((a,b)=>a+b,0);
 for(let i=0;i<lengths.length;i++){
  if(remaining<=lengths[i]&&lengths[i]>0){const t=remaining/lengths[i];return points[i].map((x,k)=>x+(points[i+1][k]-x)*t);}
  remaining-=lengths[i];
 }
 return [...points.at(-1)];
}
export const pathLength=points=>points.slice(1).reduce((sum,p,i)=>sum+Math.hypot(...p.map((x,k)=>x-points[i][k])),0);

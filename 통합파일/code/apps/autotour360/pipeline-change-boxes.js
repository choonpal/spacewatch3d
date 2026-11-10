// Presentation only: candidate status is not a new change decision.
export function changeState(row){
  const items=row?.items||[];
  return items.some(r=>r.type!=='unresolved')?'estimated':items.length?'deferred':'none';
}
export function markerColor(row){return {estimated:'#ff5757',deferred:'#ffb432',none:'#a7d4e6'}[changeState(row)];}
export function projectBox(item,project,width,height){
  if(!width||!height)return null;
  const points=(item.angular_samples||[]).map(p=>project(p[0],p[1])).filter(Boolean);
  if(points.length<2)return null;
  const x=Math.max(0,Math.min(...points.map(p=>p.x))),y=Math.max(0,Math.min(...points.map(p=>p.y)));
  const right=Math.min(width,Math.max(...points.map(p=>p.x))),bottom=Math.min(height,Math.max(...points.map(p=>p.y)));
  return right-x>=3&&bottom-y>=3?{x,y,width:right-x,height:bottom-y}:null;
}

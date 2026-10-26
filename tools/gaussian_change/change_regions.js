// Shared event IDs survive viewpoint changes; observations never add events.
export function uniqueForView(candidates,index,visibility){
 const selected=new Map();
 for(const b of candidates){const observation=visibility?.[b.id];if(!observation?.visible)continue;
  const id=index?.candidate_to_event[b.id]||b.id,values=Object.values(observation.support_rays),score=Math.min(...values),secondary=Math.max(...values);
  const previous=selected.get(id);
  if(!previous||score>previous.score||score===previous.score&&secondary>previous.secondary)selected.set(id,{box:b,score,secondary});
 }
 return [...selected.values()].map(v=>v.box);
}

export function visibleRegions(index,candidates){
 if(!index)return [];
 const boxes=new Map(candidates.map(b=>[b.id,b]));
 return index.regions.map(region=>{
  const events=region.event_ids.map(id=>index.events[id]).map(event=>({...event,candidate_ids:event.candidate_ids.filter(id=>boxes.has(id))})).filter(e=>e.candidate_ids.length);
  if(!events.length)return null;
  const keyframes=region.observing_keyframes.filter(v=>events.some(e=>e.candidate_ids.some(id=>index.observations[v].candidate_ids.includes(id))));
  const representative=keyframes.includes(region.representative_view)?region.representative_view:keyframes[0];
  return {...region,events,observing_keyframes:keyframes,representative_view:representative};
 }).filter(Boolean);
}

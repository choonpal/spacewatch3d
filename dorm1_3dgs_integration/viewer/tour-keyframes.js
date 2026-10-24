import {dot,sub,rotate,poseAt} from './trajectory-math.js';

export const DENSITIES={
  compact:{label:'적게',spacing:2.2,deviation:.5,maxSeconds:8},
  standard:{label:'보통',spacing:1.4,deviation:.3,maxSeconds:5},
  detailed:{label:'촘촘하게',spacing:.8,deviation:.18,maxSeconds:3}
};

// Fingerprint the route as well as the video: re-analysis can change its poses.
export function routeSignature(data){
  const text=JSON.stringify([data.samples.map(s=>[s.t,s.p,s.q]),data.segments,data.maxGap,data.up,data.floor]);
  let hash=2166136261;for(let i=0;i<text.length;i++)hash=Math.imul(hash^text.charCodeAt(i),16777619);
  return (hash>>>0).toString(16).padStart(8,'0');
}

export function connectedChunks(data){
  const chunks=[];
  for(const [start,end] of data.segments){
    let first=start;
    for(let i=start;i<end;i++)if(data.samples[i+1].t-data.samples[i].t>data.maxGap){chunks.push([first,i]);first=i+1;}
    chunks.push([first,end]);
  }
  return chunks;
}

function geometry(data){
  const up=data.floor?.normal||data.up;
  const horizontal=v=>sub(v,up.map(x=>x*dot(v,up)));
  const points=data.samples.map(s=>horizontal(s.p));
  const steps=connectedChunks(data).flatMap(([a,b])=>points.slice(a+1,b+1).map((p,i)=>Math.hypot(...sub(p,points[a+i])))).filter(v=>v>1e-8).sort((a,b)=>a-b);
  return {points,scale:data.floor?.cameraHeight||steps[Math.floor(steps.length/2)]*5||1};
}

function pointLineDistance(point,a,b){
  const ab=sub(b,a),length=dot(ab,ab),alpha=length?Math.max(0,Math.min(1,dot(sub(point,a),ab)/length)):0;
  return Math.hypot(...sub(point,a.map((v,i)=>v+ab[i]*alpha)));
}

// Preserve bends first, then fill long intervals by travelled distance. Time is
// a secondary bound only when the camera moves; standing still adds no nodes.
export function selectTourKeyframes(data,metrics=[],density='standard'){
  if(!DENSITIES[density])throw new Error('지원하지 않는 키프레임 간격입니다.');
  const {points,scale}=geometry(data),settings=DENSITIES[density],chosen=new Map();
  const qualities=new Map(metrics.map(m=>[m.index,m]));
  const peak=Math.max(1,...metrics.map(m=>Math.log1p(m.sharpness)));
  const quality=index=>{const m=qualities.get(index);return m?Math.log1p(m.sharpness)/peak*.8+m.exposure*.2:0;};
  for(const [start,end] of connectedChunks(data)){
    const anchors=new Map([[start,'구간 시작'],[end,'구간 끝']]),stack=[[start,end]];
    while(stack.length){
      const [a,b]=stack.pop();let furthest=-1,distance=settings.deviation*scale;
      for(let i=a+1;i<b;i++){const d=pointLineDistance(points[i],points[a],points[b]);if(d>distance){distance=d;furthest=i;}}
      if(furthest>=0){anchors.set(furthest,'동선 꺾임');stack.push([a,furthest],[furthest,b]);}
    }
    const bends=[...anchors.keys()].sort((a,b)=>a-b);
    for(let k=0;k<bends.length-1;k++){
      let last=bends[k],distance=0;
      for(let i=last+1;i<bends[k+1];i++){
        distance+=Math.hypot(...sub(points[i],points[i-1]));
        const moved=Math.hypot(...sub(points[i],points[last]));
        if(distance>=settings.spacing*scale||(data.samples[i].t-data.samples[last].t>=settings.maxSeconds&&moved>=.2*scale)){
          anchors.set(i,distance>=settings.spacing*scale?'이동 거리 확보':'이동 구간 유지');last=i;distance=0;
        }
      }
    }
    // A distance stop just before a preserved bend/end adds almost no new
    // viewpoint. Keep the structural anchor and drop the redundant soft stop.
    const all=[...anchors.keys()].sort((a,b)=>a-b);
    for(let k=1;k<all.length-1;k++){
      const index=all[k];if(['구간 시작','구간 끝','동선 꺾임'].includes(anchors.get(index)))continue;
      if([all[k-1],all[k+1]].some(neighbour=>['구간 시작','구간 끝','동선 꺾임'].includes(anchors.get(neighbour))&&Math.abs(data.samples[index].t-data.samples[neighbour].t)<1.5&&Math.hypot(...sub(points[index],points[neighbour]))<settings.spacing*scale*.45))anchors.delete(index);
    }
    const ordered=[...anchors.keys()].sort((a,b)=>a-b);
    ordered.forEach((index,k)=>{
      let best=index,reason=anchors.get(index);
      // Endpoints and bends stay exact. Only a distance anchor can shift to a
      // sharper nearby observation, without changing order or its route chunk.
      if(reason!=='구간 시작'&&reason!=='구간 끝'&&reason!=='동선 꺾임'){
        let score=quality(index);
        for(let i=Math.max(start,index-2,ordered[k-1]+1);i<=Math.min(end,index+2,ordered[k+1]-1);i++){
          if(Math.abs(data.samples[i].t-data.samples[index].t)>.8||Math.hypot(...sub(points[i],points[index]))>scale*.2)continue;
          const candidate=quality(i)-Math.abs(data.samples[i].t-data.samples[index].t)*.04;
          if(candidate>score+.001){best=i;score=candidate;}
        }
        if(best!==index)reason+=' · 주변 선명도 우선';
      }
      chosen.set(best,{index:best,t:data.samples[best].t,reason});
    });
  }
  return [...chosen.values()].sort((a,b)=>a.index-b.index);
}

export function validateTourKeyframes(input,data){
  const fail=()=>{throw new Error('키프레임 정보가 원본 영상 또는 촬영 경로와 일치하지 않습니다.');};
  if(!data||!input||input.version!==1||input.sourceSha256!==data.source.sha256||input.routeSignature!==routeSignature(data)||!DENSITIES[input.density]||!Array.isArray(input.frames)||!input.frames.length||input.frames.length>data.samples.length||!Array.isArray(input.metrics)||input.metrics.length>data.samples.length)fail();
  const eligible=new Set(connectedChunks(data).flatMap(([a,b])=>Array.from({length:b-a+1},(_,i)=>a+i)));
  let previous=-1;
  const reference=item=>{if(!item||!Number.isInteger(item.index)||!eligible.has(item.index)||item.index<=previous||item.t!==data.samples[item.index].t)fail();previous=item.index;};
  const frames=input.frames.map(frame=>{
    reference(frame);
    if(typeof frame.reason!=='string'||!frame.reason.trim()||frame.reason.length>100)fail();
    const thumbnail=typeof frame.thumbnail==='string'&&frame.thumbnail.length<60000&&/^data:image\/jpeg;base64,[A-Za-z0-9+/=]+$/.test(frame.thumbnail)?frame.thumbnail:'';
    return {index:frame.index,t:frame.t,reason:frame.reason,thumbnail};
  });
  previous=-1;
  const metrics=input.metrics.map(m=>{
    reference(m);if(!Number.isFinite(m.sharpness)||m.sharpness<0||m.sharpness>1e7||!Number.isFinite(m.exposure)||m.exposure<0||m.exposure>1)fail();
    return {index:m.index,t:m.t,sharpness:m.sharpness,exposure:m.exposure};
  });
  return {version:1,sourceSha256:input.sourceSha256,routeSignature:input.routeSignature,density:input.density,enabled:input.enabled!==false,frames,metrics};
}

// The next/previous selected observation is a navigation target, never a chord
// across an unobserved interval or a spatially nearby part of a later revisit.
export function keyframeRouteSteps(data,time,selection){
  const chunk=connectedChunks(data).find(([a,b])=>time>=data.samples[a].t&&time<=data.samples[b].t);
  if(!chunk||chunk[0]===chunk[1])return [];
  const pose=poseAt({...data,segments:[chunk]},time);if(!pose)return [];
  const up=data.floor?.normal||data.up,inverse=[-pose.q[0],-pose.q[1],-pose.q[2],pose.q[3]],steps=[];
  for(const direction of [-1,1]){
    const frames=selection.frames.filter(f=>f.index>=chunk[0]&&f.index<=chunk[1]&&(f.t-time)*direction>.025);
    if(direction<0)frames.reverse();
    for(const frame of frames){
      const sample=data.samples[frame.index],delta=sub(sample.p,pose.p),horizontal=sub(delta,up.map(x=>x*dot(delta,up)));
      if(Math.hypot(...horizontal)<1e-6)continue;
      const camera=rotate(inverse,horizontal);steps.push({...sample,direction,yaw:Math.atan2(camera[0],camera[2])*180/Math.PI});break;
    }
  }
  return steps;
}

// Compare the equatorial band to avoid the stretched poles of a 360 panorama.
// This is an image-detail heuristic, not an object-visibility or accuracy score.
export function frameQuality(pixels,width,height){
  const grey=new Float32Array(width*height);let exposed=0,count=0,sum=0,squared=0;
  for(let i=0;i<grey.length;i++)grey[i]=pixels[i*4]*.299+pixels[i*4+1]*.587+pixels[i*4+2]*.114;
  for(let y=Math.max(1,Math.floor(height*.25));y<Math.min(height-1,Math.ceil(height*.75));y++)for(let x=1;x<width-1;x++){
    const i=y*width+x,value=grey[i],lap=grey[i-1]+grey[i+1]+grey[i-width]+grey[i+width]-4*value;
    sum+=lap;squared+=lap*lap;exposed+=value>15&&value<240?1:0;count++;
  }
  return {sharpness:count?Math.max(0,squared/count-(sum/count)**2):0,exposure:count?exposed/count:0};
}

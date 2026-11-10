import {animateTransition,distance,samplePath} from './pipeline-tour.js';
import {directionAngles} from './pipeline-arrows.js';

const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const smooth=(a,b,t)=>{const x=clamp((t-a)/(b-a),0,1);return x*x*(3-2*x);};
export const travelLength=path=>path?.length>1?path.slice(1).reduce((sum,p,i)=>sum+distance(path[i],p),0):0;
export function travelDuration(path,height=1.7,reduced=false){
 if(reduced)return 0;
 const length=travelLength(path);if(length<.05)return 300;
 return Math.round(clamp(470+155*Math.log1p(length/Math.max(.5,height)),580,920));
}
export function travelFrame(progress,forward=true){
 const t=clamp(progress,0,1),pulse=Math.sin(Math.PI*t);
 return {sourceScale:1+(forward ? .20 : 0)*smooth(0,.9,t),
  targetScale:1+(forward ? .09 : 0)*(1-smooth(.12,1,t)),
  targetOpacity:smooth(.16,.86,t),sourceBlur:forward?1.25*pulse:0,targetBlur:forward ? .45*pulse : 0};
}

// A display-only travel effect using two ORIGINAL panoramas. It creates no
// intermediate capture, model output, source-image edit, or shortcut in the plan.
export class PanoramaTravel {
 constructor(source,panel,makePanorama){
  this.source=source;this.panel=panel;this.active=null;
  const canvas=source.canvas.ownerDocument.createElement('canvas');canvas.id='travel-panorama';canvas.hidden=true;canvas.setAttribute('aria-hidden','true');
  source.canvas.insertAdjacentElement('afterend',canvas);this.canvas=canvas;
  try{this.incoming=makePanorama(canvas);}catch(error){canvas.remove();throw error;}
 }
 start({image,current,targetView,path,height=1.7,onProgress=()=>{},reduced=false,clock}){
  this.cancel();
  const length=travelLength(path),forward=length>.05,sourceCanvas=this.source.canvas;
  let focus=null;
  if(forward){const next=samplePath(path,Math.min(1,.8*height/length)),direction=next.map((v,i)=>v-current.position[i]),angles=directionAngles(current.pose,direction);if(angles)focus=this.source.project(angles.yaw,angles.pitch);}
  const width=sourceCanvas.clientWidth,heightPx=sourceCanvas.clientHeight;
  sourceCanvas.style.transformOrigin=`${clamp(focus?.x??width/2,width*.15,width*.85)}px ${clamp(focus?.y??heightPx/2,heightPx*.15,heightPx*.85)}px`;
  sourceCanvas.style.willChange='transform, filter';this.canvas.style.opacity='0';this.canvas.hidden=false;
  try{this.incoming.resize();this.incoming.setView(targetView);this.incoming.setMedia(image);this.incoming.render();this.source.render();if(this.incoming.lost||this.incoming.media!==image){this.clear();return null;}}
  catch(error){this.clear();return null;}
  this.panel.classList.add('is-travelling');this.panel.setAttribute('aria-busy','true');this.panel.dataset.travelState='travelling';
  const paint=t=>{
   const frame=travelFrame(t,forward);sourceCanvas.style.transform=`scale(${frame.sourceScale})`;sourceCanvas.style.filter=`blur(${frame.sourceBlur}px)`;
   this.canvas.style.transform=`scale(${frame.targetScale})`;this.canvas.style.opacity=String(frame.targetOpacity);this.canvas.style.filter=`blur(${frame.targetBlur}px)`;
   this.panel.dataset.travelProgress=t.toFixed(3);onProgress(t);
  };
  paint(0);
  const animation=animateTransition(travelDuration(path,height,reduced),paint,clock);
  const controller={done:animation.done,cancel:()=>{animation.cancel();if(this.active===controller){this.active=null;this.clear();}}};
  this.active=controller;return controller;
 }
 clear(){
  for(const name of ['transform','transform-origin','filter','will-change'])this.source.canvas.style.removeProperty(name);
  this.canvas.hidden=true;this.canvas.style.opacity='0';this.canvas.style.removeProperty('transform');this.canvas.style.removeProperty('filter');
  this.incoming.setMedia(null);this.panel.classList.remove('is-travelling');this.panel.setAttribute('aria-busy','false');this.panel.dataset.travelState='idle';delete this.panel.dataset.travelProgress;
 }
 finish(){this.active=null;this.clear();}
 cancel(){if(this.active)this.active.cancel();else this.clear();}
}

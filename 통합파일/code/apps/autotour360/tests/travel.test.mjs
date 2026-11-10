import test from 'node:test';
import assert from 'node:assert/strict';
import {PanoramaTravel,travelDuration,travelFrame,travelLength} from '../pipeline-travel.js';

const path=[[0,0,0],[0,0,4],[3,0,4]],current={position:path[0],pose:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]};
function fixture(){
 const style=()=>({removeProperty(key){delete this[key.replace(/-([a-z])/g,(_,c)=>c.toUpperCase())];}});
 const canvas=()=>({clientWidth:800,clientHeight:400,style:style(),hidden:false,setAttribute(){},insertAdjacentElement(){}});
 const sourceCanvas=canvas();sourceCanvas.ownerDocument={createElement:canvas};
 const original={kind:'source-photo'},source={canvas:sourceCanvas,media:original,project:()=>({x:300,y:150}),render(){}};
 const classes=new Set(),attributes={},panel={dataset:{},classList:{add:c=>classes.add(c),remove:c=>classes.delete(c)},setAttribute:(k,v)=>attributes[k]=v};
 let rendererCount=0;
 const make=()=>{rendererCount++;return {resize(){},setView(v){this.view=v;},setMedia(v){this.media=v;},render(){}};};
 let now=0,last=0;const callbacks=new Map(),clock={now:()=>now,request:fn=>{callbacks.set(++last,fn);return last;},cancel:id=>callbacks.delete(id)};
 const step=time=>{now=time;const queue=[...callbacks.values()];callbacks.clear();queue.forEach(fn=>fn(time));};
 const player=new PanoramaTravel(source,panel,make),frames=[];
 const args={image:{kind:'destination-photo'},current,targetView:{yaw:20,pitch:0,fov:80},path,height:2,clock,onProgress:t=>frames.push(t)};
 return {player,source,original,panel,classes,attributes,args,frames,step,rendererCount:()=>rendererCount};
}
test('motion duration is bounded, respects reduced motion, and does not fabricate a disconnected route',()=>{
 assert.equal(travelLength(path),7);assert.equal(travelDuration(path,2,true),0);assert.equal(travelDuration(null),300);
 assert.ok(travelDuration([[0,0,0],[0,0,.2]],2)>=580);assert.ok(travelDuration([[0,0,0],[0,0,10000]],2)<=920);
});
test('travel frames end at an unblurred fully visible destination',()=>{
 const first=travelFrame(0),last=travelFrame(1),middle=travelFrame(.5);
 assert.equal(first.sourceScale,1);assert.equal(first.targetOpacity,0);assert.equal(last.targetOpacity,1);assert.equal(last.targetScale,1);assert.ok(last.targetBlur<1e-8);
 assert.ok(middle.sourceScale>1&&middle.targetOpacity>0&&middle.targetOpacity<1);assert.equal(travelFrame(.5,false).sourceScale,1);
});
test('the target stays covered until the caller commits its image and synchronized map selection',async()=>{
 const f=fixture(),duration=travelDuration(path,2),motion=f.player.start(f.args);
 assert.equal(f.player.canvas.hidden,false);assert.ok(f.classes.has('is-travelling'));assert.equal(f.source.media,f.original);
 f.step(duration/2);assert.ok(+f.player.canvas.style.opacity>0&&+f.player.canvas.style.opacity<1);assert.equal(f.source.media,f.original);
 f.step(duration);assert.equal(await motion.done,true);assert.equal(f.player.canvas.style.opacity,'1');assert.equal(f.player.canvas.hidden,false);
 f.player.finish();assert.equal(f.player.canvas.hidden,true);assert.equal(f.player.incoming.media,null);assert.equal(f.source.canvas.style.transform,undefined);assert.equal(f.attributes['aria-busy'],'false');
});
test('cancelling restores the committed view and prevents late animation frames',async()=>{
 const f=fixture(),motion=f.player.start(f.args);f.step(200);motion.cancel();const count=f.frames.length;
 f.step(2000);assert.equal(await motion.done,false);assert.equal(f.frames.length,count);assert.equal(f.source.media,f.original);assert.equal(f.source.canvas.style.filter,undefined);assert.equal(f.player.canvas.hidden,true);assert.equal(f.panel.dataset.travelState,'idle');
});
test('a second navigation cancels the first and reuses the same panorama renderer',async()=>{
 const f=fixture(),first=f.player.start(f.args);f.step(200);const second=f.player.start(f.args);
 assert.equal(await first.done,false);f.step(200+travelDuration(path,2));assert.equal(await second.done,true);assert.equal(f.rendererCount(),1);f.player.finish();
});
test('reduced motion completes in one frame without intermediate zoom frames',async()=>{
 const f=fixture(),motion=f.player.start({...f.args,reduced:true});assert.equal(await motion.done,true);assert.deepEqual(f.frames,[0,1]);f.player.finish();
});
test('a renderer failure leaves the original viewer usable',()=>{
 const f=fixture();f.player.incoming.render=()=>{throw Error('test graphics failure');};assert.equal(f.player.start(f.args),null);assert.equal(f.player.canvas.hidden,true);assert.equal(f.source.media,f.original);assert.equal(f.classes.has('is-travelling'),false);
});

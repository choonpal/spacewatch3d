import {Panorama} from './panorama.js';
import {PipelineMesh} from './pipeline-mesh.js';
import {orderedViews,decodeGeometry} from './pipeline-model.js';
import {movementPath,samplePath,projectMap,fitMapRadius} from './pipeline-tour.js';
import {linkedDestinations,projectKeyframe,directionAngles,headingAfterMove} from './pipeline-arrows.js';
import {RouteSurface,prepareRoute,routeArrivalDirection,chevronPlacements} from './pipeline-route-arrows.js';
import {RouteOverlay} from './pipeline-route-overlay.js';
import {PanoramaTravel} from './pipeline-travel.js';
import {changeState as originalChangeState,projectBox} from './pipeline-change-boxes.js';
const $=id=>document.getElementById(id),svgNS='http://www.w3.org/2000/svg';
const fail=e=>{$('error').hidden=false;$('error').textContent=e?.message||String(e);};
let manifest,capture,views=[],index=0,token=0,pano,mesh,transition,motionPoint,overlayData,focusChangeIndex=0,displayOriginal=false;
let markerGeometry=null,markerCache=null,routeOverlay,travelPlayer=null;
const navigationCache=new Map();
const text=(tag,value,parent)=>{const e=document.createElement(tag);e.textContent=value;parent?.append(e);return e;};
const table=(target,headers,rows)=>{const t=document.createElement('table'),h=t.createTHead().insertRow();headers.forEach(v=>text('th',v,h));const b=t.createTBody();rows.forEach(row=>{const r=b.insertRow();row.forEach(v=>text('td',v??'—',r));});$(target).replaceChildren(t);};
const pct=x=>Number.isFinite(x)?(100*x).toFixed(1)+'%':'—';
const duration=x=>Number.isFinite(x)?(x>=60?`${Math.floor(x/60)}분 ${(x%60).toFixed(1)}초`:`${x.toFixed(2)}초`):'—';
const changeRow=(id=capture?.id,view=views[index])=>overlayData?.captures?.[id]?.views.find(r=>r.viewpoint_id===view?.id);
const changeState=row=>row?.verified_change_ids?.length||row?.items?.some(item=>item.verified_change)?'estimated':row?.evidence_ids?.length?'deferred':originalChangeState(row);
const markerColor=row=>({estimated:'#ff5757',deferred:'#ffb432',none:'#a7d4e6'}[changeState(row)]);
const boxesEnabled=()=>$('toggle-changes').getAttribute('aria-pressed')==='true';
function changeLabels(){if(!pano)return;const layer=$('change-labels');layer.replaceChildren();const row=changeRow();
 const active=capture?.id===overlayData?.after_capture;
 $('toggle-changes').hidden=!active;$('overlay-hint').hidden=true;$('focus-changes').hidden=!active||!boxesEnabled()||!row?.items.length;
 if(!active||!boxesEnabled()||!row)return;
 for(const item of row.items){const p=projectBox(item,(yaw,pitch)=>pano.project(yaw,pitch),layer.clientWidth,layer.clientHeight);if(!p)continue;const box=text('div',null,layer);box.className='change-box '+(item.verified_change||item.type!=='unresolved'?'estimated':'deferred');box.style.left=p.x+'px';box.style.top=p.y+'px';box.style.width=p.width+'px';box.style.height=p.height+'px';}
 if(row.evidence_ids?.length&&!row.items.length){$('overlay-hint').hidden=false;$('overlay-hint').textContent='이 지점에 연결된 변화 근거가 있습니다. 변화 버튼에서 비교 사진을 확인하세요.';}}
function evidenceCards(all=false){
 const panel=$('evidence-panel'),body=$('evidence-body');body.replaceChildren();
 if(capture?.id!==overlayData?.after_capture){text('p','변화 근거는 비교 후 촬영에서 확인할 수 있습니다.',body);return;}
 const row=changeRow(),allowed=new Set(row?.evidence_ids||[]);
 const evidence=(overlayData?.evidence||[]).filter(item=>all||allowed.has(item.id));
 $('evidence-title').textContent=all?'전체 변화 근거':`K${index+1}에 연결된 변화 근거`;
 if(!evidence.length)text('p','이 지점에 연결된 비교 사진이 없습니다. 전체 변화를 확인할 수 있습니다.',body);
 const groups=new Map();for(const item of evidence){const key=item.verified_change||item.event_id;if(!groups.has(key))groups.set(key,[]);groups.get(key).push(item);}
 for(const [key,items] of groups){const group=overlayData.verified_changes?.find(group=>group.id===key),section=text('section',null,body);section.className='evidence-group';text('h3',group?.label||items[0].label,section);
  const unique=new Map();for(const item of items){const identity=item.image+'|'+JSON.stringify(item.bbox);if(!unique.has(identity))unique.set(identity,item);}
  for(const item of unique.values()){const figure=text('figure',null,section),wrap=text('div',null,figure);wrap.className='evidence-image';const image=text('img',null,wrap);image.src=item.image;image.alt=`${item.label} 변화 비교 사진`;image.loading='lazy';
   if(item.bbox?.length===4){const [x0,y0,x1,y1]=item.bbox,svg=document.createElementNS(svgNS,'svg');svg.setAttribute('viewBox',`0 0 ${item.width} ${item.height}`);svg.setAttribute('aria-hidden','true');const rect=document.createElementNS(svgNS,'rect');for(const [name,value] of Object.entries({x:x0,y:y0,width:x1-x0,height:y1-y0}))rect.setAttribute(name,value);svg.append(rect);wrap.append(svg);}
   text('figcaption',`비교 후 원본 프레임 ${item.source_frame_index} · ${item.role==='expected_location'?'이전 물건의 예상 위치':'관찰된 물건'}`,figure);
  }
 }
 panel.hidden=false;
}
function openEvidence(all=false){tab(false);$('evidence-panel').hidden=false;evidenceCards(all);}
function preparedAnchors(current,surface){
 const cached=navigationCache.get(current.id);if(surface&&cached?.surface===surface)return cached.anchors;
 const anchors=linkedDestinations(capture,views,views.indexOf(current)).map(a=>{const route=prepareRoute(surface,current,a);route.labelTips=chevronPlacements(route).filter(c=>Math.hypot(c.points[1][0]-current.position[0],c.points[1][1]-current.position[1])>.8*route.height&&surface.visible(current.position,c.points[1])).sort((a,b)=>b.arc-a.arc);return route;});
 if(surface)navigationCache.set(current.id,{surface,anchors});return anchors;
}
function arrows(){if(!pano||!capture||!views.length)return;changeLabels();const layer=$('arrows'),w=layer.clientWidth,h=layer.clientHeight;layer.replaceChildren();
 const current=views[index],surface=markerGeometry?.capture===capture?markerGeometry.surface:null;
 if(markerCache?.current!==current||markerCache?.surface!==surface){markerCache={current,surface,anchors:preparedAnchors(current,surface)};routeOverlay?.setRoutes(markerCache.anchors,index);}
 routeOverlay?.draw(current,pano.view);
 const dock=text('nav',null,layer);dock.className='keyframe-dock';dock.setAttribute('aria-label','다른 경로로 이동');
 const project=point=>projectKeyframe(directionAngles(current.pose,point.map((v,k)=>v-current.position[k])),pano.view,w,h),used=[];
 for(const a of markerCache.anchors){
  const pin=a.labelTips?.map(head=>({p:project(head.points[1]),point:head.points[1],arc:head.arc})).find(head=>head.p&&head.p.x>25&&head.p.x<w-25&&head.p.y>48&&head.p.y<h-20&&used.every(p=>Math.hypot(p.x-head.p.x,p.y-head.p.y)>45));
  const b=text('button',a.label,pin?layer:dock);b.dataset.viewpointId=a.id;
  if(pin){used.push(pin.p);b.className='route-label route-head-label';b.style.left=pin.p.x+'px';b.style.top=(pin.p.y-8)+'px';b.dataset.worldPosition=JSON.stringify(pin.point);b.dataset.routeArc=String(pin.arc);b.setAttribute('aria-label',`${a.label} 안내선 앞쪽 · 경로로 이동`);b.title=`보이는 안내선의 앞쪽 · 순찰 경로를 따라 ${a.label}로 이동`;}
  else{const note=a.status==='loading'?'경로 확인 중':a.status==='no-floor'?'경로 확인 불가':a.status==='occluded'?'안내선 가려짐':'안내선 화면 밖';b.className='keyframe-jump';b.textContent=`${a.label} · ${note}`;b.setAttribute('aria-label',`${a.label} 경로로 이동 · ${note}`);}
  b.onclick=()=>select(a.index,routeArrivalDirection(a.path)).catch(fail);
  if(a.status==='ready'){
   const svg=document.createElementNS(svgNS,'svg');svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.classList.add('route-hit-layer');layer.append(svg);let d='';
   for(const run of a.runs||[a.points]){for(const s of run){const p=project(s.point);d+=p?`${d?'L':'M'}${p.x},${p.y} `:'';if(!p&&d){const path=document.createElementNS(svgNS,'path');path.setAttribute('d',d);path.onclick=b.onclick;svg.append(path);d='';}}if(d){const path=document.createElementNS(svgNS,'path');path.setAttribute('d',d);path.onclick=b.onclick;svg.append(path);d='';}}
   if(d){const path=document.createElementNS(svgNS,'path');path.setAttribute('d',d);path.onclick=b.onclick;svg.append(path);}
  }
 }
 dock.hidden=!dock.childElementCount;}
function drawMap(){if(!mesh?.capture||!views.length)return;const svg=$('map-overlay'),w=svg.clientWidth,h=svg.clientHeight;svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.replaceChildren();const cam={center:mesh.center,radius:mesh.radius,basis:mesh.basis()},project=p=>projectMap(p,cam,w,h);
 const node=(name,attrs,parent=svg)=>{const e=document.createElementNS(svgNS,name);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));parent.append(e);return e;};
 for(const link of capture.links||[])node('polyline',{points:link.polyline_m.map(p=>project(p).join(',')).join(' ')});
 // Keyframe markers always represent their actual source-camera coordinates.
 // The separate unlabeled pulse is only a transition cue, never the current view.
 if(motionPoint){const p=project(motionPoint);node('circle',{cx:p[0],cy:p[1],r:3,fill:'#fff','aria-hidden':'true',class:'travel-pulse'});}
 views.forEach((v,i)=>{const p=project(v.position),row=changeRow(capture.id,v),state=changeState(row),note={estimated:'변화 근거 연결',deferred:'보류 근거 연결',none:'연결된 변화 없음'}[state],g=node('g',{tabindex:0,role:'button','aria-label':`K${i+1} · ${note}`,'data-change-state':state});if(i===index)node('circle',{cx:p[0],cy:p[1],r:11,style:'fill:none;stroke:#fff;stroke-width:2'},g);node('circle',{cx:p[0],cy:p[1],r:i===index?8:5,style:`fill:${markerColor(row)}`},g);node('text',{x:p[0]+12,y:p[1]+4},g).textContent='K'+(i+1);g.onclick=()=>select(i).catch(fail);g.onkeydown=e=>{if(['Enter',' '].includes(e.key)){e.preventDefault();select(i).catch(fail);}};});}
// Frame-timed held keys avoid depending on the operating system's key-repeat rate.
const heldViewKeys=new Set();let viewKeyFrame=0,viewKeyTime=0;
function stopViewKeys(){heldViewKeys.clear();cancelAnimationFrame(viewKeyFrame);viewKeyFrame=0;viewKeyTime=0;}
function typingTarget(target){return target instanceof Element&&!!target.closest('input,textarea,select,[contenteditable]:not([contenteditable="false"]),[role="textbox"],[role="combobox"]');}
function canRotateView(){return pano?.media&&!$('tour').hidden&&$('loading').hidden&&!$('view-panel').classList.contains('is-travelling')&&!document.hidden;}
function rotateHeldView(now){
 viewKeyFrame=0;if(!heldViewKeys.size||!canRotateView()){stopViewKeys();return;}
 const dt=viewKeyTime?Math.min((now-viewKeyTime)/1000,.05):0;viewKeyTime=now;
 const x=Number(heldViewKeys.has('KeyD'))-Number(heldViewKeys.has('KeyA'));
 const y=Number(heldViewKeys.has('KeyW'))-Number(heldViewKeys.has('KeyS'));
 const step=75*dt/Math.max(1,Math.hypot(x,y));
 if(x||y)pano.setView({...pano.view,yaw:pano.view.yaw+x*step,pitch:pano.view.pitch+y*step});
 viewKeyFrame=requestAnimationFrame(rotateHeldView);
}
function installViewKeys(){
 window.addEventListener('keydown',e=>{
  if(!['KeyW','KeyA','KeyS','KeyD'].includes(e.code)||e.ctrlKey||e.metaKey||e.altKey||typingTarget(e.target)||!canRotateView())return;
  e.preventDefault();heldViewKeys.add(e.code);if(!viewKeyFrame)viewKeyFrame=requestAnimationFrame(rotateHeldView);
 });
 window.addEventListener('keyup',e=>{heldViewKeys.delete(e.code);if(!heldViewKeys.size)stopViewKeys();});
 window.addEventListener('blur',stopViewKeys);window.addEventListener('pagehide',stopViewKeys);
 document.addEventListener('visibilitychange',stopViewKeys);
 document.addEventListener('focusin',e=>{if(typingTarget(e.target))stopViewKeys();});
 $('panorama').title='드래그 또는 W/A/S/D를 누르고 시선 회전';
 $('panorama').setAttribute('aria-keyshortcuts','W A S D');
}
async function select(next,navigationDirection=null){stopViewKeys();if(!views.length)return;next=Math.max(0,Math.min(views.length-1,next));const id=++token,old=views[index],target=views[next],from=index;transition?.cancel();motionPoint=null;$('view-label').textContent=`K${index+1}`;drawMap();$('loading').hidden=false;
 const img=new Image();img.src=displayOriginal&&target.original_image?target.original_image:target.image;try{await img.decode();}catch(error){if(id!==token)return;$('loading').hidden=true;throw error;}if(id!==token)return;
 const path=movementPath(capture,views,from,next),heading=old?headingAfterMove(old.pose,target.pose,pano.view,navigationDirection):null;
 const reduced=window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
 if(from!==next&&pano.media&&!reduced){
  const surface=markerGeometry?.capture===capture?markerGeometry.surface:null;if(surface)preparedAnchors(target,surface);
  if(id!==token)return;
  if(!travelPlayer){try{travelPlayer=new PanoramaTravel(pano,$('view-panel'),canvas=>new Panorama(canvas,()=>{},()=>{}));}catch(error){travelPlayer=null;}}
  if(travelPlayer){
   const height=surface?.ground(old.position)?.height||1.7;$('loading').hidden=true;$('view-label').textContent=`K${from+1} → K${next+1}`;
   transition=travelPlayer.start({image:img,current:old,targetView:heading||{...pano.view},path,height,onProgress:t=>{motionPoint=path?samplePath(path,t):null;drawMap();}});
   if(transition){const completed=await transition.done;if(!completed||id!==token)return;}
  }
 }
 if(id!==token)return;
 index=next;focusChangeIndex=0;$('focus-changes').textContent='◎';motionPoint=null;pano.setMedia(img);if(heading)pano.setView(heading);pano.render();travelPlayer?.finish();$('loading').hidden=true;$('panorama').classList.remove('moving');
 $('display-restored').disabled=!target.original_image;$('view-label').textContent=`K${index+1}`;$('counter').textContent=`${capture.id.toUpperCase()} · ${index+1} / ${views.length}`;$('frame-select').value=String(index);arrows();
 transition=null;drawMap();if(!$('evidence-panel').hidden)evidenceCards(false);}
function settings(){const s=manifest.runtime_details||{},inputs=manifest.requested_inputs||{};$('manual').replaceChildren();for(const line of inputs.manual||[])text('li',line,$('manual'));
 const units=manifest.coordinate_units||'u';$('coordinate-note').textContent=`지도 좌표 단위는 ${units}${units==='u'?'(재구성 단위)':''}입니다.${manifest.metric_scale_verified?'':' 실제 미터 단위로 검증하지 않았습니다.'}`;
 $('classes').textContent=(inputs.segmentation_classes||inputs.object_classes||[]).join(' · ');$('videos').replaceChildren();for(const v of inputs.videos||[])text('li',v.split('/').at(-1),$('videos'));
 table('coverage',['촬영','후보 / 선정','공간 관측률(전체 표면)','객체 품질 충족','전체 입력 목표'],manifest.captures.map(c=>[c.id,`${c.capture_candidates} / ${c.views.length}`,pct(c.whole_surface_coverage_min),`${c.object_observed_count}/${c.object_input_count}`,c.input_coverage_status]));
 $('pose-note').textContent='전체 영상의 0.25초 구간마다 가장 선명한 원본 프레임을 후보로 사용합니다. 현재 선정 결과의 실제 지점 수와 관측 상태를 표시합니다.';
 table('ratings',['객체','5회 점수','평균 / 5','표준편차'],(s.ratings||[]).map(r=>[r.class_name,r.ratings.join(', '),(Number(r.mean)/5).toFixed(2),r.sd]));
 table('noise',['촬영','클래스','원본 연결영역','유지','관측 의무 제외'],s.noise||[]);
 table('timing',['단계','이번 처리시간','구분'],(s.timings||[]).map(r=>[r.stage,duration(r.seconds),r.reused?'기존 산출물 재사용·확인':'실제 계산']));
 $('version').textContent=`${s.version||manifest.pipeline_version} · 실행 ${manifest.run_id}`;
 const changes=manifest.changes;$('change-link').hidden=!changes?.viewer_supported;
 $('change-status').textContent=changes?`이번 실행: 변화 후보 묶음 ${changes.verified_count??0}건 · 원본 후보 ${changes.candidates}건 · 판정 보류 ${changes.deferred??0}건. 묶음은 독립 정답 검증을 뜻하지 않습니다.`:'이번 실행에는 회차 간 변화 결과가 없습니다.';
 $('change-link').onclick=e=>{e.preventDefault();const after=manifest.captures.find(c=>c.id===overlayData?.after_capture);if(after&&capture!==after)choose(after).then(()=>openEvidence(true)).catch(fail);else openEvidence(true);};}
function tab(settingsOn){stopViewKeys();if(settingsOn)$('evidence-panel').hidden=true;$('settings').hidden=!settingsOn;$('tour').hidden=settingsOn;$('settings-tab').setAttribute('aria-pressed',String(settingsOn));$('tour-tab').setAttribute('aria-pressed',String(!settingsOn));history.replaceState(null,'',settingsOn?'#settings':'#tour');if(!settingsOn){pano.resize();mesh.draw();arrows();}}
async function choose(c){stopViewKeys();++token;transition?.cancel();travelPlayer?.cancel();navigationCache.clear();capture=c;markerGeometry=null;markerCache=null;routeOverlay?.setScene(null);views=orderedViews(c).map((v,i)=>({...v,label:'K'+(i+1)}));index=0;motionPoint=null;[...$('captures').children].forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.id===c.id)));
 $('evidence-panel').hidden=true;$('changes-tab').hidden=c.id!==overlayData?.after_capture;$('changes-tab').textContent=`변화 후보 ${overlayData?.counts?.verified_changes??0}건`;
 $('frame-select').replaceChildren(...views.map((v,i)=>new Option(`K${i+1} · 프레임 ${v.source_frame?.source_frame_index??v.id}`,String(i))));$('downloads').replaceChildren();for(const [name,url] of Object.entries(c.downloads||{})){const a=text('a',name+' ↓',$('downloads'));a.href=url;a.download='';}
 $('map-mode').querySelector('[value=instances]').disabled=!c.instance_mesh;mesh.capture=c;mesh.count=0;mesh.reset();const rect=$('mesh').getBoundingClientRect();mesh.radius=fitMapRadius(c.bounds,mesh.basis(),rect.width/Math.max(1,rect.height));
 pano.setView({yaw:0,pitch:0,fov:80});await Promise.all([mesh.load(c,$('map-mode').value).then(async()=>{if(capture!==c)return;let data=mesh.data;if(data.kind!==1||$('map-mode').value==='instances'){const r=await fetch(c.mesh);if(!r.ok)throw Error('순찰 경로 표시용 지도를 불러오지 못했습니다.');data=decodeGeometry(await r.arrayBuffer());}if(capture!==c)return;const surface=new RouteSurface(data,c.classes);routeOverlay.setScene(data.positions,surface.floorFaces);markerGeometry={capture:c,surface};markerCache=null;arrows();}),select(0)]);if(capture===c){drawMap();arrows();}}
async function start(){const r=await fetch('/api/pipeline');if(!r.ok)throw Error('결과를 불러오지 못했습니다.');manifest=await r.json();if(!manifest.captures?.length)throw Error('촬영 결과가 없습니다.');
 if(manifest.keypoint_changes){const r=await fetch(manifest.keypoint_changes.url);if(!r.ok)throw Error('변화 표시 자료를 불러오지 못했습니다.');overlayData=await r.json();}
 const routeCanvas=document.createElement('canvas');routeCanvas.id='route-canvas';routeCanvas.setAttribute('aria-hidden','true');$('panorama').insertAdjacentElement('afterend',routeCanvas);routeOverlay=new RouteOverlay(routeCanvas);pano=new Panorama($('panorama'),arrows,fail);installViewKeys();mesh=new PipelineMesh($('mesh'));mesh.afterDraw=drawMap;for(const c of manifest.captures){const b=text('button',c.id.toUpperCase(),$('captures'));b.dataset.id=c.id;b.onclick=()=>choose(c).catch(fail);}
 $('tour-tab').onclick=()=>tab(false);$('settings-tab').onclick=()=>tab(true);$('frame-select').onchange=()=>select(Number($('frame-select').value)).catch(fail);$('map-mode').onchange=()=>mesh.load(capture,$('map-mode').value).catch(fail);$('ceiling').onchange=()=>{mesh.cut=$('ceiling').checked;mesh.draw();};
 $('display-restored').onchange=()=>{displayOriginal=!$('display-restored').checked;select(index).catch(fail);};
 $('toggle-changes').onclick=()=>{const on=!boxesEnabled();$('toggle-changes').setAttribute('aria-pressed',String(on));$('toggle-changes').textContent=on?'박스 켜짐':'박스 꺼짐';changeLabels();};
 $('changes-tab').onclick=()=>openEvidence(false);$('evidence-all').onclick=()=>evidenceCards(true);$('evidence-close').onclick=()=>{$('evidence-panel').hidden=true;};
 $('focus-changes').textContent='◎';$('focus-changes').setAttribute('aria-label','박스 방향 보기');$('focus-changes').title='박스 방향 보기';$('focus-changes').onclick=()=>{const items=[...(changeRow()?.items||[])].sort((a,b)=>(a.type==='unresolved')-(b.type==='unresolved')||b.pixels-a.pixels),n=focusChangeIndex++%items.length,item=items[n];if(item)pano.setView({yaw:item.yaw,pitch:item.pitch,fov:pano.view.fov});};
 for(const [id,k] of [['zoom-in',.8],['zoom-out',1.25]])$(id).onclick=()=>{mesh.radius=Math.max(.2,Math.min(500,mesh.radius*k));mesh.draw();};$('fit').onclick=()=>{mesh.reset();const r=$('mesh').getBoundingClientRect();mesh.radius=fitMapRadius(capture.bounds,mesh.basis(),r.width/r.height);mesh.draw();};$('look-reset').onclick=()=>pano.setView({yaw:0,pitch:0,fov:80});new ResizeObserver(arrows).observe($('view-panel'));settings();const requested=new URLSearchParams(location.search).get('capture');await choose(manifest.captures.find(c=>c.id===requested)||manifest.captures[0]);tab(location.hash==='#settings');}
start().catch(fail);

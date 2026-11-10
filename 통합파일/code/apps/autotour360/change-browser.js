// Evidence navigation only. Never creates object coordinates or detector votes.
const names={chair:'의자',table:'탁자',box:'상자',person:'사람',window:'창문',door:'문'};
const node=(tag,text,parent)=>{const e=document.createElement(tag);if(text!=null)e.textContent=text;parent?.append(e);return e;};

export async function mountChangeBrowser(manifest,{navigate,currentCapture}){
 const spec=manifest.change_browser;if(!spec)return null;
 const response=await fetch(spec.catalog_url);if(!response.ok)throw Error('변화 근거 목록을 불러오지 못했습니다.');
 const data=await response.json();if(data.events.length!==manifest.changes.candidates)throw Error('변화 후보 목록 수가 다릅니다.');
 const css=node('link',null,document.head);css.rel='stylesheet';css.href=spec.stylesheet;
 const panel=node('aside',null,document.body);panel.id='change-browser';panel.setAttribute('aria-label','전체 변화 후보와 실제 검출 사진');
 const legend=document.querySelector('.map-legend');if(legend)node('span',' · 보라 테두리: 검출 근거 연결',legend);
 const header=node('div',null,panel);header.className='proof-heading';node('strong',`변화 후보 ${data.events.length}개`,header);
 const full=node('a','전체 원본 비교',header);full.href=manifest.changes.url;
 const close=node('button','닫기',header);close.onclick=()=>{panel.hidden=true;};
 node('p','후보를 누르면 가까운 키포인트로 이동하고, 아래에 실제 검출 입력 사진을 보여줍니다. 보류 후보도 숨기지 않습니다.',panel);
 const list=node('div',null,panel);list.className='proof-list';const content=node('div',null,panel);content.className='proof-content';
 let selected=null,pairIndex=0;
 function renderEvidence(event){
  content.replaceChildren();node('h3',`${event.id} · ${names[event.label]||event.label} · ${event.type==='unresolved'?'판단 보류':'변화 추정'}`,content);
  node('p','아래는 실제 검출 근거 사진입니다. 현재 키포인트 사진에 억지로 옮긴 박스가 아닙니다.',content);
  const target=event.targets[currentCapture()];
  if(target)node('p',`연결 위치: ${target.capture.toUpperCase()} K${target.keypoint} · 근거 촬영 위치와 ${target.camera_gap_m.toFixed(2)}m 차이 (물체 좌표 아님)`,content);
  node('p',event.display_note,content);
  const controls=node('div',null,content);controls.className='proof-pairs';
  if(event.pairs.length>1){const prev=node('button','이전 근거',controls);prev.onclick=()=>{pairIndex=(pairIndex+event.pairs.length-1)%event.pairs.length;renderEvidence(event);};}
  node('span',`근거 쌍 ${pairIndex+1}/${event.pairs.length}`,controls);
  if(event.pairs.length>1){const next=node('button','다음 근거',controls);next.onclick=()=>{pairIndex=(pairIndex+1)%event.pairs.length;renderEvidence(event);};}
  const photos=node('div',null,content);photos.className='proof-photos';
  for(const side of ['0','1']){
   const p=event.pairs[pairIndex][side];if(!p)continue;
   const card=node('figure',null,photos);node('figcaption',`${p.capture_id.toUpperCase()} · ${p.timestamp_s.toFixed(2)}초 · ${p.role==='observed_object'?'관측된 영역':'대응 위치'}`,card);
   const frame=node('div',null,card);frame.className='proof-frame';const image=node('img',null,frame);image.src=p.image;image.alt=`${event.id} ${p.capture_id} 실제 검출 사진`;image.loading='lazy';
   if(p.bbox){const [x0,y0,x1,y1]=p.bbox;const box=node('div',null,frame);box.className='proof-box'+(event.type==='unresolved'?' deferred':'');box.style.left=100*x0/p.width+'%';box.style.top=100*y0/p.height+'%';box.style.width=100*(x1-x0)/p.width+'%';box.style.height=100*(y1-y0)/p.height+'%';}
  }
 }
 async function select(event){
  selected=event;pairIndex=0;panel.hidden=false;
  for(const b of list.children)b.setAttribute('aria-pressed',String(b.dataset.id===event.id));
  await navigate(event.targets[currentCapture()],event.id);renderEvidence(event);
 }
 for(const event of data.events){const button=node('button',`${event.id} · ${names[event.label]||event.label}${event.type==='unresolved'?' · 보류':''}`,list);button.dataset.id=event.id;button.setAttribute('aria-pressed','false');button.onclick=()=>select(event).catch(e=>{content.replaceChildren();node('p',e.message,content);});}
 const toggle=document.getElementById('changes-tab');toggle.textContent=`변화 ${data.events.length}개 보기`;toggle.onclick=e=>{e.preventDefault();panel.hidden=!panel.hidden;};
 const current=node('button',null,document.getElementById('view-panel'));current.id='current-proof';current.hidden=true;
 function update(capture,view){
  const events=data.events.filter(e=>e.targets[capture]?.viewpoint_id===view.id);current.hidden=!events.length;
  current.textContent=events.length?`연결된 변화 근거 ${events.length}개 보기`:'';
  current.onclick=()=>select(events[0]);if(selected&&!panel.hidden)renderEvidence(selected);
 }
 node('p','12개 후보 모두 이 목록에서 확인할 수 있습니다. 사진에 박스가 안 그려지는 후보도 실제 검출 사진을 열어봅니다.',content);
 panel.hidden=!spec.open_on_load;
 return {update,openEvent:id=>select(data.events.find(e=>e.id===id))};
}

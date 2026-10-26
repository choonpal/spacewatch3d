export function changeSource(search){const value=new URLSearchParams(search).get('source');return ['dorm','rescan','gaussian'].includes(value)?value:'gaussian';}
export function isTourMessage(event,frame,origin){return event.origin===origin&&event.source===frame.contentWindow&&event.data?.type==='gaussian-tour:state';}

// Mount the same native Gaussian route renderer inside the existing app mode.
export function createGaussianChangeTour({onFrame=()=>{},onError=()=>{}}={}){
 let active=false,ready=false,frame=null,pending=null,container=null,statistics=null,failure=null;
 const send=(action,value)=>frame?.contentWindow?.postMessage({type:'gaussian-tour:command',action,value},location.origin);
 const report=(state={})=>{if(Number.isInteger(state.events))statistics={...statistics,...state};onFrame({presentation:{projectTitle:'통합 변화투어',heading:state.keyframe?`${state.keyframe} · Gaussian 경로 투어`:'Gaussian 경로 투어',
  description:'공통 3D 위치의 같은 변화를 관측 시점에 연결합니다.',resolution:'3D GAUSSIAN',
  summary:failure||(statistics?`저장된 자동분석 결과 · ${statistics.regions}구역 / 고유 후보 ${statistics.events}개 · ${statistics.keyframes}지점`:'저장된 자동분석 결과를 연결하는 중')}});};
 function mount(){
  container=document.createElement('section');container.id='gaussian-change-workspace';container.className='gaussian-change-workspace';
  const toolbar=document.createElement('div');toolbar.className='gaussian-source-bar';
  const label=document.createElement('label');label.textContent='변화탐지 입력 ';
  const picker=document.createElement('select');picker.setAttribute('aria-label','변화탐지 입력');
  for(const [value,text]of [['gaussian','Gaussian 경로 투어'],['dorm','기존 Dorm 메쉬·사진'],['rescan','기존 Rescan']]){const option=document.createElement('option');option.value=value;option.textContent=text;picker.append(option);}
  picker.onchange=()=>{const url=new URL(location.href);url.searchParams.set('mode','changes');url.searchParams.set('source',picker.value);location.href=url;};label.append(picker);toolbar.append(label);
  const note=document.createElement('span');note.textContent='전·후 저장 경로를 각각 이동 · 같은 변화는 한 번만 집계';toolbar.append(note);container.append(toolbar);
  const message=document.createElement('p');message.id='gaussian-change-status';message.setAttribute('role','status');message.textContent='Gaussian 경로 투어를 연결하는 중…';container.append(message);
  document.getElementById('viewport').append(container);
  window.addEventListener('message',event=>{if(frame&&isTourMessage(event,frame,location.origin)){if(event.data.error){failure=event.data.error;message.hidden=false;message.textContent=failure;onError(failure);}report(event.data);}});
  return fetch('/api/gaussian-change/status').then(async response=>{if(!response.ok)throw new Error('Gaussian 변화 결과를 확인하지 못했습니다.');const state=await response.json();
   if(!state.available){failure='Gaussian 맵 파일 연결 필요';message.textContent='Gaussian 맵 파일을 연결해 주세요. 서버의 --gaussian-change-dir에 준비된 결과 폴더를 지정할 수 있습니다.';message.className='gaussian-missing';report();return;}
   frame=document.createElement('iframe');frame.id='gaussian-change-frame';frame.title='Gaussian 경로 이동과 자동 변화 후보';frame.allow='fullscreen';frame.src=state.viewer_url;frame.onload=()=>{message.hidden=true;ready=true;if(!active)send('suspend');else send('resume');};container.append(frame);report({keyframes:state.keyframes,regions:state.counts.regions,events:state.counts.unique_events});
  }).catch(error=>{failure=error.message;message.textContent=failure;report();onError(failure);});
 }
 return {activate(){active=true;document.body.classList.add('gaussian-change-tour-mode');if(container)container.hidden=false;report();if(ready)send('resume');if(!pending)pending=mount();return pending;},
  deactivate(){active=false;document.body.classList.remove('gaussian-change-tour-mode');if(container)container.hidden=true;send('suspend');},
  navigate(direction){send('navigate',Math.sign(direction));},clear(){send('clear');},resize(){}};
}

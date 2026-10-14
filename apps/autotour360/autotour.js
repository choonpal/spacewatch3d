export async function inspectVideoFile(file) {
  if(!/\.(mp4|m4v|mov|webm|ogv)$/i.test(file.name))throw new Error('MP4, MOV, WebM 등의 360도 영상 파일을 선택해 주세요.');
  if(!file.size||file.size>2*1024**3)throw new Error('영상은 최대 2GB까지 입력할 수 있습니다.');
  const video=document.createElement('video'),url=URL.createObjectURL(file);video.preload='metadata';video.muted=true;
  try {
    await new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(new Error('영상 정보를 읽는 데 시간이 오래 걸립니다. MP4/H.264 파일인지 확인해 주세요.')),20000);video.onloadedmetadata=()=>{clearTimeout(timer);resolve();};video.onerror=()=>{clearTimeout(timer);reject(new Error('브라우저에서 열 수 없는 영상입니다. MP4/H.264로 변환해 주세요.'));};video.src=url;});
    if(Math.abs(video.videoWidth/video.videoHeight-2)>.03)throw new Error('2:1 비율의 스티칭된 360도 영상이 필요합니다.');
    if(!Number.isFinite(video.duration)||video.duration<3||video.duration>600)throw new Error('3초 이상, 10분 이하의 영상을 선택해 주세요.');
    return {duration:video.duration,width:video.videoWidth,height:video.videoHeight};
  } finally {video.removeAttribute('src');video.load();URL.revokeObjectURL(url);}
}

export function uploadVideo(file,onProgress,signal) {
  return new Promise((resolve,reject)=>{
    const xhr=new XMLHttpRequest();
    const abort=()=>xhr.abort();
    signal?.addEventListener('abort',abort,{once:true});
    xhr.open('POST',`/api/videos?name=${encodeURIComponent(file.name)}`);xhr.setRequestHeader('Content-Type','application/octet-stream');
    xhr.upload.onprogress=e=>onProgress(e.lengthComputable?e.loaded/e.total:0);
    xhr.onload=()=>{try{const result=JSON.parse(xhr.responseText);if(xhr.status>=400)throw new Error(result.error||'영상을 가져오지 못했습니다.');resolve(result.media);}catch(error){reject(error);}};
    xhr.onerror=()=>reject(new Error('프로그램 서버에 연결할 수 없습니다. 실행 창을 확인해 주세요.'));
    xhr.onabort=()=>reject(new DOMException('영상 입력을 취소했습니다.','AbortError'));
    xhr.onloadend=()=>signal?.removeEventListener('abort',abort);
    if(signal?.aborted){reject(new DOMException('영상 입력을 취소했습니다.','AbortError'));return;}
    xhr.send(file);
  });
}

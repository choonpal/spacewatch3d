import {validateTrajectory,validateRouteEdits} from './trajectory-math.js';
import {validateTourKeyframes} from './tour-keyframes.js';
export const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
export const wrapYaw = value => ((value + 180) % 360 + 360) % 360 - 180;
export const radians = value => value * Math.PI / 180;
export const degrees = value => value * 180 / Math.PI;
export function uid() {
  if (globalThis.crypto.randomUUID) return globalThis.crypto.randomUUID();
  // Mobile devices may access the local server over HTTP on a private LAN.
  const bytes = globalThis.crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 15) | 64; bytes[8] = (bytes[8] & 63) | 128;
  const hex = [...bytes].map(x => x.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0,8)}-${hex.slice(8,12)}-${hex.slice(12,16)}-${hex.slice(16,20)}-${hex.slice(20)}`;
}
export const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
export function formatTime(value) {
  const seconds = Math.max(0, Math.floor(Number(value) || 0));
  return `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${(seconds % 60).toString().padStart(2, '0')}`;
}
export function cameraBasis({yaw, pitch}) {
  const y = radians(yaw), p = radians(pitch);
  return {
    forward: [Math.sin(y)*Math.cos(p), Math.sin(p), Math.cos(y)*Math.cos(p)],
    right: [Math.cos(y), 0, -Math.sin(y)],
    up: [-Math.sin(y)*Math.sin(p), Math.cos(p), -Math.cos(y)*Math.sin(p)],
  };
}
export function projectPoint(yaw, pitch, view, width, height) {
  const direction = cameraBasis({yaw, pitch}).forward;
  const basis = cameraBasis(view);
  const dot = vector => vector.reduce((sum, n, i) => sum + n * direction[i], 0);
  const z = dot(basis.forward);
  if (z <= 0.01) return null;
  const scale = Math.tan(radians(view.fov) / 2);
  const x = dot(basis.right) / (z * scale * width / height);
  const y = dot(basis.up) / (z * scale);
  if (Math.abs(x) > 1.15 || Math.abs(y) > 1.15) return null;
  return {x: (x+1)*width/2, y: (1-y)*height/2};
}
export function removeScene(project, id) {
  project.scenes = project.scenes.filter(scene => scene.id !== id);
  for (const scene of project.scenes) scene.hotspots = scene.hotspots.filter(h => h.type !== 'link' || h.targetId !== id);
  const used = new Set(project.scenes.map(s => s.assetId));
  project.assets = project.assets.filter(a => used.has(a.id));
}
export function emptyProject() { return {version: 1, id: uid(), title: '나의 자동 투어', assets: [], scenes: []}; }
export function newScene(assetId, title, view = {yaw: 0, pitch: 0, fov: 75}, startTime = 0) {
  return {id: uid(), assetId, title, description: '', startTime, view: {...view}, thumbnail: '', hotspots: []};
}
// Rebuild imported data from an allowlist. Imported JSON cannot introduce a URL or local path.
export function validateProject(input) {
  const fail = reason => { throw new Error(`투어 파일을 열 수 없습니다: ${reason}`); };
  if (!input || input.version !== 1 || !Array.isArray(input.assets) || !Array.isArray(input.scenes)) fail('지원하지 않는 파일 형식입니다.');
  if (input.scenes.length > 200 || input.assets.length > 200) fail('장면과 파일은 각각 최대 200개입니다.');
  const str = (value, length, fallback = '') => typeof value === 'string' ? value.slice(0,length) : fallback;
  const num = (value, fallback, min, max) => Number.isFinite(value) ? clamp(value,min,max) : fallback;
  const assetIds = new Set(), sceneIds = new Set(), hotspotIds = new Set();
  const assets = input.assets.map(a => {
    if (!a || typeof a.id !== 'string' || !a.id || a.id.length > 100 || assetIds.has(a.id)) fail('미디어 ID가 잘못되었습니다.');
    assetIds.add(a.id);
    if (!['video','image'].includes(a.type)) fail('미디어 종류가 잘못되었습니다.');
    const asset={id:a.id, name:str(a.name,255,'미디어'), type:a.type, size:num(a.size,0,0,1e14), lastModified:num(a.lastModified,0,0,1e14)};
    if(typeof a.trajectoryKey==='string'&&/^[a-f0-9]{64}$/.test(a.trajectoryKey))asset.trajectoryKey=a.trajectoryKey;
    if(a.trajectory){const trajectory=validateTrajectory(a.trajectory);if(asset.type!=='video'||trajectory.source.size!==asset.size||trajectory.source.name!==asset.name)fail('경로와 원본 영상 정보가 일치하지 않습니다.');asset.trajectory=trajectory;}
    if(a.routeEdits)asset.routeEdits=validateRouteEdits(a.routeEdits,asset.trajectory);
    if(a.tourKeyframes)asset.tourKeyframes=validateTourKeyframes(a.tourKeyframes,asset.trajectory);
    if(asset.trajectory&&a.trajectorySettings)asset.trajectorySettings={enabled:a.trajectorySettings.enabled!==false,showMap:a.trajectorySettings.showMap!==false,windowSeconds:[6,12,24,60].includes(a.trajectorySettings.windowSeconds)?a.trajectorySettings.windowSeconds:12,heightFactor:num(a.trajectorySettings.heightFactor,1,.5,1.5)};
    return asset;
  });
  const scenes = input.scenes.map(s => {
    if (!s || typeof s.id !== 'string' || !s.id || s.id.length > 100 || sceneIds.has(s.id)) fail('장면 ID가 잘못되었습니다.');
    sceneIds.add(s.id);
    if (!assetIds.has(s.assetId)) fail('장면에 연결된 미디어가 없습니다.');
    if (!Array.isArray(s.hotspots) || s.hotspots.length > 100) fail('장면당 지점은 최대 100개입니다.');
    const thumbnail = typeof s.thumbnail === 'string' && s.thumbnail.length < 100000 && /^data:image\/jpeg;base64,[A-Za-z0-9+/=]+$/.test(s.thumbnail) ? s.thumbnail : '';
    return {id:s.id,assetId:s.assetId,title:str(s.title,80,'새 장면'),description:str(s.description,2000),startTime:num(s.startTime,0,0,864000),
      view:{yaw:wrapYaw(num(s.view?.yaw,0,-1e6,1e6)),pitch:num(s.view?.pitch,0,-85,85),fov:num(s.view?.fov,75,35,100)},thumbnail,
      hotspots:s.hotspots.map(h => {
        if (!h || !['link','info'].includes(h.type) || typeof h.id !== 'string' || !h.id || h.id.length > 100 || hotspotIds.has(h.id)) fail('지점 정보가 잘못되었습니다.');
        hotspotIds.add(h.id);
        return {id:h.id,type:h.type,title:str(h.title,80,'지점'),body:str(h.body,3000),targetId:str(h.targetId,100),yaw:wrapYaw(num(h.yaw,0,-1e6,1e6)),pitch:num(h.pitch,0,-85,85)};
      })};
  });
  for (const s of scenes) for (const h of s.hotspots) if (h.type === 'link' && !sceneIds.has(h.targetId)) fail('이동할 장면을 찾을 수 없습니다.');
  return {version:1,id:str(input.id,100,uid()),title:str(input.title,100,'나의 360° 투어'),assets,scenes};
}

// Compact preview buffers retain the original geometry and color values.
export function decodeGeometry(buffer) {
  if (buffer.byteLength < 16) throw new Error('잘못된 3D 파일입니다.');
  const h = new DataView(buffer), count = h.getUint32(4,true), kind = h.getUint32(8,true);
  if (h.getUint32(0,true) !== 0x53575031 || ![1,2].includes(kind) || !count ||
      buffer.byteLength !== 16 + count*(kind === 1 ? 30 : 15) || (kind === 1 && count%3))
    throw new Error('3D 파일의 형식 또는 크기가 맞지 않습니다.');
  return {count, kind, positions:new Float32Array(buffer,16,count*3),
    normals:kind===1?new Float32Array(buffer,16+count*12,count*3):null,
    rgb:new Uint8Array(buffer,16+count*(kind===1?24:12),count*3),
    semantic:kind===1?new Uint8Array(buffer,16+count*27,count*3):null};
}
export function orderedViews(capture) {
  const byId=new Map(capture.views.map(v=>[v.id,v]));
  const order=[...new Set([...capture.tour_order,...capture.views.map(v=>v.id)])];
  return order.map(id=>byId.get(id)).filter(Boolean);
}
export function routePolylines(capture) {
  // Only Task 5b's explicit safe paths; never bridge disconnected components.
  return capture.links.filter(l=>l.connection_method==='shortest_safe_path' && l.polyline_m?.length>1).map(l=>l.polyline_m);
}
export function mapProjection(views, width=640, height=200) {
  const xs=views.map(v=>v.position[0]), ys=views.map(v=>v.position[1]);
  const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
  const scale=Math.min((width-40)/Math.max(xmax-xmin,.1),(height-40)/Math.max(ymax-ymin,.1));
  return p=>[width/2+(p[0]-(xmin+xmax)/2)*scale,height/2-(p[1]-(ymin+ymax)/2)*scale];
}

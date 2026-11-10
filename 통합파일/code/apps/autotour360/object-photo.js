// Object-only photo reprojection. Pixels without a visible source surface stay transparent.
export function decodeObjectMesh(bytes,meta,origin,rotation){
  if(meta.stride!==32||bytes.byteLength!==meta.count*32||meta.count%3)throw new Error('객체 메쉬 형식을 읽을 수 없습니다.');
  const view=new DataView(bytes),positions=new Float64Array(meta.count*3),regionIds=new Uint8Array(meta.count/3);
  for(let t=0;t<regionIds.length;t++){
    const ids=[0,1,2].map(k=>view.getUint8((t*3+k)*32+27)%64);
    regionIds[t]=ids.every(id=>id===ids[0])?ids[0]:255;
    for(let k=0;k<3;k++){
      const p=[0,1,2].map(a=>view.getFloat32((t*3+k)*32+a*4,true)+origin[a]);
      for(let a=0;a<3;a++)positions[t*9+k*3+a]=rotation.reduce((sum,row,b)=>sum+row[a]*p[b],0);
    }
  }
  return {positions,regionIds};
}

function cameraPoint(x,y,z,frame){
  const r=frame.camera_to_world,dx=x-r[0][3],dy=y-r[1][3],dz=z-r[2][3];
  return [r[0][0]*dx+r[1][0]*dy+r[2][0]*dz,r[0][1]*dx+r[1][1]*dy+r[2][1]*dz,r[0][2]*dx+r[1][2]*dy+r[2][2]*dz,x,y,z];
}
function clipNear(vertices){
  const result=[];
  for(let i=0;i<vertices.length;i++){
    const a=vertices[i],b=vertices[(i+1)%vertices.length],insideA=a[2]>=.1,insideB=b[2]>=.1;
    if(insideA)result.push(a);
    if(insideA!==insideB){const t=(.1-a[2])/(b[2]-a[2]);result.push(a.map((v,k)=>v+t*(b[k]-v)));}
  }
  return result;
}
function rasterize(mesh,frame,regionId,worldCoordinates,visit){
  const {width,height,focal_px:f}=frame,cx=(width-1)/2,cy=(height-1)/2;
  for(let t=0;t<mesh.regionIds.length;t++){
    if(regionId!==null&&mesh.regionIds[t]!==regionId)continue;
    const vertices=[];
    for(let k=0;k<3;k++){const n=t*9+k*3;vertices.push(cameraPoint(mesh.positions[n],mesh.positions[n+1],mesh.positions[n+2],frame));}
    const polygon=clipNear(vertices);
    for(let i=1;i+1<polygon.length;i++){
      const v=[polygon[0],polygon[i],polygon[i+1]],p=v.map(a=>[f*a[0]/a[2]+cx,f*a[1]/a[2]+cy]);
      const area=(p[1][0]-p[0][0])*(p[2][1]-p[0][1])-(p[1][1]-p[0][1])*(p[2][0]-p[0][0]);
      if(Math.abs(area)<1e-8)continue;
      const x0=Math.max(0,Math.ceil(Math.min(...p.map(a=>a[0]))-.5)),x1=Math.min(width-1,Math.floor(Math.max(...p.map(a=>a[0]))-.5));
      const y0=Math.max(0,Math.ceil(Math.min(...p.map(a=>a[1]))-.5)),y1=Math.min(height-1,Math.floor(Math.max(...p.map(a=>a[1]))-.5));
      for(let y=y0;y<=y1;y++)for(let x=x0;x<=x1;x++){
        const px=x+.5,py=y+.5;
        const a=((p[1][0]-px)*(p[2][1]-py)-(p[1][1]-py)*(p[2][0]-px))/area;
        const b=((p[2][0]-px)*(p[0][1]-py)-(p[2][1]-py)*(p[0][0]-px))/area,c=1-a-b;
        if(a< -1e-7||b< -1e-7||c< -1e-7)continue;
        const qa=a/v[0][2],qb=b/v[1][2],qc=c/v[2][2],z=1/(qa+qb+qc);
        if(worldCoordinates)visit(y*width+x,z,mesh.regionIds[t],z*(qa*v[0][3]+qb*v[1][3]+qc*v[2][3]),z*(qa*v[0][4]+qb*v[1][4]+qc*v[2][4]),z*(qa*v[0][5]+qb*v[1][5]+qc*v[2][5]));
        else visit(y*width+x,z,mesh.regionIds[t]);
      }
    }
  }
}

export function objectPhotoVisibility(mesh,frame){
  const depth=new Float32Array(frame.width*frame.height).fill(Infinity),ids=new Uint8Array(depth.length).fill(255);
  // Include other objects and unlabelled surfaces in the source-camera depth test.
  rasterize(mesh,frame,null,false,(i,z,id)=>{if(z<depth[i]){depth[i]=z;ids[i]=id;}});
  return {depth,ids};
}

export function reprojectObjectPhoto(mesh,regionId,target,sources,{inset=1}={}){
  const rgba=new Uint8ClampedArray(target.width*target.height*4),depth=new Float32Array(target.width*target.height).fill(Infinity);
  const prepared=sources.map(source=>({...source,visibility:source.visibility||objectPhotoVisibility(mesh,source.frame)}));
  rasterize(mesh,target,regionId,true,(i,z,id,x,y,wz)=>{
    if(z>=depth[i])return;
    depth[i]=z;const out=i*4;rgba[out+3]=0;
    for(const source of prepared){
      const {frame,pixels,visibility}=source,p=cameraPoint(x,y,wz,frame);
      if(p[2]<.1)continue;
      const u=frame.focal_px*p[0]/p[2]+(frame.width-1)/2,v=frame.focal_px*p[1]/p[2]+(frame.height-1)/2;
      const sx=Math.floor(u),sy=Math.floor(v),n=sy*frame.width+sx;
      if(sx<inset||sy<inset||sx>=frame.width-inset||sy>=frame.height-inset||visibility.ids[n]!==regionId)continue;
      if(inset&&(visibility.ids[n-1]!==regionId||visibility.ids[n+1]!==regionId||visibility.ids[n-frame.width]!==regionId||visibility.ids[n+frame.width]!==regionId))continue;
      // Pixel footprint tolerance for the same reconstructed surface, not a change threshold.
      if(Math.abs(visibility.depth[n]-p[2])>Math.max(.012,p[2]*.004))continue;
      const color=n*4;
      if(!pixels[color+3])continue;
      rgba[out]=pixels[color];rgba[out+1]=pixels[color+1];rgba[out+2]=pixels[color+2];rgba[out+3]=255;
      break;
    }
  });
  return rgba;
}

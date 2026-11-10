import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
// A data module keeps this test independent of the host's package.json mode.
const source=await readFile(new URL('../apps/autotour360/pipeline-change-model.js',import.meta.url),'utf8');
const {patchColors,boxSegments,validateComparison}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const data={count:6,rgb:new Uint8Array(18).fill(100)},event={id:'a',epoch:0,status:'deferred',face_ids:[1]};
assert.deepEqual([...patchColors(data,[event],0,event,'selected').slice(0,9)],Array(9).fill(62));
assert.deepEqual([...patchColors(data,[event],0,event,'selected').slice(9)],Array(3).fill([255,224,160]).flat());
assert.deepEqual([...patchColors(data,[event],1,event,'selected')],Array(18).fill(62));
assert.equal(patchColors(data,[event],0,event,'rgb'),data.rgb);
assert.throws(()=>patchColors(data,[{...event,face_ids:[2]}],0,event,'all'));
const mesh={center:[0,0,0],radius:1,basis:()=>({right:[1,0,0],up:[0,1,0]})};
const lines=boxSegments([[-1,-1,-1],[1,1,1]],mesh,200,100);
assert.equal(lines.length,12);assert.deepEqual(lines[0],[[50,100],[150,100]]);
assert.throws(()=>validateComparison({schema_version:1,scope:'surface_regions_not_object_events',no_object_identity_claim:true,available:false,events:[event],maps:[]}));
console.log('PASS: exact selected faces, no invented counterpart, RGB passthrough, invalid face IDs, shared box projection, unavailable state.');

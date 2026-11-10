"""Verify discovered words with Grounding DINO, and obtain independent SAM2 masks."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from .common import read,save,sha,digest,STRUCTURE,DYNAMIC,catalog_for,canonicalize


def run(config,output,prepared_override=None,names_override=None):
    import torch
    from PIL import Image,ImageDraw
    from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
    from torchvision.ops import nms
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    assert torch.cuda.device_count()==1
    torch.set_num_threads(2);torch.manual_seed(5201)
    output=Path(output); prepared=prepared_override or read(output/'prepared.json');candidates={}; evidence=defaultdict(list)
    normalized={}
    if names_override is None:
        phrases=[c['phrase'] for s in prepared['scenes'] for r in s['views'] for c in read(output/'discovery'/f'{r["key"]}.json')['candidates']]
        from .normalize import normalize_phrases
        normalized=normalize_phrases(phrases,config)
        save(output/'normalized_phrases.json',normalized)
    for scene in prepared['scenes']:
        if names_override is not None:
            candidates[scene['id']]=names_override
            continue
        count=defaultdict(set)
        for row in scene['views']:
            for c in read(output/'discovery'/f'{row["key"]}.json')['candidates']:
                name=normalized[c['phrase']]
                if not name:continue
                count[name].add(row['frame'])
                evidence[name].append(dict(scene=scene['id'],frame=row['frame'],view=row['view']['name'],phrase=c['phrase']))
        # Preserve rare candidates for verification; reject conversational region names.
        ignored={'image','photo','picture','background','foreground','room','hallway','building','corridor','corner','area','side','front','top','bottom',
                 'surface','tile','furniture','object','edge','line','shadow','reflection','stripe','pattern','text','letter',
                 'bathroom','cement','concrete','marble','writing','sunlight','number','piece'}
        candidates[scene['id']]=sorted(set(count)-ignored)
    if names_override is None:save(output/'candidate_classes.json',dict(scenes=candidates,evidence=dict(evidence)))
    dino=config['models']['grounding_dino'];sam=config['models']['sam2']
    processor=AutoProcessor.from_pretrained(dino['path'],local_files_only=True)
    model=AutoModelForZeroShotObjectDetection.from_pretrained(dino['path'],local_files_only=True,
                                                            disable_custom_kernels=True).eval().cuda()
    predictor=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml',sam['path'],device='cuda',
                                          apply_postprocessing=False),max_hole_area=0,max_sprinkle_area=0)
    verified=defaultdict(list);allrows=[]
    for scene in prepared['scenes']:
        names=candidates[scene['id']]
        # Batching by tokenizer length avoids silently truncated class prompts.
        batches=[];batch=[]
        for name in names:
            if len(processor.tokenizer('. '.join(batch+[name])+'.')['input_ids'])>180:
                batches.append(batch);batch=[]
            batch.append(name)
        if batch:batches.append(batch)
        for row in scene['views']:
            dest=output/'grounded'/row['key']; target=dest.with_suffix('.json')
            identity=digest([row['sha256'],names,config['models']['grounding_dino'],config['models']['sam2'],config.get('box_threshold',.28)])
            if target.exists() and dest.with_suffix('.npz').exists() and read(target).get('identity')==identity and read(target).get('mask_sha256')==sha(dest.with_suffix('.npz')):
                record=read(target)
            else:
                im=Image.open(row['path']).convert('RGB');size=im.size[0]; detections=[]
                for batch in batches:
                    caption='. '.join(batch)+'.';offsets=processor.tokenizer(caption,return_offsets_mapping=True)['offset_mapping']
                    spans=[];pos=0
                    for name in batch:
                        start=caption.index(name,pos);end=start+len(name);pos=end
                        spans.append([i for i,(s,e) in enumerate(offsets) if e>s and s>=start and e<=end])
                    inputs=processor(images=im,text=caption,return_tensors='pt').to('cuda')
                    with torch.inference_mode():result=model(**inputs)
                    logits=result.logits[0].sigmoid()
                    scores=torch.stack([logits[:,ids].mean(1) for ids in spans],1)
                    score,classes=scores.max(1)
                    cx,cy,w,h=result.pred_boxes[0].unbind(-1)
                    boxes=torch.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],-1).clamp(0,1)*size
                    keep=(score>=config.get('box_threshold',.28))&((boxes[:,2]-boxes[:,0])>=10)&((boxes[:,3]-boxes[:,1])>=10)
                    for ci,name in enumerate(batch):
                        ids=torch.nonzero(keep&(classes==ci),as_tuple=True)[0]
                        for q in ids[nms(boxes[ids],score[ids],.5)].tolist():
                            detections.append(dict(name=name,box=boxes[q].tolist(),dino_score=float(score[q])))
                # One winning label per near-identical region, including across prompt batches.
                if detections:
                    ids=nms(torch.tensor([d['box'] for d in detections]),torch.tensor([d['dino_score'] for d in detections]),.8).tolist()
                    detections=[detections[i] for i in ids[:config.get('max_detections_per_view',40)]]
                masks=[];kept=[]
                with torch.inference_mode():
                    predictor.set_image(np.array(im))
                    for detection in detections:
                        mm,quality,low=predictor.predict(box=np.array(detection['box']),multimask_output=True)
                        best=int(quality.argmax());mask=mm[best].astype(bool)
                        if quality[best]<.65 or mask.sum()<64:continue
                        stability=float((low[best]>.5).sum()/max(1,(low[best]>-.5).sum()))
                        detection.update(sam_score=float(quality[best]),stability=stability,pixels=int(mask.sum()),
                                         border_touch=bool(mask[:2].any() or mask[-2:].any() or mask[:,:2].any() or mask[:,-2:].any()))
                        masks.append(mask);kept.append(detection)
                arr=np.asarray(masks,dtype=bool).reshape(-1,size,size)
                dest.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest.with_suffix('.npz'),masks_packed=np.packbits(arr.reshape(len(arr),size*size),axis=1),shape=np.array(arr.shape))
                record=dict(identity=identity,view=row,detections=kept,mask_sha256=sha(dest.with_suffix('.npz')))
                save(target,record)
                draw=ImageDraw.Draw(im)
                for d in kept:draw.rectangle(d['box'],outline='lime',width=2);draw.text(d['box'][:2],d['name'],fill='red')
                im.save(dest.with_suffix('.jpg'))
            # Infer moving body/held-object labels from actual mask overlap with person.
            with np.load(dest.with_suffix('.npz')) as z:
                shape=tuple(z['shape']);all_masks=np.unpackbits(z['masks_packed'],axis=1,count=shape[1]*shape[2]).reshape(shape).astype(bool)
            person=np.zeros(shape[1:],bool)
            for d,m in zip(record['detections'],all_masks):
                if d['name']=='person':person|=m
            for detection,mask in zip(record['detections'],all_masks):
                detection['person_overlap']=float((person&mask).sum()/max(1,mask.sum()))
                import cv2
                band=cv2.dilate(mask.astype('u1'),np.ones((5,5),np.uint8)).astype(bool)&~mask
                detection['person_boundary_contact']=float((band&person).sum()/max(1,band.sum()))
            save(target,record)
            for i,detection in enumerate(record['detections']):
                verified[detection['name']].append(dict(scene=scene['id'],frame=row['frame'],view=row['view']['name'],
                                                       key=row['key'],index=i,score=detection['dino_score'],person_overlap=detection['person_overlap']))
            allrows.append(record)
            print('GROUND',row['key'],len(record['detections']),flush=True)
    if names_override is not None:
        return dict(views=len(allrows),detections=sum(len(r['detections']) for r in allrows))
    accepted={};rejected={}
    for name,obs in verified.items():
        independent=set((o['scene'],o['frame']) for o in obs)
        if len(independent)>=config.get('minimum_class_frames',2):accepted[name]=obs
        else:rejected[name]=dict(reason='fewer_than_two_distinct_frames',observations=obs)
    # Structural fallback is declared separately, never counted as discovered evidence.
    for name in ['wall','floor','ceiling']:
        if name not in accepted: accepted[name]=[dict(reason='reserved_structure_fallback')]
    catalog=catalog_for(accepted,accepted)
    for cls in catalog:
        obs=verified.get(cls['name'],[])
        fraction=sum(o.get('person_overlap',0)>=.8 for o in obs)/max(1,len(obs))
        cls['person_contained_observation_fraction']=fraction
        if fraction>=.6:
            cls.update(static_mesh=False,instance_eligible=False,exclusion_reason='usually_inside_person_mask')
    save(output/'classes.json',catalog)
    save(output/'grounding_summary.json',dict(classes=len(catalog),accepted=accepted,rejected=rejected,
                                             candidate_classes=len(evidence),detections=sum(len(r['detections']) for r in allrows),
                                             peak_vram_gib=torch.cuda.max_memory_allocated()/2**30))


if __name__=='__main__':
    import sys
    run(read(sys.argv[1]),sys.argv[2])

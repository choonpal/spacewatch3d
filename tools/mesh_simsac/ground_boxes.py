"""Generate box masks from RGB using locally configured Grounding DINO and SAM2.

Runs in the segmentation model's environment. No change decisions or hand-picked
rectangles are inputs. Output masks retain source image hashes for mesh transfer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class BoxDetector:
    def __init__(self, config):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from sam2.build_sam import build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.torch = torch
        torch.set_num_threads(4)
        self.processor = AutoProcessor.from_pretrained(config['dino'], local_files_only=True)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            config['dino'], local_files_only=True, disable_custom_kernels=True).eval().cuda()
        sam = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_s.yaml', config['sam2'],
            device='cuda', apply_postprocessing=False, hydra_overrides_extra=['++model.fill_hole_area=0'])
        self.sam = SAM2ImagePredictor(sam, max_hole_area=0, max_sprinkle_area=0)
        labels = ['box','chair','table','door','wall','floor','ceiling','water dispenser',
                  'vacuum cleaner','cabinet','person','bag','electrical panel']
        self.caption = 'box. cardboard box. storage box.'
        self.verify_caption = '. '.join(labels)+'.'
        offsets = self.processor.tokenizer(self.verify_caption,return_offsets_mapping=True)['offset_mapping']
        self.tokens=[]
        position=0
        for name in labels:
            start=self.verify_caption.index(name,position);end=start+len(name);position=end
            self.tokens.append([i for i,(a,b) in enumerate(offsets) if b>a and a>=start and b<=end])
        if not all(self.tokens):
            raise ValueError('Segmentation label tokens are missing')

    def verify_box(self, image, bbox):
        """Recheck automatically proposed small ERP objects at useful scale."""
        torch=self.torch
        h,w=image.shape[:2]
        x0,y0,x1,y1=bbox
        cx,cy=(x0+x1)/2,(y0+y1)/2
        radius=max(32,(x1-x0)*1.5,(y1-y0)*1.5)
        left,top=max(0,int(cx-radius)),max(0,int(cy-radius))
        right,bottom=min(w,int(cx+radius)),min(h,int(cy+radius))
        crop=Image.fromarray(image[top:bottom,left:right])
        result=self.detector(**self.processor(images=crop,text=self.verify_caption,return_tensors='pt').to('cuda'))
        logits=result.logits[0].sigmoid()
        scores=torch.stack([logits[:,tokens].mean(1) for tokens in self.tokens],1)
        cx,cy,bw,bh=result.pred_boxes[0].unbind(-1)
        boxes=torch.stack([cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2],-1)
        boxes*=boxes.new_tensor([right-left,bottom-top,right-left,bottom-top])
        target=boxes.new_tensor([x0-left,y0-top,x1-left,y1-top])
        intersection=(torch.minimum(boxes[:,2:],target[2:])-torch.maximum(boxes[:,:2],target[:2])).clamp_min(0).prod(1)
        union=(boxes[:,2:]-boxes[:,:2]).clamp_min(0).prod(1)+(target[2:]-target[:2]).prod()-intersection
        iou=intersection/union.clamp_min(1e-6)
        valid=(iou>=.25)&(scores.argmax(1)==0)&(scores[:,0]>=.25)
        return bool(valid.any())

    def predict(self, image):
        from torchvision.ops import nms
        torch = self.torch
        with torch.inference_mode():
            inputs = self.processor(images=Image.fromarray(image), text=self.caption, return_tensors='pt').to('cuda')
            result = self.detector(**inputs)
            score = result.logits[0].sigmoid().max(-1).values
            cx, cy, w, h = result.pred_boxes[0].unbind(-1)
            height, width = image.shape[:2]
            boxes = torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], -1)
            boxes *= boxes.new_tensor([width, height, width, height])
            good = torch.nonzero((score >= .30) & (w > .01) & (h > .01) & (w < .65) & (h < .65), as_tuple=True)[0]
            picked = good[nms(boxes[good], score[good], .5)][:16] if len(good) else []
            detections = []
            if len(picked):
                self.sam.set_image(image)
                for q in picked:
                    if not self.verify_box(image,boxes[q].cpu().tolist()):
                        continue
                    masks, quality, _ = self.sam.predict(box=boxes[q].cpu().numpy(), multimask_output=True)
                    best = int(quality.argmax())
                    if masks[best].sum() < 32 or quality[best] < .5:
                        continue
                    detections.append(dict(mask=masks[best].astype(bool), score=float(score[q]),
                        sam_score=float(quality[best]), bbox=boxes[q].cpu().tolist()))
            return detections


def main(config_path):
    config = json.loads(Path(config_path).read_text())
    weights = list(Path(config['dino']).glob('*.safetensors')) + list(Path(config['dino']).glob('pytorch_model*.bin'))
    if not weights:
        raise ValueError('Grounding DINO weights are missing')
    model_hashes = {str(p):sha(p) for p in weights+[Path(config['sam2'])]}
    model_key = hashlib.sha256(json.dumps(dict(config={k:v for k,v in config.items() if k != 'tasks'},
        weights=model_hashes),sort_keys=True).encode()+Path(__file__).read_bytes()).hexdigest()
    detector = BoxDetector(config)
    for task in config['tasks']:
        destination = Path(task['output'])
        receipt_path = destination.with_suffix('.json')
        fingerprint = sha(task['image'])
        if destination.is_file() and receipt_path.is_file():
            previous = json.loads(receipt_path.read_text())
            if previous.get('image_sha256') == fingerprint and previous.get('model_key') == model_key and previous.get('mask_sha256') == sha(destination):
                print('SEGMENTATION_CACHE', task['image'], flush=True)
                continue
        image = np.asarray(Image.open(task['image']).convert('RGB'))
        started = time.monotonic()
        detections = detector.predict(image)
        destination.parent.mkdir(parents=True, exist_ok=True)
        masks = np.stack([d['mask'] for d in detections]) if detections else np.zeros((0, *image.shape[:2]), bool)
        np.savez_compressed(destination, masks=masks, scores=np.array([d['score'] for d in detections]),
            sam_scores=np.array([d['sam_score'] for d in detections]))
        receipt = dict(image=task['image'], image_sha256=fingerprint, model_key=model_key, model_hashes=model_hashes, mask_sha256=sha(destination),
            detections=[{k:v for k,v in d.items() if k != 'mask'} for d in detections],
            seconds=time.monotonic()-started, model='Grounding DINO base + SAM 2.1 small',
            caption=detector.caption, thresholds=dict(dino=.30, sam=.50), manual_regions_used=False)
        receipt.update(verification_caption=detector.verify_caption,verification_box_threshold=.25,
                       verification_iou_threshold=.25,verification_crop='Three proposal widths/heights, minimum 64 pixels; automatic proposals only')
        destination.with_suffix('.json').write_text(json.dumps(receipt))
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    main(parser.parse_args().config)

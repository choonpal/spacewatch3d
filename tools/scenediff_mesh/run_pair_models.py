"""Real pretrained ViewDelta / SimSaC on shared views and complete ERP frames."""
from __future__ import annotations
import argparse,ast,gc,hashlib,json,os,sys,time,types
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
import torch
import torch.nn.functional as F

ROOT=Path('/workspace/pano_change_models_20261025')
def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()

def local_correlation(tensorFirst,tensorSecond):
    """Same zero-padded 9x9 channel-mean correlation as the upstream CUDA op."""
    _,_,h,w=tensorFirst.shape
    padded=F.pad(tensorSecond,(4,4,4,4))
    result=[]
    for dy in range(9):
        for dx in range(9):
            result.append((tensorFirst*padded[:,:,dy:dy+h,dx:dx+w]).mean(1))
    return torch.stack(result,dim=1)

def project_probability(prob,yaw,size=224):
    y,x=np.mgrid[:size,:size];f=(size-1)/2
    directions=np.stack([(x-f)/f,(y-f)/f,np.ones_like(x)],-1)
    angle=np.deg2rad(yaw);pitch=np.deg2rad(10)
    s,c,sp,cp=np.sin(angle),np.cos(angle),np.sin(pitch),np.cos(pitch)
    rotation=np.array([[c,-s*sp,s*cp],[0,cp,sp],[-s,-c*sp,c*cp]])
    d=directions@rotation.T;d/=np.linalg.norm(d,axis=-1,keepdims=True)
    h,w=prob.shape
    u=(np.arctan2(d[...,0],d[...,2])/(2*np.pi)+.5)*w-.5
    v=(np.arcsin(np.clip(d[...,1],-1,1))/np.pi+.5)*h-.5
    return map_coordinates(np.pad(prob,((0,0),(1,1)),mode='wrap'),[np.clip(v,0,h-1),np.mod(u,w)+1],order=1,mode='nearest')

class ViewDelta:
    def __init__(self,setup):
        sys.path.insert(0,str(ROOT/'viewdelta'))
        from ViewDelta.model.transformer_args import TransformerModelArgs
        from ViewDelta.model.patch_embedder import PatchEmbedder
        from einops import rearrange
        from transformers import AutoModel,AutoImageProcessor,AutoProcessor
        source=ROOT/'viewdelta/ViewDelta/model/model_feature_segmentor.py'
        tree=ast.parse(source.read_text())
        tree.body=[n for n in tree.body if not(isinstance(n,ast.Import) and any(a.name=='deepspeed' for a in n.names))]
        namespace=dict(__name__='viewdelta_inference_only',TransformerModelArgs=TransformerModelArgs,PatchEmbedder=PatchEmbedder)
        exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),namespace)
        args=TransformerModelArgs(text_embeddings='siglip',image_embeddings='dinov2',use_multiscale=False,
            use_separation_tokens=False,depth=12,dim=768,mlp_dim=3072,heads=12,
            image_embedding_dim=1024,text_embedding_dim=1024,checkpoint_ff=False,checkpoint_attn=False)
        self.model=namespace['TextConditionedDecoder'](args)
        ckpt=Path(setup['checkpoints']['viewdelta'])/'viewdelta_checkpoint.pth'
        state=torch.load(ckpt,map_location='cpu',weights_only=True)
        self.model.load_state_dict(state,strict=True)
        self.model=self.model.cuda().eval()
        self.image_model=AutoModel.from_pretrained(setup['checkpoints']['dinov2'],local_files_only=True).cuda().eval()
        self.image_processor=AutoImageProcessor.from_pretrained(setup['checkpoints']['dinov2'],local_files_only=True)
        full_text_model=AutoModel.from_pretrained(setup['checkpoints']['siglip'],local_files_only=True)
        self.text_model=full_text_model.text_model.cuda().eval()
        self.text_processor=AutoProcessor.from_pretrained(setup['checkpoints']['siglip'],local_files_only=True)
        del full_text_model,state;gc.collect()
        self.prompt='What are the differences?'
        tokens=self.text_processor(text=self.prompt,padding='max_length',return_tensors='pt')
        with torch.inference_mode():self.text=self.text_model(**{k:v.cuda() for k,v in tokens.items()})['last_hidden_state']
        self.cache={}
        self.audit=dict(model='ViewDelta',source_commit=setup['sources']['viewdelta'],checkpoint=str(ckpt),checkpoint_sha256=sha(ckpt),
            loaded_strict=True,prompt=self.prompt,learned_parameters=sum(p.numel() for p in self.model.parameters()),
            input_preprocessing='Upstream PIL square resize 256, DINOv2 processor (224 crop/257 tokens), SigLIP text, FP32 eval',
            compatibility='Unused DeepSpeed import omitted; activation checkpointing remains disabled as in official inference. No model layers changed.',
            feature_models={k:setup['checkpoints'][k] for k in ['dinov2','siglip']})
    def feature(self,rgb):
        key=hashlib.sha256(rgb.tobytes()).hexdigest()
        if key not in self.cache:
            image=Image.fromarray(rgb).resize((256,256))
            inputs=self.image_processor(images=[image],return_tensors='pt')
            with torch.inference_mode():self.cache[key]=self.image_model(**{k:v.cuda() for k,v in inputs.items()}).last_hidden_state.cpu()
        return self.cache[key].cuda()
    def predict(self,a,b):
        with torch.inference_mode():
            logits=self.model(self.feature(a),self.feature(b),self.text)
            logits=F.interpolate(logits,size=a.shape[:2],mode='bilinear',align_corners=False)
            prob=logits.softmax(1)[0,1].float().cpu().numpy()
        return prob

class SimSaC:
    def __init__(self,setup,checkpoint='simsac_changesim'):
        sys.path.insert(0,str(ROOT/'simsac'))
        module=types.ModuleType('models.correlation.correlation')
        module.FunctionCorrelation=local_correlation
        sys.modules[module.__name__]=module
        from models.our_models.SimSaC import SimSaC_Model
        self.model=SimSaC_Model(evaluation=False,div=1.,batch_norm=True,pyramid_type='VGG',
            consensus_network=False,cyclic_consistency=True,dense_connection=True,
            decoder_inputs='corr_flow_feat',refinement_at_all_levels=False,refinement_at_adaptive_reso=True,
            num_class=2,use_pac=False,vpr_candidates=False)
        ckpt=Path(setup['checkpoints'][checkpoint])
        loaded=torch.load(ckpt,map_location='cpu',weights_only=False)
        state=loaded.get('state_dict',loaded.get('model',loaded))
        if all(k.startswith('module.') for k in state):state={k[7:]:v for k,v in state.items()}
        self.model.load_state_dict(state,strict=True)
        self.model.cuda().eval()
        self.mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1)
        self.std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)
        self.audit=dict(model='SimSaC',source_commit=setup['sources']['simsac'],checkpoint=str(ckpt),checkpoint_sha256=sha(ckpt),
            loaded_strict=True,checkpoint_selection='Official synthetic+ChangeSim checkpoint selected before inference because target imagery is indoors.',
            input_preprocessing='RGB float/255, ImageNet z-score, full branch 520x520 and low branch 256x256; no test augmentation',
            compatibility='Native PyTorch equivalent of upstream zero-padded 9x9 channel-mean correlation. Original train.py architecture, use_pac=False; no head replaced.')
    def prep(self,rgb):
        x=torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1)[None].float().cuda()/255
        full=F.interpolate(x,size=(520,520),mode='area');small=F.interpolate(x,size=(256,256),mode='area')
        return (full-self.mean)/self.std,(small-self.mean)/self.std
    def predict(self,a,b):
        full_a,small_a=self.prep(a);full_b,small_b=self.prep(b)
        with torch.inference_mode():
            result=self.model(full_a,full_b,small_a,small_b)
            logits=result['change'][1][-1]
            logits=F.interpolate(logits,size=a.shape[:2],mode='bilinear',align_corners=False)
            prob=logits.softmax(1)[0,1].float().cpu().numpy()
        return prob

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['viewdelta','simsac'],required=True)
    args=parser.parse_args();torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
    setup=read(ROOT/'setup-status.json');assert setup['stage']=='complete'
    data=np.load(ROOT/'inputs/rgb.npz');manifest=read(ROOT/'inputs/manifest.json');lookup={v['key']:v for v in manifest['views']}
    output=ROOT/'results'/args.model;output.mkdir(parents=True,exist_ok=True);(output/'pairs').mkdir(exist_ok=True)
    start=time.time();write(output/'status.json',dict(state='loading'))
    model=ViewDelta(setup) if args.model=='viewdelta' else SimSaC(setup)
    write(output/'model.json',model.audit)
    prob_maps={};erp_maps={};pair_audit=[];cache={}
    for mode in ['view','erp']:
        write(output/'status.json',dict(state='running',mode=mode))
        for number,view in enumerate(manifest['views']):
            a=view['key'] if mode=='view' else view['erp']
            refs=list(dict.fromkeys(ref if mode=='view' else lookup[ref]['erp'] for ref in view['refs']))
            predictions=[]
            for b in refs:
                key=mode+'__'+a+'__'+b
                filename=output/'pairs'/(key+'.npz')
                if key not in cache:
                    if filename.exists():prob=np.load(filename)['prob'].astype('f4')
                    else:
                        print(args.model,mode,a,b,flush=True)
                        prob=model.predict(data[a],data[b])
                        assert np.isfinite(prob).all()
                        np.savez_compressed(filename,prob=prob.astype('f2'))
                    cache[key]=prob
                    pair_audit.append(dict(key=key,a=a,b=b,mode=mode,source_hash=hashlib.sha256(data[a].tobytes()).hexdigest(),target_hash=hashlib.sha256(data[b].tobytes()).hexdigest(),native_shape=list(prob.shape)))
                predictions.append(cache[key])
            avg=np.mean(predictions,axis=0)
            if mode=='erp':
                erp_maps[view['key']]=avg.astype('f2')
                avg=project_probability(avg,view['yaw'])
            prob_maps[mode+'__'+view['key']]=avg.astype('f4')
            write(output/'status.json',dict(state='running',mode=mode,view=number+1,total=30,pairs=len(cache)))
    control={}
    first=manifest['views'][0]
    for mode,key in [('view',first['key']),('erp',first['erp'])]:
        prob=model.predict(data[key],data[key])
        control[mode]=dict(candidate_fraction=float((prob>.5).mean()),mean_probability=float(prob.mean()),max_probability=float(prob.max()))
        prob_maps['self__'+mode]=prob
    np.savez_compressed(output/'common_view_probabilities.npz',**prob_maps)
    np.savez_compressed(output/'erp_probabilities.npz',**erp_maps)
    write(output/'pairs.json',pair_audit);write(output/'self-control.json',control)
    write(output/'status.json',dict(state='complete',elapsed_seconds=time.time()-start,pair_predictions=len(cache),control=control))
    print(args.model,'COMPLETE',read(output/'status.json'),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        name=sys.argv[sys.argv.index('--model')+1] if '--model' in sys.argv else 'unknown'
        out=ROOT/'results'/name;out.mkdir(parents=True,exist_ok=True)
        write(out/'status.json',dict(state='error',traceback=traceback.format_exc()))
        raise

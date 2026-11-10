"""Class-once simple-v2 scoring. Network is explicit opt-in, never a hidden call."""
import json
import os
from pathlib import Path
import urllib.request
import urllib.error
from .io import ROOT, read, save, result
from . import patrol_classifier as classifier

class ApiRateLimit(RuntimeError):
    def __init__(self, metadata):
        self.quota_metadata=metadata
        super().__init__('Importance API HTTP 429; see sanitized quota metadata')


def path(value):
    p=Path(value)
    return p if p.is_absolute() else ROOT/p


def transport(request):
    from .api_credentials import api_key
    key=api_key()
    body={'contents':[{'role':'user','parts':[{'text':request['prompt']}]}],
          'generationConfig':{'temperature':0,'responseMimeType':'application/json',
                              'responseJsonSchema':request['response_schema']}}
    req=urllib.request.Request('https://generativelanguage.googleapis.com/v1beta/models/'+request['model']+':generateContent',
        data=json.dumps(body).encode(),headers={'Content-Type':'application/json','x-goog-api-key':key})
    try:
        with urllib.request.urlopen(req,timeout=90) as stream: value=json.load(stream)
    except urllib.error.HTTPError as exc:
        if exc.code==429:
            info={}
            try:
                details=json.load(exc).get('error',{}).get('details',[])
                info['quota_ids']=[v.get('quotaId','') for d in details for v in d.get('violations',[])]
                info['retry_delays']=[d['retryDelay'] for d in details if 'retryDelay' in d]
            except (ValueError,TypeError):pass
            raise ApiRateLimit(info) from None
        raise RuntimeError(f'Importance API returned HTTP {exc.code}; no response body or credentials logged') from None
    raw=''.join(p.get('text','') for p in value.get('candidates',[{}])[0].get('content',{}).get('parts',[]))
    try:decoded=json.loads(raw)
    except json.JSONDecodeError:decoded={'malformed_json_response':raw}
    return decoded, value.get('usageMetadata',{})


def importance(job):
    c=job['capture'];out=Path(job['output']);cfg=job['config'];prefix=c['id']
    scene=read(job['dependencies'][prefix+'.scene']['scene'])
    opt=cfg.get('importance',{})
    if opt.get('mode') == 'manual_1to5':
        # Frozen completed repeat study only; never silently use legacy defaults.
        from . import patrol_rating as rating
        if c.get('importance_cache') or opt.get('cache'):
            raise ValueError('manual_1to5 requires the bound repeat report, not a legacy instance cache')
        mp=path(opt['manual'])
        if opt.get('ratings_report'):
            rp=path(opt['ratings_report'])
        else:
            if not opt.get('allow_api',False):
                raise RuntimeError('New 1--5 scoring needs a completed repeat report or explicit API permission; no old/default scoring')
            rating.run([job['dependencies'][prefix+'.scene']['scene']],mp,out/'repeated_ratings',
                       allow_api=True,min_interval_s=7)
            rp=out/'repeated_ratings/ratings_report.json'
        manual=read(mp) if mp.suffix=='.json' else mp.read_text()
        names=sorted({o['class_name'] for o in scene['objects']
                      if o.get('class_name') is not None and classifier.eligible(o['class_name'])})
        classes=rating.checked_report(read(rp),manual,names)
        cache=rating.export_cache(scene,classes)
        events=[dict(code='unclassified_neutral_default',instance_id=o['instance_id'],score=.6)
                for o in scene['objects'] if o.get('class_name') is None]
        save(out/'importance_cache.json',cache)
        save(out/'importance_detail.json',dict(status='UNVERIFIED',classes=classes,policy_events=events,
            api_calls=0,required_objects_added=False,score_policy='five fresh class ratings; mean/5',
            ratings_report=str(rp),source_api_calls=read(rp)['api_calls_total']))
        result(out,cache=str(out/'importance_cache.json'),detail=str(out/'importance_detail.json'),
               calls=0,policy_events=events,verification_status='UNVERIFIED',external_artifacts=[str(mp),str(rp)])
        return
    cache_path=c.get('importance_cache') or opt.get('cache')
    if not cache_path and c.get('planning_profile') and opt.get('mode','profile')=='profile':
        cache_path=str(path(c['planning_profile']['path'])/'importance_cache.json')
    events=[];usage=[];classes={};calls=0
    if cache_path:
        p=path(cache_path);cache=read(p)
        if cache['capture_id']!=scene['capture_id'] or cache['scene_id']!=scene['scene_id']:
            raise ValueError('Importance cache belongs to another scene/capture')
        if set(cache['scores'])!={o['instance_id'] for o in scene['objects']}:
            raise ValueError('Importance cache target IDs differ; use class-once simple scoring for new targets')
        for item in cache['scores'].values():
            if type(item.get('score')) not in (int,float) or item['score'] not in (0,.5,1):
                raise ValueError('Simple-v2 importance scores must be 0, 0.5, or 1')
        external=[str(p)]
    else:
        mode=opt.get('mode','uniform');external=[]
        manual={};template=(ROOT/'spacewatch_pipeline/importance_prompt.txt').read_text()
        if mode=='simple':
            mp=path(opt['manual']);external.append(str(mp))
            manual=read(mp) if mp.suffix=='.json' else classifier.preprocess_manual(mp.read_text())
        for name in sorted({o['class_name'] for o in scene['objects'] if o.get('class_name') is not None}):
            if not classifier.eligible(name):continue
            if mode=='uniform':
                classes[name]={'level':1,'reason':'균등 기본 중요도; LLM 미호출'};continue
            request=classifier.prepare_request(name,manual,template)
            seed=ROOT/'resources/importance_seed_cache'/request['cache_key']
            directory=path(opt['class_cache']) if opt.get('class_cache') else out/'class_cache'
            if (seed/'result.json').exists():
                classes[name]=classifier.classify_cached(request,seed.parent)
                external.extend(str(p) for p in seed.glob('*.json'));continue
            def complete(req):
                nonlocal calls
                calls+=1
                raw,tokens=transport(req);usage.append(tokens)
                save(out/f'api_receipt_{calls:03d}.json',dict(response=raw,usage=tokens,cache_key=req['cache_key']))
                return raw
            try:
                classes[name]=classifier.classify_cached(request,directory,complete if opt.get('allow_api',False) else None)
            except RuntimeError as exc:
                if str(exc)!='no_authorized_transport' or opt.get('missing','error')!='default':raise
                classes[name]={'level':1,'reason':'캐시 미존재: 미채점 기본 중요도'}
                events.append(dict(code='importance_cache_missing_default',class_name=name,score=.5))
        cache=classifier.export_cache(scene,classes)
    save(out/'importance_cache.json',cache)
    save(out/'importance_detail.json',dict(status='UNVERIFIED',classes=classes,policy_events=events,
        api_calls=calls,usage=usage,required_objects_added=False,score_policy='level/2; no count normalization'))
    result(out,cache=str(out/'importance_cache.json'),detail=str(out/'importance_detail.json'),
           calls=calls,policy_events=events,verification_status='UNVERIFIED',external_artifacts=external)

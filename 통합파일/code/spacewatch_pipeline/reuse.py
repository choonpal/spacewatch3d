"""Reuse v1 Gaussian training only after validating its inputs, code and receipts."""
from pathlib import Path
from .io import ROOT,read,result,verify,sha


def gaussian_result(job):
    config=job['config'];old=Path(config['reuse_gaussian_run']);prefix=job['capture']['id'];action=job['action']
    identity=read(old/'identity.json');oldconfig=identity['config']
    def scientific_options(value):
        return {k:v for k,v in value.items() if k not in ['decoded_cache_mib']}
    if scientific_options(oldconfig.get('gaussian',{}))!=scientific_options(config.get('gaussian',{})):
        raise ValueError('Gaussian reuse requires identical training configuration')
    compatibility=read(ROOT/'configs/gaussian_reuse_compatibility.json')
    for file in ['gaussian.py','gaussian_data.py']:
        name='spacewatch_pipeline/'+file
        producer=identity['code'].get(name);current=sha(ROOT/name)
        if producer!=current:
            allowed=compatibility.get(name,{})
            if current!=allowed.get('consumer_sha256') or producer not in allowed.get('producer_sha256',[]):
                raise ValueError('Gaussian implementation changed without a reviewed compatibility record: '+name)
    oldresults=read(old/'results.json');source=job['dependencies'][prefix+'.reconstruct']
    previous=oldresults[prefix+'.reconstruct']
    if Path(source['source_run']).resolve()!=Path(previous['source_run']).resolve():
        raise ValueError('Gaussian cache uses a different reconstruction')
    source_receipt=read(old/'stages'/f'{prefix}.reconstruct'/'complete.json')
    coordinate_record=next(r for r in source_receipt['artifacts'] if r['path']==str(Path(previous['coordinates']).resolve()))
    if sha(source['coordinates'])!=coordinate_record['sha256']:raise ValueError('Gaussian coordinate system changed')
    external=[str(ROOT/'configs/gaussian_reuse_compatibility.json')]
    for name in [prefix+'.reconstruct',prefix+'.gs_prepare']+([prefix+'.gs_train'] if action=='gs_train' else []):
        path=old/'stages'/name/'complete.json';receipt=read(path)
        for record in receipt['artifacts']:
            verify(record);external.append(record['path'])
        external.append(str(path))
    values=dict(oldresults[prefix+'.'+action]);values.update(reused_from=str(old),reused=True,
                                                           producer_gaussian_sha256=identity['code'].get('spacewatch_pipeline/gaussian.py'),
                                                           external_artifacts=sorted(set(external)))
    result(job['output'],**values)

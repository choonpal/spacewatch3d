"""Explicit, verified reuse of unchanged v2 geometry stages after downstream edits."""
from pathlib import Path
from .io import ROOT,read,save,sha,digest,verify,result

GEOMETRY_ACTIONS=('inputs','infer','masks','reconstruct','filter','mesh','semantic','scene')
CONFIG_KEYS=('pipeline_version','schema_version','captures','vocabulary','runtime','mesh','early_semantics','semantic','scene','scene_id','importance_by_class')
MODULES=('geometry.py','early_semantics.py','io.py','ooops.py','planning.py')

def reuse(job):
    source=Path(job['config']['reuse_geometry_run']).resolve();identity=read(source/'identity.json')
    for key in CONFIG_KEYS:
        if identity['config'].get(key)!=job['config'].get(key):raise ValueError('Geometry reuse config differs: '+key)
    current={str(p.relative_to(ROOT)) for folder in ['vendor/segmentation','vendor/panovggt_pipeline_v2'] for p in (ROOT/folder).rglob('*.py')}
    current.update('spacewatch_pipeline/'+name for name in MODULES)
    old={name for name in identity['code'] if name.startswith(('vendor/segmentation/','vendor/panovggt_pipeline_v2/')) and name.endswith('.py')}
    if old != {name for name in current if name.startswith('vendor/')}:raise ValueError('Geometry reuse source inventory changed')
    for name in sorted(current):
        if identity['code'].get(name)!=sha(ROOT/name):raise ValueError('Geometry reuse code differs: '+name)
    action=job['reuse_action']
    if action not in GEOMETRY_ACTIONS:raise ValueError('Downstream stages may not be imported as geometry')
    stage=job['capture']['id']+'.'+action;complete=source/'stages'/stage/'complete.json';receipt=read(complete)
    if digest({k:v for k,v in receipt.items() if k!='receipt_digest'})!=receipt['receipt_digest']:raise ValueError('Invalid source receipt digest')
    paths=[str(verify(r)) for r in receipt['artifacts']]
    if str(Path(receipt['result_path']).resolve()) not in paths:raise ValueError('Result is not receipted')
    original=read(receipt['result_path'])
    original['external_artifacts']=sorted(set(paths+[str(complete),str(source/'identity.json')]))
    original['imported_geometry_stage']=dict(run=str(source),stage=stage,receipt_digest=receipt['receipt_digest'],algorithms_unchanged=True)
    result(job['output'],**original)

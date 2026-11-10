"""Run OOOPS with the generated vocabulary, in the existing OOOPS environment."""
from pathlib import Path
import sys
from .common import read,save,sha


def run(config,output):
    output=Path(output);vocabulary=output/'classes.json';names=[r['name'] for r in read(vocabulary)]
    sys.path.insert(0,config['runtime']['pipeline_root'])
    from spacewatch_pipeline.ooops import predict
    import torch
    if torch.cuda.device_count()!=1:raise RuntimeError('Exactly one GPU required')
    job=dict(config=dict(runtime=config['runtime'],vocabulary=str(vocabulary)))
    for capture in config['captures']:
        source=Path(capture['source_run']);manifest=read(source/'input/manifest.json')
        images=[source/'input'/r['file'] for r in manifest['frames']]
        dest=output/'segmentation'/capture['id'];dest.mkdir(parents=True,exist_ok=True)
        complete=dest/'complete.json'
        if complete.exists() and read(complete).get('vocabulary_sha256')==sha(vocabulary):
            prior=read(complete)
            hashes=prior.get('prediction_hashes',{})
            if len(hashes)==len(images) and prior.get('source_manifest_sha256')==sha(source/'input/manifest.json') and all((dest/p).is_file() and sha(dest/p)==h for p,h in hashes.items()):continue
        predict(job,images,dest,names)
        record=read(complete);record.update(source_manifest_sha256=sha(source/'input/manifest.json'),
                                           source_video_sha256=manifest['source_sha256'],
                                           prediction_hashes={p.name:sha(p) for p in dest.glob('[0-9][0-9][0-9][0-9][0-9].npz')})
        save(complete,record)
    save(output/'segmentation_summary.json',dict(vocabulary_sha256=sha(vocabulary),classes=names,
                                                captures=[c['id'] for c in config['captures']]))


if __name__=='__main__':
    import sys
    run(read(sys.argv[1]),sys.argv[2])

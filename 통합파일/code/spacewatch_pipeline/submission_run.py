"""One public command connects exact original raw producers to current pipeline."""
import argparse
import copy
import fcntl
from pathlib import Path
import time

from .io import read,save,sha,record,code_identity


def resolve(base,value):
    path=Path(value)
    return path.resolve() if path.is_absolute() else (base/path).resolve()


def run(request_path,output):
    from .upstream519 import run as prepare
    from .engine import execute,validate
    request_path=Path(request_path).resolve();request=read(request_path)
    if request.get('schema')!='spacewatch.integrated_fresh.v1':
        raise ValueError('Unsupported whole-pipeline request')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    with (output/'.submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.time();results={}
        from .upstream519 import source_snapshot
        source_snapshot()
        save(output/'submission-identity.json',dict(request=record(request_path),code=code_identity(),
            exact_original_source_manifest=record(Path(__file__).resolve().parents[1]/'upstream519_original/source_manifest.json')))
        save(output/'submission-state.json',dict(status='preparing',pid=__import__('os').getpid(),request=record(request_path)))
        try:
            config=read(resolve(request_path.parent,request['pipeline_config']))
            ids={capture['id'] for capture in config['captures']}
            rows=request['captures']
            if len(rows)!=len(ids) or {row['id'] for row in rows}!=ids:
                raise ValueError('Preparation requests must cover exactly current captures')
            catalog_hash=None
            for row in rows:
                prep_request=resolve(request_path.parent,row['prepare_request'])
                value=prepare(read(prep_request),output/'preparation'/row['id'])
                if value['capture_id']!=row['id']:raise ValueError('Prepared capture identity differs')
                if not value.get('captured_prediction'):
                    raise ValueError('Original raw producer did not bind its actual OOOPS frame predictions')
                this_hash=sha(value['vocabulary'])
                if catalog_hash is not None and this_hash!=catalog_hash:
                    raise ValueError('Both captures must use the same requested class vocabulary')
                catalog_hash=this_hash
                capture=next(c for c in config['captures'] if c['id']==row['id'])
                capture.pop('reconstruction_config',None);capture.pop('planning_scene',None)
                capture.update(source_run=value['source_run'],coordinates=value['coordinates'],
                    semantic_arrays=value['semantic_arrays'],semantic_frame_id=row['id']+'_mesh',
                    semantic_coordinates_sha256=sha(value['coordinates']),
                    planning_profile=value['planning_profile'],captured_prediction=value['captured_prediction'],
                    importance_cache=str(Path(value['planning_profile']['path'])/'importance_cache.json'))
                config['segmentation']['vocabulary']=value['vocabulary']
                results[row['id']]=value
                save(output/'preparation-results.json',results)
            config['paper_integration'].update(upstream_mode='fresh_original_producer_in_this_submission',
                fresh_geometry=False,preparation_results=str(output/'preparation-results.json'),
                reuse_scope=['native-stage import of newly completed exact original producer outputs in this submission',
                             'fixed model checkpoints and frozen manual-conditioned class-rating cache'])
            validate(config)
            config_path=output/'pipeline-config.json';save(config_path,config)
            save(output/'submission-state.json',dict(status='running_native_pipeline',pid=__import__('os').getpid(),
                request=record(request_path),preparation_results=str(output/'preparation-results.json')))
            execute(config_path,output/'pipeline')
            save(output/'submission-state.json',dict(status='complete',seconds=time.time()-started,
                pipeline=str(output/'pipeline'),preparation_results=str(output/'preparation-results.json'),
                execution_complete_is_not_scientific_quality_certification=True))
        except BaseException as error:
            save(output/'submission-state.json',dict(status='failed',error=str(error),seconds=time.time()-started,
                completed_preparations=results))
            raise


def main(argv=None):
    p=argparse.ArgumentParser(description='Exact original raw producers → current native selection/change/viewer')
    p.add_argument('--request',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv);run(args.request,args.output);return 0


if __name__=='__main__':raise SystemExit(main())

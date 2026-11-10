import sys
from pathlib import Path
from .io import read

def main():
    job=read(sys.argv[1]);action=job['action']
    if job['config'].get('paper_integration', {}).get('enabled'):
        from .paper_pipeline import dispatch
        if dispatch(job):
            return
    if job['config'].get('media', {}).get('mode') == 'original_erp':
        from .capture519 import dispatch
        if dispatch(job):
            return
    if action in ['gs_prepare','gs_train'] and job['config'].get('reuse_gaussian_run'):
        from .reuse import gaussian_result
        gaussian_result(job)
        return
    if action in ['instances','scene','plan'] and (job.get('capture') or {}).get('planning_profile'):
        from . import target_profile
        getattr(target_profile,action)(job)
        return
    if action=='instances':
        from . import size_instances as module
    elif action in ['mesh','discovery','semantic'] and job['config']['segmentation']['source']=='provided':
        from . import provided as module
    elif action=='scene':
        from . import size_scene as module
    elif action=='validate':
        from . import validation as module
    elif action in ['prepare','infer','observe','refine_prepare','refine_infer','objects','discovery','semantic','instances']:
        from . import florence as module
    elif action in ['reconstruct','mesh']:
        from . import geometry as module
    elif action in ['scene','plan']:
        from . import planning as module
    elif action in ['align','changes']:
        from . import change as module
    elif action=='gs_prepare':
        from . import gaussian_data as module
    elif action in ['gs_train','render']:
        from . import gaussian as module
    elif action=='publish':
        from . import report as module
    elif action=='viewer':
        from . import viewer as module
    else:
        raise ValueError('Unknown action: '+action)
    getattr(module,action)(job)

if __name__=='__main__':main()

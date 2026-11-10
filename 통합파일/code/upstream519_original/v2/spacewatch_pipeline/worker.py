import sys
from pathlib import Path
from .io import read

def main():
    job=read(sys.argv[1]);action=job['action']
    if action=='reuse':
        from . import reuse as module
    elif action in ['inputs','masks','filter']:
        from . import early_semantics as module
    elif action=='priority':
        from . import change_priority as module
    elif action in ['reconstruct','mesh','infer','semantic']:
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

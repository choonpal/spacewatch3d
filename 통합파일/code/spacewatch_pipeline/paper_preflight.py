"""Parse the actual profile-specific native model BEFORE costly video work."""
import json
from pathlib import Path
import sys
from .capture519 import planning_options
from .io import ROOT,read


def main():
    config=json.load(sys.stdin)
    capture=next(c for c in config['captures'] if c['id']==sys.argv[1])
    profile=Path(capture['planning_profile']['path']) if capture.get('planning_profile') else None
    sys.path.insert(0,str(profile/'code') if profile else str(ROOT/'vendor/spacewatch3d_src'))
    from spacewatch3d.tasks.task_5b_keyframes.models import Config
    options=planning_options(config,read(profile/'config.json') if profile else None)
    paper=options.pop('paper_selection',None)
    native=Config.from_dict(options)
    if not (native.coverage_repair and native.coverage_repair_runs==2 and
            not native.route_aware_selection and not native.lateral_candidate_refinement and native.max_link_m is None):
        raise ValueError('Native observation/selection contract differs from paper')
    print(json.dumps(dict(ok=True,capture=capture['id'],native_config_accepted=True,paper_selection=paper)))


if __name__=='__main__':main()

"""Run the unchanged M3C2/noise-screen pipeline on a real Bonn removal clip."""
import argparse
import json
from pathlib import Path

import numpy as np

from prepare_bonn_pair import prepare
from compare_mesh_methods import build as compare
from build_m3c2_independent import build as detect
from publish_rgbd_demo import publish


def run(data, output):
    prepare(data,output)
    result=compare(output/'prepared/cache/0-depth8.ply',output/'prepared/cache/1-depth8.ply',
        output/'comparison',scene='Bonn box removal · 0 → 1',display_rotation=np.eye(3))
    result['dataset_id']='bonn-removal'
    result['alignment']='Bonn provided poses with published format-12 correction; one common display transform; no per-epoch fit to changed objects.'
    (output/'comparison/comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    detect(output/'comparison',output/'prepared',output/'frames',output/'detected')
    publish(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.data,args.output)

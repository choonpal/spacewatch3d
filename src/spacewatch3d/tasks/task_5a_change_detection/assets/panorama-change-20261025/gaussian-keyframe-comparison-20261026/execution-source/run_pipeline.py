from pathlib import Path
import os,subprocess,json
root=Path('/workspace/gaussian_keyframes_20261026_v2')
with (root/'render.log').open('wb') as log:
 subprocess.run(['bash','/workspace/v1_splatfacto_comparison_20261025/env.sh',str(root/'render_maps.py'),'--root',str(root)],cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT,check=True)
with (root/'infer.log').open('wb') as log:
 subprocess.run(['/workspace/sam31_panorama_test_20261023/env/bin/python',str(root/'infer_keyframes.py'),'--root',str(root)],cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT,check=True)

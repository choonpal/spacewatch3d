"""Publish only viewer assets from a completed Bonn comparison run."""
import argparse
import json
import shutil
from pathlib import Path


def publish(root):
    output=root/'published'
    if output.exists() and any(output.iterdir()):raise ValueError('Published directory must be new and empty')
    output.mkdir(parents=True,exist_ok=True)
    prefix='/datasets/bonn-removal/'
    comparison=json.loads((root/'comparison/comparison.json').read_text(encoding='utf-8'))
    reference=json.loads((root/'prepared/changes.json').read_text(encoding='utf-8'))
    changes=json.loads((root/'detected/changes.json').read_text(encoding='utf-8'))
    if reference['dataset_id']!='bonn-removal' or comparison['dataset_id']!='bonn-removal':raise ValueError('Dataset mismatch')
    if changes['source_hashes']!=comparison['input_hashes']:raise ValueError('Mesh mismatch')
    for meta in comparison['displays']:
        name=Path(meta['url']).name;meta['url']=prefix+name
        shutil.copy2(root/'comparison'/name,output/name)
    for meta in changes['region_masks']:
        name=Path(meta['url']).name;meta['url']=prefix+name
        shutil.copy2(root/'detected'/name,output/name)
    for document in (reference,changes):
        for frame in document['keyframes']:
            name=frame['id']+'.jpg';frame['url']=prefix+name
            if not (output/name).exists():shutil.copy2(root/'frames'/name,output/name)
    changes['dataset_id']='bonn-removal'
    changes['provenance']={'dataset':reference['dataset'],'sequence':reference['sequence'],
        'source_page':reference['source_page'],'capture_relation':reference['capture_relation'],
        'windows_s':reference['windows_s'],'labels_used_for_detection':False,
        'parameters_changed_from_rescan_noise_screen':False}
    for name,data in [('comparison',comparison),('reference',reference),('changes',changes)]:
        (output/(name+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('Published',changes['summary'],'with',len(reference['keyframes']),'original frames at',output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    publish(p.parse_args().run_dir)

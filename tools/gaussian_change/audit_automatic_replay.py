"""Rebuild object decisions in a fresh folder without copying decision outputs.

This verifies reproducibility of the object stage from cached upstream inputs.
It does not rerun Gaussian reconstruction, registration, segmentation or SimSaC,
and does not measure physical change-detection accuracy or generalization.
"""
from __future__ import annotations
import argparse
import ast
from pathlib import Path
import shutil
import tempfile
from prepare import read,save,sha
from link_segmented_objects import build,classify_with_context,validate


def audit(directory,base,replay_parent):
    replay_parent.mkdir(parents=True,exist_ok=True)
    replay=Path(tempfile.mkdtemp(prefix='automatic-object-replay-',dir=replay_parent))
    required=['protocol.json','simsac-candidates.json']
    for name in required:shutil.copy2(directory/name,replay/name)
    initial_files=sorted(p.name for p in replay.iterdir())
    assert initial_files==sorted(required)
    previous={name:sha(directory/name) for name in ['automatic-matching.json','object-evidence.json']}
    result=build(replay,base)
    checks={name:sha(replay/name)==previous[name] for name in previous}
    validation=validate(replay)
    matching=read(replay/'automatic-matching.json');events=[];ablations=[]
    protocol=read(replay/'protocol.json')
    for id,record in result['regions']['filtered'].items():
        event=record['event'];pair=event.get('automatic_match')
        events.append({'candidate_id':id,'type':event['type'],'selected_pair':event['selected_pair'],
                       'shape_f1':pair['shape_f1'] if pair else None,'automatic':event['automatic']})
        if event['type']=='moved':
            original=next(b for b in read(replay/'simsac-candidates.json')['boxes']['filtered'] if b['id']==id)
            error=protocol['alignment_context']['methods']['structure']['errors']['regions'][original['zone']]['tolerance_u']
            without=classify_with_context(record['linked_objects'],record['correspondence_proposals'],record['same_position_pairs'],protocol['height_u'],error,record['same_zone_context'],[])
            ablations.append({'candidate_id':id,'with_correspondence':event['type'],
                              'without_correspondence':without['type'],'movement_requires_computed_correspondence':without['type']!='moved'})
    # This source scan is supplementary evidence, not an accuracy evaluation.
    decision_files=[Path(__file__).with_name(name) for name in ['automatic_object_matching.py','link_segmented_objects.py']]
    literals={p.name:sorted({node.value for node in ast.walk(ast.parse(p.read_text(encoding='utf-8'))) if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value in {'R12','I0-005','I1-067','S0-010','S1-024'}}) for p in decision_files}
    report={'schema':'automatic-object-replay-audit-v1','passed':all(checks.values()) and all(a['movement_requires_computed_correspondence'] for a in ablations),
        'replay_directory':str(replay),'initial_files':initial_files,'copied_previous_decision_outputs':False,
        'identical_output_sha256':checks,'published_sha256':previous,
        'decision_source_sha256':{p.name:sha(p) for p in decision_files},'case_specific_literals_in_decision_code':literals,
        'manual_override_files_present':any((p/name).exists() for p,name in [(directory,'event-review-overrides.json'),(replay,'event-review-overrides.json'),(Path(__file__).parent,'review_overrides.json')]),
        'stages_recomputed':['instance-to-mesh linkage','mask-assisted mesh partition','rigid shape search','one-to-one assignment','event classification'],
        'upstream_inputs_reused':['registered meshes','semantic components','image instance tracks and vacancy votes','Gaussian-render SimSaC candidate regions'],
        'upstream_model_inference_repeated':False,'new_maps_to_final_output_exercised':False,
        'generalization_or_physical_accuracy_evaluated':False,
        'events':events,'evidence_ablation':ablations,'object_validation':validation,
        'accepted_matches':len(matching['accepted_matches'])}
    save(directory/'automatic-replay-audit.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,default=Path('outputs/scenediff-mesh-rgb-20261024/gaussian-uncapped-tour-20261026'))
    parser.add_argument('--base',type=Path,default=Path('outputs/registration-benchmark-20261019/new-meshfirst-20261022'))
    parser.add_argument('--replay-parent',type=Path,default=Path('outputs/automation-replays-20261026'))
    args=parser.parse_args();report=audit(args.directory,args.base,args.replay_parent)
    print({key:report[key] for key in ['passed','initial_files','copied_previous_decision_outputs','identical_output_sha256','manual_override_files_present','events','evidence_ablation','new_maps_to_final_output_exercised']})
    if not report['passed']:raise SystemExit(1)

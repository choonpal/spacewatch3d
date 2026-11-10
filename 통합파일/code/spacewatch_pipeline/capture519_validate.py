"""Media-enum adapter, not a replacement for original geometry validation."""
import argparse
import copy
from pathlib import Path
import sys
from .io import ROOT,read


def validate_native(plan,requests,legacy_validate):
    views={v['viewpoint_id']:v for v in plan['viewpoints']}
    translated=copy.deepcopy(requests)
    for original,legacy in zip(requests['requests'],translated['requests']):
        view=views[original['viewpoint_id']]
        if original.get('media_origin')!='original_video_frame':raise ValueError('Native request origin mismatch')
        if original.get('source_frame')!=view.get('source_frame') or original['reference_from_camera']!=view['reference_from_camera']:
            raise ValueError('Native source frame/pose mismatch')
        # 5.1.9's enum meant virtual-camera geometry requests. Only its media
        # enum is translated for that unchanged validator, never any score,
        # target, pose, edge, denominator or failure/planning status.
        legacy['media_origin']='novel_view'
    legacy_validate(plan,translated)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--requests',type=Path,required=True)
    parser.add_argument('--profile',type=Path)
    args=parser.parse_args()
    if args.profile:sys.path[:0]=[str(args.profile/'code'),str(args.profile/'native'),str(args.profile)]
    from spacewatch3d.tasks.task_5b_keyframes.validation import validate_outputs
    validate_native(read(args.plan),read(args.requests),validate_outputs)
    print('NATIVE_MEDIA_AND_ORIGINAL_GEOMETRY_VALIDATION_OK')


if __name__=='__main__':main()

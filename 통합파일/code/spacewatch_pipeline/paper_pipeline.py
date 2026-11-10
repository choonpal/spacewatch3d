"""Paper integration boundaries; original geometry/target/visibility cores stay native.

Stored geometry is an explicit input, never a replacement for current selection
or detector execution. The raw producer entrypoint is separately provenance-bound.
"""
import copy
import json
import math
from pathlib import Path
import time

from .io import read, save, result, sha


def validate(config):
    paper = config['paper_integration']
    if paper.get('schema') != 'spacewatch.paper.20261110.v1':
        raise ValueError('Unsupported integrated paper configuration')
    if config['media'].get('mode') != 'original_erp':
        raise ValueError('Paper integration never synthesizes Gaussian images')
    if config['media'].get('candidate_interval_s') != .25:
        raise ValueError('Paper candidate intervals must be 0.25 second bins')
    selection = paper['selection']
    if set(selection) != {'max_views','deadline_seconds'}:
        raise ValueError('Only solver safety cap/deadline are configurable')
    if not isinstance(selection['max_views'],int) or selection['max_views'] < 1:
        raise ValueError('Positive selection safety cap required')
    if not 0 < selection['deadline_seconds'] <= 600:
        raise ValueError('Finite selection deadline required')
    if config['runtime'].get('overlap_cpu_gpu') is not False:
        raise ValueError('Large integration stages must run sequentially')
    if not 1 <= config['runtime'].get('cpu_threads',0) <= 4:
        raise ValueError('Use 1..4 CPU threads')
    if len(config['captures']) == 2:
        change=config.get('changes',{})
        if not change.get('enabled') or change.get('backend')!='mesh_rgb_simsac' or not change.get('keypoint_nearby'):
            raise ValueError('Paired paper execution needs current-keypoint original-RGB changes')
    for capture in config['captures']:
        if not capture.get('video'):
            raise ValueError('Each capture must bind its original ERP video')
        if not capture.get('importance_cache'):
            raise ValueError('Explicit frozen manual-conditioned importance cache required')
    if config.get('gaussian_changes',{}).get('enabled'):
        raise ValueError('Obsolete Gaussian change backend must stay disabled')


def adapt_stages(rows, config):
    """Add full-video measurement and ensure comparison uses CURRENT plans."""
    rows=copy.deepcopy(rows)
    for capture in config['captures']:
        prefix=capture['id']
        rows.append(dict(name=prefix+'.quality',action='quality',capture=prefix,
                         dependencies=[],gpu=False))
        rows.append(dict(name=prefix+'.importance',action='importance',capture=prefix,
                         dependencies=[prefix+'.scene'],gpu=False))
        for row in rows:
            if row['name']==prefix+'.trajectory':
                row['dependencies'].append(prefix+'.quality')
            if row['name']==prefix+'.plan':
                row['dependencies'].append(prefix+'.importance')
    for row in rows:
        if row['name']=='pair.rgb_inputs':
            row['dependencies']=list(dict.fromkeys(row['dependencies']+
                [c['id']+'.plan' for c in config['captures']]))
        if row['name']=='pair.rgb_prepare':
            row['dependencies']=list(dict.fromkeys(row['dependencies']+
                [c['id']+'.importance' for c in config['captures']]))
        if row['action'] in ('validate','publish','viewer'):
            row['dependencies']=list(dict.fromkeys(row['dependencies']+
                [c['id']+'.quality' for c in config['captures']]))
            row['dependencies']=list(dict.fromkeys(row['dependencies']+
                [c['id']+'.importance' for c in config['captures']]))
    return rows


def memory_snapshot():
    values={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,rest=line.split(':',1)
        values[key]=int(rest.split()[0])*1024
    vm={}
    for line in Path('/proc/vmstat').read_text().splitlines():
        key,value=line.split()
        if key in ('pswpin','pswpout'):vm[key]=int(value)
    return dict(mem_available_bytes=values['MemAvailable'],swap_free_bytes=values['SwapFree'],
                swap_total_bytes=values['SwapTotal'],**vm)


def resource_guard(config, action, report):
    """Fail before a new worker instead of silently reducing scientific quality."""
    first=memory_snapshot();time.sleep(.25);second=memory_snapshot()
    policy=config['paper_integration'].get('resources',{})
    minimum=policy.get('minimum_available_gib',1.5)
    if action=='rgb_infer':minimum=max(minimum,policy.get('inference_available_gib',3.0))
    if action in ('mesh','reconstruct') and config['paper_integration'].get('fresh_geometry'):
        minimum=max(minimum,policy.get('fresh_geometry_available_gib',16.0))
    report_data=dict(action=action,before=first,after=second,minimum_available_gib=minimum,
        local_data_written=False,quality_reduced=False,
        swap_out_delta=second['pswpout']-first['pswpout'])
    ok=second['mem_available_bytes']>=minimum*1024**3
    if report_data['swap_out_delta']>policy.get('max_swap_out_pages_per_check',4096):ok=False
    save(report,dict(report_data,ok=ok))
    if not ok:
        raise RuntimeError('Resource safety hold: insufficient available RAM or active swap pressure; '+str(report))


def quality(job):
    """Read every original video frame; no initial eligibility/blur gate."""
    out=Path(job['output']);capture=job['capture']
    from .frame_sharpness import scan
    summary=scan(Path(capture['video']),out/'quality',
                 job['config']['paper_integration'].get('quality_timeout_seconds',900))
    result(out,quality_source=str(out/'quality'),video=capture['video'],
        video_sha256=summary['source']['sha256'],frames=summary['source_frames'],
        front_quality_filter=False,metric='median_four_equatorial_Laplacian_variance_ksize3',
        seconds=summary['seconds'],external_artifacts=[str(Path(capture['video']).resolve())])


def trajectory(job):
    from .existing_path import trajectory as native
    payload=copy.deepcopy(job);prefix=payload['capture']['id']
    payload['config']['media'].update(payload['capture'].get('media',{}))
    payload['config']['media']['quality_source']=payload['dependencies'][prefix+'.quality']['quality_source']
    native(payload)


def dispatch(job):
    action=job['action']
    if action=='quality':quality(job)
    elif action=='importance':
        from .paper_importance import importance
        importance(job)
    elif action=='trajectory':trajectory(job)
    elif action in ('rgb_inputs','rgb_prepare','rgb_infer','rgb_finalize'):
        from .paper_changes import dispatch as changes
        if not changes(job):raise ValueError('Missing integrated change action '+action)
    elif action=='viewer':
        from .paper_viewer import viewer
        viewer(job)
    else:return False
    return True

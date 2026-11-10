#!/usr/bin/env python3
"""Controlled evaluator profile; this is not a full-scene speed measurement."""
import importlib.util
from pathlib import Path
import sys
import time
import numpy as np
from spacewatch_pipeline.io import ROOT, save
sys.path[:0]=[str(ROOT/'vendor/spacewatch3d_src'),str(ROOT/'vendor/native')]
from spacewatch3d.tasks.task_5b_keyframes.selection import Objective
from spacewatch3d.tasks.task_5b_keyframes.models import Config
import objective_executor as optimized


def main():
    spec=importlib.util.spec_from_file_location('v5_objective', '/workspace/spacewatch_pipeline_v5/vendor/native/objective_executor.py')
    baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
    rng=np.random.default_rng(85)
    visible=rng.random((704,6000))<.2;quality=rng.random((704,87));weights=rng.random(6000)+.1
    regions=np.array(['region_'+str(i%20) for i in range(6000)])
    objects=[dict(instance_id=str(i),must_see=False,importance=.5) for i in range(87)]
    samples=[rng.choice(704,13,replace=False).tolist() for _ in range(400)]
    report={}
    for scenario,queries in [('unique',samples),('repeated',samples[:128]*10)]:
        outputs=[];timings={}
        for name,install in [('v5',baseline.install),('v51',lambda:optimized.install(immutable=True))]:
            install();obj=Objective(visible,quality,weights,regions,objects,Config())
            start=time.perf_counter();values=[obj.measure(q) for q in queries]
            timings[name]=time.perf_counter()-start;outputs.append(values)
            optimized.restore()
        assert outputs[0]==outputs[1]
        report[scenario]=dict(calls=len(queries),seconds=timings,exact_report_parity=True)
    save(sys.argv[1],report);print(report)


if __name__=='__main__':main()

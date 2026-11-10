"""Bounded finite-pool covering proposal; original goals are always rechecked.

Binary camera MILP plus valid union-coverage cuts. No distance/ROI/physical
threshold is introduced. Runtime is a computational budget, not a new target.
"""
import time
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import csr_matrix,vstack
from spacewatch3d.tasks.task_5b_keyframes.selection import AttainableGoals

def compact(objective,selected,required=(),allowed=None,seconds=30.,evaluate=None):
    from spacewatch3d.tasks.task_5b_keyframes.coverage_repair import JointObjective
    models=objective.objectives if isinstance(objective,JointObjective) else [objective]
    n=len(models[0].visible);selected=list(selected);required=set(required)
    started=time.monotonic();goals=AttainableGoals(objective)
    info={'status':'not_attempted','before':len(selected),'after':len(selected),'budget_seconds':seconds,
          'goal_semantics_unchanged':True,'uses_final_holdout':False,'solver_calls':0,'coverage_cuts':0,
          'scope':'same finite observation pool, not continuous-space or rendered-image optimum'}
    measure=evaluate or objective.measure
    before=measure(selected)[1]
    if not selected or not goals.met(before):return selected,{**info,'status':'baseline_attainable_goals_not_met'}
    if len(selected)==1:return selected,{**info,'status':'no_smaller_nonempty_plan'}
    allowed=set(range(n)) if allowed is None else set(allowed)
    if not required<=allowed or not set(selected)<=allowed:raise ValueError('Invalid compaction candidate scope')
    rows=[];lower=[];upper=[];matrices=[];preserved_input=[];preserved_objects=[]
    def add(row,lo,hi=np.inf):
        rows.append(csr_matrix(np.asarray(row,float).reshape(1,-1)));lower.append(float(lo));upper.append(float(hi))
    for m in models:
        for j,o in enumerate(m.objects):
            if m.possible_objects[j] and (m.config.require_all_objects or o['must_see']):
                add(m.quality[:,j]>=m.object_thresholds[j],1.)
        matrices.append(csr_matrix(m.visible,dtype=float))
        covered=m.visible[selected].any(0)
        norms=[];extra=[]
        for region in m.region_ids:
            mask=m.regions==region;area=m.weights[mask].sum()
            if not area:continue
            norm=np.where(mask,m.weights/area,0.);norms.append(norm)
            if float(norm@covered)+1e-10>=m.config.region_target:
                extra.append((norm,m.config.region_target))
        if norms:
            norm=np.mean(norms,axis=0)
            if float(norm@covered)+1e-10>=m.config.space_target:
                extra.append((norm,m.config.space_target))
        preserved_input.append(extra)
        observed=m.quality>=m.config.object_quality_threshold
        total=m.importance.sum()
        norm=m.importance/total if total else np.zeros(len(m.objects))
        keep_total=bool(total and float(norm@observed[selected].any(0))+1e-10>=m.config.object_target)
        preserved_objects.append((observed,norm,m.config.object_target) if keep_total else None)
    bound_low=np.zeros(n);bound_up=np.array([float(i in allowed) for i in range(n)])
    bound_low[list(required)]=1.
    add(np.ones(n),1.,len(selected)-1)
    previous=set();candidate=None;optimal=False
    while time.monotonic()-started<seconds:
        remaining=seconds-(time.monotonic()-started)
        if remaining<=0:break
        solved=milp(np.ones(n),integrality=np.ones(n),bounds=Bounds(bound_low,bound_up),
            constraints=LinearConstraint(vstack(rows,format='csc'),lower,upper),
            options={'time_limit':remaining,'mip_rel_gap':0.0})
        info['solver_calls']+=1;info['last_solver_status']=int(solved.status)
        if solved.x is None:
            info['status']='no_smaller_cover' if solved.status==2 else 'solver_budget_or_failure'
            break
        trial=np.flatnonzero(solved.x>.5).tolist();key=tuple(trial)
        if key in previous:
            info['status']='numerical_or_repeated_proposal';break
        previous.add(key)
        if len(trial)>=len(selected) or not required<=set(trial):
            info['status']='rejected_solver_proposal';break
        measured=measure(trial)[1]
        input_preserved=all(float(norm@(m.visible[trial].any(0)))+1e-10>=target
            for m,extra in zip(models,preserved_input) for norm,target in extra)
        input_preserved=input_preserved and all(item is None or
            float(item[1]@item[0][trial].any(0))+1e-10>=item[2] for item in preserved_objects)
        if goals.met(measured) and input_preserved:
            failures=measured.get('selection_connection_failures',[])
            old_failures=before.get('selection_connection_failures',[])
            old_groups=before.get('selection_connection_groups')
            new_groups=measured.get('selection_connection_groups')
            connection_ok=set(failures)<=set(old_failures)
            if failures:
                from spacewatch3d.tasks.task_5b_keyframes.selection import _connection_not_worse
                connection_ok=connection_ok and _connection_not_worse(before,measured)
                if old_groups is not None and new_groups is not None:
                    connection_ok=connection_ok and len(new_groups)<=len(old_groups)
            images_before=before.get('selection_image_count',len(selected))
            images_after=measured.get('selection_image_count',len(trial))
            if connection_ok and images_after<images_before:
                info.update(routed_image_count_before=images_before,routed_image_count_after=images_after)
                candidate=trial;optimal=solved.status==0;info['status']='accepted';break
            # Exact no-good cut only; adding a connector may change feasibility.
            cut=np.ones(n);cut[trial]=-1.;add(cut,1.-len(trial))
            continue
        added=0
        for m,matrix,extra,object_total in zip(models,matrices,preserved_input,preserved_objects):
            covered=m.visible[trial].any(0) if trial else np.zeros(len(m.weights),bool)
            for norm,target in extra:
                score=float(norm@covered)
                if score+1e-10<target:
                    add(matrix@(norm*~covered),target-score);added+=1
            if object_total is not None:
                observation,norm,target=object_total
                seen=observation[trial].any(0);score=float(norm@seen)
                if score+1e-10<target:
                    add(observation@(norm*~seen),target-score);added+=1
            balanced=np.zeros(len(m.weights));regions=0
            for region in m.region_ids:
                mask=(m.regions==region)&m.observable
                area=m.weights[mask].sum()
                if not area:continue
                norm=np.where(mask,m.weights/area,0.);balanced+=norm;regions+=1
                score=float(norm@covered)
                if score+1e-10<m.config.region_target:
                    add(matrix@(norm*~covered),m.config.region_target-score);added+=1
            if regions:
                norm=balanced/regions;score=float(norm@covered)
                if score+1e-10<m.config.space_target:
                    add(matrix@(norm*~covered),m.config.space_target-score);added+=1
            possible=m.possible_objects
            den=m.importance[possible].sum()
            if den:
                observe=m.quality>=m.object_thresholds
                covered_objects=observe[trial].any(0) if trial else np.zeros(len(m.objects),bool)
                norm=np.where(possible,m.importance/den,0.)
                score=float(norm@covered_objects)
                if score+1e-10<m.config.object_target:
                    add(observe@(norm*~covered_objects),m.config.object_target-score);added+=1
        info['coverage_cuts']+=added
        if not added:
            info['status']='unrepresented_failed_goal';break
    else:info['status']='solver_budget_exhausted'
    info.update(seconds=time.monotonic()-started,after=len(candidate) if candidate is not None else len(selected),
                finite_cover_minimum_proved=optimal,original_input_denominators_preserved=True,
                original_unreachable_targets_retained=True,routed_image_count_globally_minimal=False)
    info['previously_met_whole_input_training_goals_preserved']=True
    return (candidate if candidate is not None else selected),info

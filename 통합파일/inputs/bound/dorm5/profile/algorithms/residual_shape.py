"""One bounded residual screen: coherent low filling AND smaller surface area.

This module consumes anonymous numeric arrays only. No names, IDs, labels,
physical sizes, reference masks or desired viewpoint counts are accepted.
The already accepted conservative exclusions are supplied as a frozen mask.

Filling is surface area / (middle PCA span * longest PCA span). A two-Gaussian
log-filling model is compared with one Gaussian using BIC. A new lower group
is allowed only with BIC gain >= 6, posterior >= .95 for EVERY member, at least
two members per group, and COMPLETE area ordering (max lower < min upper).
Thus a shape split mixing large and small surfaces does not authorize removal.
The existing equal-parameter fused-row KEEP safeguard is retained. This is one
pass, never recursive refitting until a desired count is reached.

All thresholds are common engineering assumptions, not noise probabilities.
A genuinely small sparse object can remain indistinguishable from a fragment.
"""
from dataclasses import asdict, dataclass
import itertools

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class Settings:
    minimum_members: int = 6
    minimum_group_members: int = 2
    minimum_bic_gain: float = 6.0
    minimum_posterior: float = .95
    log_variance_floor: float = 1e-4
    maximum_iterations: int = 500
    convergence_tolerance: float = 1e-8


def _fit(x, means, cfg):
    means = np.asarray(means, float)
    k, n = len(means), len(x)
    variances = np.full(k, max(float(np.var(x)), cfg.log_variance_floor))
    weights = np.full(k, 1 / k)
    previous = -np.inf
    for iteration in range(cfg.maximum_iterations):
        logs = (np.log(weights)[None, :] - .5 * np.log(2*np.pi*variances)[None, :]
                - .5 * (x[:, None]-means[None, :])**2/variances[None, :])
        normalizer = logsumexp(logs, axis=1)
        ll = float(normalizer.sum())
        posterior = np.exp(logs-normalizer[:, None])
        members = posterior.sum(0)
        if members.min() < 1e-8:
            return None
        if ll-previous <= cfg.convergence_tolerance and ll >= previous-1e-9:
            break
        weights = members/n
        means = (posterior*x[:, None]).sum(0)/members
        variances = np.maximum((posterior*(x[:, None]-means)**2).sum(0)/members,
                               cfg.log_variance_floor)
        previous = ll
    order = np.argsort(means)
    return dict(bic=float(-2*ll+(3*k-1)*np.log(n)), log_likelihood=ll,
                means=means[order], variances=variances[order],
                members=members[order], posterior=posterior[:, order],
                iterations=iteration+1)


def filling_split(values, cfg=Settings()):
    values = np.asarray(values, float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('Expected finite positive filling ratios')
    lower = np.zeros(len(values), bool)
    report = dict(status='keep_insufficient_or_no_supported_split', n=len(values),
                  settings=asdict(cfg))
    if len(values) < cfg.minimum_members:
        return lower, report
    # Sort the anonymous numeric inputs: stable summation independent of IDs
    # and input order. Transforming units cannot change dimensionless filling.
    order = np.argsort(values, kind='stable')
    x = np.log(values[order])
    offset = float(x.mean())
    x = x-offset
    if np.ptp(x) <= 1e-12:
        return lower, report
    one = _fit(x, [0.], cfg)
    initial = [np.quantile(x, q) for q in itertools.combinations([.1,.25,.4,.6,.75,.9], 2)]
    fits = [_fit(x, seed, cfg) for seed in initial]
    fits = [f for f in fits if f is not None and f['members'].min() >= .95]
    if not fits:
        return lower, report
    two = max(fits, key=lambda f: f['log_likelihood'])
    gain = one['bic']-two['bic']
    candidate = two['posterior'][:, 0] >= cfg.minimum_posterior
    upper = two['posterior'][:, 1] >= cfg.minimum_posterior
    report.update(bic_one=one['bic'], bic_two=two['bic'], bic_gain=gain,
                  geometric_means=np.exp(two['means']+offset).tolist(),
                  effective_members=two['members'].tolist(),
                  lower_probability=two['posterior'][np.argsort(order), 0].tolist())
    if gain < cfg.minimum_bic_gain:
        report['status'] = 'keep_bic_gain_insufficient'
    elif min(int(candidate.sum()), int(upper.sum())) < cfg.minimum_group_members:
        report['status'] = 'keep_group_has_too_few_members'
    elif not (candidate | upper).all():
        report['status'] = 'keep_posterior_ambiguous_members'
    else:
        lower[order] = candidate
        report['status'] = 'supported_two_filling_populations'
    return lower, report


def model_error(delta, direction):
    d, v = np.asarray(delta, float), np.asarray(direction, float)
    growth = max(0., float(d@v/(v@v)))
    return float(np.sum((d-growth*v)**2))


def extra_exclusions(areas, spans, already_excluded, cfg=Settings()):
    a, s = np.asarray(areas, float), np.asarray(spans, float)
    old = np.asarray(already_excluded, bool)
    if a.ndim != 1 or s.shape != (len(a),3) or old.shape != a.shape:
        raise ValueError('One area, ordered extent triple and baseline flag per component')
    if not np.isfinite(a).all() or np.any(a <= 0) or not np.isfinite(s).all():
        raise ValueError('Finite positive area and finite extents required')
    if np.any(s < 0) or np.any(s[:,1:] <= 0) or np.any(np.diff(s,axis=1) < -1e-12):
        raise ValueError('Ordered nonnegative spans with positive middle/long spans required')
    result = np.zeros(len(a), bool)
    eligible = np.flatnonzero(~old)
    filling = a/(s[:,1]*s[:,2])
    lo, info = filling_split(filling[eligible], cfg)
    report = dict(status='no_new_exclusion', filling=filling.tolist(),
                  retained_indices=eligible.tolist(), filling_model=info,
                  new_indices=[], complete_area_ordering=False,
                  upper_size_exclusions=False, recursive_refitting=False)
    if not lo.any():
        return result, report
    low_ids, high_ids = eligible[lo], eligible[~lo]
    report.update(lower_shape_indices=low_ids.tolist(), upper_shape_indices=high_ids.tolist(),
                  lower_max_area=float(a[low_ids].max()), upper_min_area=float(a[high_ids].min()))
    # A numerical equality/tie is not evidence of a smaller population.
    epsilon = 64*np.finfo(float).eps*max(a.max(), np.finfo(float).tiny)
    if not a[low_ids].max() + epsilon < a[high_ids].min():
        report['status'] = 'keep_shape_groups_overlap_in_area'
        return result, report
    report['complete_area_ordering'] = True
    x = np.column_stack((.5*np.log(a), np.log(s[:,1]), np.log(s[:,2])))
    anchor = x[high_ids].mean(0)
    checks = []
    for i in low_ids:
        delta = anchor-x[i]
        row = model_error(delta, [.5,0.,1.])
        isotropic = model_error(delta, [1.,1.,1.])
        tie = 32*np.finfo(float).eps*max(1.,row,isotropic)
        result[i] = row > isotropic+tie
        checks.append(dict(index=int(i), row_error=row, isotropic_error=isotropic,
                           excluded=bool(result[i])))
    report.update(status='joint_shape_size_proposal', fusion_checks=checks,
                  new_indices=np.flatnonzero(result).tolist())
    return result, report

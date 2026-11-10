"""Capture-independent planning defaults shared by both planner adapters."""
import math

DEFAULT_CLEARANCE_M = 0.20


def effective_planning_config(config, profile_config=None):
    """Keep a saved profile's observation policy; use the current route clearance.

    Historical profiles are immutable input snapshots. Their old clearance must
    not silently override the release default. An explicit planning.clearance_m
    still permits reproducing a historical run. No capture ID selects this rule.
    """
    planning = config.get('planning', {})
    clearance = planning.get('clearance_m', DEFAULT_CLEARANCE_M)
    if isinstance(clearance, bool) or not isinstance(clearance, (int, float)) or not math.isfinite(clearance) or clearance <= 0:
        raise ValueError('planning.clearance_m must be a positive finite distance in metres')
    values = dict(planning if profile_config is None else profile_config)
    values['clearance_m'] = clearance
    return values

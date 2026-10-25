"""Conservative occupancy-event policy, independent of semantic fragment IDs.

Tracking and vacancy gates are frozen from image_object_lifecycle.py, not
fitted to the 45 visual judgments. The new surface quorum only assigns an audit
category; it never creates or accepts a change candidate. Surface consistency
cannot prove that a thin object such as a notice still exists on the wall.
"""
from math import hypot, sqrt

CONFIG = {
    'version': 'occupancy-evidence-v2',
    'minimum_own_frames': 3,
    'minimum_frame_span': 4,
    'minimum_large_views': 2,
    'large_min_side_px': 20,
    'large_min_area_px': 400,
    'maximum_confidence_gate': .6,
    'mean_confidence_gate': .45,
    'mean_confidence_min_frames': 4,
    'minimum_vacancy_views': 2,
    'minimum_free_fraction': .35,
    'maximum_occluded_fraction': .35,
    'minimum_finite_fraction': .8,
    'maximum_same_class_overlap': .2,
    'maximum_person_iou': .15,
    'minimum_frame_gap': 4,
    'minimum_camera_baseline_h': .08,
    'depth_error_multiplier': 1.5,
    'nearby_guard_h': .08,
    'minimum_counterpart_observations': 2,
    'surface_consistency_quorum': .8,
    'threshold_source': 'tracking/vacancy gates inherited from image_object_lifecycle.py; surface quorum is a new unfitted audit-only setting, not a candidate gate; no candidate-ID exceptions',
}


def supported_track(track, config=CONFIG):
    obs = track['observations']
    if not obs:
        return False
    confidence = max(o['confidence'] for o in obs) >= config['maximum_confidence_gate'] or (
        len(obs) >= config['mean_confidence_min_frames'] and
        sum(o['confidence'] for o in obs)/len(obs) >= config['mean_confidence_gate'])
    large = sum(min(o['bbox_unwrapped_px'][2]-o['bbox_unwrapped_px'][0], o['bbox_unwrapped_px'][3]-o['bbox_unwrapped_px'][1]) >= config['large_min_side_px']
                and o['area_px'] >= config['large_min_area_px'] for o in obs)
    return bool(len(obs) >= config['minimum_own_frames'] and
                max(o['frame'] for o in obs)-min(o['frame'] for o in obs) >= config['minimum_frame_span'] and
                confidence and large >= config['minimum_large_views'])


def vacancy_vote(view, config=CONFIG):
    return (view.get('free_fraction', 0) >= config['minimum_free_fraction'] and
            view.get('occluded_fraction', 1) <= config['maximum_occluded_fraction'] and
            view.get('finite_fraction', 0) >= config['minimum_finite_fraction'] and
            view.get('same_class_overlap', 1) < config['maximum_same_class_overlap'] and
            view.get('person_box_iou', 1) < config['maximum_person_iou'])


def independent_votes(views, height, config=CONFIG):
    accepted = []
    for v in sorted(views, key=lambda v: (-v.get('score', 0), v['frame'])):
        if not vacancy_vote(v, config) or not v.get('camera'):
            continue
        if all(abs(v['frame']-o['frame']) >= config['minimum_frame_gap'] and
               sqrt(sum((a-b)**2 for a, b in zip(v['camera'], o['camera']))) > config['minimum_camera_baseline_h']*height
               for o in accepted):
            accepted.append(v)
    return accepted


def nearby_counterparts(track, all_tracks, tolerance, height, config=CONFIG):
    result = []
    for other in all_tracks:
        if (other['epoch'] == track['epoch'] or other['zone'] != track['zone'] or
            other['class_name'] != track['class_name'] or len(other['observations']) < config['minimum_counterpart_observations']):
            continue
        distance = hypot(track['center'][0]-other['center'][0], track['center'][2]-other['center'][2])
        guard = tolerance + track['spread_u'] + other['spread_u'] + config['nearby_guard_h']*height
        if distance <= guard:
            result.append({'id': other['id'], 'distance_u': distance, 'guard_u': guard,
                           'supported': supported_track(other, config)})
    return sorted(result, key=lambda r: (r['distance_u'], r['id']))


def decide_track(track, all_tracks, error, height, paired_ids, config=CONFIG):
    votes = independent_votes(track.get('opposite_observations', []), height, config)
    nearby = nearby_counterparts(track, all_tracks, error['tolerance_u'], height, config)
    base = {'vacancy_frames': [v['frame'] for v in votes], 'nearby_counterparts': nearby,
            'physical_identity_confirmed': False, 'event_type_confirmed': False}
    if not supported_track(track, config):
        return dict(base, decision='deferred', reason_code='weak_observation')
    if track['id'] in paired_ids:
        return dict(base, decision='suppressed', reason_code='same_position_pair')
    if not error['supported']:
        return dict(base, decision='deferred', reason_code='unsupported_map_error')
    if nearby:
        return dict(base, decision='deferred', reason_code='possible_presence_including_weak_counterpart')
    if len(votes) < config['minimum_vacancy_views']:
        return dict(base, decision='deferred', reason_code='insufficient_independent_vacancy_views')
    return dict(base, decision='candidate', reason_code='repeated_presence_and_opposite_vacancy',
                event_alternatives=['possible_relocation', 'possible_added' if track['epoch'] else 'possible_removed'])


def event_groups(tracks, decisions):
    """One class/region review group can contain many fragments and triggers.

    This is presentation aggregation, not a claim of one physical event.
    """
    groups = {}
    for t in tracks:
        if decisions[t['id']]['decision'] != 'candidate':
            continue
        key = (t['zone'], t['class_name'])
        groups.setdefault(key, []).append(t['id'])
    return [(key, sorted(ids)) for key, ids in sorted(groups.items())]


def component_disposition(coverage, linked_event_ids, paired=False, config=CONFIG):
    if linked_event_ids:
        return {'decision': 'group_context', 'reason_code': 'fragment_fused_into_observed_group', 'event_ids': linked_event_ids}
    if paired:
        return {'decision': 'suppressed', 'reason_code': 'same_position_pair', 'event_ids': []}
    if coverage >= config['surface_consistency_quorum']:
        return {'decision': 'suppressed', 'reason_code': 'surface_present_label_or_shape_gap_not_event', 'event_ids': []}
    return {'decision': 'deferred', 'reason_code': 'no_independent_object_absence_evidence', 'event_ids': []}

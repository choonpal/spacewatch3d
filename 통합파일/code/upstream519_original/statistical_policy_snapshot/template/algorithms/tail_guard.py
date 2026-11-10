"""Combined candidate; same shared separation setting plus empty-gap fallback."""
from fusion_guard import lower_components as guarded
from statistical_modes import Settings


def lower_components(areas, extents, cfg=Settings()):
    return guarded(areas, extents, cfg, support_tail_gap=True)

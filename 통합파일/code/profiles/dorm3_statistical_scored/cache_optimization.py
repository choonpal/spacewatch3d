"""Finite-input memoization, static metadata and independent-pose parallelism."""
from functools import lru_cache


def install():
    from spacewatch3d.tasks.task_5b_keyframes.object_quality import ObjectQualityRefiner
    if not getattr(ObjectQualityRefiner,'_fast_uncapped_installed',False):
        original=ObjectQualityRefiner.__init__
        def initialize(self,*args,**kwargs):
            original(self,*args,**kwargs)
            self.sample=lru_cache(maxsize=max(64,4*len(self.triangles)))(self.sample.__wrapped__)
            self._score=lru_cache(maxsize=None)(self._score.__wrapped__)
        ObjectQualityRefiner.__init__=initialize
        ObjectQualityRefiner._fast_uncapped_installed=True
    from fast_execution import install_objective_metadata_cache,install_parallel_visibility
    install_objective_metadata_cache()
    install_parallel_visibility()

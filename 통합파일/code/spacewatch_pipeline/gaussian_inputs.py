"""Byte-bounded decoded views and two pinned staging buffers on one CUDA device."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np


class DecodedViews:
    def __init__(self, dataset, views, max_bytes):
        self.dataset = Path(dataset); self.views = views; self.max_bytes = max_bytes
        self.cache = OrderedDict(); self.bytes = self.hits = self.decodes = 0

    def get(self, index):
        if index in self.cache:
            self.hits += 1; self.cache.move_to_end(index)
            return self.cache[index]
        view = self.views[index]
        image = cv2.imread(str(self.dataset/view['image']))
        mask = cv2.imread(str(self.dataset/view['mask']), 0)
        if image is None or mask is None or image.shape[:2] != mask.shape:
            raise ValueError(f'Invalid training image/mask: {index}')
        pair = cv2.cvtColor(image, cv2.COLOR_BGR2RGB), mask > 254
        size = sum(a.nbytes for a in pair); self.decodes += 1
        while self.cache and self.bytes + size > self.max_bytes:
            _, old = self.cache.popitem(last=False); self.bytes -= sum(a.nbytes for a in old)
        if size <= self.max_bytes:
            self.cache[index] = pair; self.bytes += size
        return pair


class CudaViewLoader:
    def __init__(self, dataset, views, size, max_bytes):
        import torch
        self.torch = torch
        self.cache = DecodedViews(dataset, views, max_bytes)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='view-decode')
        self.stream = torch.cuda.Stream()
        self.buffers = [(torch.empty((size,size,3), dtype=torch.uint8, pin_memory=True),
                         torch.empty((size,size), dtype=torch.bool, pin_memory=True)) for _ in range(2)]
        self.events = [None, None]; self.slot = 0; self.pending = self.staged = None

    def prefetch(self, index):
        if self.pending is not None:
            self.pending[1].result()
        self.pending = index, self.executor.submit(self.cache.get, index)

    def stage_pending(self):
        if self.pending is None:
            return
        torch = self.torch
        index, future = self.pending; self.pending = None
        rgb, mask = future.result(); slot = self.slot; self.slot = 1-slot
        if self.events[slot] is not None:
            self.events[slot].synchronize()  # Host staging storage is safe to reuse.
        host_rgb, host_mask = self.buffers[slot]
        host_rgb.copy_(torch.from_numpy(rgb)); host_mask.copy_(torch.from_numpy(mask))
        with torch.cuda.stream(self.stream):
            target = host_rgb.to('cuda', non_blocking=True).float().div_(255)
            valid = host_mask.to('cuda', non_blocking=True)
            event = torch.cuda.Event(); event.record()
        self.events[slot] = event; self.staged = index, target, valid, event

    def get(self, index):
        if self.staged is None or self.staged[0] != index:
            self.prefetch(index); self.stage_pending()
        _, target, valid, event = self.staged; self.staged = None
        current = self.torch.cuda.current_stream(); current.wait_event(event)
        target.record_stream(current); valid.record_stream(current)
        return target, valid

    def close(self):
        self.executor.shutdown(wait=True)
        self.stream.synchronize()

    def statistics(self):
        return dict(decoded_views=self.cache.decodes, cache_hits=self.cache.hits,
                    resident_bytes=self.cache.bytes, limit_bytes=self.cache.max_bytes,
                    pinned_staging_buffers=2)

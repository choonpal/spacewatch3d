"""Bounded, per-model DINO features for overlapping inference windows only."""
from collections import OrderedDict
import torch


class FrameFeatureCache(torch.nn.Module):
    def __init__(self, encoder, max_bytes=512 * 1024**2):
        super().__init__()
        self.encoder = encoder
        self.max_bytes = max_bytes
        self.cache = OrderedDict()
        self.bytes = self.hits = self.misses = 0
        self.frame_keys = None

    def set_frames(self, keys):
        # Keys bind to verified input hashes and the preprocessing version.
        self.frame_keys = tuple(keys)

    def clear(self):
        self.cache.clear(); self.bytes = 0

    def forward(self, images):
        keys, self.frame_keys = self.frame_keys, None
        if self.training or torch.is_grad_enabled() or keys is None:
            self.clear()
            return self.encoder(images)
        if len(keys) != len(images):
            raise ValueError('DINO frame identity count differs from flattened batch')
        signature = (tuple(images.shape[1:]), images.dtype, images.device,
                     torch.is_autocast_enabled())
        keys = [(key, signature) for key in keys]
        found = {}; missing = []
        for i, key in enumerate(keys):
            if key in self.cache:
                found[key] = self.cache[key]; self.cache.move_to_end(key)
                self.hits += 1
            elif key not in found:
                found[key] = None; missing.append(i)
        if missing:
            tokens = self.encoder(images[missing])
            if isinstance(tokens, dict):
                tokens = tokens['x_norm_patchtokens']
            self.misses += len(missing)
            for i, token in zip(missing, tokens):
                # Clone each frame: a slice must not retain the entire encoder batch.
                value = token.detach().clone(); key = keys[i]
                found[key] = value
                size = value.numel() * value.element_size()
                while self.cache and self.bytes + size > self.max_bytes:
                    _, old = self.cache.popitem(last=False)
                    self.bytes -= old.numel() * old.element_size()
                if size <= self.max_bytes:
                    self.cache[key] = value; self.bytes += size
        return torch.stack([found[key] for key in keys])

    def statistics(self):
        return dict(hits=self.hits, encoded_frames=self.misses, resident_bytes=self.bytes,
                    limit_bytes=self.max_bytes)

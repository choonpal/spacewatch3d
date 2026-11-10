"""Strict official SimSaC checkpoint adapter; no Gaussian renderer dependencies."""
from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .core import sha


def local_correlation(tensorFirst, tensorSecond):
    _, _, h, w = tensorFirst.shape
    padded = F.pad(tensorSecond, (4, 4, 4, 4))
    return torch.stack([(tensorFirst * padded[:, :, dy:dy+h, dx:dx+w]).mean(1)
                        for dy in range(9) for dx in range(9)], dim=1)


class SimSaC:
    def __init__(self, source, checkpoint, device="cuda"):
        source = Path(source).resolve()
        sys.path.insert(0, str(source))
        module = types.ModuleType("models.correlation.correlation")
        module.FunctionCorrelation = local_correlation
        sys.modules[module.__name__] = module
        from models.our_models.SimSaC import SimSaC_Model
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device.index or 0)
        elif self.device.type == "cpu" and torch.cuda.is_available():
            raise ValueError("For CPU inference hide GPUs with CUDA_VISIBLE_DEVICES=''; upstream SimSaC allocates CUDA tensors when GPUs are exposed")
        self.model = SimSaC_Model(evaluation=False, div=1., batch_norm=True, pyramid_type="VGG",
                                 consensus_network=False, cyclic_consistency=True, dense_connection=True,
                                 decoder_inputs="corr_flow_feat", refinement_at_all_levels=False,
                                 refinement_at_adaptive_reso=True, num_class=2, use_pac=False,
                                 vpr_candidates=False)
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = loaded.get("state_dict", loaded.get("model", loaded))
        if all(k.startswith("module.") for k in state):
            state = {k[7:]: v for k, v in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.mean = torch.tensor([.485, .456, .406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([.229, .224, .225], device=self.device).view(1, 3, 1, 1)
        commit = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                                capture_output=True, text=True)
        self.audit = dict(model="SimSaC", source_commit=commit.stdout.strip() or None,
                          checkpoint_sha256=sha(checkpoint), loaded_strict=True,
                          source=str(source), checkpoint=str(checkpoint), device=str(self.device),
                          preprocessing="ImageNet RGB; 520x520 + 256x256 branches",
                          correlation="Native Torch zero-padded 9x9 channel mean",
                          input_source="Original captured RGB perspective views",
                          output_coordinates="First (target/query) image", gaussian_inputs_used=False)

    def prep(self, rgb):
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].to(self.device).float() / 255
        return tuple((F.interpolate(x, size=(s, s), mode="area") - self.mean) / self.std
                     for s in [520, 256])

    def predict(self, query, reference):
        full_q, small_q = self.prep(query)
        full_r, small_r = self.prep(reference)
        with torch.inference_mode():
            result = self.model(full_q, full_r, small_q, small_r)
            logits = F.interpolate(result["change"][1][-1], size=query.shape[:2],
                                   mode="bilinear", align_corners=False)
            prob = logits.softmax(1)[0, 1].float().cpu().numpy()
        if not np.isfinite(prob).all():
            raise ValueError("Non-finite SimSaC output")
        return prob

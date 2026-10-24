"""Windows build compatibility hooks for the local Nerfstudio environment."""

try:
    import torch
    import torch.utils.cpp_extension as cpp_extension

    # Codex/ConPTY can report an invalid OEM code page (``cp1``). CUDA extension
    # discovery only needs ASCII version numbers, so UTF-8 with replacement is
    # safe for compiler diagnostics and avoids failing before compilation starts.
    cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")

    # Nerfstudio 1.1.5 predates PyTorch 2.6's weights_only=True default. Its
    # locally generated checkpoints include NumPy scalar metadata, so retain
    # the legacy behavior when callers do not specify weights_only explicitly.
    _torch_load = torch.load

    def _nerfstudio_compatible_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _torch_load(*args, **kwargs)

    torch.load = _nerfstudio_compatible_load
except Exception:
    pass

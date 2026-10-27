# Shared mesh / SimSaC components

`core.py` and `model.py` are reusable components from the original mesh SimSaC
implementation. `structural.py` carries the robust regional ICP and structural
residual functions from `tools/registration_benchmark/refine_tail.py` and
`compare_objects.py`. Capture-specific launchers and saved outcomes are excluded.

The fresh mesh + Gaussian pipeline is documented in
[`../gaussian_change/README.md`](../gaussian_change/README.md). SimSaC expects
RGB images, so that pipeline renders RGB from the corrected Gaussian maps and
records that input provenance explicitly. It does not feed mesh vertices or
Gaussian parameters directly into the SimSaC network.

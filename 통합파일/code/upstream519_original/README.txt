Original upstream519 preparation component

Public API
  from spacewatch_pipeline.upstream519 import run
  result = run(request_dict, output_dir, gpu=None)
The result contains source_run, coordinates, semantic_arrays, vocabulary,
semantic_frame_id, semantic_coordinates_sha256, planning_profile{path,sha256},
captured_prediction{predictions,inference{path,bytes,sha256},vocabulary{...}},
and execution/provenance receipts. Main binds these to the native capture DAG.
planning_scene is diagnostic output; native capture planning should use the
new planning_profile and must not route through a saved virtual plan.

One-capture CLI, from code directory
  python3 -B -m spacewatch_pipeline.upstream519 --prepare-request \
    upstream519_original/requests/dorm4-fresh-request.json \
    --output ../runs/unique-upstream-dorm4
Append --check-only and use an output JSON filename for non-NN validation.
Dorm5 and explicit verified-reuse requests are in the same requests directory.
Main's prepare-and-run entry point can call both captures then the native DAG.

Fresh raw path
1. Original prepare_new.extract: every15th captured frame,1036x518 ERP,JPEG95.
2. Original v2 OOOPS implementation, existing strict-loaded checkpoint and
   current explicitly supplied vocabulary; no replacement model or API.
3. Original prepare_new.masks: temporal +/-2 person union; capture-wide person
   envelope>=10% with original lower/outer ROI;21x31 periodic dilation; original
   rows492: nadir exclusion. The original source's TWO label==16 comparisons
   are bound to the vocabulary's person ID by name. Current19class person15
   and umbrella16 are tested explicitly. Original source stays byte-identical;
   derived AST/source hashes and exact comparison edits are recorded per stage.
4. Exact actual Dorm4/5 V5 Pano2 runner and its internal component filtering.
   This differs from the clone's v2 prepare_inputs only by absence of an extra
   optional semantic-mask append. No extra early_semantics rescue is inserted.
5. Provided checksummed coordinates OR original prepare_new.coordinates using
   floor1/ceiling2, RANSAC floor, assumed2.7m ceiling gap. Metric scale remains
   explicitly unverified. Floor/ceiling IDs and source shapes fail closed.
6. Original v2 geometry.reconstruct validation, Poisson mesh, semantic voting,
   and planning.scene geometry adapter. No virtual planning or Gaussian stage.
7. Unchanged original prepare_map_profile.py/statistical_adapter.py. New mesh,
   scene and reconstruction are bridged with ONLY the old context plan's camera
   height/gridorigin/fallbacktileorigin; old selected views and V/Q are absent.
   The exact accepted statistical algorithms, manual/prompt and class-score
   cache are hash-bound; existing class scores are remapped by original script.
   Missing class scores retain its declared default, with no new LLM/API call.
8. captured_prediction receipt binds actual OOOPS prediction NPZ SHA256 values
   to the new source frame manifest/RGB SHA values and original checkpoint,
   vocabulary and strict-load completion receipt. No probabilities are invented.

Prepared-bundle mode consumes a checksummed existing RGB/base-mask assets bundle
instead of re-extracting/remasking. Fresh inference is still explicit. Neither
mode claims a universal mask policy or automatically verified physical scale.

Verified reuse
Reused profiles, semantic NPZs, vocabulary, coordinates, reconstruction manifest,
poses/status/config and optional prediction lineage have explicit expected SHA
records. Semantic array digests are independently streamed and compared with
the original profile input_binding. This is labelled artifact/profile reuse,
not fresh NN execution or strict originalv2 stage replay. inputs19_02 provenance
is retained; its bridge alone cannot establish exact originalv2 producer origin.

Resources and current validation limits
No cold NN, Pano reconstruction, Poisson or full semantic projection was executed
while implementing this component. Current~6GiB available RAM is below default
16GiB cold-run guard, so cold execution is explicitly blocked/unverified.
Subprocesses inherit one-CPU affinity, one-thread environment and finite stage
timeouts. An existing cooperative GPU lease row can be passed to run(); without
one, only infer/Pano acquire the existing per-conversation GPU lease. Installed
model files are required; Python network access is denied in producer children.
No installation, model download, new API call or original workspace mutation.

Eight small contract tests pass, including a real synthetic mask execution,
class-ID disambiguation, source hash mutation rejection, streamed array digest
parity, sanitized baseline context, memory guard and native prediction import.
Raw request preflight validates actual
Dorm4/5 source hashes, ERP metadata, current19vocabulary IDs and runtime/source
bindings. These checks do not establish successful cold execution.

External accepted dependencies
The unchanged original profile script still reads the accepted external release
/workspace/spacewatch_pipeline_v5_1 template/config/class-score files. Exact
source/policy snapshots are packaged and external paths are checked against
their hashes before profile execution. Installed runtime/model repositories and
the explicit original context plan are also external, recorded dependencies.
If any are absent or changed, fail rather than substitute another policy.

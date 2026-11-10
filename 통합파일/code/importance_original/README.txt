Original importance producer — source provenance, not the default execution path

The spacewatch_pipeline/ directory here preserves exact original source bytes
from /workspace/spacewatch_objects19_v7_20261027_52i774iz/code/spacewatch_pipeline.
It is isolated under importance_original and is not imported by the current DAG.

The current program defaults to spacewatch_pipeline.paper_importance.importance:
it validates the manual, requested classes, requested model and saved five-run
response identities, reconstructs class means, and maps mean/5 to current scene
object IDs. It does not call an API and does not generate new model responses.
The self-contained saved importance records are in ../../inputs/importance.

Actual 1–5 x 5 producer:
  spacewatch_pipeline/patrol_rating.py
  VERSION = patrol-importance-manual-1to5-v1
  requested model = gemini-3.1-flash-lite
  temperature = 0; five independent recorded responses per class; score=mean/5

Its exact inline PROMPT and SCHEMA are additionally extracted without execution
as prompts/manual_1to5_prompt.txt and prompts/manual_1to5_schema.json. The prompt
file contains the literal template, without an added trailing newline. Its hash
must equal the preserved report's prompt_sha256. The source manifest also checks
that this prompt/schema reconstruct all 14 saved request identity hashes.

spacewatch_pipeline/importance_stage.py is the original stage/transport adapter.
spacewatch_pipeline/patrol_classifier.py supplies its original eligibility and
manual preprocessing functions; io.py and __init__.py preserve local helpers.
api_credentials.py is also the exact original source helper. It contains code
for reading an environment variable or an owner-only private credential file,
plus an interactive configure function; no actual credential value is embedded.
spacewatch_pipeline/importance_prompt.txt is the older simple-v2 classification
prompt used by a different branch. It is retained as an exact source dependency;
it is not the manual 1–5 prompt and is not substituted into the current method.

Current authorization permits cache validation only. A different manual/class
list cannot silently reuse this cache. Creating a new cache with the preserved
original producer requires new explicit API authority and a separately prepared
safe environment, including deliberately provisioned credentials and a new
output directory. No key value, .env, or credential store is included. The
original producer and its source dependencies are preserved and importable;
externally provisioned credentials and fresh explicit API authority are still
required for live calls. No automatic live-API fallback is wired into the DAG.

The publication label "Gemini 3 Flash-Lite" differs from the preserved exact
requested ID gemini-3.1-flash-lite. The original transport kept parsed response
content and usageMetadata but did not retain provider modelVersion. Neither
the default adapter nor this source collection relabels those records.

This collection performed source copying, AST literal extraction, hashing,
offline module import, request preparation and saved-report validation only.
It did not invoke api_key(), configure(), run(), or transport(), contact a model
service, copy credential values, or include human response files.

See source_manifest.json for exact producer/derived-file hashes and the closure
checks. See LICENSE_CONTEXT.txt for the license evidence available in this
producing snapshot.

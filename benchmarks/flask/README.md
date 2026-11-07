# Pinned Flask questions, version 2

This is a corrected copy of the 18 original seminar questions. Question wording
and row order are preserved from [`original-validation.csv`](original-validation.csv).
The original question CSV is retained unchanged. The earlier experiment code,
result CSVs and plots are preserved in the archive branch.

The target is Flask commit
[`49943b5a94413a530761350f04e896f76e8d620e`](https://github.com/pallets/flask/tree/49943b5a94413a530761350f04e896f76e8d620e),
the `main` revision obtained by pulling the existing checkout for this update.
This is a pinned development revision, not a claim about the latest release.
Do not pull again during a benchmark run. Use a separate checkout of this commit
if your normal Flask checkout has moved on.

## Files and checks

- `validation.csv`: questions with matching lists of immutable source URLs and
  inclusive, one-based start/end lines. All paths include `src/` and are relative
  to the Flask repository root, matching the MCP indexer's metadata.
- `manifest.json`: dataset ID, revision, metric version, CSV checksum, historical
  CSV checksum and hashes of all 83 Python files in the indexed corpus. This
  includes tests and examples, using the indexer's normal exclusions.
- `benchmarks/dataset.py`: `load_benchmark` validates the checksums,
  corpus, URL revisions, list lengths and line bounds before evaluation. No
  GitHub downloads are needed. The agent runner checks corpus hashes again before
  and after each condition.

The content hashes identify the Python corpus even when it is supplied as a
source archive without Git metadata. Non-Python files are outside these checks.

## Label review

Ranges were inspected against this checkout. They are selected evidence for
retrieval evaluation, not exhaustive answer rubrics or proof that the agent
understands the code. Class-introduction ranges deliberately avoid requiring
retrieval of an entire class. Several questions require multiple snippets.

| Question | Review and correction |
|---|---|
| 1 | Relocated `send_from_directory` to its current function range. |
| 2 | Relocated the CLI `--host` option. |
| 3 | Checked `jsonify` and `DefaultJSONProvider`; retained their relevant ranges. |
| 4 | Relocated the helper and application redirect implementations. |
| 5 | Retained the concrete Blueprint constructor and added the sans-I/O class description explaining grouped routes. |
| 6 | Checked Response defaults and relocated `Flask.make_response`. Setting inherited response properties may require Werkzeug context outside this corpus. |
| 7 | Replaced route-registration code with `routes_command` using `url_map.iter_rules`; retained URL-map context. |
| 8 | Removed unrelated `stream_with_context` and `url_for` documentation plus a duplicate. Labels now cover `route` and the three relevant `add_url_rule` implementations. |
| 9 | Relocated `Flask.run` and `get_debug_flag`. |
| 10 | Checked the `Request.max_content_length` getter and configuration documentation. |
| 11 | Checked the concrete Blueprint, sans-I/O Blueprint and Scaffold inheritance declarations. |
| 12 | Checked AppGroup and FlaskGroup class descriptions. External Click behavior is outside the corpus. |
| 13 | Removed the repeated App range and repaired the missing end-line entry; retained Scaffold and its direct subclasses. |
| 14 | Interpreted the ambiguous wording as attributes defined or overridden by Flask itself: `default_config`, `request_class`, `response_class`, `session_interface` and instance `cli`. Replaced a partial constructor range with this evidence. |
| 15 | Relocated the helper and application response-conversion functions. |
| 16 | Relocated the route decorator and registration methods; removed unrelated `View` documentation. |
| 17 | Labelled the default values, `App.make_config` and its constructor call instead of only the config-class description. |
| 18 | Replaced unrelated View-instance lifecycle text. In this revision request creation is in `AppContext.from_environ`; labels also cover `Flask.request_context` and its WSGI call site. |

These are manually reviewed seed labels, not an independently adjudicated
benchmark. A second review and broader task coverage would strengthen them.

## Validate the local checkout

From the Codeaware project root, with its virtual environment active:

```python
from pathlib import Path
from benchmarks.dataset import load_benchmark

repository = Path("/absolute/path/to/flask")
data = load_benchmark("benchmarks/flask", repository)
print(data["metadata"])
print(f"Validated {len(data['questions'])} questions")
```

Added, deleted or edited Python files cause a mismatch. Adjusting labels also
requires an intentional new manifest/dataset version; do not silently reuse the
existing identity for different inputs.

## Run the current agent evaluation

Build the Neo4j index from this exact corpus using the main README. Then follow
[the paired agent benchmark instructions](../README.md).
Record new results in a new directory. Saved historical measurements predate
the Neo4j-only migration and must not be presented as measurements of this code.

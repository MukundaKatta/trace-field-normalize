# trace-field-normalize

Normalize inconsistent field names in agent trace JSONL events. Zero dependencies.

Different agent frameworks use different names for the same semantic fields.
`trace-field-normalize` maps known variants to canonical names so your
downstream tooling sees a consistent schema.

## Install

```bash
pip install trace-field-normalize
```

## Usage

```python
from trace_field_normalize import normalize_event, normalize_events, normalize_file

# Single event
event = {"input_tokens": 42, "latency_ms": 350, "model_id": "claude-sonnet-4-5"}
normalized = normalize_event(event)
# {"tokens_in": 42, "duration_ms": 350, "model": "claude-sonnet-4-5"}

# List of events
events = normalize_events([...])

# JSONL file
events = normalize_file("traces.jsonl", "normalized.jsonl")
```

## Knowing what changed

Use `normalize_event_verbose` when you want to log or count how much
normalization happened. It returns a `NormalizeResult` with the normalized
`event` and a `renamed` map of `{old_name: canonical_name}`.

```python
from trace_field_normalize import normalize_event_verbose

result = normalize_event_verbose({"input_tokens": 42, "latency_ms": 350})
result.event        # {"tokens_in": 42, "duration_ms": 350}
result.renamed      # {"input_tokens": "tokens_in", "latency_ms": "duration_ms"}
result.rename_count # 2
```

A field where the canonical name already exists (canonical wins) is *not*
reported as a rename.

## Default field map

| Canonical     | Variants accepted |
|---------------|-------------------|
| `kind`        | `event_type`, `type`, `event_kind` |
| `name`        | `step`, `tool`, `tool_name`, `function_name` |
| `tokens_in`   | `input_tokens`, `prompt_tokens`, `tokens_prompt` |
| `tokens_out`  | `output_tokens`, `completion_tokens`, `tokens_completion` |
| `cost_usd`    | `cost`, `price_usd`, `usd`, `price` |
| `duration_ms` | `latency_ms`, `elapsed_ms`, `duration`, `latency` |
| `error`       | `err`, `exception`, `error_message` |
| `model`       | `model_id`, `model_name` |
| `lane`        | `worker`, `agent_id`, `thread` |
| `timestamp`   | `ts`, `time`, `created_at`, `event_time` |

## Custom field maps

```python
from trace_field_normalize import FieldMap, normalize_event

fm = FieldMap({"score": ["rating", "confidence"]}, include_defaults=True)
event = {"rating": 0.95}
normalize_event(event, fm)  # {"score": 0.95}
```

## Canonical wins

If the canonical name is already present, variants are left unchanged.

```python
event = {"tokens_in": 99, "input_tokens": 42}
normalize_event(event)  # {"tokens_in": 99, "input_tokens": 42}
```

## keep_original

```python
normalize_event({"ts": 100}, keep_original=True)
# {"timestamp": 100, "ts": 100}
```

## API reference

| Object | Description |
|--------|-------------|
| `normalize_event(event, field_map=None, *, keep_original=False)` | Return a new dict with variant keys renamed to canonical names. The input is never mutated. |
| `normalize_event_verbose(event, field_map=None, *, keep_original=False)` | Like `normalize_event` but returns a `NormalizeResult` reporting which keys were renamed. |
| `normalize_events(events, field_map=None, *, keep_original=False)` | Apply `normalize_event` to a list, returning a new list. |
| `normalize_file(source, dest=None, field_map=None, *, keep_original=False)` | Read a JSONL file, normalize each object, and optionally write the result to `dest`. Returns the list of normalized dicts. |
| `FieldMap(mapping=None, *, include_defaults=True)` | Mapping of canonical names to variant lists. Use `.add(canonical, variants)` and `.get(canonical)`. |
| `NormalizeResult(event, renamed)` | Result object from `normalize_event_verbose`; exposes `.event`, `.renamed`, and `.rename_count`. |

### Errors

`normalize_file` raises:

- `FileNotFoundError` if `source` does not exist.
- `ValueError` if a non-blank line is not valid JSON, or decodes to something
  other than a JSON object (an array, number, string, etc.). The message
  includes the 1-based line number so malformed traces are easy to locate.

## Zero dependencies

Standard library only: `json`, `dataclasses`, `pathlib`. Nothing else.

## Development

The test suite uses only the Python standard library (`unittest`), so no
third-party packages are required to run it:

```bash
python3 -m unittest discover -s tests -v
```

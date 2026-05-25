"""Tests for trace-field-normalize."""

import json
import tempfile
from pathlib import Path

import pytest

from trace_field_normalize import FieldMap, NormalizeResult, normalize_event, normalize_events, normalize_file


# ---------------------------------------------------------------------------
# normalize_event — basic renaming
# ---------------------------------------------------------------------------

def test_renames_tokens_in():
    event = {"input_tokens": 42, "some_field": "x"}
    result = normalize_event(event)
    assert result["tokens_in"] == 42
    assert "input_tokens" not in result

def test_renames_tokens_out():
    event = {"completion_tokens": 10}
    result = normalize_event(event)
    assert result["tokens_out"] == 10

def test_renames_cost_usd():
    event = {"cost": 0.001}
    result = normalize_event(event)
    assert result["cost_usd"] == 0.001

def test_renames_duration_ms():
    event = {"latency_ms": 350}
    result = normalize_event(event)
    assert result["duration_ms"] == 350

def test_renames_kind():
    event = {"event_type": "llm_call"}
    result = normalize_event(event)
    assert result["kind"] == "llm_call"

def test_renames_name():
    event = {"tool_name": "web_search"}
    result = normalize_event(event)
    assert result["name"] == "web_search"

def test_renames_error():
    event = {"err": "timeout"}
    result = normalize_event(event)
    assert result["error"] == "timeout"

def test_renames_model():
    event = {"model_id": "claude-sonnet-4-5"}
    result = normalize_event(event)
    assert result["model"] == "claude-sonnet-4-5"

def test_renames_lane():
    event = {"agent_id": "worker-3"}
    result = normalize_event(event)
    assert result["lane"] == "worker-3"

def test_renames_timestamp():
    event = {"ts": 1700000000}
    result = normalize_event(event)
    assert result["timestamp"] == 1700000000

def test_renames_multiple_fields():
    event = {
        "input_tokens": 100,
        "completion_tokens": 50,
        "latency_ms": 400,
        "model_id": "claude-sonnet-4-5",
    }
    result = normalize_event(event)
    assert result["tokens_in"] == 100
    assert result["tokens_out"] == 50
    assert result["duration_ms"] == 400
    assert result["model"] == "claude-sonnet-4-5"

# ---------------------------------------------------------------------------
# canonical already present — no rename
# ---------------------------------------------------------------------------

def test_canonical_wins():
    event = {"tokens_in": 99, "input_tokens": 42}
    result = normalize_event(event)
    assert result["tokens_in"] == 99
    assert "input_tokens" in result  # not consumed

def test_canonical_already_present_skips_all_variants():
    event = {"kind": "existing", "event_type": "other", "type": "third"}
    result = normalize_event(event)
    assert result["kind"] == "existing"

# ---------------------------------------------------------------------------
# unknown / non-canonical fields preserved
# ---------------------------------------------------------------------------

def test_unknown_fields_preserved():
    event = {"foo": "bar", "baz": 123}
    result = normalize_event(event)
    assert result["foo"] == "bar"
    assert result["baz"] == 123

def test_no_variants_present():
    event = {"some_unrelated": "value"}
    result = normalize_event(event)
    assert result == {"some_unrelated": "value"}

def test_empty_event():
    assert normalize_event({}) == {}

# ---------------------------------------------------------------------------
# keep_original flag
# ---------------------------------------------------------------------------

def test_keep_original_preserves_old_key():
    event = {"input_tokens": 42}
    result = normalize_event(event, keep_original=True)
    assert result["tokens_in"] == 42
    assert result["input_tokens"] == 42

def test_keep_original_false_removes_old_key():
    event = {"input_tokens": 42}
    result = normalize_event(event, keep_original=False)
    assert "input_tokens" not in result

def test_keep_original_with_multiple_fields():
    event = {"ts": 1, "model_id": "x"}
    result = normalize_event(event, keep_original=True)
    assert result["timestamp"] == 1
    assert result["ts"] == 1
    assert result["model"] == "x"
    assert result["model_id"] == "x"

# ---------------------------------------------------------------------------
# only first matching variant is renamed
# ---------------------------------------------------------------------------

def test_only_first_variant_renamed():
    # Both prompt_tokens and input_tokens are variants of tokens_in
    event = {"prompt_tokens": 5, "input_tokens": 10}
    result = normalize_event(event)
    assert "tokens_in" in result
    # One variant is renamed; other might still be present since canonical is now set
    # Either way, canonical has a value
    assert result["tokens_in"] in (5, 10)

# ---------------------------------------------------------------------------
# does not modify original event
# ---------------------------------------------------------------------------

def test_does_not_mutate_original():
    event = {"input_tokens": 42, "foo": "bar"}
    original = dict(event)
    normalize_event(event)
    assert event == original

# ---------------------------------------------------------------------------
# normalize_events
# ---------------------------------------------------------------------------

def test_normalize_events_empty():
    assert normalize_events([]) == []

def test_normalize_events_single():
    events = [{"input_tokens": 5}]
    result = normalize_events(events)
    assert result == [{"tokens_in": 5}]

def test_normalize_events_multiple():
    events = [
        {"input_tokens": 10},
        {"completion_tokens": 20},
        {"latency_ms": 300},
    ]
    result = normalize_events(events)
    assert result[0]["tokens_in"] == 10
    assert result[1]["tokens_out"] == 20
    assert result[2]["duration_ms"] == 300

def test_normalize_events_returns_new_list():
    events = [{"input_tokens": 1}]
    result = normalize_events(events)
    assert result is not events

def test_normalize_events_does_not_mutate_originals():
    events = [{"input_tokens": 1}]
    normalize_events(events)
    assert events[0] == {"input_tokens": 1}

# ---------------------------------------------------------------------------
# normalize_file
# ---------------------------------------------------------------------------

def test_normalize_file_basic():
    events = [{"input_tokens": 5}, {"latency_ms": 100}]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
        path = f.name
    result = normalize_file(path)
    assert result[0]["tokens_in"] == 5
    assert result[1]["duration_ms"] == 100

def test_normalize_file_with_dest():
    events = [{"input_tokens": 7}]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(events[0]) + "\n")
        src = f.name
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        dst = f.name
    normalize_file(src, dst)
    written = [json.loads(line) for line in Path(dst).read_text().splitlines() if line.strip()]
    assert written[0]["tokens_in"] == 7

def test_normalize_file_skips_blank_lines():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"input_tokens": 1}) + "\n")
        f.write("\n")
        f.write(json.dumps({"latency_ms": 200}) + "\n")
        path = f.name
    result = normalize_file(path)
    assert len(result) == 2

def test_normalize_file_returns_list():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"ts": 100}) + "\n")
        path = f.name
    result = normalize_file(path)
    assert isinstance(result, list)
    assert result[0]["timestamp"] == 100

# ---------------------------------------------------------------------------
# FieldMap
# ---------------------------------------------------------------------------

def test_fieldmap_defaults():
    fm = FieldMap()
    assert len(fm) == 10  # 10 canonical names

def test_fieldmap_no_defaults():
    fm = FieldMap(include_defaults=False)
    assert len(fm) == 0

def test_fieldmap_custom_mapping():
    fm = FieldMap({"my_field": ["variant_a", "variant_b"]}, include_defaults=False)
    event = {"variant_a": "hello"}
    result = normalize_event(event, fm)
    assert result["my_field"] == "hello"

def test_fieldmap_overrides_default():
    fm = FieldMap({"tokens_in": ["my_custom_variant"]})
    event = {"my_custom_variant": 99}
    result = normalize_event(event, fm)
    assert result["tokens_in"] == 99

def test_fieldmap_add():
    fm = FieldMap(include_defaults=False)
    fm.add("score", ["rating", "value"])
    event = {"rating": 0.95}
    result = normalize_event(event, fm)
    assert result["score"] == 0.95

def test_fieldmap_get():
    fm = FieldMap(include_defaults=False)
    fm.add("x", ["a", "b"])
    assert fm.get("x") == ["a", "b"]
    assert fm.get("unknown") == []

def test_fieldmap_len():
    fm = FieldMap(include_defaults=False)
    fm.add("a", ["x"])
    fm.add("b", ["y"])
    assert len(fm) == 2

# ---------------------------------------------------------------------------
# NormalizeResult
# ---------------------------------------------------------------------------

def test_normalize_result_rename_count():
    nr = NormalizeResult(event={"tokens_in": 5}, renamed={"input_tokens": "tokens_in"})
    assert nr.rename_count == 1

def test_normalize_result_zero_renames():
    nr = NormalizeResult(event={}, renamed={})
    assert nr.rename_count == 0

# ---------------------------------------------------------------------------
# variant priority — first variant in list wins
# ---------------------------------------------------------------------------

def test_first_variant_in_list_wins():
    # "step" is listed before "tool" for "name"
    event = {"step": "call_llm", "tool": "search"}
    result = normalize_event(event)
    assert result["name"] == "call_llm"

def test_alternative_timestamp_variants():
    for variant in ("ts", "time", "created_at", "event_time"):
        event = {variant: 12345}
        result = normalize_event(event)
        assert result["timestamp"] == 12345, f"Failed for variant: {variant}"

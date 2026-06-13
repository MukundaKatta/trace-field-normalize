"""Tests for trace-field-normalize.

Uses only the Python standard library (``unittest``) so the suite runs with
no third-party dependencies::

    python3 -m unittest discover -s tests
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Allow running the suite from a source checkout without installing the
# package: make ``src/`` importable when the package is not already on the path.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import trace_field_normalize
from trace_field_normalize import (
    FieldMap,
    NormalizeResult,
    normalize_event,
    normalize_event_verbose,
    normalize_events,
    normalize_file,
)


class TestNormalizeEventBasicRenaming(unittest.TestCase):
    def test_renames_tokens_in(self):
        result = normalize_event({"input_tokens": 42, "some_field": "x"})
        self.assertEqual(result["tokens_in"], 42)
        self.assertNotIn("input_tokens", result)

    def test_renames_tokens_out(self):
        self.assertEqual(normalize_event({"completion_tokens": 10})["tokens_out"], 10)

    def test_renames_cost_usd(self):
        self.assertEqual(normalize_event({"cost": 0.001})["cost_usd"], 0.001)

    def test_renames_duration_ms(self):
        self.assertEqual(normalize_event({"latency_ms": 350})["duration_ms"], 350)

    def test_renames_kind(self):
        self.assertEqual(normalize_event({"event_type": "llm_call"})["kind"], "llm_call")

    def test_renames_name(self):
        self.assertEqual(normalize_event({"tool_name": "web_search"})["name"], "web_search")

    def test_renames_error(self):
        self.assertEqual(normalize_event({"err": "timeout"})["error"], "timeout")

    def test_renames_model(self):
        self.assertEqual(
            normalize_event({"model_id": "claude-sonnet-4-5"})["model"],
            "claude-sonnet-4-5",
        )

    def test_renames_lane(self):
        self.assertEqual(normalize_event({"agent_id": "worker-3"})["lane"], "worker-3")

    def test_renames_timestamp(self):
        self.assertEqual(normalize_event({"ts": 1700000000})["timestamp"], 1700000000)

    def test_renames_multiple_fields(self):
        event = {
            "input_tokens": 100,
            "completion_tokens": 50,
            "latency_ms": 400,
            "model_id": "claude-sonnet-4-5",
        }
        result = normalize_event(event)
        self.assertEqual(result["tokens_in"], 100)
        self.assertEqual(result["tokens_out"], 50)
        self.assertEqual(result["duration_ms"], 400)
        self.assertEqual(result["model"], "claude-sonnet-4-5")


class TestCanonicalWins(unittest.TestCase):
    def test_canonical_wins(self):
        result = normalize_event({"tokens_in": 99, "input_tokens": 42})
        self.assertEqual(result["tokens_in"], 99)
        self.assertIn("input_tokens", result)  # not consumed

    def test_canonical_already_present_skips_all_variants(self):
        event = {"kind": "existing", "event_type": "other", "type": "third"}
        self.assertEqual(normalize_event(event)["kind"], "existing")


class TestUnknownFieldsPreserved(unittest.TestCase):
    def test_unknown_fields_preserved(self):
        result = normalize_event({"foo": "bar", "baz": 123})
        self.assertEqual(result["foo"], "bar")
        self.assertEqual(result["baz"], 123)

    def test_no_variants_present(self):
        self.assertEqual(normalize_event({"some_unrelated": "value"}), {"some_unrelated": "value"})

    def test_empty_event(self):
        self.assertEqual(normalize_event({}), {})


class TestKeepOriginal(unittest.TestCase):
    def test_keep_original_preserves_old_key(self):
        result = normalize_event({"input_tokens": 42}, keep_original=True)
        self.assertEqual(result["tokens_in"], 42)
        self.assertEqual(result["input_tokens"], 42)

    def test_keep_original_false_removes_old_key(self):
        result = normalize_event({"input_tokens": 42}, keep_original=False)
        self.assertNotIn("input_tokens", result)

    def test_keep_original_with_multiple_fields(self):
        result = normalize_event({"ts": 1, "model_id": "x"}, keep_original=True)
        self.assertEqual(result["timestamp"], 1)
        self.assertEqual(result["ts"], 1)
        self.assertEqual(result["model"], "x")
        self.assertEqual(result["model_id"], "x")


class TestVariantSelection(unittest.TestCase):
    def test_only_first_variant_renamed(self):
        # Both prompt_tokens and input_tokens are variants of tokens_in.
        result = normalize_event({"prompt_tokens": 5, "input_tokens": 10})
        self.assertIn("tokens_in", result)
        self.assertIn(result["tokens_in"], (5, 10))

    def test_first_variant_in_list_wins(self):
        # "step" is listed before "tool" for "name".
        result = normalize_event({"step": "call_llm", "tool": "search"})
        self.assertEqual(result["name"], "call_llm")

    def test_alternative_timestamp_variants(self):
        for variant in ("ts", "time", "created_at", "event_time"):
            with self.subTest(variant=variant):
                self.assertEqual(normalize_event({variant: 12345})["timestamp"], 12345)


class TestNoMutation(unittest.TestCase):
    def test_does_not_mutate_original(self):
        event = {"input_tokens": 42, "foo": "bar"}
        original = dict(event)
        normalize_event(event)
        self.assertEqual(event, original)


class TestNormalizeEvents(unittest.TestCase):
    def test_normalize_events_empty(self):
        self.assertEqual(normalize_events([]), [])

    def test_normalize_events_single(self):
        self.assertEqual(normalize_events([{"input_tokens": 5}]), [{"tokens_in": 5}])

    def test_normalize_events_multiple(self):
        events = [{"input_tokens": 10}, {"completion_tokens": 20}, {"latency_ms": 300}]
        result = normalize_events(events)
        self.assertEqual(result[0]["tokens_in"], 10)
        self.assertEqual(result[1]["tokens_out"], 20)
        self.assertEqual(result[2]["duration_ms"], 300)

    def test_normalize_events_returns_new_list(self):
        events = [{"input_tokens": 1}]
        self.assertIsNot(normalize_events(events), events)

    def test_normalize_events_does_not_mutate_originals(self):
        events = [{"input_tokens": 1}]
        normalize_events(events)
        self.assertEqual(events[0], {"input_tokens": 1})


class TestNormalizeFile(unittest.TestCase):
    def _write_jsonl(self, lines: list[str]) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        self.addCleanup(lambda: os.path.exists(f.name) and os.unlink(f.name))
        f.write("\n".join(lines) + "\n")
        f.close()
        return f.name

    def test_normalize_file_basic(self):
        path = self._write_jsonl([json.dumps({"input_tokens": 5}), json.dumps({"latency_ms": 100})])
        result = normalize_file(path)
        self.assertEqual(result[0]["tokens_in"], 5)
        self.assertEqual(result[1]["duration_ms"], 100)

    def test_normalize_file_with_dest(self):
        src = self._write_jsonl([json.dumps({"input_tokens": 7})])
        dst_fd, dst = tempfile.mkstemp(suffix=".jsonl")
        os.close(dst_fd)
        self.addCleanup(lambda: os.path.exists(dst) and os.unlink(dst))
        normalize_file(src, dst)
        written = [json.loads(ln) for ln in Path(dst).read_text().splitlines() if ln.strip()]
        self.assertEqual(written[0]["tokens_in"], 7)

    def test_normalize_file_skips_blank_lines(self):
        path = self._write_jsonl([json.dumps({"input_tokens": 1}), "", json.dumps({"latency_ms": 200})])
        self.assertEqual(len(normalize_file(path)), 2)

    def test_normalize_file_returns_list(self):
        path = self._write_jsonl([json.dumps({"ts": 100})])
        result = normalize_file(path)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0]["timestamp"], 100)

    def test_normalize_file_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            normalize_file("/tmp/trace-field-normalize-does-not-exist.jsonl")

    def test_normalize_file_non_object_line_raises_valueerror(self):
        path = self._write_jsonl([json.dumps({"ts": 1}), "[1, 2, 3]"])
        with self.assertRaises(ValueError) as ctx:
            normalize_file(path)
        self.assertIn("line 2", str(ctx.exception))

    def test_normalize_file_invalid_json_raises_valueerror(self):
        path = self._write_jsonl([json.dumps({"ts": 1}), "{not valid json"])
        with self.assertRaises(ValueError) as ctx:
            normalize_file(path)
        self.assertIn("line 2", str(ctx.exception))


class TestFieldMap(unittest.TestCase):
    def test_fieldmap_defaults(self):
        self.assertEqual(len(FieldMap()), 10)

    def test_fieldmap_no_defaults(self):
        self.assertEqual(len(FieldMap(include_defaults=False)), 0)

    def test_fieldmap_custom_mapping(self):
        fm = FieldMap({"my_field": ["variant_a", "variant_b"]}, include_defaults=False)
        self.assertEqual(normalize_event({"variant_a": "hello"}, fm)["my_field"], "hello")

    def test_fieldmap_overrides_default(self):
        fm = FieldMap({"tokens_in": ["my_custom_variant"]})
        self.assertEqual(normalize_event({"my_custom_variant": 99}, fm)["tokens_in"], 99)

    def test_fieldmap_add(self):
        fm = FieldMap(include_defaults=False)
        fm.add("score", ["rating", "value"])
        self.assertEqual(normalize_event({"rating": 0.95}, fm)["score"], 0.95)

    def test_fieldmap_add_returns_self(self):
        fm = FieldMap(include_defaults=False)
        self.assertIs(fm.add("x", ["a"]), fm)

    def test_fieldmap_get(self):
        fm = FieldMap(include_defaults=False)
        fm.add("x", ["a", "b"])
        self.assertEqual(fm.get("x"), ["a", "b"])
        self.assertEqual(fm.get("unknown"), [])

    def test_fieldmap_len(self):
        fm = FieldMap(include_defaults=False)
        fm.add("a", ["x"])
        fm.add("b", ["y"])
        self.assertEqual(len(fm), 2)


class TestNormalizeResult(unittest.TestCase):
    def test_normalize_result_rename_count(self):
        nr = NormalizeResult(event={"tokens_in": 5}, renamed={"input_tokens": "tokens_in"})
        self.assertEqual(nr.rename_count, 1)

    def test_normalize_result_zero_renames(self):
        self.assertEqual(NormalizeResult(event={}, renamed={}).rename_count, 0)


class TestNormalizeEventVerbose(unittest.TestCase):
    def test_returns_normalize_result(self):
        self.assertIsInstance(normalize_event_verbose({"ts": 1}), NormalizeResult)

    def test_reports_renamed_map(self):
        result = normalize_event_verbose({"input_tokens": 42, "model_id": "x"})
        self.assertEqual(result.event["tokens_in"], 42)
        self.assertEqual(result.event["model"], "x")
        self.assertEqual(result.renamed, {"input_tokens": "tokens_in", "model_id": "model"})
        self.assertEqual(result.rename_count, 2)

    def test_no_renames_empty_map(self):
        result = normalize_event_verbose({"already": "canonical"})
        self.assertEqual(result.renamed, {})
        self.assertEqual(result.rename_count, 0)

    def test_canonical_wins_not_reported_as_rename(self):
        result = normalize_event_verbose({"tokens_in": 99, "input_tokens": 42})
        self.assertEqual(result.renamed, {})
        self.assertEqual(result.event["tokens_in"], 99)

    def test_does_not_mutate_original(self):
        event = {"input_tokens": 42}
        original = dict(event)
        normalize_event_verbose(event)
        self.assertEqual(event, original)

    def test_keep_original_still_reported(self):
        result = normalize_event_verbose({"ts": 5}, keep_original=True)
        self.assertEqual(result.event["timestamp"], 5)
        self.assertEqual(result.event["ts"], 5)
        self.assertEqual(result.renamed, {"ts": "timestamp"})


class TestPackageMetadata(unittest.TestCase):
    def test_version_is_string(self):
        self.assertIsInstance(trace_field_normalize.__version__, str)

    def test_version_matches_expected(self):
        self.assertEqual(trace_field_normalize.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()

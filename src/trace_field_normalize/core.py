"""Normalize inconsistent field names in agent trace JSONL events.

Different agent frameworks use different names for the same semantic fields.
This module maps known variants to canonical names.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Default mapping: canonical_name -> [list of variant names to try]
# The first variant found in the event is renamed to the canonical name.
_DEFAULT_FIELD_MAP: dict[str, list[str]] = {
    "kind": ["event_type", "type", "event_kind"],
    "name": ["step", "tool", "tool_name", "function_name"],
    "tokens_in": ["input_tokens", "prompt_tokens", "tokens_prompt"],
    "tokens_out": ["output_tokens", "completion_tokens", "tokens_completion"],
    "cost_usd": ["cost", "price_usd", "usd", "price"],
    "duration_ms": ["latency_ms", "elapsed_ms", "duration", "latency"],
    "error": ["err", "exception", "error_message"],
    "model": ["model_id", "model_name"],
    "lane": ["worker", "agent_id", "thread"],
    "timestamp": ["ts", "time", "created_at", "event_time"],
}


class FieldMap:
    """A mapping from canonical field names to lists of known variant names.

    Example::

        fm = FieldMap({"kind": ["event_type", "type"]})
        event = {"event_type": "llm_call"}
        normalized = normalize_event(event, field_map=fm)
        # {"kind": "llm_call"}
    """

    def __init__(
        self,
        mapping: dict[str, list[str]] | None = None,
        *,
        include_defaults: bool = True,
    ) -> None:
        self._map: dict[str, list[str]] = {}
        if include_defaults:
            self._map.update(_DEFAULT_FIELD_MAP)
        if mapping:
            for canonical, variants in mapping.items():
                self._map[canonical] = variants

    def add(self, canonical: str, variants: list[str]) -> "FieldMap":
        """Add or replace a canonical → variants mapping."""
        self._map[canonical] = variants
        return self

    def get(self, canonical: str) -> list[str]:
        """Return the variants for a canonical name."""
        return self._map.get(canonical, [])

    def items(self):
        return self._map.items()

    def __len__(self) -> int:
        return len(self._map)


_DEFAULT = FieldMap()


@dataclass
class NormalizeResult:
    """Result of normalizing a single event.

    Produced by :func:`normalize_event_verbose` when you need to know which
    keys were renamed (for logging, metrics, or debugging).

    Attributes:
        event: the normalized event dict.
        renamed: dict of ``{old_name: canonical_name}`` for fields that were
            renamed.
    """

    event: dict[str, Any]
    renamed: dict[str, str]

    @property
    def rename_count(self) -> int:
        """Number of fields that were renamed."""
        return len(self.renamed)


def _normalize(
    event: dict[str, Any],
    field_map: FieldMap | None,
    keep_original: bool,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Core normalization routine.

    Returns the normalized event and a ``{old_name: canonical_name}`` map of
    the renames that were applied.
    """
    fm = field_map or _DEFAULT
    result = dict(event)
    renamed: dict[str, str] = {}

    for canonical, variants in fm.items():
        if canonical in result:
            # Already has the canonical name — skip
            continue
        for variant in variants:
            if variant in result:
                value = result.pop(variant)
                result[canonical] = value
                renamed[variant] = canonical
                if keep_original:
                    result[variant] = value
                break  # only rename the first matching variant

    return result, renamed


def normalize_event(
    event: dict[str, Any],
    field_map: FieldMap | None = None,
    *,
    keep_original: bool = False,
) -> dict[str, Any]:
    """Normalize field names in a single event dict.

    For each canonical name in the field map, checks if any variant is
    present in the event. If found, the variant key is renamed to the
    canonical name. If the canonical name already exists, the variant
    is left unchanged (canonical wins).

    Args:
        event: the raw event dict.
        field_map: custom FieldMap; uses defaults if None.
        keep_original: if True, keep the original field under its old name
            in addition to the canonical name.

    Returns:
        New dict with normalized field names. The input ``event`` is never
        mutated.
    """
    result, _ = _normalize(event, field_map, keep_original)
    return result


def normalize_event_verbose(
    event: dict[str, Any],
    field_map: FieldMap | None = None,
    *,
    keep_original: bool = False,
) -> NormalizeResult:
    """Normalize a single event and report which fields were renamed.

    Behaves exactly like :func:`normalize_event` but returns a
    :class:`NormalizeResult` carrying both the normalized ``event`` and a
    ``renamed`` map of ``{old_name: canonical_name}``. Useful when you want to
    log or count how much normalization actually happened.

    Args:
        event: the raw event dict.
        field_map: custom FieldMap; uses defaults if None.
        keep_original: if True, keep the original field under its old name
            in addition to the canonical name.

    Returns:
        A :class:`NormalizeResult`. The input ``event`` is never mutated.
    """
    result, renamed = _normalize(event, field_map, keep_original)
    return NormalizeResult(event=result, renamed=renamed)


def normalize_events(
    events: list[dict[str, Any]],
    field_map: FieldMap | None = None,
    *,
    keep_original: bool = False,
) -> list[dict[str, Any]]:
    """Normalize field names across a list of events.

    Returns a new list — the originals are not modified.
    """
    return [normalize_event(e, field_map, keep_original=keep_original) for e in events]


def normalize_file(
    source: str | Path,
    dest: str | Path | None = None,
    field_map: FieldMap | None = None,
    *,
    keep_original: bool = False,
) -> list[dict[str, Any]]:
    """Load a JSONL file, normalize field names, and optionally write to dest.

    Args:
        source: input JSONL file path.
        dest: optional output JSONL file path; if None, result is returned only.
        field_map: custom FieldMap; uses defaults if None.
        keep_original: if True, keep original field names alongside canonical.

    Returns:
        List of normalized event dicts.

    Raises:
        FileNotFoundError: if ``source`` does not exist.
        ValueError: if a non-blank line is not valid JSON or does not decode
            to a JSON object (the error message includes the 1-based line
            number).
    """
    p = Path(source)
    events: list[dict[str, Any]] = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p}: line {lineno}: invalid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(
                f"{p}: line {lineno}: expected a JSON object, got {type(obj).__name__}"
            )
        events.append(obj)

    normalized = normalize_events(events, field_map, keep_original=keep_original)

    if dest is not None:
        Path(dest).write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in normalized) + "\n",
            encoding="utf-8",
        )

    return normalized

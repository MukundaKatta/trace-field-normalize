"""trace-field-normalize: normalize inconsistent field names in agent trace events."""

from .core import FieldMap, NormalizeResult, normalize_event, normalize_events, normalize_file

__all__ = ["FieldMap", "NormalizeResult", "normalize_event", "normalize_events", "normalize_file"]

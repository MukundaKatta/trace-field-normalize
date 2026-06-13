"""trace-field-normalize: normalize inconsistent field names in agent trace events."""

from .core import (
    FieldMap,
    NormalizeResult,
    normalize_event,
    normalize_event_verbose,
    normalize_events,
    normalize_file,
)

__version__ = "0.1.0"

__all__ = [
    "FieldMap",
    "NormalizeResult",
    "normalize_event",
    "normalize_event_verbose",
    "normalize_events",
    "normalize_file",
    "__version__",
]

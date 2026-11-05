"""The single validator (D6): full reference implementation and its incremental variant."""

from slotor.validation.full import validate_layout
from slotor.validation.incremental import IncrementalValidator
from slotor.validation.issues import Issue, IssueCode, Surface, ValidationResult

__all__ = [
    "IncrementalValidator",
    "Issue",
    "IssueCode",
    "Surface",
    "ValidationResult",
    "validate_layout",
]

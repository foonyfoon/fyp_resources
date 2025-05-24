from dataclasses import dataclass, field
from typing import Any, Dict
# ---------------------------------------------------------------------------
# Core data container
# ---------------------------------------------------------------------------

@dataclass
# constructed per node basis
class PromptPackage:
    """Carries the evolving prompt plus arbitrary metadata/state."""

    text: str
    state: Dict[str, Any] = field(default_factory=dict)
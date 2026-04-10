"""StubMrcAdapter — configurable canned MRC responses for local dev/testing.

Default behavior: readiness=True, confidence=0.95.
Per-file overrides can be configured at construction time for testing
specific failure scenarios.
"""
from __future__ import annotations

from validator_agent.ports.mrc_port import MrcPort, MrcResult


class StubMrcAdapter(MrcPort):
    """Stub adapter for the Material Readiness Checker.

    Args:
        default_readiness: Default readiness flag for all files.
        default_confidence: Default confidence score for all files.
        overrides: Per-file overrides keyed by storage_path.
            Values are dicts with optional "readiness" and "confidence" keys.
    """

    def __init__(
        self,
        *,
        default_readiness: bool = True,
        default_confidence: float = 0.95,
        overrides: dict[str, dict] | None = None,
    ) -> None:
        self._default_readiness = default_readiness
        self._default_confidence = default_confidence
        self._overrides = overrides or {}

    async def predict(self, storage_path: str) -> MrcResult:
        """Return a canned MRC result, with optional per-file override."""
        if storage_path in self._overrides:
            override = self._overrides[storage_path]
            return MrcResult(
                readiness=override.get("readiness", self._default_readiness),
                confidence=override.get("confidence", self._default_confidence),
            )

        return MrcResult(
            readiness=self._default_readiness,
            confidence=self._default_confidence,
        )

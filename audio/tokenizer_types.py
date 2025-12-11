"""Shared tokenizer typing for the Kani audio stack."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class TokenizerLike(Protocol):
    """Minimal interface shared between generation and playback."""

    def decode(self, tokens: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        """Convert token IDs back into a string."""
        ...

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """Convert text into token IDs."""
        ...

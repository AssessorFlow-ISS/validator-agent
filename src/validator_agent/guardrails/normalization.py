"""Text normalization to defeat obfuscation-based evasion attacks.

Inlined from af_shared.guardrails.normalization.
Pre-processes text before regex scanning to handle:
- Zero-width character insertion
- HTML entity encoding
- Unicode homoglyph substitution (Cyrillic/Greek lookalikes)
- Base64-encoded instruction segments
- Code comment delimiter wrapping
- Whitespace/punctuation interleaving

All normalization is deterministic and zero-cost (no LLM calls).
"""

from __future__ import annotations

import base64
import html
import re
import unicodedata

# Cyrillic/Greek/fullwidth -> ASCII homoglyph mapping
_HOMOGLYPH_MAP: dict[str, str] = {
    # Cyrillic lookalikes
    "\u0430": "a",
    "\u0435": "e",
    "\u043e": "o",
    "\u0440": "p",
    "\u0441": "c",
    "\u0443": "y",
    "\u0445": "x",
    "\u0456": "i",
    "\u0458": "j",
    "\u04bb": "h",
    "\u0410": "A",
    "\u0412": "B",
    "\u0415": "E",
    "\u041a": "K",
    "\u041c": "M",
    "\u041d": "H",
    "\u041e": "O",
    "\u0420": "P",
    "\u0421": "C",
    "\u0422": "T",
    "\u0425": "X",
    # Greek lookalikes
    "\u03b1": "a",
    "\u03b5": "e",
    "\u03bf": "o",
    "\u03c1": "p",
    "\u0391": "A",
    "\u0392": "B",
    "\u0395": "E",
    "\u0397": "H",
    "\u0399": "I",
    "\u039a": "K",
    "\u039c": "M",
    "\u039d": "N",
    "\u039f": "O",
    "\u03a1": "P",
    "\u03a4": "T",
    "\u03a7": "X",
    "\u03a5": "Y",
    "\u0396": "Z",
}

_HOMOGLYPH_TABLE = str.maketrans(_HOMOGLYPH_MAP)

# 15 zero-width and invisible characters to strip
_ZERO_WIDTH_CHARS = frozenset(
    {
        "\u200b",
        "\u200c",
        "\u200d",
        "\u200e",
        "\u200f",
        "\u2060",
        "\u2061",
        "\u2062",
        "\u2063",
        "\u2064",
        "\ufeff",
        "\u00ad",
        "\u034f",
        "\u061c",
        "\u180e",
    }
)

_ZERO_WIDTH_RE = re.compile("[" + "".join(_ZERO_WIDTH_CHARS) + "]")

# Base64 heuristic: sequences of 20+ base64 chars that decode to ASCII text
_BASE64_SEGMENT_RE = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")

# Punctuation/separator interleaving: "i.g.n.o.r.e" -> "ignore"
_INTERLEAVE_RE = re.compile(r"(?<=\w)[.\-_](?=\w)")


def normalize_for_evasion(text: str) -> str:
    """Normalize text to defeat obfuscation-based evasion attacks.

    8-step pipeline:
    1. Strip zero-width characters
    2. Decode HTML entities
    3. Translate Unicode homoglyphs to ASCII
    4. Apply NFKD normalization (fullwidth -> ASCII)
    5. Attempt base64 decoding of suspicious segments
    6. Strip code comment delimiters
    7. Remove punctuation interleaving
    8. Collapse excessive whitespace
    """
    if not text:
        return text

    result = _ZERO_WIDTH_RE.sub("", text)
    result = html.unescape(result)
    result = result.translate(_HOMOGLYPH_TABLE)
    result = unicodedata.normalize("NFKD", result)
    result = _decode_base64_segments(result)
    result = _strip_comments(result)
    result = _INTERLEAVE_RE.sub("", result)
    result = re.sub(r"\s+", " ", result).strip()

    return result


def _decode_base64_segments(text: str) -> str:
    """Attempt to decode base64 segments and append decoded text."""

    def _try_decode(match: re.Match[str]) -> str:
        segment = match.group()
        try:
            decoded = base64.b64decode(segment, validate=True).decode("ascii")
            if decoded.isprintable() and len(decoded) >= 5:
                return f"{segment} {decoded}"
        except Exception:
            pass
        return segment

    return _BASE64_SEGMENT_RE.sub(_try_decode, text)


def _strip_comments(text: str) -> str:
    """Remove comment delimiters but keep the content inside."""
    result = text
    result = re.sub(r"/\*(.*?)\*/", r" \1 ", result, flags=re.DOTALL)
    result = re.sub(r"<!--(.*?)-->", r" \1 ", result, flags=re.DOTALL)
    result = re.sub(r"^//\s*", "", result, flags=re.MULTILINE)
    result = re.sub(r"^#\s+(?=[a-zA-Z])", "", result, flags=re.MULTILINE)
    result = re.sub(r"^--\s+(?=[a-zA-Z])", "", result, flags=re.MULTILINE)
    return result

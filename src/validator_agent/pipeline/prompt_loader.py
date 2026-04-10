"""Shared prompt loader for pipeline YAML prompt templates (ADR-39).

Reads YAML files from the repo-root ``prompts/`` directory, splitting the
``---`` delimited frontmatter from the body text.  This is the single
canonical loader used by all pipeline modules — the original
``_load_prompt_template`` in ``domain/content_safety.py`` predates this
module and handles the same format for the domain layer prompts.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# Resolve to <repo-root>/prompts/ regardless of the working directory
PROMPT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "prompts"


def load_prompt(name: str) -> tuple[str, dict]:
    """Load a YAML prompt template and return ``(body_text, frontmatter_dict)``.

    The file is expected at ``prompts/{name}.yaml`` with an optional YAML
    frontmatter block delimited by ``---``.

    Parameters
    ----------
    name:
        Stem of the YAML file (without ``.yaml`` extension).

    Returns
    -------
    tuple[str, dict]
        The prompt body text and the parsed frontmatter dictionary.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    """
    path = PROMPT_DIR / f"{name}.yaml"
    raw = path.read_text(encoding="utf-8")

    # Split YAML frontmatter from body
    parts = raw.split("---", 2)
    if len(parts) >= 3:
        frontmatter = yaml.safe_load(parts[1]) or {}
        body = parts[2].strip()
    else:
        frontmatter = {}
        body = raw.strip()

    return body, frontmatter

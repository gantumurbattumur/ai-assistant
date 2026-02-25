"""Central configuration constants for the AI assistant."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

# LLM model used by all multi-agent nodes and standalone helpers.
# Change this one constant to upgrade or swap the model project-wide.
MODEL_NAME = "gpt-4o-mini"

# Lazy singleton — created on first call, *after* setup_environment() has
# loaded the .env file and set OPENAI_API_KEY.  Never instantiated at import time.
_openai_client: "OpenAI | None" = None


def get_openai_client() -> "OpenAI":
    """Return the shared OpenAI client, creating it on first call."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI()
    return _openai_client

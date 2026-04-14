"""
utils.py
Shared utility functions: API key validation, OpenAI client factory,
friendly error formatting.
"""

import re
from openai import OpenAI, AuthenticationError, APIConnectionError


def validate_api_key(api_key: str) -> tuple[bool, str]:
    """
    Lightweight format check then a cheap live ping to verify the key is valid.

    Returns:
        (is_valid: bool, error_message: str)
    """
    api_key = (api_key or "").strip()

    if not api_key:
        return False, "API key is empty."

    if not re.match(r"^sk-[A-Za-z0-9\-_]{20,}$", api_key):
        return False, "Key format looks wrong — it should start with 'sk-'."

    try:
        client = OpenAI(api_key=api_key)
        # Cheapest possible call to validate credentials
        client.models.list()
        return True, ""
    except AuthenticationError:
        return False, "Invalid API key — authentication failed."
    except APIConnectionError:
        return False, "Network error while validating key. Check your connection."
    except Exception as e:
        return False, f"Unexpected error: {e}"


def make_client(api_key: str) -> OpenAI:
    """Return a configured OpenAI client."""
    return OpenAI(api_key=api_key.strip())


def friendly_error(e: Exception) -> str:
    """Convert common exceptions to user-friendly messages."""
    msg = str(e)
    if "RateLimitError" in type(e).__name__ or "rate limit" in msg.lower():
        return "⚠️ OpenAI rate limit reached. Please wait a moment and try again."
    if "AuthenticationError" in type(e).__name__ or "authentication" in msg.lower():
        return "🔑 API key authentication failed. Please check your key in the sidebar."
    if "insufficient_quota" in msg.lower():
        return "💳 Your OpenAI account has insufficient quota. Please check your billing."
    if "context_length_exceeded" in msg.lower():
        return "📄 The document context is too large for the model. Try a shorter PDF."
    return f"❌ Error: {msg}"

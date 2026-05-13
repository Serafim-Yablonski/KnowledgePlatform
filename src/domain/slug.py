import re
import secrets
import string

_ALPHABET = string.ascii_lowercase + string.digits


def generate_slug(name: str) -> str:
    """Return a URL-safe slug from *name* with a 4-char random suffix for uniqueness."""
    lowered = name.lower()
    hyphenated = lowered.replace(" ", "-")
    cleaned = re.sub(r"[^a-z0-9-]", "", hyphenated)
    # Collapse consecutive hyphens and strip leading/trailing ones.
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    truncated = cleaned[:50]
    suffix = "".join(secrets.choice(_ALPHABET) for _ in range(4))
    if truncated:
        return f"{truncated}-{suffix}"
    return suffix

"""URL content extraction via trafilatura."""

from __future__ import annotations

import trafilatura


def extract_content(url: str) -> str | None:
    """Extract main text content from a URL. Returns None on failure."""
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        return None
    return trafilatura.extract(downloaded)

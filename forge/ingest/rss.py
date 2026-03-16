"""RSS/Atom feed polling and article storage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from time import mktime
from typing import TYPE_CHECKING

import feedparser

if TYPE_CHECKING:
    from forge.db.models import Article, Feed
    from forge.db.store import Store


def poll_feed(feed: Feed, store: Store) -> list[Article]:
    """Poll a single feed and save new articles. Returns newly saved articles."""
    parsed = feedparser.parse(feed.url)
    new_articles: list[Article] = []

    for entry in parsed.entries:
        url = getattr(entry, "link", None)
        if not url:
            continue

        # Skip duplicates
        if store.get_article_by_url(url) is not None:
            continue

        title = entry.get("title", None)
        published_at = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            published_at = datetime.fromtimestamp(
                mktime(entry.published_parsed), tz=UTC
            ).isoformat()

        article = store.save_article(
            url=url,
            feed_id=feed.id,
            title=title,
            published_at=published_at,
        )
        new_articles.append(article)

    # Update last_polled_at
    store.update_feed(feed.id, last_polled_at=datetime.now(UTC).isoformat())
    return new_articles


def poll_all_feeds(store: Store) -> dict[str, int]:
    """Poll all active feeds due for refresh. Returns {feed_url: article_count}."""
    now = datetime.now(UTC)
    results: dict[str, int] = {}

    for feed in store.list_feeds(active=True):
        if feed.last_polled_at:
            last = datetime.fromisoformat(feed.last_polled_at)
            if now - last < timedelta(minutes=feed.poll_interval_minutes):
                continue

        articles = poll_feed(feed, store)
        results[feed.url] = len(articles)

    return results

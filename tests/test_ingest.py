"""Tests for ingestion: Feed/Article CRUD, RSS polling, URL extraction."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from forge.db.models import Article, Feed

if TYPE_CHECKING:
    from forge.db.store import Store


# ------------------------------------------------------------------
# Feed CRUD
# ------------------------------------------------------------------


@pytest.mark.unit
class TestFeedCRUD:
    def test_store_save_feed_returns_feed_model(self, db: Store) -> None:
        feed = db.save_feed(name="TechCrunch", url="https://techcrunch.com/feed")
        assert isinstance(feed, Feed)
        assert feed.id.startswith("f_")
        assert feed.name == "TechCrunch"
        assert feed.url == "https://techcrunch.com/feed"
        assert feed.feed_type == "rss"
        assert feed.active == 1
        assert feed.poll_interval_minutes == 240

    def test_store_save_feed_custom_interval(self, db: Store) -> None:
        feed = db.save_feed(
            name="Fast Feed", url="https://fast.com/rss",
            poll_interval_minutes=60,
        )
        assert feed.poll_interval_minutes == 60

    def test_store_save_feed_empty_name_raises(self, db: Store) -> None:
        with pytest.raises(ValueError, match="name must not be empty"):
            db.save_feed(name="", url="https://example.com/feed")

    def test_store_save_feed_empty_url_raises(self, db: Store) -> None:
        with pytest.raises(ValueError, match="url must not be empty"):
            db.save_feed(name="Test", url="")

    def test_store_save_feed_duplicate_url_raises(self, db: Store) -> None:
        db.save_feed(name="Feed 1", url="https://example.com/feed")
        with pytest.raises(sqlite3.IntegrityError):
            db.save_feed(name="Feed 2", url="https://example.com/feed")

    def test_store_get_feed_returns_feed(self, db: Store) -> None:
        feed = db.save_feed(name="Test", url="https://example.com/feed")
        result = db.get_feed(feed.id)
        assert result is not None
        assert result.id == feed.id
        assert result.name == "Test"

    def test_store_get_feed_not_found_returns_none(self, db: Store) -> None:
        assert db.get_feed("f_nonexistent") is None

    def test_store_list_feeds_returns_all(self, db: Store) -> None:
        db.save_feed(name="Feed 1", url="https://a.com/feed")
        db.save_feed(name="Feed 2", url="https://b.com/feed")
        feeds = db.list_feeds()
        assert len(feeds) == 2

    def test_store_list_feeds_filter_active(self, db: Store) -> None:
        f1 = db.save_feed(name="Active", url="https://a.com/feed")
        f2 = db.save_feed(name="Inactive", url="https://b.com/feed")
        db.update_feed(f2.id, active=0)

        active = db.list_feeds(active=True)
        assert len(active) == 1
        assert active[0].id == f1.id

    def test_store_update_feed(self, db: Store) -> None:
        feed = db.save_feed(name="Old", url="https://example.com/feed")
        updated = db.update_feed(feed.id, name="New")
        assert updated is not None
        assert updated.name == "New"

    def test_store_update_feed_not_found_returns_none(self, db: Store) -> None:
        assert db.update_feed("f_nonexistent", name="X") is None

    def test_store_list_feeds_empty_db(self, db: Store) -> None:
        assert db.list_feeds() == []


# ------------------------------------------------------------------
# Article CRUD
# ------------------------------------------------------------------


@pytest.mark.unit
class TestArticleCRUD:
    def test_store_save_article_returns_article_model(self, db: Store) -> None:
        article = db.save_article(
            url="https://example.com/article1",
            title="Test Article",
            content="Article content here.",
        )
        assert isinstance(article, Article)
        assert article.id.startswith("a_")
        assert article.url == "https://example.com/article1"
        assert article.title == "Test Article"
        assert article.content == "Article content here."
        assert article.claims_extracted == 0

    def test_store_save_article_with_feed_id(self, db: Store) -> None:
        feed = db.save_feed(name="Test", url="https://example.com/feed")
        article = db.save_article(
            url="https://example.com/a1",
            feed_id=feed.id,
            title="From Feed",
        )
        assert article.feed_id == feed.id

    def test_store_save_article_duplicate_url_raises(self, db: Store) -> None:
        db.save_article(url="https://example.com/a1")
        with pytest.raises(sqlite3.IntegrityError):
            db.save_article(url="https://example.com/a1")

    def test_store_get_article_returns_article(self, db: Store) -> None:
        article = db.save_article(url="https://example.com/a1", title="Test")
        result = db.get_article(article.id)
        assert result is not None
        assert result.title == "Test"

    def test_store_get_article_not_found_returns_none(self, db: Store) -> None:
        assert db.get_article("a_nonexistent") is None

    def test_store_get_article_by_url(self, db: Store) -> None:
        article = db.save_article(url="https://example.com/a1", title="Found")
        result = db.get_article_by_url("https://example.com/a1")
        assert result is not None
        assert result.id == article.id

    def test_store_get_article_by_url_not_found(self, db: Store) -> None:
        assert db.get_article_by_url("https://nonexistent.com") is None

    def test_store_list_articles_by_feed(self, db: Store) -> None:
        feed = db.save_feed(name="F1", url="https://f1.com/feed")
        db.save_article(url="https://f1.com/a1", feed_id=feed.id)
        db.save_article(url="https://f1.com/a2", feed_id=feed.id)
        db.save_article(url="https://other.com/a3")

        articles = db.list_articles(feed_id=feed.id)
        assert len(articles) == 2

    def test_store_list_articles_unextracted(self, db: Store) -> None:
        a1 = db.save_article(url="https://example.com/a1")
        a2 = db.save_article(url="https://example.com/a2")
        db.update_article(a2.id, claims_extracted=3)

        unextracted = db.list_articles(unextracted=True)
        assert len(unextracted) == 1
        assert unextracted[0].id == a1.id

    def test_store_update_article(self, db: Store) -> None:
        article = db.save_article(url="https://example.com/a1")
        updated = db.update_article(article.id, claims_extracted=5)
        assert updated is not None
        assert updated.claims_extracted == 5

    def test_store_update_article_not_found_returns_none(self, db: Store) -> None:
        assert db.update_article("a_nonexistent", title="X") is None

    def test_store_list_articles_empty(self, db: Store) -> None:
        assert db.list_articles() == []


# ------------------------------------------------------------------
# RSS Polling
# ------------------------------------------------------------------


@pytest.mark.unit
class TestRSSPolling:
    def test_poll_feed_saves_new_articles(self, db: Store) -> None:
        from forge.ingest.rss import poll_feed

        feed = db.save_feed(name="Test", url="https://example.com/feed")

        mock_entries = [
            MagicMock(
                link="https://example.com/post1",
                title="Post 1",
                get=lambda k, d=None, _t="Post 1": _t if k == "title" else d,
                published_parsed=None,
            ),
            MagicMock(
                link="https://example.com/post2",
                title="Post 2",
                get=lambda k, d=None, _t="Post 2": _t if k == "title" else d,
                published_parsed=None,
            ),
        ]
        mock_parsed = MagicMock()
        mock_parsed.entries = mock_entries
        mock_parsed.bozo = False

        with patch("forge.ingest.rss.feedparser.parse", return_value=mock_parsed):
            articles = poll_feed(feed, db)

        assert len(articles) == 2
        assert all(isinstance(a, Article) for a in articles)

    def test_poll_feed_skips_duplicates(self, db: Store) -> None:
        from forge.ingest.rss import poll_feed

        feed = db.save_feed(name="Test", url="https://example.com/feed")
        db.save_article(url="https://example.com/post1", feed_id=feed.id)

        mock_entries = [
            MagicMock(
                link="https://example.com/post1",
                title="Post 1",
                get=lambda k, d=None, _t="Post 1": _t if k == "title" else d,
                published_parsed=None,
            ),
        ]
        mock_parsed = MagicMock()
        mock_parsed.entries = mock_entries
        mock_parsed.bozo = False

        with patch("forge.ingest.rss.feedparser.parse", return_value=mock_parsed):
            articles = poll_feed(feed, db)

        assert len(articles) == 0

    def test_poll_feed_empty_feed_returns_empty(self, db: Store) -> None:
        from forge.ingest.rss import poll_feed

        feed = db.save_feed(name="Test", url="https://example.com/feed")
        mock_parsed = MagicMock()
        mock_parsed.entries = []
        mock_parsed.bozo = False

        with patch("forge.ingest.rss.feedparser.parse", return_value=mock_parsed):
            articles = poll_feed(feed, db)

        assert articles == []

    def test_poll_feed_skips_entries_without_link(self, db: Store) -> None:
        from forge.ingest.rss import poll_feed

        feed = db.save_feed(name="Test", url="https://example.com/feed")
        mock_entries = [MagicMock(link=None)]  # No link attribute
        mock_parsed = MagicMock()
        mock_parsed.entries = mock_entries
        mock_parsed.bozo = False

        with patch("forge.ingest.rss.feedparser.parse", return_value=mock_parsed):
            articles = poll_feed(feed, db)

        assert articles == []

    def test_poll_feed_extracts_published_date(self, db: Store) -> None:
        import time

        from forge.ingest.rss import poll_feed
        feed = db.save_feed(name="Test", url="https://example.com/feed")
        published_time = time.strptime("2024-01-15", "%Y-%m-%d")
        entry = MagicMock(
            link="https://example.com/dated-post",
            published_parsed=published_time,
        )
        entry.get = lambda k, d=None: "Dated Post" if k == "title" else d

        mock_parsed = MagicMock()
        mock_parsed.entries = [entry]
        mock_parsed.bozo = False

        with patch("forge.ingest.rss.feedparser.parse", return_value=mock_parsed):
            articles = poll_feed(feed, db)

        assert len(articles) == 1
        assert articles[0].published_at is not None
        assert "2024-01-15" in articles[0].published_at

    def test_poll_all_feeds_skips_recently_polled(self, db: Store) -> None:
        from datetime import UTC, datetime

        from forge.ingest.rss import poll_all_feeds

        feed = db.save_feed(
            name="Recent", url="https://recent.com/feed",
            poll_interval_minutes=240,
        )
        # Mark as just polled
        db.update_feed(feed.id, last_polled_at=datetime.now(UTC).isoformat())

        mock_parsed = MagicMock()
        mock_parsed.entries = []
        mock_parsed.bozo = False

        with patch("forge.ingest.rss.feedparser.parse", return_value=mock_parsed):
            result = poll_all_feeds(db)

        # Feed should be skipped
        assert result == {}

    def test_poll_all_feeds_polls_due_feeds(self, db: Store) -> None:
        from forge.ingest.rss import poll_all_feeds

        db.save_feed(
            name="F1", url="https://f1.com/feed", poll_interval_minutes=0,
        )

        mock_parsed = MagicMock()
        mock_parsed.entries = []
        mock_parsed.bozo = False

        with patch("forge.ingest.rss.feedparser.parse", return_value=mock_parsed):
            result = poll_all_feeds(db)

        assert isinstance(result, dict)
        assert "https://f1.com/feed" in result


# ------------------------------------------------------------------
# URL Content Extraction
# ------------------------------------------------------------------


@pytest.mark.unit
class TestURLExtraction:
    def test_extract_content_returns_text(self) -> None:
        from forge.ingest.url import extract_content

        html = "<html><body><p>Hello world</p></body></html>"
        with (
            patch("forge.ingest.url.trafilatura.fetch_url", return_value=html),
            patch("forge.ingest.url.trafilatura.extract", return_value="Hello world"),
        ):
            result = extract_content("https://example.com/article")

        assert result == "Hello world"

    def test_extract_content_returns_none_on_failure(self) -> None:
        from forge.ingest.url import extract_content

        with patch("forge.ingest.url.trafilatura.fetch_url", return_value=None):
            result = extract_content("https://example.com/404")

        assert result is None

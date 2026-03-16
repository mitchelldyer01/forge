"""Tests for forge/export/obsidian.py — One-way Obsidian vault renderer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from forge.export.obsidian import render_vault

if TYPE_CHECKING:
    from pathlib import Path

    from forge.db.store import Store


def test_render_vault_creates_directory(db: Store, tmp_path: Path) -> None:
    """Render creates the vault directory if it doesn't exist."""
    vault_path = tmp_path / "obsidian"
    render_vault(db, str(vault_path))
    assert vault_path.exists()
    assert vault_path.is_dir()


def test_render_vault_creates_hypothesis_files(db: Store, tmp_path: Path) -> None:
    """Each hypothesis gets a markdown file."""
    db.save_hypothesis("AI will dominate SaaS", "manual", confidence=75, tags=["ai"])
    db.save_hypothesis("Crypto winter ends 2026", "manual", confidence=40, tags=["crypto"])

    vault_path = tmp_path / "obsidian"
    render_vault(db, str(vault_path))

    hyp_dir = vault_path / "hypotheses"
    assert hyp_dir.exists()
    md_files = list(hyp_dir.glob("*.md"))
    assert len(md_files) == 2


def test_render_vault_hypothesis_content(db: Store, tmp_path: Path) -> None:
    """Hypothesis file contains claim, confidence, status, and metadata."""
    db.save_hypothesis(
        "AI agents replace SaaS",
        "manual",
        confidence=72,
        tags=["ai", "saas"],
        context="Enterprise software trends",
    )

    vault_path = tmp_path / "obsidian"
    render_vault(db, str(vault_path))

    hyp_dir = vault_path / "hypotheses"
    md_files = list(hyp_dir.glob("*.md"))
    assert len(md_files) == 1

    content = md_files[0].read_text()
    assert "AI agents replace SaaS" in content
    assert "72" in content
    assert "alive" in content
    assert "#ai" in content
    assert "#saas" in content


def test_render_vault_includes_relations(db: Store, tmp_path: Path) -> None:
    """Hypothesis file includes links to related hypotheses."""
    h1 = db.save_hypothesis("Parent claim", "manual", confidence=70)
    h2 = db.save_hypothesis("Supporting claim", "manual", confidence=60)
    db.save_relation(h2.id, h1.id, "supports", strength=0.8)

    vault_path = tmp_path / "obsidian"
    render_vault(db, str(vault_path))

    hyp_dir = vault_path / "hypotheses"
    # Find h1's file and check it references h2
    md_files = list(hyp_dir.glob("*.md"))
    contents = [f.read_text() for f in md_files]
    all_content = "\n".join(contents)
    assert "supports" in all_content


def test_render_vault_creates_predictions_dir(db: Store, tmp_path: Path) -> None:
    """Predictions get rendered in their own directory."""
    sim = db.save_simulation("scenario", "test seed")
    db.save_prediction(
        sim.id, "BTC hits 100k by Q3", 65,
        resolution_deadline="2026-09-30T00:00:00+00:00",
    )

    vault_path = tmp_path / "obsidian"
    render_vault(db, str(vault_path))

    pred_dir = vault_path / "predictions"
    assert pred_dir.exists()
    md_files = list(pred_dir.glob("*.md"))
    assert len(md_files) == 1

    content = md_files[0].read_text()
    assert "BTC hits 100k by Q3" in content
    assert "65" in content


def test_render_vault_empty_db(db: Store, tmp_path: Path) -> None:
    """Empty database renders empty vault without errors."""
    vault_path = tmp_path / "obsidian"
    render_vault(db, str(vault_path))

    assert vault_path.exists()
    hyp_dir = vault_path / "hypotheses"
    assert hyp_dir.exists()
    assert list(hyp_dir.glob("*.md")) == []


def test_render_vault_overwrites_existing(db: Store, tmp_path: Path) -> None:
    """Re-rendering overwrites existing files (one-way sync)."""
    h = db.save_hypothesis("Original claim", "manual", confidence=50)

    vault_path = tmp_path / "obsidian"
    render_vault(db, str(vault_path))

    # Update hypothesis
    db.update_hypothesis(h.id, confidence=80)
    render_vault(db, str(vault_path))

    hyp_dir = vault_path / "hypotheses"
    md_files = list(hyp_dir.glob("*.md"))
    assert len(md_files) == 1
    content = md_files[0].read_text()
    assert "80" in content


def test_render_vault_simulation_index(db: Store, tmp_path: Path) -> None:
    """Simulations get an index file."""
    db.save_simulation("scenario", "What if EU bans AI?", agent_count=30, rounds=3)

    vault_path = tmp_path / "obsidian"
    render_vault(db, str(vault_path))

    sim_dir = vault_path / "simulations"
    assert sim_dir.exists()
    md_files = list(sim_dir.glob("*.md"))
    assert len(md_files) == 1
    content = md_files[0].read_text()
    assert "What if EU bans AI?" in content

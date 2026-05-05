"""
Skill Library: Voyager-style persistent skill store for PWM.

Philosophical grounding:
  Sthāpanā (TA 6.1, Abhinavagupta): Establishment — the creative act that
  instantiates a potential into a stable form. Each narrated creative event
  (camatkāra flash) that achieves quality > θ is "established" in the skill
  library, available for future retrieval and re-instantiation.

  Smṛti (PHr sūtra 9): Episodic memory — the WM's ability to recognise
  (pratyabhijñā) that a current creative challenge is similar to a past one
  and retrieve the corresponding skill.

Architecture (Voyager §3.2 adapted):
  - SQLite table: skill metadata (id, name, description, quality, created_at)
  - FAISS index: dense embedding retrieval over skill descriptions
  - Skill entries: name + description + code_hint + camatkāra_score

Usage:
  lib = SkillLibrary("data/skill_lib")
  lib.add_skill(name="metaphor_bridge", description="...", code_hint="...", score=0.85)
  skills = lib.retrieve("retrieve skills for bridging abstract concepts", k=3)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Skill:
    """
    A single skill entry in the library.

    Fields:
      name:          short camelCase identifier (e.g. "metaphor_bridge")
      description:   natural-language description of what the skill does
      code_hint:     optional pseudocode or prompt fragment
      camatk_score:  camatkāra quality score at time of creation [0, 1]
      embedding:     dense vector for FAISS retrieval (None until indexed)
      created_at:    Unix timestamp
      n_retrieved:   retrieval count (popularity signal)
    """
    name: str
    description: str
    code_hint: str = ""
    camatk_score: float = 0.0
    embedding: list[float] | None = None
    created_at: float = field(default_factory=time.time)
    n_retrieved: int = 0


class SkillLibrary:
    """
    Persistent skill library with SQLite backing and FAISS vector retrieval.

    Thread-safe for single-process use (SQLite in WAL mode).
    FAISS index is rebuilt from SQLite on load — no separate index file needed.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS skills (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT UNIQUE NOT NULL,
        description TEXT NOT NULL,
        code_hint   TEXT DEFAULT '',
        camatk_score REAL DEFAULT 0.0,
        embedding   BLOB,
        created_at  REAL NOT NULL,
        n_retrieved INTEGER DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_skills_camatk ON skills(camatk_score DESC);
    CREATE INDEX IF NOT EXISTS idx_skills_name ON skills(name);
    """

    def __init__(
        self,
        db_dir: str | Path = "data/skill_lib",
        embedding_dim: int = 384,
        quality_threshold: float = 0.6,
    ) -> None:
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.db_dir / "skills.db"
        self.embedding_dim = embedding_dim
        self.quality_threshold = quality_threshold

        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(self._SCHEMA)
        self._conn.commit()

        self._faiss_index: Any = None
        self._faiss_ids: list[int] = []
        self._build_faiss_index()

        log.info("SkillLibrary: %d skills loaded from %s", len(self), self.db_path)

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM skills").fetchone()
        return int(row[0])

    def _get_embedder(self) -> Any:
        """Lazy-load a simple text embedder for skill descriptions."""
        if not hasattr(self, "_embedder"):
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import]
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                self._embedder = None
        return self._embedder

    def _embed(self, text: str) -> np.ndarray | None:
        """Embed text to dense vector. Returns None if embedder unavailable."""
        embedder = self._get_embedder()
        if embedder is None:
            return None
        vec = embedder.encode([text], normalize_embeddings=True)[0]
        return vec.astype(np.float32)

    def _build_faiss_index(self) -> None:
        """Rebuild FAISS index from all skills in DB with embeddings."""
        try:
            import faiss  # type: ignore[import]
        except ImportError:
            log.warning("faiss-cpu not installed — semantic retrieval unavailable. pip install faiss-cpu")
            return

        rows = self._conn.execute(
            "SELECT id, embedding FROM skills WHERE embedding IS NOT NULL"
        ).fetchall()

        if not rows:
            return

        ids = [r[0] for r in rows]
        vecs = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows], axis=0)

        index = faiss.IndexFlatIP(self.embedding_dim)  # inner product (cosine on normalised)
        index.add(vecs)  # type: ignore[arg-type]
        self._faiss_index = index
        self._faiss_ids = ids
        log.debug("FAISS index rebuilt with %d skill vectors.", len(ids))

    def add_skill(
        self,
        name: str,
        description: str,
        code_hint: str = "",
        camatk_score: float = 0.0,
        force: bool = False,
    ) -> bool:
        """
        Add a skill to the library.

        Only stores skills with camatk_score >= quality_threshold (or force=True).
        Returns True if the skill was stored, False if rejected.
        """
        if camatk_score < self.quality_threshold and not force:
            log.debug("Skill '%s' rejected: camatk=%.3f < threshold=%.3f", name, camatk_score, self.quality_threshold)
            return False

        embedding_bytes: bytes | None = None
        emb = self._embed(description)
        if emb is not None:
            embedding_bytes = emb.tobytes()

        try:
            self._conn.execute(
                """
                INSERT INTO skills (name, description, code_hint, camatk_score, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    description=excluded.description,
                    code_hint=excluded.code_hint,
                    camatk_score=MAX(skills.camatk_score, excluded.camatk_score),
                    embedding=excluded.embedding
                """,
                (name, description, code_hint, camatk_score, embedding_bytes, time.time()),
            )
            self._conn.commit()
        except sqlite3.IntegrityError as exc:
            log.warning("Skill insert failed: %s", exc)
            return False

        # Rebuild FAISS index to include new skill
        if embedding_bytes is not None:
            self._build_faiss_index()

        log.info("Skill stored: '%s' (camatk=%.3f)", name, camatk_score)
        return True

    def retrieve(self, query: str, k: int = 5) -> list[Skill]:
        """
        Retrieve k most relevant skills for a query string.

        Uses FAISS semantic search if available, falls back to SQLite
        ordered by camatkāra score.

        Returns list of Skill objects (up to k, possibly fewer).
        """
        if self._faiss_index is not None and self._faiss_ids:
            emb = self._embed(query)
            if emb is not None:
                scores, indices = self._faiss_index.search(
                    emb.reshape(1, -1), min(k, len(self._faiss_ids))
                )
                skill_ids = [self._faiss_ids[i] for i in indices[0] if i >= 0]
                rows = self._conn.execute(
                    f"SELECT * FROM skills WHERE id IN ({','.join(['?'] * len(skill_ids))})",
                    skill_ids,
                ).fetchall()
                self._increment_retrieved(skill_ids)
                return [self._row_to_skill(r) for r in rows]

        # Fallback: return top-k by camatk_score
        rows = self._conn.execute(
            "SELECT * FROM skills ORDER BY camatk_score DESC LIMIT ?", (k,)
        ).fetchall()
        return [self._row_to_skill(r) for r in rows]

    def _increment_retrieved(self, ids: list[int]) -> None:
        """Increment retrieval counter for retrieved skills."""
        self._conn.executemany(
            "UPDATE skills SET n_retrieved = n_retrieved + 1 WHERE id = ?",
            [(i,) for i in ids],
        )
        self._conn.commit()

    def _row_to_skill(self, row: tuple) -> Skill:
        """Convert a SQLite row to a Skill dataclass."""
        _id, name, description, code_hint, camatk_score, embedding_bytes, created_at, n_retrieved = row
        embedding = None
        if embedding_bytes is not None:
            arr = np.frombuffer(embedding_bytes, dtype=np.float32)
            embedding = arr.tolist()
        return Skill(
            name=name,
            description=description,
            code_hint=code_hint or "",
            camatk_score=float(camatk_score),
            embedding=embedding,
            created_at=float(created_at),
            n_retrieved=int(n_retrieved),
        )

    def list_skills(self, min_score: float = 0.0, limit: int = 100) -> list[Skill]:
        """Return all skills above min_score, ordered by score."""
        rows = self._conn.execute(
            "SELECT * FROM skills WHERE camatk_score >= ? ORDER BY camatk_score DESC LIMIT ?",
            (min_score, limit),
        ).fetchall()
        return [self._row_to_skill(r) for r in rows]

    def delete_skill(self, name: str) -> bool:
        """Remove a skill by name. Returns True if it existed."""
        cur = self._conn.execute("DELETE FROM skills WHERE name = ?", (name,))
        self._conn.commit()
        if cur.rowcount > 0:
            self._build_faiss_index()
            return True
        return False

    def export_json(self, path: str | Path) -> None:
        """Export all skills to JSON for inspection / transfer."""
        skills = self.list_skills()
        data = [
            {k: v for k, v in asdict(s).items() if k != "embedding"}
            for s in skills
        ]
        Path(path).write_text(json.dumps(data, indent=2))
        log.info("Exported %d skills to %s", len(data), path)

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()

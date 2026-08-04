"""SQLite metadata store.

A thin, dependency-free layer over Python's built-in ``sqlite3`` module that
records the *structured* facts about the retrieval system:

* **images** -- every patch in the dataset: its class label/name and split.
* **galleries** -- per-modality searchable galleries that have been built,
  with the vector count and the on-disk index path.
* **retrieval_logs** -- a bounded history of retrieval queries so the web app
  can show "recently performed retrievals".

The actual embedding vectors and FAISS indices live in :mod:`embedding_store`
and :mod:`index_store`; this store holds only small relational metadata.

**Thread safety.** FastAPI runs sync endpoints in a worker threadpool, so this
store opens a fresh SQLite connection per operation rather than reusing a
connection bound to the startup thread. SQLite serializes writes internally;
readers/writers are coordinated by the database file.
"""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id          INTEGER PRIMARY KEY,
    class_label INTEGER,
    class_name  TEXT,
    split       TEXT
);

CREATE TABLE IF NOT EXISTS galleries (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT UNIQUE,
    modality      TEXT,
    num_vectors   INTEGER,
    embedding_dim INTEGER,
    index_path    TEXT,
    config_hash   TEXT,
    created_at    REAL
);

CREATE TABLE IF NOT EXISTS retrieval_logs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id     INTEGER,
    query_mod    TEXT,
    gallery_mod  TEXT,
    k            INTEGER,
    avg_time_ms  REAL,
    n_retrieved  INTEGER,
    created_at   REAL
);
"""


class MetadataStore:
    """Persistent SQLite-backed metadata store (thread-safe per operation)."""

    def __init__(self, db_path: str = "database/metadata.db") -> None:
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.db_path = db_path
        # Create the schema once; subsequent operations open their own conn.
        conn = self._connect_db()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _connect_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _fetch(self, sql: str, params: Tuple = ()) -> List[Dict[str, Any]]:
        conn = self._connect_db()
        try:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def _scalar(self, sql: str, params: Tuple = ()):
        conn = self._connect_db()
        try:
            return conn.execute(sql, params).fetchone()[0]
        finally:
            conn.close()

    def _execute(self, sql: str, params: Tuple = ()) -> None:
        conn = self._connect_db()
        try:
            conn.execute(sql, params)
            conn.commit()
        finally:
            conn.close()

    # -- context management --------------------------------------------------
    def close(self) -> None:
        # Connections are short-lived per operation; nothing to close.
        pass

    def __enter__(self) -> "MetadataStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- images --------------------------------------------------------------
    def save_images(self, labels: np.ndarray, class_names: Sequence[str], splits: Sequence[str]) -> None:
        """Bulk upsert image rows. ``splits`` is per-image 'train'|'val'|'test'."""
        conn = self._connect_db()
        try:
            conn.executemany(
                "INSERT OR REPLACE INTO images (id, class_label, class_name, split) VALUES (?, ?, ?, ?)",
                [
                    (int(i), int(labels[i]), str(class_names[int(labels[i])]), str(splits[i]))
                    for i in range(int(labels.shape[0]))
                ],
            )
            conn.commit()
        finally:
            conn.close()

    def get_image(self, image_id: int) -> Optional[Dict[str, Any]]:
        rows = self._fetch("SELECT * FROM images WHERE id = ?", (int(image_id),))
        return rows[0] if rows else None

    def image_count(self) -> int:
        return int(self._scalar("SELECT COUNT(*) FROM images"))

    def class_counts(self) -> Dict[str, int]:
        rows = self._fetch("SELECT class_name, COUNT(*) AS n FROM images GROUP BY class_name")
        return {r["class_name"]: int(r["n"]) for r in rows}

    # -- galleries -----------------------------------------------------------
    def save_gallery(
        self,
        name: str,
        modality: str,
        num_vectors: int,
        embedding_dim: int,
        index_path: str,
        config_hash: str,
    ) -> None:
        self._execute(
            "INSERT OR REPLACE INTO galleries "
            "(name, modality, num_vectors, embedding_dim, index_path, config_hash, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, modality, int(num_vectors), int(embedding_dim), index_path, config_hash, time.time()),
        )

    def get_gallery(self, name: str) -> Optional[Dict[str, Any]]:
        rows = self._fetch("SELECT * FROM galleries WHERE name = ?", (name,))
        return rows[0] if rows else None

    def list_galleries(self) -> List[Dict[str, Any]]:
        return self._fetch("SELECT * FROM galleries ORDER BY created_at DESC")

    # -- retrieval logs ------------------------------------------------------
    def log_retrieval(
        self,
        query_id: int,
        query_mod: str,
        gallery_mod: str,
        k: int,
        avg_time_ms: float,
        n_retrieved: int = 0,
    ) -> None:
        self._execute(
            "INSERT INTO retrieval_logs "
            "(query_id, query_mod, gallery_mod, k, avg_time_ms, n_retrieved, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (int(query_id), query_mod, gallery_mod, int(k), float(avg_time_ms), int(n_retrieved), time.time()),
        )

    def recent_retrievals(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self._fetch(
            "SELECT * FROM retrieval_logs ORDER BY created_at DESC LIMIT ?", (int(limit),)
        )

    def retrieval_stats(self) -> Dict[str, Any]:
        total = int(self._scalar("SELECT COUNT(*) FROM retrieval_logs"))
        avg = self._scalar(
            "SELECT AVG(avg_time_ms) FROM retrieval_logs WHERE avg_time_ms IS NOT NULL"
        )
        return {"total_queries": total, "avg_time_ms": float(avg) if avg is not None else None}

    # -- misc ----------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        return {
            "db_path": self.db_path,
            "images": self.image_count(),
            "galleries": int(self._scalar("SELECT COUNT(*) FROM galleries")),
            "queries_logged": int(self._scalar("SELECT COUNT(*) FROM retrieval_logs")),
        }
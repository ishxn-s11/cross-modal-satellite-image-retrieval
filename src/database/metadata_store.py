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
    id               INTEGER PRIMARY KEY,
    class_label      INTEGER,
    class_name       TEXT,
    split            TEXT,
    dataset          TEXT,
    sensor           TEXT,
    latitude         REAL,
    longitude        REAL,
    acquisition_date TEXT,
    cloud_cover      REAL,
    resolution       REAL,
    orbit            TEXT,
    file_path        TEXT
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

CREATE TABLE IF NOT EXISTS datasets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT UNIQUE,
    sensor      TEXT,
    n_images    INTEGER,
    n_classes   INTEGER,
    modalities  TEXT,
    metadata    TEXT,
    created_at  REAL
);

CREATE TABLE IF NOT EXISTS embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    modality    TEXT,
    config_hash TEXT,
    dim         INTEGER,
    n_vectors   INTEGER,
    path        TEXT,
    created_at  REAL
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    config_hash TEXT,
    pair        TEXT,
    kind        TEXT,
    k           INTEGER,
    metric      TEXT,
    value       REAL,
    created_at  REAL
);

CREATE TABLE IF NOT EXISTS model_versions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT UNIQUE,
    config_hash TEXT,
    path        TEXT,
    metrics     TEXT,
    created_at  REAL
);
"""

# Columns added by migrations (kept so re-running is a no-op).
_IMAGE_EXTRA_COLUMNS = {
    "dataset": "TEXT",
    "sensor": "TEXT",
    "latitude": "REAL",
    "longitude": "REAL",
    "acquisition_date": "TEXT",
    "cloud_cover": "REAL",
    "resolution": "REAL",
    "orbit": "TEXT",
    "file_path": "TEXT",
}


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
            self._migrate(conn)
            conn.commit()
        finally:
            conn.close()

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Idempotent migration: add nullable metadata columns to ``images``.

        Existing databases created before the metadata schema are upgraded in
        place; the columns are simply missing on very old tables and `None`.
        """
        existing = {r[1] for r in conn.execute("PRAGMA table_info(images)")}
        for col, decl in _IMAGE_EXTRA_COLUMNS.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE images ADD COLUMN {col} {decl}")

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
    def save_images(
        self,
        labels: np.ndarray,
        class_names: Sequence[str],
        splits: Sequence[str],
        metadata: Optional[Sequence] = None,
    ) -> None:
        """Bulk upsert image rows.

        ``splits`` is per-image 'train'|'val'|'test'. ``metadata`` (optional) is
        a sequence of :class:`~src.data.metadata.ImageMetadata` used to fill the
        nullable geographic/radiometric columns (None when a dataset has none).
        """
        meta = list(metadata or [])
        conn = self._connect_db()
        try:
            rows = []
            for i in range(int(labels.shape[0])):
                m = meta[i] if i < len(meta) else None
                rows.append(
                    (
                        int(i),
                        int(labels[i]),
                        str(class_names[int(labels[i])]),
                        str(splits[i]),
                        getattr(m, "dataset", None) if m else None,
                        getattr(m, "sensor", None) if m else None,
                        getattr(m, "latitude", None) if m else None,
                        getattr(m, "longitude", None) if m else None,
                        getattr(m, "acquisition_date", None) if m else None,
                        getattr(m, "cloud_cover", None) if m else None,
                        getattr(m, "resolution", None) if m else None,
                        getattr(m, "orbit", None) if m else None,
                        getattr(m, "file_path", None) if m else None,
                    )
                )
            conn.executemany(
                "INSERT OR REPLACE INTO images "
                "(id, class_label, class_name, split, dataset, sensor, latitude, longitude, "
                " acquisition_date, cloud_cover, resolution, orbit, file_path) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
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

    # -- datasets ------------------------------------------------------------
    def save_dataset(
        self,
        name: str,
        sensor: Optional[str],
        n_images: int,
        n_classes: int,
        modalities: Sequence[str],
        metadata: Optional[Dict] = None,
    ) -> None:
        self._execute(
            "INSERT OR REPLACE INTO datasets "
            "(name, sensor, n_images, n_classes, modalities, metadata, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, sensor, int(n_images), int(n_classes),
             ",".join(modalities), (metadata or None), time.time()),
        )

    def get_dataset(self, name: str) -> Optional[Dict[str, Any]]:
        rows = self._fetch("SELECT * FROM datasets WHERE name = ?", (name,))
        return rows[0] if rows else None

    # -- evaluation ----------------------------------------------------------
    def save_evaluation_result(
        self, config_hash: str, pair: str, kind: str, k: int, metric: str, value: float
    ) -> None:
        self._execute(
            "INSERT INTO evaluation_results (config_hash, pair, kind, k, metric, value, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (config_hash, pair, kind, int(k), metric, float(value), time.time()),
        )

    def list_evaluation_results(self, config_hash: Optional[str] = None, limit: int = 500) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM evaluation_results"
        params: Tuple = ()
        if config_hash:
            sql += " WHERE config_hash = ?"
            params = (config_hash,)
        sql += " ORDER BY created_at DESC LIMIT ?"
        return self._fetch(sql, params + (int(limit),))

    # -- model versions ------------------------------------------------------
    def save_model_version(self, name: str, config_hash: str, path: str, metrics: Optional[Dict] = None) -> None:
        import json as _json

        self._execute(
            "INSERT OR REPLACE INTO model_versions (name, config_hash, path, metrics, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, config_hash, path, _json.dumps(metrics) if metrics else None, time.time()),
        )

    def list_model_versions(self) -> List[Dict[str, Any]]:
        return self._fetch("SELECT * FROM model_versions ORDER BY created_at DESC")

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
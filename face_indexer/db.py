from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path


SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS images (
 image_id TEXT PRIMARY KEY, file_path TEXT NOT NULL, file_name TEXT NOT NULL,
 file_hash TEXT NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
 file_size INTEGER NOT NULL, face_count INTEGER NOT NULL DEFAULT 0,
 status TEXT NOT NULL, error_message TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS faces (
 face_id TEXT PRIMARY KEY, image_id TEXT NOT NULL, face_crop_path TEXT NOT NULL,
 bbox_x REAL NOT NULL, bbox_y REAL NOT NULL, bbox_width REAL NOT NULL, bbox_height REAL NOT NULL,
 detection_score REAL, embedding BLOB NOT NULL, embedding_dim INTEGER NOT NULL,
 embedding_norm REAL, group_id TEXT, quality_score REAL, status TEXT NOT NULL,
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
 FOREIGN KEY(image_id) REFERENCES images(image_id));
CREATE TABLE IF NOT EXISTS face_groups (
 group_id TEXT PRIMARY KEY, representative_face_id TEXT, face_count INTEGER NOT NULL DEFAULT 0,
 image_count INTEGER NOT NULL DEFAULT 0, group_center_embedding BLOB, embedding_dim INTEGER,
 status TEXT NOT NULL, note TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS group_images (
 group_id TEXT NOT NULL, image_id TEXT NOT NULL, face_count_in_image INTEGER NOT NULL DEFAULT 1,
 PRIMARY KEY(group_id,image_id), FOREIGN KEY(group_id) REFERENCES face_groups(group_id),
 FOREIGN KEY(image_id) REFERENCES images(image_id));
CREATE TABLE IF NOT EXISTS queries (
 query_id TEXT PRIMARY KEY, query_image_path TEXT NOT NULL, query_face_crop_path TEXT,
 matched_group_id TEXT, best_face_id TEXT, best_distance REAL, confidence TEXT NOT NULL,
 status TEXT NOT NULL, message TEXT, result_json_path TEXT, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_images_file_hash ON images(file_hash);
CREATE INDEX IF NOT EXISTS idx_faces_image_id ON faces(image_id);
CREATE INDEX IF NOT EXISTS idx_faces_group_id ON faces(group_id);
CREATE INDEX IF NOT EXISTS idx_group_images_group_id ON group_images(group_id);
CREATE INDEX IF NOT EXISTS idx_group_images_image_id ON group_images(image_id);
"""


class Database:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    def next_id(self, table: str, column: str, prefix: str) -> str:
        allowed = {("images", "image_id"), ("faces", "face_id"), ("queries", "query_id")}
        if (table, column) not in allowed:
            raise ValueError("Unsupported ID target")
        with self.connect() as connection:
            rows = connection.execute(f"SELECT {column} FROM {table}").fetchall()
        highest = max((int(row[0].rsplit("_", 1)[1]) for row in rows), default=0)
        return f"{prefix}_{highest + 1:06d}"

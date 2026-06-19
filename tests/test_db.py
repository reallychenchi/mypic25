import tempfile
import unittest
from pathlib import Path

import numpy as np

from face_indexer.db import Database


class DatabaseTests(unittest.TestCase):
    def test_embedding_round_trip(self):
        with tempfile.TemporaryDirectory() as value:
            db = Database(Path(value) / "index.db")
            db.initialize()
            timestamp = "2026-01-01T00:00:00+00:00"
            vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            with db.connect() as connection:
                connection.execute("INSERT INTO images VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                                   ("img_000001", "/a.jpg", "a.jpg", "hash", 10, 10, 1, 1, "success", None, timestamp, timestamp))
                connection.execute("INSERT INTO faces VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                   ("face_000001", "img_000001", "/face.jpg", 1, 2, 3, 4, 0.9,
                                    vector.tobytes(), vector.size, 1.0, None, 0.9, "valid", timestamp, timestamp))
            with db.connect() as connection:
                row = connection.execute("SELECT embedding,embedding_dim FROM faces").fetchone()
            actual = np.frombuffer(row["embedding"], dtype=np.float32, count=row["embedding_dim"])
            np.testing.assert_array_equal(vector, actual)


if __name__ == "__main__":
    unittest.main()

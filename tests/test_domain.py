import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from face_indexer.domain import cosine_distances, crop_face, dbscan_cosine, scan_images, vote


class DomainTests(unittest.TestCase):
    def test_scan_images_filters_and_recurses(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            (root / "nested").mkdir()
            (root / "a.JPG").touch()
            (root / "notes.txt").touch()
            (root / "nested" / "b.png").touch()
            self.assertEqual(2, len(scan_images(root, [".jpg", ".png"], True)))
            self.assertEqual(1, len(scan_images(root, [".jpg", ".png"], False)))

    def test_crop_does_not_cross_image_boundary(self):
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        crop = crop_face(image, (0, 0, 40, 50), 0.25)
        self.assertEqual((63, 50, 3), crop.shape)

    def test_cosine_distance(self):
        distances = cosine_distances(np.array([1, 0]), np.array([[1, 0], [0, 1], [-1, 0]]))
        np.testing.assert_allclose(distances, [0, 1, 2], atol=1e-6)

    def test_group_vote_and_tie_break(self):
        matches = [
            {"group_id": "group_b", "distance": 0.1},
            {"group_id": "group_a", "distance": 0.2},
            {"group_id": "group_a", "distance": 0.3},
            {"group_id": "group_b", "distance": 0.4},
        ]
        result = vote(matches, 2, 0.5)
        self.assertEqual("group_b", result["group_id"])
        self.assertTrue(result["passes_vote"])

    def test_dbscan_and_noise(self):
        vectors = np.array([[1, 0], [0.999, 0.04], [0, 1]], dtype=np.float32)
        labels = dbscan_cosine(vectors, eps=0.01, min_samples=2)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(-1, labels[2])


if __name__ == "__main__":
    unittest.main()

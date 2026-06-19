import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from face_indexer.domain import DetectedFace
from face_indexer.services import BuildIndexService, ReClusterService, ReportService, SearchFaceService


class FakeEngine:
    def __init__(self, config):
        self.config = config

    def detect_and_extract(self, image):
        if image.mean() < 1:
            return []
        return [DetectedFace((10, 10, 50, 50), 0.99, np.array([1, 0, 0], dtype=np.float32))]


class ServiceIntegrationTests(unittest.TestCase):
    @patch("face_indexer.services.InsightFaceEngine", FakeEngine)
    def test_build_search_report_and_recluster(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            photos = root / "photos"
            photos.mkdir()
            white = np.full((80, 80, 3), 255, dtype=np.uint8)
            black = np.zeros((80, 80, 3), dtype=np.uint8)
            cv2.imwrite(str(photos / "person_1.jpg"), white)
            cv2.imwrite(str(photos / "person_2.jpg"), white)
            cv2.imwrite(str(photos / "no_face.jpg"), black)
            config_path = root / "config.json"
            config_path.write_text(json.dumps({
                "cluster": {"eps": 0.1, "min_samples": 2},
                "search": {"top_k": 2, "min_group_vote_count": 1, "min_group_vote_ratio": 0.5},
            }), encoding="utf-8")
            workspace = root / "workspace"

            build = BuildIndexService().run(photos, workspace, config_path=config_path)
            self.assertEqual(3, build["total_images"])
            # Identical white JPEGs are deduplicated by content hash.
            self.assertEqual(1, build["valid_faces"])
            self.assertEqual(1, build["total_groups"])
            report = ReportService().report(workspace)
            self.assertEqual(1, report["duplicated_images"])
            self.assertEqual(1, report["no_face_images"])

            export = root / "export"
            result = SearchFaceService().run(photos / "person_1.jpg", workspace, export=export)
            self.assertTrue(result["matched"])
            self.assertEqual("group_000001", result["matched_group_id"])
            self.assertTrue((export / "result.json").is_file())
            self.assertTrue((export / "matched_images.csv").is_file())
            self.assertTrue((workspace / "queries" / result["query_id"] / "result.json").is_file())

            clustered = ReClusterService().run(workspace)
            self.assertEqual(1, clustered["total_groups"])


if __name__ == "__main__":
    unittest.main()

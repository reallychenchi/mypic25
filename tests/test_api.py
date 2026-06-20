import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from face_indexer.api import create_app
from face_indexer.services import QueryFaceCountError


class ApiTests(unittest.TestCase):
    def make_app(self, root: Path, *, download_limit: int = 3):
        workspace = root / "workspace"
        workspace.mkdir()
        (workspace / "config.json").write_text(json.dumps({
            "api": {
                "download_max_concurrency": download_limit,
                "zip_retention_hours": 48,
                "max_upload_bytes": 1024 * 1024,
            }
        }), encoding="utf-8")
        return create_app(workspace)

    def test_search_uses_isolated_directories_and_stored_zip(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            source = root / "original.jpg"
            source.write_bytes(b"original-image")
            missing = root / "missing.jpg"
            app = self.make_app(root)

            def fake_search(*args, **kwargs):
                self.assertTrue(kwargs["require_exactly_one_face"])
                self.assertFalse(kwargs["write_exports"])
                return {
                    "matched": True,
                    "matched_group_id": "group_000001",
                    "matched_images": [
                        {"image_id": "img_000001", "file_path": str(source)},
                        {"image_id": "img_000002", "file_path": str(missing)},
                    ],
                }

            app.state.runtime.search_service.run = fake_search
            with TestClient(app) as client:
                first = client.post("/api/v1/face-searches", files={"image": ("query.jpg", b"one", "image/jpeg")})
                second = client.post("/api/v1/face-searches", files={"image": ("query.jpg", b"two", "image/jpeg")})

                self.assertEqual(200, first.status_code)
                self.assertEqual(1, first.json()["data"]["matched_image_count"])
                self.assertEqual(2, first.json()["data"]["group_image_count"])
                self.assertEqual(1, len(first.json()["data"]["warnings"]))
                self.assertNotEqual(first.json()["request_id"], second.json()["request_id"])

                archive_path = (app.state.runtime.requests_dir / first.json()["request_id"] / "matched-images.zip")
                with zipfile.ZipFile(archive_path) as archive:
                    self.assertEqual(["img_000001_original.jpg"], archive.namelist())
                    self.assertEqual(zipfile.ZIP_STORED, archive.infolist()[0].compress_type)

                download = client.get(first.json()["data"]["download_url"])
                self.assertEqual(200, download.status_code)
                self.assertEqual("application/zip", download.headers["content-type"])
                self.assertIn("attachment", download.headers["content-disposition"])

    def test_frontend_is_served_at_root(self):
        with tempfile.TemporaryDirectory() as value:
            app = self.make_app(Path(value))
            with TestClient(app) as client:
                response = client.get("/")
            self.assertEqual(200, response.status_code)
            self.assertIn("照片匹配", response.text)

    def test_download_limit_returns_429(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            app = self.make_app(root, download_limit=1)
            archive_id = "a" * 32
            directory = app.state.runtime.requests_dir / archive_id
            directory.mkdir()
            with zipfile.ZipFile(directory / "matched-images.zip", "w"):
                pass

            self.assertTrue(app.state.runtime.acquire_download("occupied"))
            try:
                with TestClient(app) as client:
                    response = client.get(f"/api/v1/downloads/{archive_id}")
                self.assertEqual(429, response.status_code)
                self.assertEqual("DOWNLOAD_LIMIT_EXCEEDED", response.json()["code"])
            finally:
                app.state.runtime.release_download("occupied")

    def test_same_archive_remains_active_until_all_downloads_finish(self):
        with tempfile.TemporaryDirectory() as value:
            app = self.make_app(Path(value), download_limit=3)
            runtime = app.state.runtime
            self.assertTrue(runtime.acquire_download("same"))
            self.assertTrue(runtime.acquire_download("same"))
            runtime.release_download("same")
            self.assertTrue(runtime.is_active("same"))
            runtime.release_download("same")
            self.assertFalse(runtime.is_active("same"))

    def test_no_match_has_no_archive(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            app = self.make_app(root)
            app.state.runtime.search_service.run = lambda *args, **kwargs: {
                "matched": False, "matched_group_id": None, "matched_images": []
            }
            with TestClient(app) as client:
                response = client.post("/api/v1/face-searches", files={"image": ("query.jpg", b"image", "image/jpeg")})
            self.assertEqual(200, response.status_code)
            self.assertEqual("NO_MATCH_FOUND", response.json()["code"])
            self.assertIsNone(response.json()["data"]["download_url"])
            self.assertFalse((app.state.runtime.requests_dir / response.json()["request_id"]).exists())

    def test_face_count_errors_are_explicit(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            app = self.make_app(root)
            with TestClient(app) as client:
                for count, code in ((0, "NO_FACE_DETECTED"), (2, "MULTIPLE_FACES_DETECTED")):
                    def fail(*args, **kwargs):
                        raise QueryFaceCountError(count)

                    app.state.runtime.search_service.run = fail
                    response = client.post(
                        "/api/v1/face-searches",
                        files={"image": ("query.jpg", b"image", "image/jpeg")},
                    )
                    self.assertEqual(422, response.status_code)
                    self.assertEqual(code, response.json()["code"])
                    self.assertEqual(count, response.json()["details"]["detected_faces"])


if __name__ == "__main__":
    unittest.main()

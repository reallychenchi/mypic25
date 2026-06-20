from __future__ import annotations

import json
import logging
import shutil
import threading
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from .config import load_config
from .engine import InsightFaceEngine
from .services import QueryFaceCountError, SearchFaceService


LOGGER = logging.getLogger("face_indexer.api")
MAX_RETENTION_HOURS = 48


def _error(status: int, code: str, message: str, request_id: str | None = None,
           details: dict[str, Any] | None = None) -> JSONResponse:
    body: dict[str, Any] = {"code": code, "message": message, "data": None}
    if request_id:
        body["request_id"] = request_id
    if details:
        body["details"] = details
    return JSONResponse(status_code=status, content=body)


class LazySynchronizedEngine:
    """Load InsightFace once and serialize access to the shared model instance."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._engine: InsightFaceEngine | None = None
        self._lock = threading.Lock()

    def detect_and_extract(self, image):
        with self._lock:
            if self._engine is None:
                self._engine = InsightFaceEngine(self.config)
            return self._engine.detect_and_extract(image)


class ApiRuntime:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        config = load_config(self.workspace)
        api_config = config["api"]
        self.download_limit = int(api_config["download_max_concurrency"])
        self.retention_hours = float(api_config["zip_retention_hours"])
        self.max_upload_bytes = int(api_config["max_upload_bytes"])
        if self.download_limit < 1:
            raise ValueError("api.download_max_concurrency must be at least 1")
        if not 0 < self.retention_hours <= MAX_RETENTION_HOURS:
            raise ValueError("api.zip_retention_hours must be greater than 0 and no more than 48")
        if self.max_upload_bytes < 1:
            raise ValueError("api.max_upload_bytes must be at least 1")

        model_config = dict(config["model"])
        model_config.setdefault("model_root", str(self.workspace))
        self.search_service = SearchFaceService(LazySynchronizedEngine(model_config))
        self.requests_dir = self.workspace / "api_requests"
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        self._downloads = threading.BoundedSemaphore(self.download_limit)
        self._state_lock = threading.Lock()
        self._active_archives: dict[str, int] = {}

    def acquire_download(self, archive_id: str) -> bool:
        if not self._downloads.acquire(blocking=False):
            return False
        with self._state_lock:
            self._active_archives[archive_id] = self._active_archives.get(archive_id, 0) + 1
        return True

    def release_download(self, archive_id: str) -> None:
        with self._state_lock:
            remaining = self._active_archives.get(archive_id, 0) - 1
            if remaining > 0:
                self._active_archives[archive_id] = remaining
            else:
                self._active_archives.pop(archive_id, None)
        self._downloads.release()

    def is_active(self, archive_id: str) -> bool:
        with self._state_lock:
            return self._active_archives.get(archive_id, 0) > 0

    def remove_archive_if_inactive(self, archive_id: str) -> bool:
        with self._state_lock:
            if self._active_archives.get(archive_id, 0) > 0:
                return False
            shutil.rmtree(self.requests_dir / archive_id, ignore_errors=True)
            return True

    def cleanup_expired(self) -> None:
        cutoff = datetime.now(timezone.utc).timestamp() - self.retention_hours * 3600
        for directory in self.requests_dir.iterdir():
            archive = directory / "matched-images.zip"
            try:
                if archive.is_file() and archive.stat().st_mtime < cutoff:
                    self.remove_archive_if_inactive(directory.name)
            except OSError:
                LOGGER.warning("Could not clean expired API request directory %s", directory, exc_info=True)


def _archive_images(result: dict[str, Any], archive_path: Path) -> tuple[int, list[str]]:
    warnings: list[str] = []
    count = 0
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for item in result["matched_images"]:
            source = Path(item["file_path"])
            if not source.is_file():
                warnings.append(f"原图不存在，已跳过：{source}")
                continue
            archive_name = f"{item['image_id']}_{source.name}"
            try:
                archive.write(source, arcname=archive_name, compress_type=zipfile.ZIP_STORED)
                count += 1
            except OSError as error:
                warnings.append(f"原图读取失败，已跳过：{source}（{error}）")
    return count, warnings


def create_app(workspace: Path) -> FastAPI:
    runtime = ApiRuntime(workspace)
    app = FastAPI(title="Face Indexer API", version="1.0.0")
    app.state.runtime = runtime

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, error: RequestValidationError):
        return _error(422, "INVALID_REQUEST", "请求参数不合法",
                      details={"errors": jsonable_encoder(error.errors())})

    @app.post("/api/v1/face-searches", name="search_face")
    async def search_face(request: Request, image: UploadFile):
        request_id = uuid.uuid4().hex
        request_dir = runtime.requests_dir / request_id
        request_dir.mkdir(parents=False, exist_ok=False)
        runtime.cleanup_expired()

        original_name = Path(image.filename or "upload").name
        suffix = Path(original_name).suffix.lower()
        allowed = {value.lower() for value in load_config(runtime.workspace)["image"]["supported_extensions"]}
        if suffix not in allowed:
            shutil.rmtree(request_dir, ignore_errors=True)
            return _error(415, "UNSUPPORTED_IMAGE_TYPE", "不支持的图片格式", request_id,
                          {"supported_extensions": sorted(allowed)})

        upload_path = request_dir / f"query{suffix}"
        size = 0
        try:
            with upload_path.open("wb") as output:
                while chunk := await image.read(1024 * 1024):
                    size += len(chunk)
                    if size > runtime.max_upload_bytes:
                        raise OverflowError
                    output.write(chunk)
        except OverflowError:
            shutil.rmtree(request_dir, ignore_errors=True)
            return _error(413, "IMAGE_TOO_LARGE", "上传图片超过大小限制", request_id,
                          {"max_upload_bytes": runtime.max_upload_bytes})
        except OSError:
            shutil.rmtree(request_dir, ignore_errors=True)
            LOGGER.exception("Could not persist upload for request %s", request_id)
            return _error(500, "UPLOAD_SAVE_FAILED", "保存上传图片失败", request_id)
        finally:
            await image.close()

        if size == 0:
            shutil.rmtree(request_dir, ignore_errors=True)
            return _error(400, "EMPTY_IMAGE", "上传图片内容为空", request_id)

        try:
            result = await run_in_threadpool(
                runtime.search_service.run,
                upload_path,
                runtime.workspace,
                copy_images=False,
                require_exactly_one_face=True,
                write_exports=False,
            )
        except QueryFaceCountError as error:
            shutil.rmtree(request_dir, ignore_errors=True)
            code = "NO_FACE_DETECTED" if error.detected_count == 0 else "MULTIPLE_FACES_DETECTED"
            message = "图片中未检测到人脸" if error.detected_count == 0 else "图片中检测到多张人脸，必须恰好包含一张人脸"
            return _error(422, code, message, request_id, {"detected_faces": error.detected_count})
        except ValueError as error:
            shutil.rmtree(request_dir, ignore_errors=True)
            if (str(error).startswith("NO_INDEX_DATA")
                    or str(error).startswith("Database does not exist")
                    or str(error).startswith("Workspace does not exist")):
                return _error(503, "NO_INDEX_DATA", "人脸索引中没有可检索的数据", request_id)
            return _error(400, "INVALID_IMAGE", "图片无法读取或处理", request_id)
        except (RuntimeError, ImportError) as error:
            shutil.rmtree(request_dir, ignore_errors=True)
            LOGGER.exception("Inference unavailable for request %s", request_id)
            return _error(503, "INFERENCE_UNAVAILABLE", str(error), request_id)
        except Exception:
            shutil.rmtree(request_dir, ignore_errors=True)
            LOGGER.exception("Face search failed for request %s", request_id)
            return _error(500, "INTERNAL_ERROR", "人脸检索失败", request_id)

        if not result["matched"]:
            shutil.rmtree(request_dir, ignore_errors=True)
            return JSONResponse(status_code=200, content={
                "code": "NO_MATCH_FOUND",
                "message": "未找到匹配分组",
                "request_id": request_id,
                "data": {"group_id": None, "matched_image_count": 0, "download_url": None,
                         "expires_at": None, "warnings": []},
            })

        archive_path = request_dir / "matched-images.zip"
        try:
            packaged_count, warnings = await run_in_threadpool(_archive_images, result, archive_path)
        except (OSError, zipfile.BadZipFile):
            shutil.rmtree(request_dir, ignore_errors=True)
            LOGGER.exception("Archive creation failed for request %s", request_id)
            return _error(500, "ARCHIVE_CREATE_FAILED", "ZIP 文件创建失败", request_id)

        expires_at = datetime.now(timezone.utc) + timedelta(hours=runtime.retention_hours)
        data = {
            "group_id": result["matched_group_id"],
            "matched_image_count": packaged_count,
            "group_image_count": len(result["matched_images"]),
            "download_url": str(request.url_for("download_archive", archive_id=request_id)),
            "expires_at": expires_at.isoformat(),
            "warnings": warnings,
        }
        response_body = {"code": "OK", "message": "匹配及打包完成", "request_id": request_id, "data": data}
        (request_dir / "response.json").write_text(
            json.dumps(response_body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return response_body

    @app.get("/api/v1/downloads/{archive_id}", name="download_archive")
    async def download_archive(archive_id: str):
        try:
            uuid.UUID(hex=archive_id)
        except ValueError:
            return _error(404, "ARCHIVE_NOT_FOUND", "下载文件不存在")

        archive_path = runtime.requests_dir / archive_id / "matched-images.zip"
        if not archive_path.is_file():
            return _error(404, "ARCHIVE_NOT_FOUND", "下载文件不存在")
        expires_at = datetime.fromtimestamp(archive_path.stat().st_mtime, timezone.utc) + timedelta(hours=runtime.retention_hours)
        if datetime.now(timezone.utc) >= expires_at:
            runtime.remove_archive_if_inactive(archive_id)
            return _error(410, "ARCHIVE_EXPIRED", "下载文件已过期")
        if not runtime.acquire_download(archive_id):
            return _error(429, "DOWNLOAD_LIMIT_EXCEEDED", "当前下载数量已达到上限，请稍后重试")
        if not archive_path.is_file():
            runtime.release_download(archive_id)
            return _error(404, "ARCHIVE_NOT_FOUND", "下载文件不存在")
        return FileResponse(
            archive_path,
            media_type="application/zip",
            filename=f"face-group-{archive_id}.zip",
            background=BackgroundTask(runtime.release_download, archive_id),
        )

    frontend_dir = Path(__file__).with_name("frontend")
    if frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

    return app

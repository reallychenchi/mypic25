from __future__ import annotations

import csv
import hashlib
import json
import logging
import shutil
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

import cv2
import numpy as np

from .config import initialize_workspace, load_config, save_config
from .db import Database
from .domain import (
    choose_face, cosine_distances, crop_face, dbscan_cosine, normalize,
    save_jpeg, scan_images, vote,
)
from .engine import InsightFaceEngine


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_logging(workspace: Path, operation: str) -> logging.Logger:
    logger = logging.getLogger(f"face_indexer.{operation}.{workspace}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        for filename, level in ((f"{operation}.log", logging.INFO), ("error.log", logging.ERROR)):
            handler = logging.FileHandler(workspace / "logs" / filename, encoding="utf-8")
            handler.setLevel(level)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    return logger


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _thumbnail(image: np.ndarray, max_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    ratio = min(1.0, max_size / max(height, width))
    if ratio == 1.0:
        return image
    return cv2.resize(image, (round(width * ratio), round(height * ratio)), interpolation=cv2.INTER_AREA)


def _runtime_model_config(config: dict, workspace: Path) -> dict:
    model_config = dict(config["model"])
    model_config.setdefault("model_root", str(workspace.resolve()))
    return model_config


def rebuild_groups(workspace: Path, config: dict, database: Database) -> dict:
    cluster_config = config["cluster"]
    statuses = ["valid"]
    if cluster_config.get("cluster_include_low_quality_faces", False):
        statuses.append("low_quality")
    placeholders = ",".join("?" for _ in statuses)
    with database.connect() as connection:
        rows = connection.execute(
            f"SELECT face_id,image_id,face_crop_path,embedding,embedding_dim FROM faces WHERE status IN ({placeholders}) ORDER BY face_id",
            statuses,
        ).fetchall()
    embeddings = [np.frombuffer(row["embedding"], dtype=np.float32, count=row["embedding_dim"]).copy() for row in rows]
    labels = dbscan_cosine(np.asarray(embeddings), float(cluster_config["eps"]), int(cluster_config["min_samples"]))

    assignments: dict[str, list[int]] = defaultdict(list)
    positive_labels = sorted(set(labels.tolist()) - {-1})
    for group_number, label in enumerate(positive_labels, 1):
        assignments[f"group_{group_number:06d}"] = np.flatnonzero(labels == label).tolist()
    next_number = len(assignments) + 1
    if cluster_config.get("noise_as_singleton_group", True):
        for index in np.flatnonzero(labels == -1):
            assignments[f"group_{next_number:06d}"] = [int(index)]
            next_number += 1

    groups_dir = workspace / "groups"
    shutil.rmtree(groups_dir, ignore_errors=True)
    groups_dir.mkdir(parents=True, exist_ok=True)
    timestamp = now()
    with database.connect() as connection:
        connection.execute("UPDATE faces SET group_id=NULL, updated_at=?", (timestamp,))
        connection.execute("DELETE FROM group_images")
        connection.execute("DELETE FROM face_groups")
        for group_id, indexes in assignments.items():
            group_embeddings = np.stack([embeddings[index] for index in indexes])
            center = normalize(group_embeddings.mean(axis=0))
            distances = cosine_distances(center, group_embeddings)
            representative = rows[indexes[int(np.argmin(distances))]]["face_id"]
            image_counts: dict[str, int] = defaultdict(int)
            for index in indexes:
                row = rows[index]
                image_counts[row["image_id"]] += 1
                connection.execute("UPDATE faces SET group_id=?,updated_at=? WHERE face_id=?", (group_id, timestamp, row["face_id"]))
                source = Path(row["face_crop_path"])
                if source.exists():
                    target_dir = groups_dir / group_id
                    target_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target_dir / source.name)
            connection.execute(
                "INSERT INTO face_groups VALUES (?,?,?,?,?,?,?,?,?,?)",
                (group_id, representative, len(indexes), len(image_counts), center.tobytes(), center.size, "active", None, timestamp, timestamp),
            )
            connection.executemany(
                "INSERT INTO group_images(group_id,image_id,face_count_in_image) VALUES (?,?,?)",
                [(group_id, image_id, count) for image_id, count in image_counts.items()],
            )
    singleton_count = sum(len(indexes) == 1 for indexes in assignments.values())
    return {"total_faces": len(rows), "total_groups": len(assignments), "singleton_groups": singleton_count}


class BuildIndexService:
    def run(self, input_dir: Path, workspace: Path, *, config_path: Path | None = None,
            recursive: bool = True, force: bool = False, copy_originals: bool | None = None,
            device: str | None = None) -> dict:
        if not input_dir.is_dir():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        initialize_workspace(workspace)
        config = load_config(workspace, config_path)
        if device:
            config["model"]["device"] = device
        if copy_originals is not None:
            config["image"]["copy_originals"] = copy_originals
        save_config(workspace, config)
        logger = configure_logging(workspace, "build")
        database = Database(workspace / "database" / "face_index.db")
        if force and database.path.exists():
            database.path.unlink()
            for folder in (workspace / "faces", workspace / "groups", workspace / "images" / "thumbnails", workspace / "images" / "originals"):
                shutil.rmtree(folder, ignore_errors=True)
                folder.mkdir(parents=True, exist_ok=True)
        database.initialize()
        paths = scan_images(input_dir, config["image"]["supported_extensions"], recursive)
        logger.info("Scanned %d image files from %s", len(paths), input_dir)
        try:
            engine = InsightFaceEngine(_runtime_model_config(config, workspace))
        except Exception:
            logger.exception("Model initialization failed")
            raise
        summary = {"total_images": len(paths), "valid_images": 0, "duplicated_images": 0, "failed_images": 0,
                   "no_face_images": 0, "total_faces": 0, "valid_faces": 0, "low_quality_faces": 0}
        with database.connect() as connection:
            known_hashes = {row[0] for row in connection.execute("SELECT file_hash FROM images WHERE status != 'duplicated'")}
        for path in paths:
            image_id = database.next_id("images", "image_id", "img")
            timestamp = now()
            file_hash = ""
            try:
                file_hash = _hash_file(path)
                if file_hash in known_hashes:
                    stat = path.stat()
                    with database.connect() as connection:
                        connection.execute("INSERT INTO images VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                                           (image_id, str(path), path.name, file_hash, 0, 0, stat.st_size, 0, "duplicated", None, timestamp, timestamp))
                    summary["duplicated_images"] += 1
                    logger.info("Duplicated image: %s", path)
                    continue
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError("OpenCV could not decode image")
                height, width = image.shape[:2]
                stat = path.stat()
                stored_path = path
                if config["image"]["copy_originals"]:
                    stored_path = (workspace / "images" / "originals" / f"{image_id}{path.suffix.lower()}").resolve()
                    shutil.copy2(path, stored_path)
                save_jpeg(workspace / "images" / "thumbnails" / f"{image_id}.jpg", _thumbnail(image, int(config["image"]["thumbnail_max_size"])))
                faces = engine.detect_and_extract(image)
                status = "success" if faces else "no_face"
                with database.connect() as connection:
                    connection.execute("INSERT INTO images VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                                       (image_id, str(stored_path), path.name, file_hash, width, height, stat.st_size, len(faces), status, None, timestamp, timestamp))
                known_hashes.add(file_hash)
                summary["valid_images"] += 1
                summary["no_face_images"] += not faces
                for detected in faces:
                    face_id = database.next_id("faces", "face_id", "face")
                    x, y, face_width, face_height = detected.bbox
                    face_status = "valid" if face_width >= config["face"]["min_face_width"] and face_height >= config["face"]["min_face_height"] else "low_quality"
                    crop_path = (workspace / "faces" / f"{face_id}.jpg").resolve()
                    if face_status == "valid" or config["face"]["save_low_quality_faces"]:
                        save_jpeg(crop_path, crop_face(image, detected.bbox, float(config["face"]["crop_margin_ratio"])))
                    embedding = normalize(detected.embedding)
                    with database.connect() as connection:
                        connection.execute(
                            "INSERT INTO faces VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (face_id, image_id, str(crop_path), x, y, face_width, face_height, detected.detection_score,
                             embedding.tobytes(), embedding.size, float(np.linalg.norm(embedding)), None, detected.detection_score,
                             face_status, timestamp, timestamp),
                        )
                    summary["total_faces"] += 1
                    summary["valid_faces"] += face_status == "valid"
                    summary["low_quality_faces"] += face_status == "low_quality"
                logger.info("Processed %s: %d faces", path, len(faces))
            except Exception as error:
                summary["failed_images"] += 1
                logger.exception("Failed image %s: %s", path, error)
                try:
                    stat = path.stat()
                    with database.connect() as connection:
                        connection.execute("DELETE FROM faces WHERE image_id=?", (image_id,))
                        connection.execute(
                            "INSERT INTO images VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
                            "ON CONFLICT(image_id) DO UPDATE SET face_count=0,status='failed',error_message=excluded.error_message,updated_at=excluded.updated_at",
                            (image_id, str(path), path.name, file_hash, 0, 0, stat.st_size, 0, "failed", str(error), timestamp, timestamp),
                        )
                except Exception:
                    logger.exception("Could not persist image failure for %s", path)
        with database.connect() as connection:
            image_counts = {row["status"]: row["count"] for row in connection.execute("SELECT status,COUNT(*) count FROM images GROUP BY status")}
            face_counts = {row["status"]: row["count"] for row in connection.execute("SELECT status,COUNT(*) count FROM faces GROUP BY status")}
        summary.update({
            "valid_images": image_counts.get("success", 0) + image_counts.get("no_face", 0),
            "duplicated_images": image_counts.get("duplicated", 0),
            "failed_images": image_counts.get("failed", 0),
            "no_face_images": image_counts.get("no_face", 0),
            "total_faces": sum(face_counts.values()),
            "valid_faces": face_counts.get("valid", 0),
            "low_quality_faces": face_counts.get("low_quality", 0),
        })
        summary.update(rebuild_groups(workspace, config, database))
        summary["database"] = str(database.path.resolve())
        summary["workspace"] = str(workspace.resolve())
        logger.info("Build finished: %s", summary)
        return summary


class ReClusterService:
    def run(self, workspace: Path) -> dict:
        database = _existing_database(workspace)
        config = load_config(workspace)
        result = rebuild_groups(workspace, config, database)
        configure_logging(workspace, "build").info("Re-cluster finished: %s", result)
        return result


def _existing_database(workspace: Path) -> Database:
    if not workspace.is_dir():
        raise ValueError(f"Workspace does not exist: {workspace}")
    path = workspace / "database" / "face_index.db"
    if not path.is_file():
        raise ValueError(f"Database does not exist: {path}")
    return Database(path)


class SearchFaceService:
    def __init__(self, engine: InsightFaceEngine | None = None):
        self.engine = engine

    def run(self, image_path: Path, workspace: Path, *, export: Path | None = None,
            top_k: int | None = None, target_face: str = "largest", copy_images: bool | None = None,
            require_exactly_one_face: bool = False, write_exports: bool = True) -> dict:
        started = monotonic()
        if not image_path.is_file():
            raise ValueError(f"Query image does not exist: {image_path}")
        database = _existing_database(workspace)
        config = load_config(workspace)
        logger = configure_logging(workspace, "search")
        # UUIDs keep concurrent API requests isolated without a read-max/write race.
        query_id = f"query_{uuid.uuid4().hex}"
        query_dir = workspace / "queries" / query_id
        query_dir.mkdir(parents=True)
        query_copy = query_dir / f"query{image_path.suffix.lower()}"
        shutil.copy2(image_path, query_copy)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read query image: {image_path}")
        try:
            engine = self.engine or InsightFaceEngine(_runtime_model_config(config, workspace))
            faces = engine.detect_and_extract(image)
        except Exception:
            logger.exception("Model initialization or query analysis failed")
            raise
        logger.info("Query %s: detected %d faces", image_path, len(faces))
        if require_exactly_one_face and len(faces) != 1:
            shutil.rmtree(query_dir, ignore_errors=True)
            raise QueryFaceCountError(len(faces))
        if not faces:
            result = self._no_face_result(query_id, image_path, query_dir)
            if export:
                self._export(result, export.resolve(), config, copy_images)
            self._save_result(database, result, "no_face", "NO_FACE_DETECTED", query_dir)
            logger.info("Query finished with no face in %.3fs", monotonic() - started)
            return result
        selected = choose_face(faces, target_face, image.shape)
        crop_path = (query_dir / "query_face_000001.jpg").resolve()
        save_jpeg(crop_path, crop_face(image, selected.bbox, float(config["face"]["crop_margin_ratio"])))
        with database.connect() as connection:
            rows = connection.execute(
                "SELECT f.face_id,f.image_id,f.group_id,f.embedding,f.embedding_dim,i.file_path "
                "FROM faces f JOIN images i ON i.image_id=f.image_id "
                "WHERE f.status='valid' AND f.group_id IS NOT NULL ORDER BY f.face_id"
            ).fetchall()
        if not rows:
            raise ValueError("NO_INDEX_DATA: no valid grouped faces in database")
        embeddings = np.stack([np.frombuffer(row["embedding"], dtype=np.float32, count=row["embedding_dim"]) for row in rows])
        distances = cosine_distances(selected.embedding, embeddings)
        actual_k = min(int(top_k or config["search"]["top_k"]), len(rows))
        if actual_k < 1:
            raise ValueError("top-k must be greater than zero")
        indexes = np.argsort(distances)[:actual_k]
        matches = [{
            "face_id": rows[index]["face_id"], "group_id": rows[index]["group_id"],
            "distance": float(distances[index]), "image_id": rows[index]["image_id"],
            "image_path": rows[index]["file_path"],
        } for index in indexes]
        voting = vote(matches, int(config["search"]["min_group_vote_count"]), float(config["search"]["min_group_vote_ratio"]))
        best = matches[0]
        distance_passes = best["distance"] <= float(config["search"]["max_best_distance"])
        matched = bool(distance_passes and voting["passes_vote"])
        if not distance_passes:
            confidence = "none"
        elif voting["vote_ratio"] >= 0.7:
            confidence = "high"
        elif voting["vote_ratio"] >= 0.5:
            confidence = "medium"
        else:
            confidence = "low"
        candidate_group = voting["group_id"]
        group_summary, matched_images = self._group_details(database, candidate_group) if candidate_group else (None, [])
        x, y, width, height = selected.bbox
        result = {
            "query_id": query_id,
            "query_image_path": str(image_path.resolve()),
            "query_face_crop_path": str(crop_path),
            "query_face_bbox": {"x": x, "y": y, "width": width, "height": height},
            "query_has_multiple_faces": len(faces) > 1,
            "matched": matched,
            "matched_group_id": candidate_group if matched else None,
            "candidate_group_id": candidate_group if not matched and confidence == "low" else None,
            "best_face_id": best["face_id"],
            "best_distance": best["distance"],
            "confidence": confidence,
            "vote_count": voting["vote_count"],
            "vote_ratio": voting["vote_ratio"],
            "top_k": matches,
            "group_summary": group_summary if matched else None,
            "matched_images": matched_images if matched else [],
        }
        destination = export.resolve() if export else query_dir.resolve()
        if write_exports:
            self._export(result, destination, config, copy_images)
        internal_json = query_dir / "result.json"
        if write_exports and destination != query_dir.resolve():
            with internal_json.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        status = "matched" if matched else "not_matched"
        self._save_result(database, result, status, None if matched else "NO_MATCH_FOUND", query_dir)
        logger.info("Query finished in %.3fs: group=%s matched=%s top_k=%s", monotonic() - started, candidate_group, matched, matches)
        return result

    @staticmethod
    def _no_face_result(query_id: str, image_path: Path, query_dir: Path) -> dict:
        return {
            "query_id": query_id, "query_image_path": str(image_path.resolve()),
            "query_face_crop_path": None, "query_face_bbox": None,
            "query_has_multiple_faces": False, "matched": False,
            "matched_group_id": None, "best_face_id": None, "best_distance": None,
            "confidence": "none", "top_k": [], "group_summary": None,
            "matched_images": [], "error": "NO_FACE_DETECTED",
        }

    @staticmethod
    def _group_details(database: Database, group_id: str) -> tuple[dict | None, list[dict]]:
        with database.connect() as connection:
            group = connection.execute("SELECT * FROM face_groups WHERE group_id=?", (group_id,)).fetchone()
            images = connection.execute(
                "SELECT i.image_id,i.file_path,i.file_name FROM group_images gi JOIN images i ON i.image_id=gi.image_id "
                "WHERE gi.group_id=? ORDER BY i.image_id", (group_id,),
            ).fetchall()
            output = []
            for image in images:
                faces = connection.execute(
                    "SELECT face_id,face_crop_path,bbox_x,bbox_y,bbox_width,bbox_height FROM faces WHERE group_id=? AND image_id=? ORDER BY face_id",
                    (group_id, image["image_id"]),
                ).fetchall()
                output.append({
                    "image_id": image["image_id"], "file_path": image["file_path"], "file_name": image["file_name"],
                    "faces": [{"face_id": face["face_id"], "face_crop_path": face["face_crop_path"], "bbox": {"x": face["bbox_x"], "y": face["bbox_y"], "width": face["bbox_width"], "height": face["bbox_height"]}} for face in faces],
                })
        summary = None if not group else {
            "group_id": group["group_id"], "face_count": group["face_count"], "image_count": group["image_count"],
            "representative_face_id": group["representative_face_id"],
        }
        return summary, output

    @staticmethod
    def _export(result: dict, destination: Path, config: dict, copy_images: bool | None) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        warnings = result.setdefault("export_warnings", [])
        should_copy = config["export"].get("copy_matched_images", True) if copy_images is None else copy_images
        if should_copy and result["matched"]:
            image_dir = destination / "matched_images"
            image_dir.mkdir(exist_ok=True)
            for item in result["matched_images"]:
                source = Path(item["file_path"])
                try:
                    shutil.copy2(source, image_dir / f"{item['image_id']}_{source.name}")
                except OSError as error:
                    warnings.append(f"Could not copy image {source}: {error}")
        if config["export"].get("copy_matched_faces", True) and result["matched"]:
            face_dir = destination / "matched_faces"
            face_dir.mkdir(exist_ok=True)
            for item in result["matched_images"]:
                for face in item["faces"]:
                    source = Path(face["face_crop_path"])
                    try:
                        shutil.copy2(source, face_dir / source.name)
                    except OSError as error:
                        warnings.append(f"Could not copy face {source}: {error}")
        if config["export"].get("write_json", True):
            with (destination / "result.json").open("w", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        if config["export"].get("write_csv", True):
            with (destination / "matched_images.csv").open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image_id", "file_name", "file_path"])
                writer.writeheader()
                writer.writerows({key: item[key] for key in writer.fieldnames} for item in result["matched_images"])

    @staticmethod
    def _save_result(database: Database, result: dict, status: str, message: str | None, query_dir: Path) -> None:
        result_path = query_dir / "result.json"
        if not result_path.exists():
            with result_path.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        with database.connect() as connection:
            connection.execute(
                "INSERT INTO queries VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (result["query_id"], result["query_image_path"], result["query_face_crop_path"], result["matched_group_id"],
                 result["best_face_id"], result["best_distance"], result["confidence"], status, message,
                 str(result_path.resolve()), now()),
            )


class QueryFaceCountError(ValueError):
    def __init__(self, detected_count: int):
        self.detected_count = detected_count
        super().__init__(f"Expected exactly one face, detected {detected_count}")


class ReportService:
    def report(self, workspace: Path) -> dict:
        database = _existing_database(workspace)
        with database.connect() as connection:
            image_counts = {row["status"]: row["count"] for row in connection.execute("SELECT status,COUNT(*) count FROM images GROUP BY status")}
            face_counts = {row["status"]: row["count"] for row in connection.execute("SELECT status,COUNT(*) count FROM faces GROUP BY status")}
            group = connection.execute(
                "SELECT COUNT(*) total_groups,COALESCE(SUM(face_count=1),0) singleton_groups,"
                "COALESCE(MAX(face_count),0) max_group_faces,COALESCE(MAX(image_count),0) max_group_images FROM face_groups"
            ).fetchone()
        return {
            "total_images": sum(image_counts.values()), "successful_images": image_counts.get("success", 0),
            "duplicated_images": image_counts.get("duplicated", 0), "failed_images": image_counts.get("failed", 0),
            "no_face_images": image_counts.get("no_face", 0), "total_faces": sum(face_counts.values()),
            "valid_faces": face_counts.get("valid", 0), "low_quality_faces": face_counts.get("low_quality", 0),
            "total_groups": group["total_groups"], "singleton_groups": group["singleton_groups"],
            "max_group_faces": group["max_group_faces"], "max_group_images": group["max_group_images"],
            "database": str(database.path.resolve()), "workspace": str(workspace.resolve()),
        }

    def list_groups(self, workspace: Path, min_face_count: int = 1, sort: str = "face_count_desc") -> list[dict]:
        database = _existing_database(workspace)
        orders = {"face_count_desc": "face_count DESC,group_id", "face_count_asc": "face_count,group_id", "group_id": "group_id"}
        if sort not in orders:
            raise ValueError(f"Unsupported sort: {sort}")
        with database.connect() as connection:
            rows = connection.execute(
                f"SELECT group_id,face_count,image_count,representative_face_id FROM face_groups WHERE face_count>=? ORDER BY {orders[sort]}",
                (min_face_count,),
            ).fetchall()
        return [dict(row) for row in rows]


class ExportGroupLargestService:
    """Export one highest-resolution source image for every active face group."""

    def run(self, workspace: Path, export: Path | None = None) -> dict:
        database = _existing_database(workspace)
        destination = (export or workspace / "exports" / "group_largest_images").resolve()
        images_dir = destination / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        with database.connect() as connection:
            rows = connection.execute(
                "WITH ranked AS ("
                " SELECT gi.group_id,i.image_id,i.file_path,i.file_name,i.width,i.height,i.file_size,"
                " ROW_NUMBER() OVER (PARTITION BY gi.group_id "
                " ORDER BY (i.width * i.height) DESC,i.file_size DESC,i.image_id) rank"
                " FROM group_images gi JOIN images i ON i.image_id=gi.image_id"
                " JOIN face_groups fg ON fg.group_id=gi.group_id"
                " WHERE fg.status='active' AND i.status='success'"
                ") SELECT * FROM ranked WHERE rank=1 ORDER BY group_id"
            ).fetchall()

        manifest = []
        warnings = []
        for row in rows:
            source = Path(row["file_path"])
            target = images_dir / f"{row['group_id']}__{source.name}"
            copied = False
            try:
                shutil.copy2(source, target)
                copied = True
            except OSError as error:
                warnings.append(f"Could not copy {source}: {error}")
            manifest.append({
                "group_id": row["group_id"],
                "image_id": row["image_id"],
                "file_name": row["file_name"],
                "source_path": str(source),
                "export_path": str(target) if copied else None,
                "width": row["width"],
                "height": row["height"],
                "pixel_count": row["width"] * row["height"],
                "file_size": row["file_size"],
            })
        result = {
            "selection_rule": "pixel_count_desc_then_file_size_desc",
            "total_groups": len(rows),
            "copied_images": sum(item["export_path"] is not None for item in manifest),
            "warnings": warnings,
            "images": manifest,
        }
        with (destination / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        with (destination / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            fieldnames = ["group_id", "image_id", "file_name", "source_path", "export_path", "width", "height", "pixel_count", "file_size"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest)
        return {**result, "export": str(destination)}


class ExportGroupLargestFaceService:
    """Export the face crop with the highest pixel resolution from every group."""

    def run(self, workspace: Path, export: Path | None = None) -> dict:
        database = _existing_database(workspace)
        destination = (export or workspace / "exports" / "group_largest_faces").resolve()
        faces_dir = destination / "faces"
        faces_dir.mkdir(parents=True, exist_ok=True)
        with database.connect() as connection:
            rows = connection.execute(
                "SELECT f.group_id,f.face_id,f.image_id,f.face_crop_path,"
                "f.bbox_x,f.bbox_y,f.bbox_width,f.bbox_height,i.file_path original_image_path "
                "FROM faces f JOIN face_groups fg ON fg.group_id=f.group_id "
                "JOIN images i ON i.image_id=f.image_id "
                "WHERE fg.status='active' AND f.group_id IS NOT NULL "
                "ORDER BY f.group_id,f.face_id"
            ).fetchall()

        winners: dict[str, dict] = {}
        warnings = []
        for row in rows:
            source = Path(row["face_crop_path"])
            crop = cv2.imread(str(source))
            if crop is None:
                warnings.append(f"Could not read face crop {source}")
                continue
            height, width = crop.shape[:2]
            file_size = source.stat().st_size
            candidate = {
                "group_id": row["group_id"],
                "face_id": row["face_id"],
                "image_id": row["image_id"],
                "face_crop_path": str(source),
                "original_image_path": row["original_image_path"],
                "width": width,
                "height": height,
                "pixel_count": width * height,
                "file_size": file_size,
                "bbox_x": row["bbox_x"],
                "bbox_y": row["bbox_y"],
                "bbox_width": row["bbox_width"],
                "bbox_height": row["bbox_height"],
            }
            current = winners.get(row["group_id"])
            candidate_rank = (candidate["pixel_count"], candidate["file_size"], candidate["face_id"])
            current_rank = None if current is None else (current["pixel_count"], current["file_size"], current["face_id"])
            if current_rank is None or candidate_rank > current_rank:
                winners[row["group_id"]] = candidate

        manifest = []
        for group_id in sorted(winners):
            item = winners[group_id]
            source = Path(item["face_crop_path"])
            target = faces_dir / f"{group_id}__{item['face_id']}{source.suffix.lower()}"
            try:
                shutil.copy2(source, target)
                item["export_path"] = str(target)
            except OSError as error:
                item["export_path"] = None
                warnings.append(f"Could not copy {source}: {error}")
            manifest.append(item)

        result = {
            "selection_rule": "face_crop_pixel_count_desc_then_file_size_desc",
            "total_groups": len(winners),
            "copied_faces": sum(item["export_path"] is not None for item in manifest),
            "warnings": warnings,
            "faces": manifest,
        }
        with (destination / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        with (destination / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            fieldnames = [
                "group_id", "face_id", "image_id", "face_crop_path", "original_image_path", "export_path",
                "width", "height", "pixel_count", "file_size", "bbox_x", "bbox_y", "bbox_width", "bbox_height",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest)
        return {**result, "export": str(destination)}

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "provider": "insightface",
        "model_name": "buffalo_l",
        "det_size": [640, 640],
        "device": "cuda",
        "gpu_device_id": 0,
        "allow_cpu_fallback": False,
        "detection_threshold": 0.5,
    },
    "image": {
        "supported_extensions": [".jpg", ".jpeg", ".png", ".webp"],
        "copy_originals": False,
        "thumbnail_max_size": 512,
    },
    "face": {
        "min_face_width": 40,
        "min_face_height": 40,
        "crop_margin_ratio": 0.25,
        "save_low_quality_faces": True,
    },
    "cluster": {
        "algorithm": "dbscan",
        "metric": "cosine",
        "eps": 0.42,
        "min_samples": 2,
        "noise_as_singleton_group": True,
        "cluster_include_low_quality_faces": False,
    },
    "search": {
        "metric": "cosine",
        "top_k": 10,
        "max_best_distance": 0.42,
        "min_group_vote_count": 3,
        "min_group_vote_ratio": 0.5,
    },
    "export": {
        "copy_matched_images": True,
        "copy_matched_faces": True,
        "write_json": True,
        "write_csv": True,
    },
    "api": {
        "download_max_concurrency": 3,
        "zip_retention_hours": 48,
        "max_upload_bytes": 20 * 1024 * 1024,
    },
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(workspace: Path, config_path: Path | None = None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    stored = workspace / "config.json"
    source = config_path or (stored if stored.exists() else None)
    if source:
        with source.open("r", encoding="utf-8") as handle:
            _merge(config, json.load(handle))
    return config


def save_config(workspace: Path, config: dict[str, Any]) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def initialize_workspace(workspace: Path) -> None:
    for relative in (
        "database", "images/originals", "images/thumbnails", "faces",
        "groups", "queries", "logs", "models",
    ):
        (workspace / relative).mkdir(parents=True, exist_ok=True)

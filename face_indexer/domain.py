from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class DetectedFace:
    bbox: tuple[float, float, float, float]
    detection_score: float | None
    embedding: np.ndarray
    landmarks: object | None = None


def scan_images(directory: Path, extensions: list[str], recursive: bool = True) -> list[Path]:
    iterator = directory.rglob("*") if recursive else directory.glob("*")
    allowed = {value.lower() for value in extensions}
    return sorted(path.resolve() for path in iterator if path.is_file() and path.suffix.lower() in allowed)


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("Embedding has zero norm")
    return np.ascontiguousarray(vector / norm, dtype=np.float32)


def cosine_distances(query: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    query = normalize(query)
    matrix = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, np.finfo(np.float32).eps)
    return np.clip(1.0 - matrix @ query, 0.0, 2.0)


def crop_face(image: np.ndarray, bbox: tuple[float, float, float, float], margin: float) -> np.ndarray:
    x, y, width, height = bbox
    image_height, image_width = image.shape[:2]
    x1 = max(0, int(np.floor(x - width * margin)))
    y1 = max(0, int(np.floor(y - height * margin)))
    x2 = min(image_width, int(np.ceil(x + width * (1 + margin))))
    y2 = min(image_height, int(np.ceil(y + height * (1 + margin))))
    return image[y1:y2, x1:x2].copy()


def choose_face(faces: list[DetectedFace], strategy: str, shape: tuple[int, ...]) -> DetectedFace:
    if strategy == "largest":
        return max(faces, key=lambda face: face.bbox[2] * face.bbox[3])
    if strategy == "highest-score":
        return max(faces, key=lambda face: face.detection_score or 0.0)
    if strategy == "center-most":
        center_x, center_y = shape[1] / 2, shape[0] / 2
        return min(faces, key=lambda face: (face.bbox[0] + face.bbox[2] / 2 - center_x) ** 2 + (face.bbox[1] + face.bbox[3] / 2 - center_y) ** 2)
    raise ValueError(f"Unsupported target-face strategy: {strategy}")


def dbscan_cosine(embeddings: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Run sklearn DBSCAN, with a small NumPy fallback for minimal installations."""
    count = len(embeddings)
    if count == 0:
        return np.empty(0, dtype=np.int32)
    matrix = np.stack([normalize(item) for item in embeddings])
    try:
        from sklearn.cluster import DBSCAN
        return DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(matrix).astype(np.int32)
    except ImportError:
        pass
    distances = np.clip(1.0 - matrix @ matrix.T, 0.0, 2.0)
    neighbors = [np.flatnonzero(distances[index] <= eps).tolist() for index in range(count)]
    labels = np.full(count, -99, dtype=np.int32)  # -99 unvisited, -1 noise
    cluster = 0
    for point in range(count):
        if labels[point] != -99:
            continue
        if len(neighbors[point]) < min_samples:
            labels[point] = -1
            continue
        labels[point] = cluster
        queue = deque(neighbors[point])
        queued = set(neighbors[point])
        while queue:
            candidate = queue.popleft()
            if labels[candidate] == -1:
                labels[candidate] = cluster
            if labels[candidate] != -99:
                continue
            labels[candidate] = cluster
            if len(neighbors[candidate]) >= min_samples:
                for neighbor in neighbors[candidate]:
                    if neighbor not in queued:
                        queued.add(neighbor)
                        queue.append(neighbor)
        cluster += 1
    return labels


def vote(top_matches: list[dict], minimum_count: int, minimum_ratio: float) -> dict:
    if not top_matches:
        return {"group_id": None, "vote_count": 0, "vote_ratio": 0.0}
    counts = Counter(item["group_id"] for item in top_matches)
    candidate = min(
        counts,
        key=lambda group: (-counts[group], min(item["distance"] for item in top_matches if item["group_id"] == group)),
    )
    count = counts[candidate]
    ratio = count / len(top_matches)
    return {
        "group_id": candidate,
        "vote_count": count,
        "vote_ratio": ratio,
        "passes_vote": count >= minimum_count and ratio >= minimum_ratio,
    }


def save_jpeg(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.size == 0 or not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write image: {path}")

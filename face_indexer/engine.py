from __future__ import annotations

import numpy as np

from .domain import DetectedFace, normalize


class InsightFaceEngine:
    """Device-aware boundary around InsightFace; application code stays runtime-agnostic."""

    def __init__(self, config: dict):
        if config.get("provider") != "insightface":
            raise ValueError(f"Unsupported model provider: {config.get('provider')}")
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        device = config.get("device", "cpu").lower()
        available = ort.get_available_providers()
        provider_options = None
        if device == "cuda":
            if "CUDAExecutionProvider" not in available:
                if not config.get("allow_cpu_fallback", False):
                    raise RuntimeError("CUDA requested but CUDAExecutionProvider is unavailable")
                providers = ["CPUExecutionProvider"]
                ctx_id = -1
                actual_device = "cpu"
            else:
                device_id = int(config.get("gpu_device_id", 0))
                if device_id < 0:
                    raise ValueError("model.gpu_device_id must be zero or greater")
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                provider_options = [{"device_id": str(device_id)}, {}]
                ctx_id = device_id
                actual_device = "cuda"
        elif device == "cpu":
            providers = ["CPUExecutionProvider"]
            ctx_id = -1
            actual_device = "cpu"
        else:
            raise ValueError("model.device must be 'cpu' or 'cuda'")
        self.device = actual_device
        self.providers = providers
        self.app = FaceAnalysis(
            name=config["model_name"],
            root=config.get("model_root", "~/.insightface"),
            providers=providers,
            provider_options=provider_options,
        )
        self.app.prepare(
            ctx_id=ctx_id,
            det_thresh=float(config.get("detection_threshold", 0.5)),
            det_size=tuple(config["det_size"]),
        )

    def detect_and_extract(self, image: np.ndarray) -> list[DetectedFace]:
        results: list[DetectedFace] = []
        for face in self.app.get(image):
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                embedding = getattr(face, "embedding", None)
            if embedding is None:
                continue
            x1, y1, x2, y2 = (float(value) for value in face.bbox)
            results.append(DetectedFace(
                bbox=(x1, y1, x2 - x1, y2 - y1),
                detection_score=float(face.det_score) if getattr(face, "det_score", None) is not None else None,
                embedding=normalize(embedding),
                landmarks=getattr(face, "kps", None),
            ))
        return results

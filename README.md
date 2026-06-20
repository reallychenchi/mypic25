# Face Indexer

本地照片人脸建库、自动分组和检索命令行程序。数据与 embedding 保存在 SQLite 中。

完整的 Python 版本、CPU/GPU 依赖和安装验证说明见 [REQUIREMENTS.md](REQUIREMENTS.md)。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate

# GPU 版本（默认）
pip install -r requirements-gpu.txt

# CPU 版本（二选一）
pip install -r requirements-cpu.txt
```

首次执行时 InsightFace 会将配置的模型（默认 `buffalo_l`，约 280 MB）下载到 workspace 的 `models/` 目录。

## 使用

```bash
python -m face_indexer build --input ./photos --workspace ./workspace --device cuda
python -m face_indexer report --workspace ./workspace
python -m face_indexer list-groups --workspace ./workspace
python -m face_indexer search --image ./query.jpg --workspace ./workspace --export ./result
python -m face_indexer re-cluster --workspace ./workspace
python -m face_indexer export-group-largest --workspace ./workspace --export ./group_largest
python -m face_indexer export-group-largest-face --workspace ./workspace --export ./group_largest_faces
python -m face_indexer serve --workspace ./workspace --host 0.0.0.0 --port 8000
```

HTTP 接口、响应字段和完整错误码见 [API.md](API.md)。

查询阶段只处理查询图片，并读取 SQLite 中已保存的 embedding；不会扫描或重新处理原始照片。修改聚类阈值后执行 `re-cluster`。更换模型需要重新 `build --force true`；新增照片可再次执行 `build`，程序按文件哈希跳过已有照片。

## CPU / GPU 切换

默认配置使用 GPU。安装与服务器 CUDA/cuDNN 版本匹配的 `onnxruntime-gpu` 后，使用 `--device cuda`，或在 `workspace/config.json` 中设置 `model.device=cuda`。多 GPU 机器通过 `model.gpu_device_id` 选择显卡，默认 `0`。

CPU 与 GPU runtime 不应同时安装。切回 CPU 时卸载 `onnxruntime-gpu`、安装 `requirements-cpu.txt`，并设置 `--device cpu`。模型之外的业务代码无需修改。请求 CUDA 但 ONNX Runtime 未提供 CUDA provider 时程序会明确报错；只有配置 `allow_cpu_fallback=true` 才会回退 CPU。

阈值、图片格式、质量规则和导出行为均在 `workspace/config.json` 配置。原始照片默认不复制，数据库保存绝对路径；若照片可能移动，请在建库时设置 `--copy-originals true`。

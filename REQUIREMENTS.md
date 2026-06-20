# Python 依赖说明

## 运行环境

- Python 3.12（当前开发环境为 3.12.3）
- pip 与 Python 必须来自同一个虚拟环境
- GPU 模式需要主机已安装与 `onnxruntime-gpu` 兼容的 NVIDIA 驱动、CUDA 和 cuDNN

建议创建独立环境：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 依赖文件

项目按公共依赖和推理后端拆分依赖：

| 文件 | 用途 | 直接依赖 |
| --- | --- | --- |
| `requirements-common.txt` | CLI 与 HTTP API 的公共依赖 | InsightFace、NumPy、OpenCV、scikit-learn、FastAPI、Uvicorn、python-multipart |
| `requirements-cpu.txt` | CPU 推理 | 公共依赖、`onnxruntime` |
| `requirements-gpu.txt` | NVIDIA GPU 推理 | 公共依赖、`onnxruntime-gpu` |
| `requirements.txt` | 默认安装入口 | 引用 GPU 依赖文件 |

主依赖用途：

- `insightface`：人脸检测、关键点和特征向量提取。
- `numpy`：特征向量、距离计算及图像数组处理。
- `opencv-python`：图片读取、缩放、裁剪和写出。
- `scikit-learn`：DBSCAN 聚类；代码包含 NumPy 降级实现，但正常安装仍建议保留。
- `onnxruntime` / `onnxruntime-gpu`：执行 InsightFace ONNX 模型。
- `fastapi`：提供 HTTP API、请求校验和响应处理。
- `uvicorn`：运行 FastAPI 服务。
- `python-multipart`：解析人脸查询接口上传的 `multipart/form-data` 图片。

## 安装

GPU 环境（项目默认）：

```bash
python -m pip install -r requirements-gpu.txt
```

CPU 环境：

```bash
python -m pip install -r requirements-cpu.txt
```

也可以使用默认入口安装 GPU 版本：

```bash
python -m pip install -r requirements.txt
```

不要在同一个环境中同时安装 `onnxruntime` 和 `onnxruntime-gpu`。切换后端时，建议重新创建虚拟环境；至少应先卸载另一种运行时。

安装完成后可启动 HTTP API：

```bash
python -m face_indexer serve --workspace ./workspace --host 0.0.0.0 --port 8000
```

服务应保持单个 Python 进程运行，以保证下载并发上限是进程内全局限制。接口详情见 `API.md`。

## 可选验证工具

根目录的 `verify.py` 用于验证另一套 `face_recognition`/dlib 安装，不参与 `face_indexer` 主程序。只有需要运行该脚本时才安装：

```bash
python -m pip install face-recognition==1.3.0
python verify.py
```

`dlib` 在部分平台需要系统编译工具和 CMake，因此不应加入主应用依赖。

## 安装验证

运行单元测试：

```bash
python -m unittest discover -v
```

检查 ONNX Runtime 后端：

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

GPU 环境的输出应包含 `CUDAExecutionProvider`；CPU 环境至少应包含 `CPUExecutionProvider`。

## 版本维护

当前依赖文件使用兼容版本范围，适合跨机器安装。部署前可在经过测试的干净环境中生成精确版本快照：

```bash
python -m pip freeze > requirements-lock.txt
```

锁定文件会包含大量间接依赖，不应使用包含 CPU 和 GPU 两种 ONNX Runtime 的现有开发环境直接生成。

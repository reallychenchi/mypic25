# Face Indexer HTTP API

## 1. 服务说明

服务接收一张恰好包含一张人脸的图片，在已有 InsightFace 分组中查找匹配分组，将该分组关联的可用原图打包成无压缩 ZIP，并提供临时下载地址。

- API 版本：`v1`
- 数据格式：除文件下载外均为 JSON
- 上传格式：`multipart/form-data`
- 鉴权：无
- 运行模型：单个 Python 进程；不要配置多个 Uvicorn worker
- 请求隔离：每个请求使用随机 UUID，并保存在独立目录 `workspace/api_requests/{request_id}/`
- ZIP 压缩方式：`ZIP_STORED`（仅打包，不压缩）
- ZIP 有效期：默认 48 小时，配置值不得超过 48 小时
- 下载并发：默认最多 3 个；达到上限时立即返回 HTTP 429

## 2. 启动服务

安装与当前 CPU/GPU 环境对应的依赖后运行：

```bash
python -m face_indexer serve \
  --workspace ./workspace \
  --host 0.0.0.0 \
  --port 8000
```

服务固定使用一个 worker。下载并发计数位于当前 Python 进程内，多 worker 或多实例之间不会共享计数。

配置位于 `workspace/config.json`：

```json
{
  "api": {
    "download_max_concurrency": 3,
    "zip_retention_hours": 48,
    "max_upload_bytes": 20971520
  }
}
```

修改配置后需要重启服务。`zip_retention_hours` 必须大于 0 且不能超过 48。

## 3. 统一响应结构

成功或业务未匹配：

```json
{
  "code": "OK",
  "message": "匹配及打包完成",
  "request_id": "9b13059beb44405886fc77e18bd648ef",
  "data": {}
}
```

错误：

```json
{
  "code": "MULTIPLE_FACES_DETECTED",
  "message": "图片中检测到多张人脸，必须恰好包含一张人脸",
  "request_id": "9b13059beb44405886fc77e18bd648ef",
  "data": null,
  "details": {
    "detected_faces": 2
  }
}
```

`request_id` 是本次上传请求的唯一标识。请求参数尚未解析成功，或下载请求没有对应上传请求时，该字段可能不存在。

## 4. 人脸匹配及打包

### `POST /api/v1/face-searches`

请求类型：`multipart/form-data`

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `image` | file | 是 | 必须恰好包含一张人脸；默认最大 20 MiB |

支持的后缀由 `image.supported_extensions` 配置，默认 `.jpg`、`.jpeg`、`.png`、`.webp`。服务还会实际解码图片，只有后缀正确但内容无效的文件仍会被拒绝。

示例：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/face-searches \
  -F 'image=@./query.jpg'
```

匹配成功响应（HTTP 200）：

```json
{
  "code": "OK",
  "message": "匹配及打包完成",
  "request_id": "9b13059beb44405886fc77e18bd648ef",
  "data": {
    "group_id": "group_000001",
    "matched_image_count": 21,
    "group_image_count": 23,
    "download_url": "http://127.0.0.1:8000/api/v1/downloads/9b13059beb44405886fc77e18bd648ef",
    "expires_at": "2026-06-22T08:00:00+00:00",
    "warnings": [
      "原图不存在，已跳过：/photos/missing.jpg"
    ]
  }
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `group_id` | 匹配到的已有分组 ID |
| `matched_image_count` | 实际成功写入 ZIP 的图片数 |
| `group_image_count` | 数据库中该分组关联的去重图片总数 |
| `download_url` | ZIP 下载地址 |
| `expires_at` | 下载过期时间，ISO 8601 UTC 时间 |
| `warnings` | 原图缺失或读取失败等跳过信息 |

原图缺失不会导致整个请求失败。服务会跳过该文件，因此 `matched_image_count` 可能小于 `group_image_count`，甚至为 0。ZIP 内文件名格式为 `{image_id}_{原文件名}`，用于避免同名图片互相覆盖。

未匹配响应（HTTP 200）：

```json
{
  "code": "NO_MATCH_FOUND",
  "message": "未找到匹配分组",
  "request_id": "9b13059beb44405886fc77e18bd648ef",
  "data": {
    "group_id": null,
    "matched_image_count": 0,
    "download_url": null,
    "expires_at": null,
    "warnings": []
  }
}
```

未匹配属于正常业务结果，不生成 ZIP。

## 5. 下载 ZIP

### `GET /api/v1/downloads/{archive_id}`

`archive_id` 就是匹配接口返回的 `request_id`。

成功时返回 HTTP 200：

- `Content-Type: application/zip`
- `Content-Disposition: attachment; filename="face-group-{archive_id}.zip"`
- 响应体为 ZIP 文件内容

浏览器访问 `download_url` 会触发正常文件下载。

并发限制统计正在传输的文件响应。达到配置上限时，新请求不会排队，而是返回 HTTP 429；客户端应稍后重试。文件响应完成或连接中断后会释放下载槽位。

## 6. HTTP 状态码与业务错误码

| HTTP | `code` | 含义 | 建议处理 |
| --- | --- | --- | --- |
| 200 | `OK` | 匹配成功且 ZIP 已生成 | 使用 `download_url` 下载 |
| 200 | `NO_MATCH_FOUND` | 没有符合阈值和投票规则的分组 | 提示用户未匹配，无需重试下载 |
| 400 | `EMPTY_IMAGE` | 上传文件为空 | 重新选择图片 |
| 400 | `INVALID_IMAGE` | 图片无法解码或处理 | 检查文件内容后重新上传 |
| 404 | `ARCHIVE_NOT_FOUND` | 下载 ID 无效或文件不存在 | 重新执行匹配请求 |
| 410 | `ARCHIVE_EXPIRED` | ZIP 已超过有效期 | 重新执行匹配请求 |
| 413 | `IMAGE_TOO_LARGE` | 超过 `max_upload_bytes` | 压缩或缩小上传图片 |
| 415 | `UNSUPPORTED_IMAGE_TYPE` | 图片后缀不受支持 | 转换为支持的格式 |
| 422 | `INVALID_REQUEST` | 缺少 `image` 等请求结构错误 | 修正表单字段 |
| 422 | `NO_FACE_DETECTED` | 图片中没有检测到人脸 | 更换清晰的单人图片 |
| 422 | `MULTIPLE_FACES_DETECTED` | 图片中检测到多张人脸 | 裁剪为只包含一张人脸 |
| 429 | `DOWNLOAD_LIMIT_EXCEEDED` | 当前下载数已达上限 | 稍后重试，建议客户端退避 |
| 500 | `UPLOAD_SAVE_FAILED` | 服务端无法保存上传文件 | 检查磁盘和目录权限 |
| 500 | `ARCHIVE_CREATE_FAILED` | 创建 ZIP 失败 | 检查磁盘空间和原图读取权限 |
| 500 | `INTERNAL_ERROR` | 未分类的检索内部错误 | 查看服务日志 |
| 503 | `NO_INDEX_DATA` | 索引不存在有效分组人脸 | 先构建或修复索引 |
| 503 | `INFERENCE_UNAVAILABLE` | InsightFace、ONNX Runtime 或 CPU/GPU 推理后端不可用 | 检查模型及运行时配置 |

## 7. 文件生命周期与隔离

每个成功匹配请求分别写入：

```text
workspace/api_requests/{request_id}/
├── query.jpg
├── matched-images.zip
└── response.json
```

不同用户和并发请求不会复用目录或文件名。InsightFace 内部查询记录同样使用独立 UUID 目录：

```text
workspace/queries/query_{uuid}/
```

服务会在后续匹配请求到来时清理过期 ZIP 目录；下载时也会检查有效期并删除已过期文件。该清理方式不是精确定时任务，因此磁盘上的过期文件可能保留到下一次请求，但任何超过有效期的 ZIP 都不能继续下载。

## 8. 部署注意事项

- 必须保持单 Python 进程，否则下载并发上限会变成“每进程最多 N 个”。
- `download_url` 根据当前请求地址生成。部署在反向代理后时，应正确传递 Host 和协议转发头。
- 当前服务不包含鉴权，暴露到公网前应由网关限制可访问网络范围。
- 当前 workspace 配置的 `model.device` 为 `cuda`。若部署要求 CPU，需要改为 `cpu` 并安装 `onnxruntime`，不要同时安装 `onnxruntime-gpu`。

# 本地人脸建库、分组与检索系统需求与技术架构文档

## 1. 项目目标

本项目目标是实现一个本地运行的 Python 程序，用于对一批照片进行人脸提取、特征保存、自动分组，并支持后续通过一张查询照片检索对应人物分组及关联原始照片。

第一阶段目标是实现一个命令行可验证程序，能够完整跑通以下流程：

1. 批量读取本地照片目录；
2. 检测每张照片中的所有人脸；
3. 保存每张人脸的人脸框位置、人脸截图、特征向量、所属原图等信息；
4. 对所有人脸进行自动聚类分组；
5. 保存分组结果；
6. 用户输入一张查询照片后，只对查询照片做人脸检测和特征提取；
7. 使用已保存的人脸特征数据进行相似度比对；
8. 找到对应的人脸分组；
9. 输出该分组关联的所有原始照片；
10. 支持将结果照片复制到指定结果目录。

第二阶段目标是在第一阶段的代码基础上改造为 Python API 服务。服务化后应提供建库、查询、查看分组、查看图片、导出结果等接口。

本系统不是实名身份识别系统，不负责判断人物真实姓名。系统只负责判断：

```text
查询照片中的人脸，与已建库数据中的哪一个人脸分组最相似。
```

如需姓名，需要后续人工为 `group_id` 添加备注或标签。

---

## 2. 核心设计原则

### 2.1 人脸建库与查询必须分离

系统必须分为两个阶段：

```text
阶段一：建库阶段
阶段二：查询阶段
```

建库阶段负责处理全部原始照片，包括人脸检测、特征提取、截图保存、数据库写入和人脸分组。

查询阶段只处理用户新提供的一张查询照片，不允许重新扫描或重新处理全部原始照片。

### 2.2 查询时不得重新处理原始照片

查询时禁止执行以下操作：

1. 重新遍历原始照片目录；
2. 重新读取全部原始照片做人脸检测；
3. 重新对全部原始照片提取人脸特征；
4. 重新对全部数据做人脸分组。

查询时只能执行以下操作：

1. 读取查询照片；
2. 检测查询照片中的人脸；
3. 提取查询人脸特征；
4. 从数据库读取已保存的人脸特征；
5. 进行向量相似度比对；
6. 根据匹配到的 `group_id` 查询关联照片。

### 2.3 所有中间结果必须持久化

建库完成后，以下数据必须保存：

1. 原始图片信息；
2. 每张图片检测到的人脸数量；
3. 每张人脸在原图中的坐标位置；
4. 每张人脸对应的人脸截图文件；
5. 每张人脸的特征向量；
6. 每张人脸所属的人脸分组；
7. 每个分组关联的原始图片；
8. 每次查询的结果记录。

这样后续查询、导出、人工检查、重新聚类都可以基于已保存数据完成。

---

## 3. 术语定义

### 3.1 Image

指一张原始照片文件。

一张 Image 中可以包含 0 张、1 张或多张人脸。

### 3.2 Face

指从某一张 Image 中检测出来的一张人脸。

每一张 Face 都必须有唯一 `face_id`。

即使两张 Face 属于同一个人，只要它们来自不同照片，或者来自同一张照片中的不同人脸，也必须分别保存为不同 Face 记录。

### 3.3 Face Embedding

指人脸识别模型提取出来的人脸特征向量。

该向量用于判断两张人脸是否相似。

系统内部应将 embedding 保存为 `float32` 数组，并记录 embedding 维度。

### 3.4 Face Group

指系统根据 embedding 聚类后得到的人脸分组。

同一个 Face Group 应尽量代表同一个人。

每个 Face Group 必须有唯一 `group_id`。

### 3.5 Query Image

指用户用于检索的输入照片。

查询照片不直接加入人脸库，除非用户显式执行“新增照片入库”操作。

### 3.6 Match

指查询人脸与人脸库中的某个已有分组达到相似度阈值，系统认为它属于该组。

---

## 4. 第一阶段功能范围

第一阶段实现命令行程序，不要求提供 Web 页面，不要求提供前端界面。

第一阶段必须实现以下命令：

```bash
python -m face_indexer build --input ./photos --workspace ./workspace
```

```bash
python -m face_indexer search --image ./query.jpg --workspace ./workspace --export ./result
```

```bash
python -m face_indexer report --workspace ./workspace
```

```bash
python -m face_indexer list-groups --workspace ./workspace
```

可选实现：

```bash
python -m face_indexer re-cluster --workspace ./workspace
```

```bash
python -m face_indexer add --input ./new_photos --workspace ./workspace
```

第一阶段的核心验收标准是：给定一批原始照片，程序可以建库、分组；给定一张查询照片，程序可以返回匹配分组和该分组关联照片。

---

## 5. 目录结构要求

系统默认使用一个 `workspace` 目录保存所有产物。

目录结构如下：

```text
workspace/
  config.json

  database/
    face_index.db

  images/
    originals/
      不强制复制原图，可为空
    thumbnails/
      img_000001.jpg
      img_000002.jpg

  faces/
    face_000001.jpg
    face_000002.jpg
    face_000003.jpg

  groups/
    group_000001/
      face_000001.jpg
      face_000037.jpg
    group_000002/
      face_000004.jpg
      face_000109.jpg

  queries/
    query_000001/
      query.jpg
      query_face_000001.jpg
      result.json
      matched_images/
        img_000001.jpg
        img_000037.jpg

  logs/
    build.log
    search.log
    error.log
```

说明：

1. `database/face_index.db` 是 SQLite 数据库文件；
2. `faces/` 保存所有检测到的人脸截图；
3. `groups/` 保存按分组归类后的人脸截图副本或软链接；
4. `queries/` 保存每次查询的输入、查询人脸截图、结果 JSON 和导出的匹配照片；
5. 原始照片可以不复制到 workspace，但数据库必须保存原始照片绝对路径；
6. 如果需要保证数据可迁移，可配置 `copy_originals=true`，将原图复制到 workspace。

---

## 6. 配置文件要求

系统必须支持配置文件 `workspace/config.json`。

默认配置如下：

```json
{
  "model": {
    "provider": "insightface",
    "model_name": "buffalo_l",
    "det_size": [640, 640],
    "device": "cpu"
  },
  "image": {
    "supported_extensions": [".jpg", ".jpeg", ".png", ".webp"],
    "copy_originals": false,
    "thumbnail_max_size": 512
  },
  "face": {
    "min_face_width": 40,
    "min_face_height": 40,
    "crop_margin_ratio": 0.25,
    "save_low_quality_faces": true
  },
  "cluster": {
    "algorithm": "dbscan",
    "metric": "cosine",
    "eps": 0.42,
    "min_samples": 2,
    "noise_as_singleton_group": true
  },
  "search": {
    "metric": "cosine",
    "top_k": 10,
    "max_best_distance": 0.42,
    "min_group_vote_count": 3,
    "min_group_vote_ratio": 0.5
  },
  "export": {
    "copy_matched_images": true,
    "copy_matched_faces": true,
    "write_json": true,
    "write_csv": true
  }
}
```

说明：

1. `model.provider` 第一版固定为 `insightface`；
2. `model.model_name` 第一版默认使用 `buffalo_l`；
3. `model.device` 可取值为 `cpu` 或 `cuda`；
4. `cluster.eps` 是聚类距离阈值，必须支持调整；
5. `search.max_best_distance` 是查询匹配阈值，必须支持调整；
6. `noise_as_singleton_group=true` 表示 DBSCAN 中无法归类的人脸也会单独形成一个 group；
7. 所有阈值都不得在代码中写死，必须从配置中读取。

---

## 7. 数据库设计

第一版使用 SQLite。

数据库文件路径：

```text
workspace/database/face_index.db
```

### 7.1 images 表

用于保存原始图片信息。

```sql
CREATE TABLE images (
    image_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    file_size INTEGER NOT NULL,
    face_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

字段说明：

| 字段            | 说明                                              |
| ------------- | ----------------------------------------------- |
| image_id      | 图片唯一 ID，例如 `img_000001`                         |
| file_path     | 原始图片绝对路径                                        |
| file_name     | 文件名                                             |
| file_hash     | 文件内容哈希，用于去重                                     |
| width         | 图片宽度                                            |
| height        | 图片高度                                            |
| file_size     | 文件大小                                            |
| face_count    | 检测到的人脸数量                                        |
| status        | `success` / `no_face` / `failed` / `duplicated` |
| error_message | 失败原因                                            |
| created_at    | 创建时间                                            |
| updated_at    | 更新时间                                            |

### 7.2 faces 表

用于保存每张人脸信息。

```sql
CREATE TABLE faces (
    face_id TEXT PRIMARY KEY,
    image_id TEXT NOT NULL,
    face_crop_path TEXT NOT NULL,
    bbox_x REAL NOT NULL,
    bbox_y REAL NOT NULL,
    bbox_width REAL NOT NULL,
    bbox_height REAL NOT NULL,
    detection_score REAL,
    embedding BLOB NOT NULL,
    embedding_dim INTEGER NOT NULL,
    embedding_norm REAL,
    group_id TEXT,
    quality_score REAL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (image_id) REFERENCES images(image_id)
);
```

字段说明：

| 字段              | 说明                                  |
| --------------- | ----------------------------------- |
| face_id         | 人脸唯一 ID，例如 `face_000001`            |
| image_id        | 所属图片 ID                             |
| face_crop_path  | 人脸截图路径                              |
| bbox_x          | 人脸框左上角 x 坐标                         |
| bbox_y          | 人脸框左上角 y 坐标                         |
| bbox_width      | 人脸框宽度                               |
| bbox_height     | 人脸框高度                               |
| detection_score | 模型检测置信度                             |
| embedding       | 人脸特征向量，float32 序列化为 BLOB            |
| embedding_dim   | 特征向量维度                              |
| embedding_norm  | 特征向量归一化前或归一化后的范数                    |
| group_id        | 所属人脸分组                              |
| quality_score   | 人脸质量分                               |
| status          | `valid` / `low_quality` / `ignored` |
| created_at      | 创建时间                                |
| updated_at      | 更新时间                                |

### 7.3 face_groups 表

用于保存人脸分组信息。

```sql
CREATE TABLE face_groups (
    group_id TEXT PRIMARY KEY,
    representative_face_id TEXT,
    face_count INTEGER NOT NULL DEFAULT 0,
    image_count INTEGER NOT NULL DEFAULT 0,
    group_center_embedding BLOB,
    embedding_dim INTEGER,
    status TEXT NOT NULL,
    note TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

字段说明：

| 字段                     | 说明                              |
| ---------------------- | ------------------------------- |
| group_id               | 分组唯一 ID，例如 `group_000001`       |
| representative_face_id | 代表人脸 ID                         |
| face_count             | 该组人脸数量                          |
| image_count            | 该组关联原始图片数量                      |
| group_center_embedding | 该组中心特征向量                        |
| embedding_dim          | 特征维度                            |
| status                 | `active` / `merged` / `ignored` |
| note                   | 人工备注，第一版可为空                     |
| created_at             | 创建时间                            |
| updated_at             | 更新时间                            |

### 7.4 group_images 表

用于保存分组和图片的关系。

```sql
CREATE TABLE group_images (
    group_id TEXT NOT NULL,
    image_id TEXT NOT NULL,
    face_count_in_image INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (group_id, image_id),
    FOREIGN KEY (group_id) REFERENCES face_groups(group_id),
    FOREIGN KEY (image_id) REFERENCES images(image_id)
);
```

说明：

同一张原始图片中可能有同一 group 的多张人脸，因此需要 `face_count_in_image`。

### 7.5 queries 表

用于保存每次查询记录。

```sql
CREATE TABLE queries (
    query_id TEXT PRIMARY KEY,
    query_image_path TEXT NOT NULL,
    query_face_crop_path TEXT,
    matched_group_id TEXT,
    best_face_id TEXT,
    best_distance REAL,
    confidence TEXT NOT NULL,
    status TEXT NOT NULL,
    message TEXT,
    result_json_path TEXT,
    created_at TEXT NOT NULL
);
```

字段说明：

| 字段                   | 说明                                                              |
| -------------------- | --------------------------------------------------------------- |
| query_id             | 查询唯一 ID                                                         |
| query_image_path     | 查询照片路径                                                          |
| query_face_crop_path | 查询人脸截图路径                                                        |
| matched_group_id     | 匹配到的分组                                                          |
| best_face_id         | 最相似的已知人脸                                                        |
| best_distance        | 最小距离                                                            |
| confidence           | `high` / `medium` / `low` / `none`                              |
| status               | `matched` / `not_matched` / `no_face` / `multi_face` / `failed` |
| message              | 查询说明                                                            |
| result_json_path     | 查询结果 JSON 路径                                                    |
| created_at           | 查询时间                                                            |

### 7.6 indexes

必须创建以下索引：

```sql
CREATE INDEX idx_images_file_hash ON images(file_hash);
CREATE INDEX idx_faces_image_id ON faces(image_id);
CREATE INDEX idx_faces_group_id ON faces(group_id);
CREATE INDEX idx_group_images_group_id ON group_images(group_id);
CREATE INDEX idx_group_images_image_id ON group_images(image_id);
```

---

## 8. 命令行功能需求

## 8.1 build 命令

命令：

```bash
python -m face_indexer build --input ./photos --workspace ./workspace
```

可选参数：

```bash
--config ./config.json
--recursive true
--force false
--copy-originals false
--device cpu
```

### 8.1.1 build 输入

| 参数                 | 必填 | 说明                     |
| ------------------ | -- | ---------------------- |
| `--input`          | 是  | 原始照片目录                 |
| `--workspace`      | 是  | 工作目录                   |
| `--config`         | 否  | 配置文件路径                 |
| `--recursive`      | 否  | 是否递归扫描子目录，默认 true      |
| `--force`          | 否  | 是否清空已有数据库重新建库，默认 false |
| `--copy-originals` | 否  | 是否复制原图到 workspace      |
| `--device`         | 否  | `cpu` 或 `cuda`         |

### 8.1.2 build 处理流程

build 命令必须按以下顺序执行：

1. 初始化 workspace；
2. 加载配置；
3. 初始化数据库；
4. 扫描输入目录；
5. 过滤支持的图片格式；
6. 计算文件哈希；
7. 跳过重复图片；
8. 读取图片；
9. 调用 InsightFace 检测人脸；
10. 对每张检测到的人脸保存 bbox；
11. 对每张人脸保存人脸截图；
12. 保存 embedding；
13. 保存 image 和 face 记录；
14. 对全部有效 face 执行聚类；
15. 更新每张 face 的 group_id；
16. 生成 face_groups 记录；
17. 生成 group_images 记录；
18. 生成 groups 目录预览；
19. 输出 build summary。

### 8.1.3 build 输出

命令执行完成后，控制台必须输出：

```text
Build finished.

Input directory: ./photos
Total image files scanned: 642
Valid images: 638
Duplicated images: 2
Failed images: 2
Images with no face: 34
Total faces detected: 1286
Valid faces: 1210
Low quality faces: 76
Total groups: 57
Singleton groups: 8
Database: ./workspace/database/face_index.db
```

实际数字根据运行结果动态生成。

### 8.1.4 build 验收要求

build 执行完成后必须满足：

1. `face_index.db` 存在；
2. `images` 表中有图片记录；
3. `faces` 表中有有效人脸记录；
4. 每条 face 记录都有 bbox；
5. 每条有效 face 记录都有 embedding；
6. 每条有效 face 记录都有 group_id；
7. `face_groups` 表中有分组记录；
8. `group_images` 表中有分组与原图关系；
9. `faces/` 目录下存在人脸截图；
10. `groups/` 目录下存在分组预览。

---

## 8.2 search 命令

命令：

```bash
python -m face_indexer search --image ./query.jpg --workspace ./workspace --export ./result
```

可选参数：

```bash
--top-k 10
--target-face largest
--copy-images true
```

### 8.2.1 search 输入

| 参数              | 必填 | 说明             |
| --------------- | -- | -------------- |
| `--image`       | 是  | 查询照片路径         |
| `--workspace`   | 是  | 已建库的 workspace |
| `--export`      | 否  | 查询结果导出目录       |
| `--top-k`       | 否  | 参与投票的最相似人脸数量   |
| `--target-face` | 否  | 多人脸时选择策略       |
| `--copy-images` | 否  | 是否复制匹配原图到结果目录  |

### 8.2.2 search 处理流程

search 命令必须按以下顺序执行：

1. 加载 workspace 配置；
2. 打开 SQLite 数据库；
3. 初始化 InsightFace 模型；
4. 读取查询照片；
5. 检测查询照片中的人脸；
6. 如果没有人脸，返回 `no_face`；
7. 如果有多张人脸，根据 `target-face` 策略选择一张；
8. 保存查询人脸截图；
9. 提取查询人脸 embedding；
10. 从数据库读取所有有效 face 的 embedding；
11. 计算查询 embedding 与所有已知 embedding 的距离；
12. 按距离升序取 Top K；
13. 对 Top K 结果按 group_id 进行投票；
14. 根据阈值和投票结果判断是否匹配；
15. 如果匹配成功，查询该 group_id 关联的所有原始照片；
16. 输出匹配结果；
17. 写入 queries 表；
18. 导出 result.json；
19. 如启用导出，则复制匹配照片到结果目录。

### 8.2.3 多人脸处理规则

查询照片中如果检测到多张人脸，第一版默认使用 `target-face=largest`。

可选值：

| 值               | 说明                       |
| --------------- | ------------------------ |
| `largest`       | 选择 bbox 面积最大的人脸          |
| `highest-score` | 选择 detection_score 最高的人脸 |
| `center-most`   | 选择最靠近图片中心的人脸             |

第一版不要求实现交互式选择人脸。

如果查询照片检测到多张人脸，结果中必须标记：

```json
"query_has_multiple_faces": true
```

并记录被选择的人脸框位置。

### 8.2.4 匹配判定规则

查询匹配采用“全量 embedding 比对 + Top K 分组投票”。

具体规则：

1. 读取所有 `status='valid'` 且 `group_id IS NOT NULL` 的 face embedding；
2. 使用配置中的 `search.metric` 计算距离；
3. 按距离从小到大排序；
4. 取前 `top_k` 条；
5. 统计 Top K 中各 `group_id` 的出现次数；
6. 取票数最高的 group 作为候选分组；
7. 候选分组必须同时满足：

   * Top 1 最小距离 `best_distance <= max_best_distance`；
   * 候选组票数 `vote_count >= min_group_vote_count`；
   * 候选组票数占比 `vote_count / top_k >= min_group_vote_ratio`。
8. 如果满足条件，则返回匹配成功；
9. 如果不满足条件，则返回未匹配。

示例：

```text
top_k = 10
max_best_distance = 0.42
min_group_vote_count = 3
min_group_vote_ratio = 0.5

Top 10 中 group_000007 出现 7 次
best_distance = 0.31

=> 匹配 group_000007
```

如果：

```text
Top 10 中 group_000007 出现 4 次
best_distance = 0.31
vote_ratio = 0.4

=> 不匹配，或返回 low confidence
```

### 8.2.5 search 输出

控制台输出示例：

```text
Search finished.

Query image: ./query.jpg
Selected face bbox: [120, 80, 220, 220]
Matched: true
Matched group: group_000007
Best face: face_000341
Best distance: 0.31
Confidence: high
Group face count: 26
Matched original images: 18

Matched images:
1. /photos/001.jpg
2. /photos/014.jpg
3. /photos/093.jpg
...
Result JSON: ./result/result.json
Copied images: ./result/matched_images/
```

### 8.2.6 result.json 格式

```json
{
  "query_id": "query_000001",
  "query_image_path": "./query.jpg",
  "query_face_crop_path": "./workspace/queries/query_000001/query_face_000001.jpg",
  "query_face_bbox": {
    "x": 120,
    "y": 80,
    "width": 220,
    "height": 220
  },
  "query_has_multiple_faces": false,
  "matched": true,
  "matched_group_id": "group_000007",
  "best_face_id": "face_000341",
  "best_distance": 0.31,
  "confidence": "high",
  "top_k": [
    {
      "face_id": "face_000341",
      "group_id": "group_000007",
      "distance": 0.31,
      "image_id": "img_000088",
      "image_path": "/photos/088.jpg"
    }
  ],
  "group_summary": {
    "group_id": "group_000007",
    "face_count": 26,
    "image_count": 18,
    "representative_face_id": "face_000113"
  },
  "matched_images": [
    {
      "image_id": "img_000001",
      "file_path": "/photos/001.jpg",
      "file_name": "001.jpg",
      "faces": [
        {
          "face_id": "face_000003",
          "bbox": {
            "x": 102,
            "y": 88,
            "width": 96,
            "height": 96
          }
        }
      ]
    }
  ]
}
```

---

## 8.3 report 命令

命令：

```bash
python -m face_indexer report --workspace ./workspace
```

输出内容：

1. 图片总数；
2. 成功处理图片数；
3. 失败图片数；
4. 无人脸图片数；
5. 检测到的人脸总数；
6. 有效人脸数；
7. 低质量人脸数；
8. 分组数；
9. 单人脸分组数；
10. 最大分组的人脸数；
11. 最大分组关联图片数；
12. 数据库路径；
13. workspace 路径。

---

## 8.4 list-groups 命令

命令：

```bash
python -m face_indexer list-groups --workspace ./workspace
```

输出：

```text
group_id       face_count   image_count   representative_face
group_000001   38           21            face_000008
group_000002   24           15            face_000041
group_000003   1            1             face_000077
```

支持可选排序：

```bash
--sort face_count_desc
--min-face-count 2
```

---

## 8.5 re-cluster 命令

命令：

```bash
python -m face_indexer re-cluster --workspace ./workspace
```

作用：

1. 不重新读取原图；
2. 不重新做人脸检测；
3. 不重新提取 embedding；
4. 只读取数据库中已保存的 embedding；
5. 重新执行聚类；
6. 更新 `faces.group_id`；
7. 重建 `face_groups`；
8. 重建 `group_images`；
9. 重建 `groups/` 预览目录。

该命令用于调整聚类参数后快速验证效果。

---

## 9. 技术架构设计

## 9.1 总体架构

系统按分层架构设计：

```text
CLI / API Layer
      ↓
Application Service Layer
      ↓
Domain Service Layer
      ↓
Infrastructure Layer
      ↓
Storage / Model Runtime
```

### 9.1.1 CLI / API Layer

职责：

1. 接收命令行参数；
2. 校验参数；
3. 调用应用服务；
4. 输出执行结果。

后续服务化时，该层替换或扩展为 FastAPI Controller。

### 9.1.2 Application Service Layer

职责：

1. 编排完整业务流程；
2. 实现 build/search/report/re-cluster 等用例；
3. 控制事务边界；
4. 记录日志；
5. 生成最终输出。

核心类：

```text
BuildIndexService
SearchFaceService
ReportService
ReClusterService
ExportService
```

### 9.1.3 Domain Service Layer

职责：

1. 图片扫描；
2. 人脸检测；
3. 人脸裁剪；
4. 特征提取；
5. 人脸质量判断；
6. 人脸聚类；
7. 相似度计算；
8. Top K 投票。

核心类：

```text
ImageScanner
FaceAnalyzer
FaceCropper
FaceQualityEvaluator
FaceClusterer
FaceMatcher
GroupVoter
```

### 9.1.4 Infrastructure Layer

职责：

1. SQLite 读写；
2. 文件系统读写；
3. 图片读取与保存；
4. InsightFace 模型初始化；
5. 配置加载；
6. 日志记录。

核心类：

```text
SQLiteRepository
FileStorage
ImageIO
InsightFaceEngine
ConfigLoader
LoggerFactory
```

---

## 9.2 模块划分

推荐代码目录：

```text
face_indexer/
  __init__.py
  __main__.py

  cli/
    main.py
    build_cmd.py
    search_cmd.py
    report_cmd.py
    cluster_cmd.py

  app/
    build_index_service.py
    search_face_service.py
    report_service.py
    recluster_service.py
    export_service.py

  domain/
    image_scanner.py
    face_analyzer.py
    face_cropper.py
    face_quality.py
    face_clusterer.py
    face_matcher.py
    group_voter.py
    ids.py
    models.py

  infra/
    config.py
    db.py
    repositories.py
    file_storage.py
    image_io.py
    insightface_engine.py
    logging.py

  api/
    main.py
    schemas.py
    routes_index.py
    routes_search.py
    routes_groups.py
    routes_images.py

  tests/
    test_build_index.py
    test_search_face.py
    test_clusterer.py
    test_matcher.py
```

第一阶段可以先实现 `cli/`、`app/`、`domain/`、`infra/`。

`api/` 目录可以先放接口草稿和 Pydantic Schema，第二阶段再正式启用。

---

## 9.3 InsightFaceEngine 设计

`InsightFaceEngine` 负责封装 InsightFace。

对外提供统一方法：

```python
class InsightFaceEngine:
    def detect_and_extract(self, image: np.ndarray) -> list[DetectedFace]:
        ...
```

`DetectedFace` 数据结构：

```python
@dataclass
class DetectedFace:
    bbox: tuple[float, float, float, float]  # x, y, width, height
    detection_score: float | None
    embedding: np.ndarray
    landmarks: Any | None = None
```

要求：

1. 业务层不得直接调用 InsightFace 原始对象；
2. 所有人脸检测和特征提取必须通过 `InsightFaceEngine`；
3. `InsightFaceEngine` 初始化时读取配置；
4. device 为 `cpu` 时使用 CPU；
5. device 为 `cuda` 时使用 GPU；
6. 如果 GPU 不可用，必须报错，不得静默回退，除非配置 `allow_cpu_fallback=true`；
7. 输出的 embedding 必须转换为 `np.float32`；
8. 输出的 embedding 必须归一化后再保存和比对；
9. bbox 坐标统一转换为 `x, y, width, height`。

---

## 9.4 人脸裁剪规则

裁剪时基于 bbox 扩大一定边距。

配置：

```json
"crop_margin_ratio": 0.25
```

计算方式：

```text
margin_x = bbox_width * crop_margin_ratio
margin_y = bbox_height * crop_margin_ratio

crop_x1 = max(0, x - margin_x)
crop_y1 = max(0, y - margin_y)
crop_x2 = min(image_width, x + width + margin_x)
crop_y2 = min(image_height, y + height + margin_y)
```

裁剪图保存为：

```text
workspace/faces/face_000001.jpg
```

查询人脸裁剪图保存为：

```text
workspace/queries/query_000001/query_face_000001.jpg
```

---

## 9.5 人脸质量规则

第一版至少实现以下质量规则：

1. 人脸宽度小于 `min_face_width`，标记为 `low_quality`；
2. 人脸高度小于 `min_face_height`，标记为 `low_quality`；
3. embedding 缺失，标记为 `ignored`；
4. 检测置信度过低，标记为 `low_quality`。

第一版默认 `save_low_quality_faces=true`。

如果人脸是低质量，仍保存截图和记录，但聚类时可以根据配置决定是否参与：

```json
"cluster_include_low_quality_faces": false
```

默认不参与聚类。

---

## 9.6 聚类设计

第一版使用 DBSCAN。

输入：

```text
所有 status='valid' 的 face embedding
```

输出：

```text
face_id -> group_id
```

聚类步骤：

1. 从数据库读取所有有效 face；
2. 将 embedding 组成矩阵；
3. 使用配置中的 `cluster.metric`；
4. 使用配置中的 `cluster.eps`；
5. 使用配置中的 `cluster.min_samples`；
6. 执行 DBSCAN；
7. 对每个 cluster label 创建一个 group；
8. 对 label = -1 的噪声点，如果 `noise_as_singleton_group=true`，每个噪声点单独创建一个 group；
9. 更新所有 face 的 group_id；
10. 计算每个 group 的 center embedding；
11. 选择每个 group 的 representative_face_id；
12. 写入 face_groups；
13. 写入 group_images。

### 9.6.1 representative_face_id 选择规则

每个 group 的代表人脸选择规则：

1. 计算该组所有 embedding 的中心向量；
2. 找出距离中心最近的 face；
3. 将其作为 `representative_face_id`。

如果组内只有一张 face，则该 face 就是代表人脸。

### 9.6.2 group_center_embedding 计算规则

计算方式：

```text
center = mean(group_embeddings)
center = normalize(center)
```

保存为 float32 BLOB。

---

## 9.7 查询匹配设计

查询时不使用聚类算法。

查询时执行：

```text
query embedding
    ↓
与数据库中所有有效 face embedding 计算距离
    ↓
取 Top K
    ↓
按 group_id 投票
    ↓
返回候选 group
```

### 9.7.1 距离计算

第一版统一使用 cosine distance。

如果 embedding 已归一化，cosine similarity 可通过点积计算：

```text
similarity = dot(query_embedding, known_embedding)
distance = 1 - similarity
```

距离越小，表示越相似。

### 9.7.2 Top K 投票规则

设：

```text
top_k = 10
```

如果数据库中有效 face 数量少于 10，则使用全部有效 face。

投票规则：

1. 取距离最近的 K 张已知人脸；
2. 统计每个 group_id 出现次数；
3. 票数最多的 group_id 作为候选；
4. 如果有多个 group 票数相同，则选择其中 best distance 最小的 group；
5. 最终候选 group 必须满足匹配阈值要求。

### 9.7.3 置信度规则

返回 confidence：

```text
high
medium
low
none
```

建议规则：

```text
high:
  best_distance <= max_best_distance
  且 vote_ratio >= 0.7

medium:
  best_distance <= max_best_distance
  且 vote_ratio >= 0.5

low:
  best_distance <= max_best_distance
  但 vote_ratio < 0.5

none:
  best_distance > max_best_distance
```

只有 `high` 和 `medium` 默认视为匹配成功。

`low` 默认返回候选分组，但 `matched=false`，供人工复核。

---

## 10. 后续 Python API 服务设计

第二阶段将命令行程序改造为 Python 后端服务。

建议使用 FastAPI。

API 服务应复用第一阶段的 Application Service 和 Domain Service，不得在 API 层重复实现业务逻辑。

---

## 10.1 API 基础约定

### 10.1.1 请求格式

上传图片使用：

```text
multipart/form-data
```

普通查询使用：

```text
application/json
```

### 10.1.2 响应格式

所有接口统一返回：

```json
{
  "success": true,
  "data": {},
  "error": null
}
```

失败返回：

```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "NO_FACE_DETECTED",
    "message": "No face detected in query image."
  }
}
```

### 10.1.3 错误码

| 错误码                   | 说明              |
| --------------------- | --------------- |
| `INVALID_ARGUMENT`    | 参数错误            |
| `WORKSPACE_NOT_FOUND` | workspace 不存在   |
| `DATABASE_NOT_FOUND`  | 数据库不存在          |
| `IMAGE_NOT_FOUND`     | 图片不存在           |
| `IMAGE_READ_FAILED`   | 图片读取失败          |
| `NO_FACE_DETECTED`    | 未检测到人脸          |
| `MULTI_FACE_DETECTED` | 检测到多张人脸且未指定选择策略 |
| `NO_INDEX_DATA`       | 人脸库为空           |
| `NO_MATCH_FOUND`      | 未找到匹配分组         |
| `MODEL_INIT_FAILED`   | 模型初始化失败         |
| `INTERNAL_ERROR`      | 未知错误            |

---

## 10.2 API 接口列表

### 10.2.1 创建或重建索引

```http
POST /api/v1/index/build
```

请求：

```json
{
  "input_dir": "./photos",
  "workspace": "./workspace",
  "force": false,
  "recursive": true,
  "device": "cpu"
}
```

响应：

```json
{
  "success": true,
  "data": {
    "workspace": "./workspace",
    "status": "finished",
    "summary": {
      "total_images": 642,
      "valid_images": 638,
      "total_faces": 1286,
      "valid_faces": 1210,
      "total_groups": 57
    }
  },
  "error": null
}
```

第一版 API 可同步执行。后续如果照片数量增大，应改为异步任务。

---

### 10.2.2 查询人脸

```http
POST /api/v1/search
```

请求方式：

```text
multipart/form-data
```

字段：

| 字段              | 类型     | 必填 | 说明            |
| --------------- | ------ | -- | ------------- |
| `workspace`     | string | 是  | 已建库 workspace |
| `image`         | file   | 是  | 查询照片          |
| `top_k`         | int    | 否  | Top K 数量      |
| `target_face`   | string | 否  | 多人脸选择策略       |
| `export_result` | bool   | 否  | 是否导出结果        |

响应：

```json
{
  "success": true,
  "data": {
    "query_id": "query_000001",
    "matched": true,
    "matched_group_id": "group_000007",
    "best_face_id": "face_000341",
    "best_distance": 0.31,
    "confidence": "high",
    "group_summary": {
      "group_id": "group_000007",
      "face_count": 26,
      "image_count": 18
    },
    "matched_images": [
      {
        "image_id": "img_000001",
        "file_path": "/photos/001.jpg",
        "file_name": "001.jpg"
      }
    ]
  },
  "error": null
}
```

---

### 10.2.3 查看分组列表

```http
GET /api/v1/groups
```

查询参数：

| 参数               | 必填 | 说明           |
| ---------------- | -- | ------------ |
| `workspace`      | 是  | workspace 路径 |
| `min_face_count` | 否  | 最小人脸数        |
| `sort`           | 否  | 排序方式         |

响应：

```json
{
  "success": true,
  "data": {
    "groups": [
      {
        "group_id": "group_000001",
        "face_count": 38,
        "image_count": 21,
        "representative_face_id": "face_000008"
      }
    ]
  },
  "error": null
}
```

---

### 10.2.4 查看某个分组详情

```http
GET /api/v1/groups/{group_id}
```

查询参数：

| 参数          | 必填 | 说明           |
| ----------- | -- | ------------ |
| `workspace` | 是  | workspace 路径 |

响应：

```json
{
  "success": true,
  "data": {
    "group_id": "group_000007",
    "face_count": 26,
    "image_count": 18,
    "representative_face_id": "face_000113",
    "faces": [
      {
        "face_id": "face_000341",
        "image_id": "img_000088",
        "face_crop_path": "./workspace/faces/face_000341.jpg",
        "bbox": {
          "x": 120,
          "y": 80,
          "width": 100,
          "height": 100
        }
      }
    ],
    "images": [
      {
        "image_id": "img_000088",
        "file_path": "/photos/088.jpg",
        "file_name": "088.jpg"
      }
    ]
  },
  "error": null
}
```

---

### 10.2.5 重新聚类

```http
POST /api/v1/index/re-cluster
```

请求：

```json
{
  "workspace": "./workspace",
  "eps": 0.42,
  "min_samples": 2,
  "metric": "cosine"
}
```

响应：

```json
{
  "success": true,
  "data": {
    "workspace": "./workspace",
    "total_faces": 1210,
    "total_groups": 57,
    "singleton_groups": 8
  },
  "error": null
}
```

---

### 10.2.6 查看建库报告

```http
GET /api/v1/report
```

查询参数：

```text
workspace=./workspace
```

响应：

```json
{
  "success": true,
  "data": {
    "total_images": 642,
    "valid_images": 638,
    "failed_images": 2,
    "images_with_no_face": 34,
    "total_faces": 1286,
    "valid_faces": 1210,
    "total_groups": 57
  },
  "error": null
}
```

---

## 11. 性能设计

### 11.1 第一版性能预期

在 600 多张照片、几百到几千张人脸的规模下：

1. 建库阶段允许耗时较长；
2. 查询阶段必须较快；
3. 查询阶段不得重新处理全部照片；
4. 查询阶段全量 embedding 比对即可，不需要引入向量数据库；
5. 当有效 face 数量低于 10000 时，可直接使用 numpy 矩阵计算距离。

### 11.2 后续扩展

当 face 数量超过 100000 时，可考虑引入：

1. FAISS；
2. Annoy；
3. Milvus；
4. PostgreSQL + pgvector。

第一版不引入这些组件，避免增加复杂度。

---

## 12. 日志要求

系统至少输出三类日志：

### 12.1 build.log

记录：

1. 扫描到的图片数量；
2. 每张图片处理状态；
3. 每张图片检测到的人脸数量；
4. 失败图片路径和错误原因；
5. 聚类参数；
6. 聚类结果。

### 12.2 search.log

记录：

1. 查询图片路径；
2. 查询图片检测到的人脸数量；
3. 被选中的查询人脸 bbox；
4. Top K 匹配结果；
5. 最终分组；
6. 查询耗时。

### 12.3 error.log

记录：

1. 图片读取失败；
2. 模型初始化失败；
3. 数据库异常；
4. 文件写入异常；
5. 其他未捕获异常。

---

## 13. 异常处理要求

| 场景            | 处理方式                                        |
| ------------- | ------------------------------------------- |
| 输入目录不存在       | build 失败，返回明确错误                             |
| workspace 不存在 | search 失败，返回明确错误                            |
| 数据库不存在        | search 失败，返回明确错误                            |
| 图片格式不支持       | build 时跳过并记录                                |
| 图片损坏          | build 时标记 `failed`                          |
| 图片无人脸         | images.status=`no_face`                     |
| 图片多人脸         | 每张脸分别保存                                     |
| 查询照片无人脸       | 返回 `NO_FACE_DETECTED`                       |
| 查询照片多人脸       | 按策略选择一张，并记录 `query_has_multiple_faces=true` |
| 没有任何有效 face   | search 返回 `NO_INDEX_DATA`                   |
| 距离超过阈值        | 返回 `NO_MATCH_FOUND`                         |
| group 投票不足    | 返回低置信度或不匹配                                  |
| 文件复制失败        | 查询成功，但导出警告                                  |
| 模型初始化失败       | 终止任务并返回错误                                   |

---

## 14. 测试要求

第一版至少编写以下测试。

### 14.1 单元测试

| 测试对象             | 测试内容                         |
| ---------------- | ---------------------------- |
| ImageScanner     | 能正确扫描图片，过滤非图片                |
| FaceCropper      | bbox 扩边裁剪不越界                 |
| FaceMatcher      | cosine distance 计算正确         |
| GroupVoter       | Top K 投票结果正确                 |
| SQLiteRepository | embedding 能正确写入和读取           |
| FaceClusterer    | DBSCAN label 能正确映射到 group_id |

### 14.2 集成测试

准备一个小型测试目录：

```text
test_photos/
  person_a_1.jpg
  person_a_2.jpg
  person_b_1.jpg
  person_b_2.jpg
  no_face.jpg
```

测试：

1. build 能成功完成；
2. 能检测到人脸；
3. 能生成至少 2 个 group；
4. search 查询 `person_a_1.jpg` 能返回 person_a 所在 group；
5. 查询无人脸图片时返回 no_face；
6. 查询结果中包含关联原图路径。

### 14.3 人工验收

工程交付时需要提供：

1. 一份 build 输出日志；
2. 一份 report 输出；
3. 至少 3 次 search 示例；
4. 每次 search 的 result.json；
5. 查询结果导出的 matched_images 目录；
6. groups 目录供人工检查分组效果。

---

## 15. 第一版实施步骤

建议按以下顺序实施：

### 第一步：项目骨架

完成：

1. Python 包结构；
2. CLI 入口；
3. 配置加载；
4. workspace 初始化；
5. 日志初始化。

### 第二步：图片扫描与数据库

完成：

1. 图片扫描；
2. 文件哈希；
3. SQLite 初始化；
4. images 表写入；
5. 重复图片跳过。

### 第三步：InsightFace 接入

完成：

1. 模型初始化；
2. 单张图片人脸检测；
3. bbox 统一格式；
4. embedding 提取；
5. embedding 归一化。

### 第四步：人脸截图与保存

完成：

1. bbox 扩边；
2. 裁剪人脸；
3. 保存 face crop；
4. faces 表写入。

### 第五步：聚类分组

完成：

1. 从数据库读取 embedding；
2. DBSCAN 聚类；
3. 创建 group_id；
4. 更新 faces.group_id；
5. 写入 face_groups；
6. 写入 group_images；
7. 生成 groups 预览目录。

### 第六步：查询功能

完成：

1. 查询图片读取；
2. 查询人脸检测；
3. 多人脸选择策略；
4. 查询 embedding 提取；
5. 与全量已知 embedding 比对；
6. Top K 分组投票；
7. 查询 group 关联照片；
8. 输出 result.json；
9. 复制 matched_images。

### 第七步：报告与验证

完成：

1. report 命令；
2. list-groups 命令；
3. 集成测试；
4. 人工验收材料。

### 第八步：API 服务化准备

完成：

1. 将 CLI 调用的业务逻辑沉淀到 Service；
2. API 层只调用 Service；
3. 定义 Pydantic 请求和响应模型；
4. 初步实现 `/api/v1/search`；
5. 初步实现 `/api/v1/groups` 和 `/api/v1/report`。

---

## 16. 第一版不做的内容

第一版明确不做：

1. 不做人脸实名识别；
2. 不做用户账号系统；
3. 不做 Web 管理后台；
4. 不做移动端 SDK；
5. 不做实时摄像头识别；
6. 不做分布式任务队列；
7. 不做大型向量数据库；
8. 不做人脸活体检测；
9. 不做隐私权限管理系统；
10. 不做复杂人工合并/拆分 UI。

但数据库设计需要预留：

1. group 备注；
2. group 人工修正；
3. group 合并；
4. group 拆分；
5. 后续 API 服务化。

---

## 17. 交付物清单

工程第一版交付时必须包含：

1. Python 源码；
2. requirements.txt 或 pyproject.toml；
3. README.md；
4. config.example.json；
5. SQLite 初始化脚本或自动初始化逻辑；
6. build 命令；
7. search 命令；
8. report 命令；
9. list-groups 命令；
10. 至少一组测试图片运行结果；
11. build.log 示例；
12. search.log 示例；
13. result.json 示例；
14. groups 预览目录示例；
15. 使用说明。

---

## 18. README 最小使用说明

README 中至少包含以下内容：

```bash
# 1. 创建环境
python -m venv .venv
source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 建库
python -m face_indexer build --input ./photos --workspace ./workspace

# 4. 查看报告
python -m face_indexer report --workspace ./workspace

# 5. 查询
python -m face_indexer search --image ./query.jpg --workspace ./workspace --export ./result

# 6. 查看分组
python -m face_indexer list-groups --workspace ./workspace
```

README 中必须说明：

1. 查询阶段不会重新处理全部原始照片；
2. 查询阶段只使用已保存 embedding；
3. 如果修改聚类阈值，应执行 re-cluster；
4. 如果更换模型，需要重新 build；
5. 如果新增照片，应执行 add 或重新 build。

---

## 19. 最终架构结论

第一版采用以下方案：

```text
InsightFace
+ ONNX Runtime
+ Python CLI
+ SQLite
+ NumPy
+ scikit-learn DBSCAN
+ OpenCV 或 Pillow
```

系统核心逻辑：

```text
建库时：
原始照片 -> 人脸检测 -> 人脸截图 -> embedding -> SQLite -> DBSCAN 聚类 -> group_id

查询时：
查询照片 -> 人脸检测 -> query embedding -> 读取 SQLite 已保存 embedding -> Top K 比对和投票 -> group_id -> 查询关联原图
```

该方案的关键点是：

```text
人脸识别、特征提取、分组是一次性建库工作；
查询时只处理查询照片，不重复处理原始照片；
查询使用已保存 embedding，效率高，结果可复用，可验证，可服务化。
```


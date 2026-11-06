# 施工人员工效智能感知

通过施工现场视频检测人员姿态、跟踪人员位置，结合相机标定和工作区域配置，统计区域人数、人员状态及班组合格工效时长。

**当前版本：2026-11-03。**实时与历史分析的使用流程按物料工程对齐，工效保留自己的人员识别、多区域配置和计工规则。默认使用本地 JSON 持久化资源、任务和分析结果，服务重启后可继续查询已保存数据。

## 设计思路

视频输入、场景配置、分析任务和结果查询分开管理。前端负责选择视频与配置、标定划区、启动停止和展示结果；后端负责保存配置、调度任务、执行人员分析及累计区域工效。

实时与历史分析共用人员姿态检测、ByteTrack 跟踪、坐标投影和工效计算逻辑。每次新建配置保留独立 ID，任务创建时保存配置快照，通过 taskId 追溯本次使用的配置和结果。

## 第一版的五个功能

### 1. 相机管理

登记 cameraId 与 RTSP/RTMP 视频流地址，为实时分析提供输入。历史视频上传无需 cameraId，也不要求登记或连接实时相机。

### 2. 实时工效分析

选择相机及适用配置，创建 CAMERA_STREAM 任务。系统检测人员、跟踪轨迹，将人员位置投影到场景图，判断工作区域状态并累计人数和工效。

实时任务须使用与 cameraId 匹配的实时配置，不能使用视频专属配置。同一相机同时只允许一个实时任务，可查询状态、请求停止并查看结果。

### 3. 历史视频分析

按照物料工程的流程，先上传录像取得 videoId，再选择已有 configurationId。可以复用相机配置、其他视频配置或当前视频配置；没有适用配置时，再上传场景图、标定划区并新建配置。

```text
上传 MP4 → videoId
               ↓
选择已有 configurationId，或上传场景图并新建配置
               ↓
确定有效且带时区的录像开始时间
               ↓
创建 VIDEO_FILE 任务 → taskId → 查询进度和结果 / 请求停止
```

历史任务不再要求配置绑定当前 videoId，也不要求配置所属相机与录像来源一致；选择配置时应确认标定、区域和实际画面适用。工效支持多个工作区域，不复制物料的单区域限制。

**全局同时只允许一个历史任务，历史与实时任务可并行，即使来自同一相机。**历史任务排队、运行或处于 STOPPING 时，新历史任务返回 HTTP 409 / WE0203；需等到 STOPPED、COMPLETED 或 FAILED 后再启动。新旧历史接口共用这一名额。

录像开始时间优先使用启动时的 recordingStartedAt；未传时使用视频上传信息中的 capturedAt。新 VIDEO_FILE 任务必须具备有效、带时区的录像时间，例如 `2026-11-03T09:00:00+08:00`，否则返回 422。结果时间按“录像开始时间 + 视频内时间”计算，任务创建时间另行保留。

停止、失败或完成后，可以用同一视频和配置创建新的 taskId，从头重新分析，不是断点续跑。旧 historical-analyses 接口暂时保留兼容，新对接统一使用 VIDEO_FILE 流程。

### 4. 工效结果查询

实时与历史结果使用同一套保存和查询方式，通过 taskId 区分分析批次。可查询任务状态、处理进度、最新人员与区域数据、区域工效汇总和已保存的帧结果，也支持 WebSocket 推送。

历史任务的时间查询使用录像开始时间筛选任务，返回该任务的完整汇总，不按查询窗口重新切分工时。工作台可加载重启前已保存的任务；目前没有可视化历史轨迹回放页面。

### 5. 场景与配置管理

上传 PNG/JPEG 场景图，通过至少四对不共线的对应点建立视频与场景的转换关系，再绘制工作区域，配置计划人数、要求工时和识别参数。

已有配置可直接选用；重新标定时创建新的 configurationId。新建视频专属场景和配置时仍检查资源绑定关系，任务创建时则允许选择已有适用配置。

任务保存独立的 configurationSnapshot，执行和归档使用该快照，后续配置变化不会改变已提交任务的分析条件。

## 2026-11-03 更新

| 方面 | 当前修改 |
| --- | --- |
| 录像上传 | 提供 multipart `/videos`，返回 videoId；补齐 MP4、空文件、大小校验、SHA256 去重和失败清理 |
| 历史流程 | 使用 videoId + configurationId 启动 VIDEO_FILE 任务，支持复用已有配置 |
| 任务冲突 | 历史全局单任务，实时按相机互斥；实时与历史独立调度，停止中继续占位 |
| 录像时间 | 新增 recordingStartedAt，回退到 capturedAt；新历史任务要求有效且带时区的时间 |
| 配置与持久化 | 在已有 JSON 持久化基础上，补齐任务配置快照、录像时间和资源元数据的保存及恢复 |
| 测试工作台 | 支持视频上传预览、新建或复用配置、录像时间输入、启动停止及已保存结果查询 |
| 文档与验证 | 更新接口说明、1103 Word 和修改记录；87 项自动化测试通过，两条浏览器操作流程通过 |

本次调整接口、调度、配置和时间输入，人员识别及工效计算规则保持不变。真实 YOLO 效果和远端 Java 联调仍需部署后验收。详细过程见[今日修改记录](docs/2026-11-03历史视频流程对齐记录.md)。

## 持久化与重启恢复

默认入口 `work_efficiency_perception.main:app` 已启用本地 JSON 持久化，**无需安装数据库即可保存和读取结果**。目前使用文件仓储；MySQL 仅有初版表结构，尚未接入运行，旧 SQLite 数据也不会自动迁移。

| 保存内容 | 用途 |
| --- | --- |
| 相机、视频、场景和配置 | 恢复资源索引、读取素材及选择配置 |
| 任务信息与配置快照 | 记录本次输入、配置、转换矩阵和录像时间 |
| 最新结果与进度 | 服务重启后继续查询已提交结果 |
| 人员与区域帧结果 | 保存实际推理帧的人员数据及区域统计，支持分页读取 |
| 累计工效与 30 秒快照 | 查看任务整体工效、达标/不足区间和过程快照 |

默认目录：

```text
data/
  videos/                              # 上传的原始 MP4
  results/
    resources.json                     # 资源与任务恢复索引，包含场景图及配置
    tasks/<taskId>/
      task_configuration.json          # 任务、配置快照、单应矩阵、视频信息及场景图
      snapshot.json                    # 最新结果、进度及区间恢复数据
      efficiency_overall.json          # 整体累计工效与达标/不足区间
      persons/frame_*.json             # 实际推理帧的人员及区域数据
      efficiency_30s/at_*.json          # 视频时间 30、60、90 秒等节点的工效快照
```

场景图以 Base64 保存，原始视频单独保留；不重复保存原始画面或生成带骨架的视频。关闭 persistPoseAndTrajectory 时，保存的结果剔除人员明细，区域统计继续保存。30 秒快照按视频/输入帧时间触发，不按程序运行耗时触发；只记录实际推理结果，不补造跳过的节点。

服务重启后，已完成、已停止或已失败任务及结果可继续查询；重启时仍未结束的任务标为 FAILED。重新分析会创建新任务，从头处理，不恢复跟踪器、计工人数或执行进度。

当前同一保存目录仅支持一个服务进程。每个 JSON 文件采用临时文件和原子替换写入，多个文件间没有数据库事务；异常断电时最新帧与汇总可能不同步。备份时应正常停止服务后同时复制 results 和 videos 目录；视频路径为绝对路径，迁移后需要调整。文件目前不会自动清理。

## 启动与联调

使用 Python 3.10 及以上版本，在工程根目录执行：

```powershell
python -m pip install -r requirements-api.txt
$env:PYTHONPATH = "src"
python -m uvicorn work_efficiency_perception.main:app --host 127.0.0.1 --port 8000
```

打开[测试工作台](http://127.0.0.1:8000/test-ui/) 或 [API 文档](http://127.0.0.1:8000/docs)。工作台与 API 同源，无需额外构建前端；启用鉴权后在页面填写 Bearer Token。

默认模型为 `model/yolo11m-pose.pt`。仓库中的权重由 Git LFS 管理，需确保本地取得实际模型文件；也可通过环境变量指定部署权重。模型文件不存在时不会启用分析工作器，不能把任务创建成功当作人员识别已经开始。

| 环境变量 | 默认值或作用 |
| --- | --- |
| WORK_EFFICIENCY_MODEL_PATH | `model/yolo11m-pose.pt`，姿态模型路径 |
| WORK_EFFICIENCY_DEVICE | 未设置，使用模型运行时默认设备 |
| WORK_EFFICIENCY_IMAGE_SIZE | `1280`，推理尺寸 |
| WORK_EFFICIENCY_RESULTS_DIRECTORY | `data/results`，JSON 保存目录 |
| WORK_EFFICIENCY_VIDEO_DIRECTORY | `data/videos`，原始视频目录 |
| WORK_EFFICIENCY_MAX_VIDEO_BYTES | `5368709120`，默认上传上限 5 GiB |
| WORK_EFFICIENCY_BEARER_TOKEN | 未设置，启用时要求 Bearer Token |

直接调用 `create_app()` 默认使用内存仓储供测试；要使用上述持久化能力，应启动 `main:app`。部署更新后需重启服务，Java 端按新规范传入所选配置和录像时间。

## 接口流程速查

公共前缀：`/api/work-efficiency/v1`。

| 操作 | 接口及关键输入 |
| --- | --- |
| 上传录像 | `POST /videos`：multipart file，可选 capturedAt，返回 videoId |
| 读取视频 | `GET /videos`、`GET /videos/{videoId}`、`GET /videos/{videoId}/preview` |
| 新建场景与配置 | `POST /scenes`、`POST /configurations`；已有配置可跳过此步骤 |
| 读取已有配置 | `GET /configurations/{configurationId}` |
| 登记实时相机 | `POST /cameras`：cameraId、streamUrl |
| 启动历史任务 | `POST /tasks`：inputType=VIDEO_FILE、videoId、configurationId、有效录像时间 |
| 启动实时任务 | `POST /tasks`：inputType=CAMERA_STREAM、cameraId、configurationId |
| 停止任务 | `PATCH /tasks/{taskId}`：`{"action":"STOP"}`，等待终态后再启动 |
| 查询已保存任务 | `GET /tasks`、`GET /tasks/{taskId}` |
| 查询进度和结果 | `GET /tasks/{taskId}/progress`、`GET /tasks/{taskId}/latest-result` |
| 查询工效及帧明细 | `GET /team-efficiency`、`GET /tasks/{taskId}/frames` |

请求字段、响应示例、时间优先级及 PowerShell 调用方式见[历史视频接口规范](docs/history_video_api.md)。旧 historical-analyses 接口保留原有兼容规则，新调用方不需要先创建 analysisId。

## 工效计算口径

当前 WORKING 表示经进入防抖确认后处于工作区域内，不直接证明人员正在执行某一具体工序。姿态模型输出与跟踪结果仍需现场验证。

班组连续计工沿用 2026-10-31 规则：区内丢失且未观察到出界时保留人数；确认出界后保留 20 分钟宽限，新成员可按规则抵消或接替保留名额。actualWorkingCount 是有效计工人数，可能大于当前可见 persons 数量；人数达标时累计合格工时。exitDetectionDurationSec 仍是出界防抖，不要改为 1200。

完整计工规则、采样方式、包结构和早期部署说明保留在[原首页技术说明存档](docs/README_technical_reference_2026-11-03.md)。

## 测试与详细文档

开发验证可安装 `requirements-dev.txt` 后运行 `python -m pytest`。2026-11-03 最终代码回归为 **87 passed**，测试不依赖真实模型或 GPU；真实模型精度与现场流仍需单独验收。

- [2026-11-03 修改记录](docs/2026-11-03历史视频流程对齐记录.md)：当天最终结果、接口调整及验证边界。
- [历史视频接口规范](docs/history_video_api.md)：以物料流程为基准的新对接说明。
- [1103 Word 规范](docs/施工人员工效智能感知历史视频接口规范-1103.docx)：可供同事查阅的接口文档。
- [配置 JSON 示例](docs/examples/history-configuration.json)：新建配置时按实际画面替换 ID、标定点和区域。
- [开发进度](docs/development_progress.md)：当前更新和早期开发记录。

0825/0902 Word 保留为旧版资料，历史视频的新对接流程以当前首页、1103 规范和运行服务的 OpenAPI 为准。

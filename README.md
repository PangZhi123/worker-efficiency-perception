# 施工人员工效智能感知后端算法

独立 Python 工程，依据《施工人员工效智能感知功能及前后端接口规范-0825》实现。首版采用 Ultralytics YOLO Pose + ByteTrack，输出姿态骨架、视频内人员轨迹、总平图位置、工作区状态、区域人数和班组合格工效时长。

## 能力边界

通用姿态模型可以可靠支撑“人员在哪里、是否进入工作区、人数是否达标”的计算与证据回放，但不能直接证明人员正在执行某个具体工序。当前 `WORKING` 定义为经过进入滞回确认后处于工作区域内。具体动作识别需要后续使用现场标注数据训练时序分类模型。

## 本地环境

工程不包含任何虚拟环境、模型权重或运行缓存。开发人员可以使用已有 Python 环境直接安装依赖：

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```

如需隔离环境，可在工程外创建环境，或自行创建本地 `.venv`；`.gitignore` 已确保它不会进入版本库。

## 启动 API

在源码目录模式下可直接启动开发服务：

```powershell
$env:PYTHONPATH = "src"
python -m uvicorn work_efficiency_perception.main:app --reload
```

默认使用 `model/yolo11m-pose.pt`（文件存在时自动启用真实分析工作器）。也可以通过环境变量配置：

```powershell
$env:WORK_EFFICIENCY_MODEL_PATH = "model/yolo11m-pose.pt"
$env:WORK_EFFICIENCY_DEVICE = "0"
$env:WORK_EFFICIENCY_IMAGE_SIZE = "1280"
```

当前已提供：

- `GET /health`：健康检查；
- `POST /api/work-efficiency/v1/scenes`：上传实时分析使用的 PNG/JPEG 场景图；
- `GET /files/scenes/{sceneId}`：读取已上传的场景图；
- `POST /api/work-efficiency/v1/configurations`：提交实时相机标定和工作区配置；
- `POST /api/work-efficiency/v1/historical-analyses`：创建独立历史分析记录；
- `GET /api/work-efficiency/v1/historical-analyses/{analysisId}`：查询历史分析准备状态；
- `POST /api/work-efficiency/v1/historical-analyses/{analysisId}/video`：上传专属历史 MP4；
- `POST /api/work-efficiency/v1/historical-analyses/{analysisId}/scene`：上传专属场景图；
- `GET /api/work-efficiency/v1/historical-analyses/{analysisId}/video/preview`：获取代表帧；
- `PUT /api/work-efficiency/v1/historical-analyses/{analysisId}/configuration`：保存历史标定和区域配置；
- `POST /api/work-efficiency/v1/historical-analyses/{analysisId}/tasks`：启动或重试历史任务；
- `POST /api/work-efficiency/v1/cameras`：登记实时 RTSP/RTMP 地址；
- `POST /api/work-efficiency/v1/tasks`：创建实时相机任务；
- `GET /api/work-efficiency/v1/tasks/{taskId}`：查询任务状态；
- `PATCH /api/work-efficiency/v1/tasks/{taskId}`：使用 `{"action":"STOP"}` 幂等停止任务。
- `GET /api/work-efficiency/v1/tasks/{taskId}/progress`：查询处理进度；
- `GET /api/work-efficiency/v1/tasks/{taskId}/latest-result`：查询最新人员与区域结果；
- `GET /api/work-efficiency/v1/team-efficiency`：使用 `taskId` 或 `startTime/endTime` 查询区域工效汇总。
- `WS /ws/work-efficiency/v1/tasks/{taskId}`：实时推送 `FRAME_RESULT/ERROR`；启用鉴权时可使用 Bearer Header 或 `access_token` 查询参数。

现阶段场景、配置、任务和分析结果使用线程安全的进程内仓储，仅用于开发联调，服务重启后数据会清空。模型文件存在时任务会自动进入后台分析；模型未配置时任务保持 `CREATED`。后续将由数据库仓储替换。

实时分析和历史视频分析使用独立的资源与配置流程。第一版每个历史分析记录包含一个视频、一张专属场景图和一套历史配置；任务启动后输入和配置锁定。历史视频默认按 200 ms 间隔抽帧推理（最高约 5 FPS），并始终分析首帧和末帧；实时流继续采用只保留最新帧的自适应丢帧策略。进度接口中的 `processedFrames` 表示已解码到的原始帧位置，`analyzedFrames` 表示实际执行模型推理的帧数。

MySQL 8 初版表结构位于 `db/mysql_schema.sql`，正式数据库连接参数确认后再实现持久化仓储。

## 包结构

- `tracker.py`：Ultralytics Pose + ByteTrack 适配器；
- `video.py`：统一帧源、历史视频读取、实时流最新帧缓冲和断流重连；
- `analysis.py`：历史视频/实时相机共用的分析工作器和最新结果快照；
- `geometry.py`：人员锚点、单应性投影、多边形判定；
- `engine.py`：人员进出滞回、丢轨容忍、人数及工效时长累计；
- `models.py`：领域模型和枚举；
- `docs/`：可行性、算法链路和验收建议；
- `tests/`：不依赖模型权重或 GPU 的核心逻辑测试。

## 模型要求

使用 Ultralytics Pose 模型，例如官方 COCO 17 点姿态模型或基于现场数据微调后的模型。权重通过部署配置传入，禁止复制到源码包或提交到版本库。

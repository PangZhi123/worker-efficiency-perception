# 施工人员工效智能感知后端算法

独立 Python 工程，依据《施工人员工效智能感知功能及前后端接口规范-0825》实现。首版采用 Ultralytics YOLO Pose + ByteTrack，输出姿态骨架、视频内人员轨迹、总平图位置、工作区状态、区域人数和班组合格工效时长。

## 能力边界

通用姿态模型可以可靠支撑“人员在哪里、是否进入工作区、人数是否达标”的计算与证据回放，但不能直接证明人员正在执行某个具体工序。当前 `WORKING` 定义为经过进入滞回确认后处于工作区域内。具体动作识别需要后续使用现场标注数据训练时序分类模型。

### 班组连续计工规则（2026-10-31）

- 已确认进入工作区的人员，未观察到出界而检测丢失时，计工人数无限期保留，直到任务结束或观察到对应出界；不因丢轨超时扣减。
- 出界仍按 `exitDetectionDurationSec` 防抖确认，防抖期间继续计工；确认离开后进入固定 1200 秒（20 分钟）宽限期。若出界候选人员随后消失，按出界候选起点加防抖时长确认，避免无限保留已观察到的出界。
- 同一区域有人经 `entryDetectionDurationSec` 确认进入时，优先抵消最早到期的临时离开人数，不要求同一人员 ID。宽限期到期即扣减人数，前 20 分钟保留计工，不追溯扣时。恰好到期的进入按新进入处理。
- 新 ID 直接出现在区内、没有观察到外部进入过程时，可接替同区域当前未检测到的保留名额；同时可见人员不会互相合并。明确观察到从外部进入、且没有临时离开人数时，增加人数。
- `actualWorkingCount` 表示班组有效计工人数，包含区内丢失保留人数和临时离开宽限人数，可能大于当前 `persons` 数量。`persons` 只输出当前检测到的人员，不伪造缺失人员的骨架或位置；已确认在区外的人员仍显示 `OUTSIDE`。
- 人数达到 `expectedWorkerCount` 时累计班组合格工时。即使帧落在宽限期到期时间两侧，也按到期时间切分计时；断流/长帧间隔的工时仍受原有保护上限限制。
- 实时与历史分析共用此规则，前端 JSON、上传表单和响应字段结构均不变，20 分钟规则由后端默认启用。原有 `exitDetectionDurationSec` 仍表示出界防抖秒数，不要改传 1200。

该规则是班组人数保留策略，不是长期身份识别。遮挡住的进出、跟踪 ID 变化和重复检测仍可能造成误差：新 ID 直接出现在区内时，无法可靠区分“遮挡后重现”与“未拍到进入过程的新成员”。应保证出入口可见并进行现场验证。服务重启、任务重试或新任务不会继承之前的保留人数。

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

默认启动入口现已使用本地 JSON 文件 保存场景、配置、任务和分析结果，服务重启后可查询。模型文件存在时任务会自动进入后台分析；模型未配置时任务保持 `CREATED`。MySQL 仓储仍待接入。

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

## 本地前端测试工作台

从工程根目录启动（保持根目录为工作目录，以便加载 model 下的模型）：

```powershell
$env:PYTHONPATH = "src"
python -m uvicorn work_efficiency_perception.main:app --host 127.0.0.1 --port 8000
```

打开 http://127.0.0.1:8000/test-ui/ 。页面与 API 同源，无需 Node 或额外前端依赖。启用鉴权时在页面填写 Bearer Token。

推荐先测“历史分析”：新建记录，上传 `cam1_5min_clip.mp4`，上传对应场景图；在两张图片上按同一顺序点击至少四对不共线的对应点；切换到绘制区域模式，在右图点击至少三个顶点并完成区域；保存配置，再启动分析。结果查询页支持进度、最新人员数据、区域工效、自动刷新、停止任务和 JSON 下载。没有现场相机也能完成此流程。

相机管理仅提供登记入口；实时分析通过可编辑 JSON 提交相机配置并启动任务；场景管理支持上传和预览实时场景图。实时 JSON 初始坐标只是结构示例，需改为实际画面与场景对应坐标。页面不是完整的生产管理系统，人员结果以 JSON 展示，不包含历史轨迹回放。已有记录可以按 analysisId 读取素材和状态，画布编辑点不会从旧配置自动恢复。默认启动入口会保存记录到本地 JSON 文件。

## 本地结果持久化（JSON 文件）

默认启动入口 `main:app` 自动保存至 `data/results/`，可通过环境变量 `WORK_EFFICIENCY_RESULTS_DIRECTORY` 指定目录，无需数据库。仅支持单个服务进程使用同一目录。直接调用 `create_app()` 默认仍使用内存仓储供测试。

```text
data/results/
  resources.json                       # 资源索引与恢复数据，含场景图和配置
  tasks/<taskId>/
    task_configuration.json            # 任务、完整配置、转换矩阵、视频信息、场景图
    snapshot.json                      # 最新结果与进度，供重启后恢复查询
    efficiency_overall.json            # 整体累计工效及达标/不足区间
    persons/
      frame_000000000000.json           # 每个实际推理帧的人员及区域结果
    efficiency_30s/
      at_000000000030.json              # 30秒节点工效快照
      at_000000000060.json
```

`task_configuration.json` 明确保存 `homography` 3×3 单应转换矩阵，完整配置内也保留矩阵和标定误差。矩阵将视频像素坐标投影到场景归一化坐标，需进行齐次除法。场景图片以 Base64 存在 JSON 中，原始视频仍保存在 `data/videos`。

人员文件保存每个实际推理帧的检测框、骨架、人员 ID、轨迹增量、投影点和区域状态，同时包含对应区域统计；不重复存储原始画面。历史视频默认每200ms推理一次。`persistPoseAndTrajectory=false` 时不落盘人员明细，但区域统计仍保存；`poseSamplingMs` 暂不用于二次抽样。

工效30秒快照按照输入帧的时间戳触发，不按程序运行耗时触发：历史视频处理得再快，也按视频的30、60、90秒等节点保存。在节点之后首个实际推理帧保存，文件同时记录 `scheduledTimestampSec` 和实际 `timestampSec`。遇到跳帧跨过多个节点，只保存实际到达节点所在时段的真实结果，不插值或伪造缺失节点。末尾不足30秒的数据仍保存在整体结果中。整体结果和恢复快照随分析更新，正常完成时进度归一为100%；停止或失败保留已提交结果。

每个文件通过临时文件、flush/fsync和原子替换写入，避免留下半个JSON。不同文件间不是数据库事务，异常断电时最后一次帧文件与汇总可能不同步。任务状态以 `task_configuration.json` 的任务状态为准。重启后中断任务标记为FAILED，历史任务可以从头重试，不续接跟踪器和计工状态。

页面“结果查询 → 加载已保存任务”可找回任务。接口：

- `GET /api/work-efficiency/v1/tasks`：任务列表。
- `GET /api/work-efficiency/v1/tasks/{taskId}/frames?afterSeq=-1&limit=100`：按帧号分页读取，使用 `nextAfterSeq` 继续，limit最大1000。

旧 SQLite 文件会保留，但不自动迁移或继续写入。升级前内存中的任务也不会自动迁移，应先导出结果再重启，新建任务验证此保存方式。备份时正常停止服务后复制 `data/results` 和 `data/videos`；原始视频路径是绝对路径，目录迁移需调整。文件会持续增加，当前不自动清理。JSON里包含人员数据和场景图，应按业务数据保管。尚未提供可视化历史回放。

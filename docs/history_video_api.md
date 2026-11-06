# 历史视频上传与分析接口

更新日期：2026-11-06。历史视频流程以物料工程现有 `/videos → /scenes → /configurations → /tasks` 实现为基准。本文替代 0902 Word 文档第 14 节作为历史视频的新对接说明；旧 `historical-analyses` 接口暂时保留兼容。

接口前缀为 `/api/work-efficiency/v1`，工效配置、算法、结果字段与物料各自独立。历史视频无需调用方创建或传递 `analysisId`。Swagger 位于 `/docs`，OpenAPI 位于 `/openapi.json`。

## 标准调用顺序

1. 上传 MP4，取得 videoId。
2. 选择一个已有的 configurationId，可以是相机配置、其他视频配置或当前视频配置；也可上传新场景图并新建配置。
3. 确定带时区的录像开始时间，启动时传 recordingStartedAt，或使用视频已有的有效 capturedAt。
4. 用 videoId + configurationId 创建 VIDEO_FILE 任务，取得 taskId。
5. 用 taskId 查询状态、进度和工效结果，或请求停止。

历史任务按调用方选择的配置版本分析，不再要求配置绑定同一个 videoId，上传录像无需 cameraId。选择已有配置时，须根据实际画面确认标定、区域和视频视角适用；本次不自动缩放或重新标定。

## 上传视频

`POST /videos`，`multipart/form-data`，成功返回 HTTP 201。

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| file | 是 | 可解码的 MP4 文件，文件名扩展名 `.mp4` |
| capturedAt | 否 | ISO 8601 录像开始时间，例如 `2026-11-03T09:00:00+08:00`，建议携带时区 |

文件部分接受 `video/mp4`、`application/mp4`、`application/octet-stream`。默认最大 5 GiB，可通过 `WORK_EFFICIENCY_MAX_VIDEO_BYTES` 修改字节数。反向代理的请求体限制和超时须另外配置。

响应沿用工效统一信封：`requestId`、`success`、`code`、`message`、`data`。`data` 包含：

| 字段 | 说明 |
| --- | --- |
| videoId、cameraId | 视频标识；cameraId 为兼容字段，新上传视频为 null，旧记录保留原值 |
| originalFileName、contentType | 文件名和规范化的 `video/mp4` |
| sizeBytes、sha256 | 实际字节数和内容摘要 |
| width、height、fps、frameCount、durationSec | 视频元数据 |
| capturedAt、uploadedAt | 录像开始时间和登记时间 |
| status | `READY` |
| deduplicated | 是否命中已有内容 |
| previewUrl | 工效测试工作台使用的代表帧接口 |

按文件 SHA256 去重，同内容返回已有 videoId，不再按 cameraId 分别登记。去重命中同一 videoId 时保留原登记元数据。空文件、无效 MP4、超过大小限制或登记失败会清理本次临时文件。

补充查询接口：

- `GET /videos`：返回 `data.items`，列出新协议登记的视频。
- `GET /videos/{videoId}`：查询单个视频。
- `GET /videos/{videoId}/preview?timestampSec=0`：返回 JPEG；时间须处于 `[0, durationSec)`。后两项为工效工作台扩展。

## 新建视频专属场景图

`POST /scenes`，`multipart/form-data`，成功返回 HTTP 201。

复用已有配置时可以跳过此步骤。新建场景时，字段为 `file`、`sceneName`、`siteId`，另可传 `description` 和 `videoId`。创建视频专属场景时传已登记的 videoId；相机场景不传 videoId。仅接受有效 PNG/JPEG，最大 20 MiB。

返回 `sceneId`、`sceneName`、`imageUrl`、`width`、`height`、`sha256`、`videoId`。`imageUrl` 可用于预览。不同视频可以使用内容相同的图片，但应分别上传，形成各自绑定的 sceneId。

## 保存标定及区域配置

`POST /configurations`，JSON。历史模式成功返回 HTTP 200；原实时模式继续返回 HTTP 201。

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| cameraId | 是 | 配置所属相机标识；无需与视频匹配，不参与视频上传 |
| videoId | 否 | 新建视频专属配置时绑定视频；复用已有配置无需重新创建 |
| sceneId | 是 | 新建视频专属配置时须绑定同一个 videoId；相机配置使用未绑定视频的场景 |
| featurePairs | 是 | 至少四对不共线的 `{cameraPoint:{x,y}, scenePoint:{x,y}}` |
| workAreas | 是 | 工效区域列表，支持多个区域 |
| algorithm | 否 | 工效识别及出入防抖参数 |
| storage | 否 | 工效存储参数 |

`cameraPoint` 表示视频原始帧像素坐标，`scenePoint` 表示场景图原始像素坐标。区域 polygon 使用 `[0,1]` 归一化坐标，至少三个顶点。完整示例见 `examples/history-configuration.json`，使用时必须替换 ID 和实际标定点。

响应包括 `configurationId`、`cameraId`、`videoId`、`status: ACTIVE`、标定版本、区域版本、质量和工效默认值信息。`GET /configurations/{configurationId}` 可读取响应信息及 `request` 原始配置，此查询为工效工作台扩展。

配置创建后不覆盖修改；重新标定时创建新的 configurationId。历史任务可以选择已有相机或视频配置；实时任务仍只接受匹配 cameraId 的实时配置，不能使用视频配置。工效的多区域、班组人数、要求工时及 20 分钟计工规则保留，不复制物料的单区域限制和物料计算参数。

## 启动与停止

`POST /tasks`，JSON，历史任务成功返回 HTTP 200。创建即提交后台执行，不另设 START 接口。

```json
{
  "inputType": "VIDEO_FILE",
  "videoId": "video_xxx",
  "configurationId": "cfg_xxx",
  "recordingStartedAt": "2026-11-03T09:00:00+08:00",
  "options": {"persistPoseAndTrajectory": true}
}
```

历史任务不传 cameraId。响应包括 taskId、inputType、videoId、configurationId、recordingStartedAt、status 等；返回的 cameraId 来自所选配置，无需对应视频来源。新协议不依赖 analysisId，该兼容字段可为 null。`options` 保留工效字段，不接受物料的 persistTrajectory/pushIntervalMs。

历史任务使用独立的全局分析名额，最多同时接受一个历史任务，新 VIDEO_FILE 和旧 HISTORICAL_VIDEO 共用此名额。实时任务继续按 cameraId 互斥；实时与历史任务可同时运行，即使来自同一 cameraId。历史名额被占用或同一相机已有实时任务时，返回 HTTP 409、WE0203，并说明冲突原因。线程池容量仍决定实际执行并发。同一视频停止、失败或完成后均可创建新任务，产生新 taskId，从头分析，不是断点续跑。重复停止终态任务不会改变终态和更新时间。

`PATCH /tasks/{taskId}`，JSON：

```json
{"action":"STOP"}
```

成功返回 HTTP 200，包含 taskId、action 和当前状态。停止是异步请求，应继续查询到 STOPPED；请求到达时已完成的任务保持 COMPLETED。

新 VIDEO_FILE 任务的公开状态为 `QUEUED → RUNNING → COMPLETED/FAILED`，停止路径经过 `STOPPING → STOPPED`。内部 CREATED/STARTING 对新协议统一展示为 QUEUED；原实时及旧 HISTORICAL_VIDEO 的状态命名保持兼容。

## 查询进度与结果

- `GET /tasks/{taskId}`：任务状态。
- `GET /tasks/{taskId}/progress`：状态、processedFrames、analyzedFrames、progress 等。
- `GET /tasks/{taskId}/latest-result`：最新人员与区域工效；尚无结果时返回 404。
- `GET /team-efficiency?taskId={taskId}`：区域工效汇总。
- `GET /tasks/{taskId}/frames`：已保存的帧结果。
- `WS /ws/work-efficiency/v1/tasks/{taskId}`：工效帧结果推送。

VIDEO_FILE 启动时优先使用 recordingStartedAt，未传则使用视频 capturedAt；两者都没有有效带时区时间时返回 HTTP 422 / WE0101。显式传入的 recordingStartedAt 不合法或不带时区时直接拒绝，不回退到 capturedAt。不要把任务创建或执行时间当作录像时间。

帧 capturedAt、WebSocket、归档帧时间及工效区间按“录像开始时间 + 视频内时间”计算。按 startTime/endTime 查询汇总时，以录像开始时间筛选新历史任务，返回完整任务汇总，不按查询窗口重新切分工时。createdAt 仍表示任务实际创建时间。

每个新实时或历史任务都深拷贝配置为 configurationSnapshot；执行和任务配置归档使用此快照。后续创建配置、替换同 ID 的仓储记录或工作器修改自己的配置副本，不会改变已提交任务的快照。GET /tasks/{taskId} 可查看 recordingStartedAt 和 configurationSnapshot。

旧 historical-analyses 接口保持原有资源绑定流程，并支持可选 recordingStartedAt；为了兼容旧调用，旧接口及旧已保存记录未强制补齐录像时间。新对接统一使用上述严格的 VIDEO_FILE 规则。

## PowerShell 调用示例

在项目根目录执行，下列 ID 均需替换为前一步响应值。不要手动设置 multipart 的 Content-Type/boundary，交给 curl.exe 生成。

```powershell
$base = 'http://127.0.0.1:8000/api/work-efficiency/v1'
$video = (curl.exe -sS "$base/videos" -F 'file=@cam1_5min_clip.mp4;type=video/mp4' | ConvertFrom-Json).data
$scene = (curl.exe -sS "$base/scenes" -F 'file=@scene.png;type=image/png' -F 'sceneName=历史场景' -F 'siteId=site_01' -F "videoId=$($video.videoId)" | ConvertFrom-Json).data
```

复制 `docs/examples/history-configuration.json` 到自己的配置文件，将 videoId、sceneId 填为响应值，cameraId 填为 cam_01，并按实际图片重新填写标定点和区域，再提交：

```powershell
$config = (curl.exe -sS "$base/configurations" -H 'Content-Type: application/json' --data-binary '@history-configuration.json' | ConvertFrom-Json).data
$body = @{inputType='VIDEO_FILE'; videoId=$video.videoId; configurationId=$config.configurationId; recordingStartedAt='2026-11-03T09:00:00+08:00'} | ConvertTo-Json -Compress
$task = (Invoke-RestMethod -Uri "$base/tasks" -Method Post -ContentType 'application/json' -Body $body).data
curl.exe -sS "$base/tasks/$($task.taskId)/progress"
Invoke-RestMethod -Uri "$base/tasks/$($task.taskId)" -Method Patch -ContentType 'application/json' -Body '{"action":"STOP"}'
```

也可以直接把已有相机配置的 configurationId 填入任务请求，跳过上传新场景与新建配置；录像开始时间应改为该段录像的真实时间。

如果服务配置了 Bearer Token，所有请求须添加 `Authorization: Bearer <token>`。工作台可直接填写 Token。

## 错误与部署注意事项

| HTTP | 典型情况 |
| --- | --- |
| 401 | Token 缺失或不正确 |
| 404 | 视频、任务或配置不存在；尚无结果 |
| 409 | 视频与场景/配置不匹配；历史名额被占用；同相机已有实时任务 |
| 413 | 视频超过上传限制 |
| 415 | 不支持的扩展名或视频 MIME 类型 |
| 422 | 缺字段、空视频、无法解码、无效时间或参数 |
| 500 | 视频存储失败 |

错误码使用工效 WE 前缀，物料 MP 错误码不复制。旧实时接口的部分校验仍返回原来的 400，以保留兼容性。现有 historical-analyses 的 LOCKED/重试规则也保持原语义，不用于新对接。

默认入口使用本地 JSON 保存资源及结果，单服务进程部署；资源占用检查不支持多个独立进程共享 JSON。服务重启时未完成任务标为 FAILED，可新建任务重跑。需配置真实模型后才能执行人员识别；只创建无工作器的 API 应用时，任务会保持排队状态。

部署时必须更新 Python 服务并重启后端，仅修改 Java 的 URL 不会启用这些接口。Java 除调用 videosUrl() 外，还需传入所选 configurationId、VIDEO_FILE 任务参数及有效的录像开始时间。

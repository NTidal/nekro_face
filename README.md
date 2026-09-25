# 人脸识别 (nekro_face)

NekroAgent 插件：让 AI 在聊天中**自己会认人**（动漫角色 / 真人），并把「认不准」的图
自动收集到待审核队列，供人工复核补图。

- **AI 侧**：提供两个 AGENT 工具，识别结论进入 AI 上下文，由 AI 按当前角色人格自然表达；不注册任何聊天指令
- **人工侧**：Aurora 主题 WebUI（总览 / 待审核队列 / 角色库），深色优先、支持日间模式
- **识别**：纯本地离线推理，动漫 YOLOv8 + CCIP（768 维）、真人 insightface buffalo_l（512 维），双库自动分流

版本：**1.0.0**　作者：**NTidal**

---

## 功能

- 🤖 **AI 自主认人**：`face_identify` 工具直接认图，无需先用 `view_image`；判断权交给模型，用户不必说固定口令
- 📋 **待审核闭环**：置信度不足的图自动入队（含检测框、候选与相似度），WebUI 一键注册补图
- 👥 **逐脸注册**：合照可逐张脸注册到不同角色，全部注册完自动出队
- 🗂️ **角色库管理**：按作品分组/筛选/搜索/排序，改名并入、删除误注册特征、代表图预览
- 🎚️ **阈值可调**：WebUI 实时调整动漫/真人阈值，写入数据目录的 `config.json`
- 🔒 **默认安全**：常驻服务只监听 `127.0.0.1`；上传校验扩展名与大小；写配置原子替换并留 `.bak`

---

## 架构

```
AI 聊天 ──工具调用──> __init__.py（插件）──HTTP 127.0.0.1:8766──> tools/face_server.py（常驻）
                        配置 / 路由 / 代理                                    模型推理 + 库单源
                         │                                                        │
                         └────── WebUI（审核 / 注册 / 角色库）──────> {数据目录}/
```

| 层 | 位置 | 说明 |
|---|---|---|
| 插件 | 本包 `__init__.py` + `face_webui.html` | 配置、LLM 工具、WebUI 路由、反向代理；**不直接读写库文件** |
| 引擎 | 本包 `tools/` | `face_server.py` 常驻服务、`face_engine.py` 推理与库管理、`face_identify.py` / `face_register.py` 命令行兜底、`face_review_queue.py` 队列写入 |
| 数据 | 见「数据目录」 | 特征库 JSON、代表图、待审核队列、模型文件 |
| 运行环境 | 独立 venv（默认 `/opt/face_venv`） | 引擎依赖与 NA 主环境隔离 |

库访问为**单源设计**：插件进程不解析数十 MB 的库 JSON，一切经常驻服务供数（服务内持有解析后的库，
插件侧再加 5 秒 TTL 缓存），既快又避免与引擎并发写库。

---

## 依赖

**1. 引擎解释器（独立 venv）**

```
onnxruntime, insightface, opencv-python, numpy
```

默认路径 `/opt/face_venv/bin/python`；可用配置项 `FACE_VENV_PYTHON` 指向任何已装好上述依赖的解释器。

**2. 模型文件**（放在数据目录下，不随插件分发）

| 路径 | 内容 |
|---|---|
| `{数据目录}/anime_models/anime_face_detect.onnx` | 动漫脸检测（YOLOv8） |
| `{数据目录}/anime_models/anime_real_cls.onnx` | 动漫 / 真人分流 |
| `{数据目录}/anime_models/ccip_feat.onnx` | CCIP 动漫特征（768 维） |
| `{数据目录}/models/buffalo_l/det_10g.onnx`、`w600k_r50.onnx` | insightface 真人检测 + 识别 |

**3. 插件侧依赖**：`httpx`、`Pillow`（队列缩略图），NA 主环境自带。

---

## 安装

1. 上传本包（`__init__.py`、`face_webui.html`、`tools/`）或整体放入 NA 插件目录；
2. **完全重启 NekroAgent**；
3. 按需设置配置项（默认值即可跑通，只要解释器与模型就位）；
4. 打开 WebUI：
   - `/plugins/<插件key>/` —— 总览（指标 / 名单 / 注册 / 测试识别 / 阈值）
   - `/plugins/<插件key>/review` —— 待审核队列
   - `/plugins/<插件key>/library` —— 角色库
5. 首次启动日志会打印解析后的实际路径：

```
[face] 数据目录 /var/lib/docker/nekro_agent_data/face | 引擎 <插件目录>/tools | 解释器 /opt/face_venv/bin/python | 服务 http://127.0.0.1:8766
[face] 识别服务已就绪 | 待审核 0 条 | review_enabled=True
```

---

## 配置项

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `THRESHOLD` | `0.5` | 真人识别阈值（认错人调高，认不出调低） |
| `ANIME_THRESHOLD` | `0.78` | 动漫识别阈值（CCIP 推荐 0.75~0.85） |
| `REVIEW_ENABLED` | `true` | 启用待审核收集 |
| `REVIEW_MAX_ITEMS` | `500` | 队列上限，超出丢弃最旧的（图片一并删除） |
| `REVIEW_DEDUP_SECONDS` | `3600` | 同图重复失败的去重窗口，避免刷屏 |
| `FACE_DATA_DIR` | 空 | 数据目录。留空自动：`{NA数据目录}/face` 存在则复用，否则用插件数据目录下的 `face/` |
| `FACE_TOOLS_DIR` | 空 | 引擎目录。留空用本插件包内 `tools/` |
| `FACE_VENV_PYTHON` | `/opt/face_venv/bin/python` | 引擎解释器（需自备依赖） |
| `FACE_UPLOAD_ROOT` | 空 | NA 上传目录（引擎按文件名兜底找图）。留空自动探测 `{NA数据目录}/uploads` |
| `FACE_SERVER_URL` | `http://127.0.0.1:8766` | 常驻服务地址，建议保持 127.0.0.1 |
| `FACE_SERVER_AUTOSTART` | `true` | 服务不可用时自动拉起；关闭则回退子进程（每次重载模型，较慢） |

阈值优先级：`{数据目录}/config.json`（WebUI 阈值表单写入）> 插件配置。

---

## 数据目录

```
{数据目录}/
├── anime_db.json          动漫特征库（含向量）
├── face_db.json           真人特征库
├── config.json            WebUI 保存的阈值（另有 .bak 备份）
├── anime_models/          动漫模型（3 个 onnx）
├── models/buffalo_l/      insightface 模型
├── thumbs/{kind}/{角色}/cover.jpg   角色代表图（每条目仅一张，随注册覆盖为最新）
├── review/
│   ├── queue.json         待审核队列
│   └── images/            待审核原图（缩略图缓存 t_{id}.jpg 同目录）
├── web_uploads/           WebUI 上传的临时图（用完即删）
└── server.log             常驻服务日志
```

特征库只存向量；`thumbs/` 下每条目一张代表图用于 WebUI 标识，
代表图与特征索引解耦——改名、并入、删除特征都不需要重排图片。

---

## 使用方式

**AI 侧（无需配置触发词）**

| 工具 | 用途 |
|---|---|
| `face_identify` | 识别图片中的人脸是谁；可传路径/文件名，留空则取当前会话最近收到的图 |
| `face_registered_list` | 查看人脸库里已登记了哪些人 |

两个都是 `AGENT` 类型：结果回灌上下文后 AI 会继续说话，因此识别结论会由 AI 用当前人格表达，
而不是插件直接回一句机械回执。用户正常发图、正常聊天即可。

**人工侧（WebUI）**

- **总览**：待审核数、动漫/真人条目与特征总量；注册新角色（可新建作品分类）、测试识别、调整阈值
- **待审核队列**：卡片显示原图（走缩略图缓存）、检测到的每张脸、候选与相似度；
  可一键按候选注册、手填角色名注册、逐脸注册，或「忽略 / 完成」出队
- **角色库**：按作品分组/搜索/筛选/排序；点开查看代表图与特征列表，可删除误注册的单张特征、
  改名（同名自动并入）、删除整个条目

---

## 常见问题

**识别服务起不来**
看 `{数据目录}/server.log`。常见原因：`FACE_VENV_PYTHON` 指向错误、venv 缺依赖、模型文件缺失。

**识别总是失败 / 报错找不到图**
聊天图片的落盘根目录由 `FACE_UPLOAD_ROOT` 决定，留空自动探测；自定义数据目录布局时建议显式填写。

**队列占用磁盘**
队列上限 `REVIEW_MAX_ITEMS`（默认 500），原图中位约 2.5 MB，满队列可达数 GB。
WebUI 卡片走 480 px 缩略图缓存，但原图仍占用磁盘，处理完请及时「忽略 / 完成」。

**认错人 / 认不出**
库内只有 1~2 个特征的角色对视角与表情极敏感，容易认不准。
建议用「待审核队列」把用户实际发过的图注册进对应角色——这是最有效的提升手段。
阈值可先在总览页调低观察，再逐步收紧。

**想让 AI 更主动认人**
在系统提示或人设里说明「看到图片想确认身份时可以直接调用认人工具」即可；
工具本身已声明判断权在模型，不依赖固定口令。

---

## 版权

MIT

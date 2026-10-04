# 人脸识别 (nekro_face)

> **v1.2.0 起：真人识别已移除。**
> 本插件现在**只识别动漫角色**（CCIP 特征，768 维）。原因与收益：
> 真人库从未注册过任何条目，该链路却常驻约 400 MB 内存；移除后
> 实测常驻内存 **1.17 GB → 574 MB（−49%）**、线程数 **167 → 93（−44%）**，
> 识别行为对动漫图**完全不变**（30 张实图回归：原 30 认出 / 新 30 认出，0 差异）。


NekroAgent 插件：让 AI 在聊天中**自己会认人**（动漫角色），并把「认不准」的图
自动收集到待审核队列，供人工复核补图。

- **AI 侧**：提供两个 AGENT 工具，识别结论进入 AI 上下文，由 AI 按当前角色人格自然表达；不注册任何聊天指令
- **人工侧**：Aurora 主题 WebUI（总览 / 待审核队列 / 角色库），深色优先、支持日间模式
- **识别**：纯本地离线推理，动漫 YOLOv8 检测 + CCIP 特征（768 维）

版本：**1.2.4**　作者：**NTidal**

---

## 功能

- 🤖 **AI 自主认人**：`face_identify` 工具直接认图，无需先用 `view_image`；判断权交给模型，用户不必说固定口令
- 📋 **待审核闭环**：置信度不足的图自动入队（含检测框、候选与相似度），WebUI 一键注册补图
- 👥 **逐脸注册**：合照可逐张脸注册到不同角色，全部注册完自动出队
- 🗂️ **角色库管理**：按作品分组/筛选/搜索/排序，改名并入、删除误注册特征、代表图预览
- 🎚️ **阈值可调**：WebUI 实时调整动漫识别阈值，写入数据目录的 `config.json`
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
onnxruntime, opencv-python, numpy
```

默认路径 `/opt/face_venv/bin/python`；可用配置项 `FACE_VENV_PYTHON` 指向任何已装好上述依赖的解释器。

**2. 模型文件**（放在数据目录下，不随插件分发）

| 路径 | 内容 |
|---|---|
| `{数据目录}/anime_models/anime_face_detect.onnx` | 动漫脸检测（YOLOv8） |
| `{数据目录}/anime_models/anime_real_cls.onnx` | 图片类型判定（是否动漫）|
| `{数据目录}/anime_models/ccip_feat.onnx` | CCIP 动漫特征（768 维） |

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
| `ANIME_THRESHOLD` | `0.78` | 动漫识别阈值（CCIP 推荐 0.75~0.85） |
| `REVIEW_ENABLED` | `true` | 启用待审核收集 |
| `REVIEW_MAX_ITEMS` | `500` | 队列上限，超出丢弃最旧的（图片一并删除） |
| `REVIEW_DEDUP_SECONDS` | `3600` | 同图重复失败的去重窗口，避免刷屏 |
| `FACE_DATA_DIR` | 空 | 数据目录。留空自动：`{NA数据目录}/face` 存在则复用，否则用插件数据目录下的 `face/` |
| `FACE_TOOLS_DIR` | 空 | 引擎目录。留空用本插件包内 `tools/` |
| `FACE_VENV_PYTHON` | `/opt/face_venv/bin/python` | 引擎解释器（需自备依赖） |
| `FACE_UPLOAD_ROOT` | 空 | NA 上传目录（引擎按文件名兜底找图）。留空自动探测 `{NA数据目录}/uploads` |
| `FACE_SERVER_URL` | `http://127.0.0.1:8766` | 常驻服务地址。单实例保持 127.0.0.1；多实例共享见[共享服务模式](#共享服务模式v110) |
| `FACE_SERVER_AUTOSTART` | `true` | 服务不可用时自动拉起；关闭则回退子进程（每次重载模型，较慢） |

阈值优先级：`{数据目录}/config.json`（WebUI 阈值表单写入）> 插件配置。

---

## 共享服务模式（v1.2.0）

默认是**每个 NA 实例各自拉起一个常驻服务**，各自加载模型、各自一份角色库。
跑多个 NA 实例时，可以把识别服务拆成**一个独立容器**，多个实例共用：

- 模型只加载一份，省内存、省启动时间；
- 角色库只有一份，任一侧注册 / 改名 / 并入，所有实例立刻生效；
- 各实例不必再装引擎依赖（依赖只在服务容器里）。

### 服务端

`tools/face_server.py` 支持 `FACE_SERVER_HOST`（v1.1.0 新增，默认 `127.0.0.1` ＝ 仅本机）：

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `FACE_SERVER_HOST` | `127.0.0.1` | 独立容器对外提供服务时设 `0.0.0.0` |
| `FACE_SERVER_PORT` | `8766` | 监听端口 |
| `FACE_DATA_DIR` | `{NA数据目录}/face` | 特征库 / 模型 / 缩略图目录 |
| `FACE_UPLOAD_ROOT` | `{NA数据目录}/uploads` | NA 上传目录（引擎按文件名兜底找图） |

示例（把插件自带的 `tools/` 与一份数据目录挂进去）：

```bash
docker run -d --name nekro_face_service --restart unless-stopped \
  -e FACE_SERVER_HOST=0.0.0.0 -e FACE_SERVER_PORT=8766 \
  -e FACE_DATA_DIR=/face_data -e FACE_UPLOAD_ROOT=/uploads \
  -v /opt/face_venv:/opt/face_venv \
  -v {NA数据目录}/plugins/packages/nekro_face/tools:/tools:ro \
  -v {NA数据目录}/face:/face_data \
  -v {NA数据目录}/uploads:/uploads:ro \
  kromiose/nekro-agent:latest \
  /opt/face_venv/bin/python /tools/face_server.py
```

`tools/` 以**只读**挂入即可（服务只读代码，写操作都落在数据目录）。
已经部署过旧版服务容器、只想补这个开关的，可以直接用包内补丁：

```bash
patch -p1 < face_server_host.patch
```

连通性检查：`curl http://<服务地址>:8766/health` → `{"ok": true, "uptime": …, "counts": {…}}`。

### 客户端（各 NA 实例）

| 配置项 | 共享模式下的取值 |
| --- | --- |
| `FACE_SERVER_URL` | 服务容器地址，如 `http://nekro_face_service:8766`（同 Docker 网络）或 `http://<宿主机IP>:8766` |
| `FACE_SERVER_AUTOSTART` | 建议 **`false`** —— 不在本实例里再拉一个服务 |
| `FACE_VENV_PYTHON` / `FACE_TOOLS_DIR` | 共享模式下用不到（模型在服务容器里跑） |

> **各实例必须指向同一份 `FACE_DATA_DIR`**（通常就是把同一个宿主机目录挂给每个实例）。
> 若各自留在自己的数据目录里，会出现"这台认得出、那台认不出"的分裂现象。

---

## 数据目录

```
{数据目录}/
├── anime_db.json          动漫特征库（含向量）
├── config.json            WebUI 保存的阈值（另有 .bak 备份）
├── anime_models/          动漫模型（3 个 onnx）
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

- **总览**：待审核数、动漫角色与特征总量；注册新角色（可新建作品分类）、测试识别、调整阈值
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

---

## ⚠️ 多实例部署注意

插件默认配置是**单实例**用的：

```yaml
FACE_SERVER_URL: http://127.0.0.1:8766   # 指向自己容器内
FACE_SERVER_AUTOSTART: true              # 服务不在就自己拉起子进程
```

**如果多个 NA 实例共享同一份 face 数据目录**（例如都 bind mount 了同一个
`{数据目录}/face`），而每个实例都保持默认的 `AUTOSTART: true`，就会出现
**多个 face_server 进程同时读写同一个 `anime_db.json`** —— 有写坏特征库的风险。

多实例必须改成**共享服务模式**：

| 配置项 | 多实例取值 |
| --- | --- |
| `FACE_SERVER_URL` | 指向服务容器，如 `http://nekro_face_service:8766` |
| `FACE_SERVER_AUTOSTART` | **`false`**（不在本地再起一个）|

> 另外注意：插件**不会自动创建容器**。`subprocess.Popen` 只在本容器内拉起
> `face_server.py` 子进程。要让多个实例共享，需要自己用
> `docker-compose.face-service.yml` 之类的方式单起一个服务容器。
>
> 全新安装（单实例）时无需任何 Docker 操作：插件会按
> `配置 → {NA数据目录}/face → 插件数据目录/face` 的顺序解析数据目录并自动创建。

### ⚠️ 每个实例的绝对路径都要挂进服务容器

引擎解析图片时，**「绝对路径存在就直接用」**。而各实例看到的路径前缀不同：

| 实例 | 传给服务的图片路径 |
| --- | --- |
| instance1 | `/var/lib/docker/nekro_agent_data/...` |
| instance2 | `/var/lib/docker/nekro_agent_data2/...` |
| instance3 | `/var/lib/docker/nekro_agent_data3/...` |

所以**同一份宿主目录必须在服务容器里按「各实例看到的绝对路径」分别挂载**：

```yaml
volumes:
  - /var/lib/docker/nekro_agent_data/face:/face_data
  - /var/lib/docker/nekro_agent_data/face:/var/lib/docker/nekro_agent_data/face
  - /var/lib/docker/nekro_agent_data/face:/var/lib/docker/nekro_agent_data2/plugin_data/NTidal.nekro_face/face
  - /var/lib/docker/nekro_agent_data/face:/var/lib/docker/nekro_agent_data3/face
```

**新增实例时最容易漏这一步**（实测踩过：第 3 个实例加进来时没补挂载，
它的「测试识别」一直返回**完全无关的角色**）。上表的别名行必须与
`FACE_SERVER_URL` 指向的服务容器一一对应。

> 插件侧 `FACE_DATA_DIR` 也要指向该实例能看到的那份共享目录，
> 否则上传图会落到服务读不到的地方（同上症状）。

同理，`FACE_UPLOAD_ROOT`（各实例的 `{数据目录}/uploads`）也要按各自前缀挂进去，
且**务必确认挂的是各实例真正在写的那份宿主目录** —— 挂错成另一个目录时，
引擎按文件名兜底会命中旧图或找不到图。

### 引擎行为：解析不到就报错，不会拿别的图冒充

`resolve_image_path()` 的兜底分两种情况：

- **调用方没传图片路径**（留空 = 「识别当前会话最近一张」）→ 取最近一张，这是设计功能
- **调用方传了具体路径但解析不到** → 返回 `None`，上层报「找不到图片」

第二种情况**绝不会**回退到「最近一张图」。这条约束是有意为之：
早期版本会静默回退，导致路径没挂载时拿 uploads 里最新的一张**无关图片**
去识别，还返回 `confidence=certain` —— 静默给出错误答案比直接报错危险得多。

排查识别结果异常时，先确认服务容器能否看到调用方传来的那个绝对路径。

---

## 变更记录

### v1.2.4

- **修复同名角色注册串条目**：手动输入的角色名如果同时存在于两个作品
  （如「椿」既在 `鸣潮·椿` 也在 `蔚蓝档案·椿`），旧逻辑会**静默**把特征写进
  特征数最多的那个条目，不提示、不报错。实测已因此污染过
  `原神·妮露` / `明日方舟·爱丽丝` / `明日方舟·杜林` 三个条目
  （内容 100% 属于另一作品，相似度 1.000）。

  现在改为**拒绝并要求指定作品**：

  - `/api/register`、`/api/item/{id}/register` 命中同名歧义时返回
    **HTTP 409** + 候选列表（`{"detail": {"ambiguous": true, "candidates": [...]}}`）
  - WebUI 提示「在库里有 N 个同名条目：鸣潮（20 张）、蔚蓝档案（7 张）」，
    引导用户改用「作品·角色」全名或在「作品 / 分类」下拉里选定
  - 传入全名（`鸣潮·椿`）时无歧义，照常通过

- **作品名改为动态推导**：原先插件侧 `_display_name()` 用硬编码的
  `WORK_PREFIXES` 白名单，新增作品（如「碧蓝航线」）会被当成角色名，
  新建出无前缀的裸条目，再次形成同名重复。现在从库概览动态收集，
  并新增 `GET /api/works` 返回作品列表与**重名角色清单**。
  （引擎侧 `work_prefixes()` 早已支持 `work_prefixes.json` 动态扩展，
  本次是把插件侧对齐。）

### v1.2.3

- **修复 WebUI「测试识别」点了没反应**：`face_webui.html` 里残留两处已删除的
  真人阈值引用（`$('#identifyRealThreshold').value` 与未定义变量 `r`），
  提交时抛 JS 异常，请求根本没发出。v1.2.0 移除真人链路时只删了表单元素、
  漏删了这段脚本。
- **加固 `resolve_image_path()`**：调用方明确传了图片路径却解析不到时，
  返回 `None` 报「找不到图片」，不再静默回退到「取最近一张图」。
  原先的行为会在路径未挂载时拿一张**无关图片**去识别并给出
  `confidence=certain`，把配置问题伪装成识别结果。留空路径
  （= 识别当前会话最近一张）的语义保持不变。

### v1.2.2

- 修复 `/identify` 全线 HTTP 500：v1.2.0 改了 `identify_ex` / `identify`
  的签名，`face_server.py` 与 `face_identify.py` 两个调用点没跟着改。

### v1.2.1

- 修复启动崩溃：清理真人链路时误删 `_LIB_CACHE` / `_Q_CACHE` 定义，
  插件初始化在 `_load_queue()` 抛 `NameError`，整个插件加载失败。

### v1.2.0

- 移除真人识别链路（真人库为空、从未注册），常驻内存约 −49%。


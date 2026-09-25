"""
# 人脸识别 (Face Recognition) —— 含待审核补图闭环

让 AI 在聊天中**自己会认人**，并把「认不准」的图收集起来供人工复核补图。

## 一、AI 工具（沙盒方法）

- `face_identify`（AGENT）：传入图片路径/文件名 → 返回人物姓名
- `face_registered_list`（AGENT）：查看当前库里有谁

**不注册任何聊天指令**。用户正常发图、正常聊天，由 AI 自行判断是否需要认人，
识别结果作为工具返回值进入 AI 上下文，再由 AI 按当前角色人格自然表达。

## 二、两类图，自动分流

- **动漫/二次元头像**：YOLOv8 动漫脸检测 + CCIP 特征（768 维）
- **真人照片**：insightface buffalo_l（SCRFD + ArcFace，512 维）

全部纯本地离线推理。

## 三、待审核补图闭环（为什么需要）

实测：留出法（同形态变体）准确率 99.6%，但**真实流量只有 35%**。根因不是模型弱：

- 官方立绘在库里（类内相似度 0.95+），用户却发**私设图 / 同人图 / 截图**；
- 这些图分数落在 0.58~0.80，够不到笃定线；
- **只有 1-2 个特征的角色**（如流萤）对视角/表情极敏感，实测 3/3 全错。

按方法论，「**把用户认不出的图注册进对应角色**」是已验证有效的修复
（实测某角色 0.668 → 1.000）。故把未确信的图自动收集到队列，供人工一键注册。

```
识别 → confidence != certain（mid/low/unknown/noface）
     → 写入 {DATA}/face/review/queue.json + images/
     → 本插件 WebUI 呈现 → 你确认角色 → 一键注册
```

## 四、WebUI

三个 URL 由**同一 SPA**（`face_webui.html`，aurora 主题）承载，按路径切换视图：

- `/plugins/<插件key>/`        —— 总览（指标 / 名单 / 注册 / 测试识别 / 阈值）
- `/plugins/<插件key>/review`  —— 待审核队列（审核补图）
- `/plugins/<插件key>/library` —— 角色库（代表图 / 特征管理）

角色库说明：特征库本身只存特征向量；注册时另存该角色的**代表图**
（`{DATA}/face/thumbs/{kind}/{角色}/cover.jpg`，每条目仅一张，随注册覆盖为最新），
供 WebUI 标识与预览。代表图与特征索引解耦，改名 / 并入 / 删除特征均无需重排图片。

## 五、路径与引擎

识别引擎（`tools/`）随插件包分发；数据目录、引擎目录、解释器、服务地址均可配置：

| 配置项 | 留空时的自动探测 |
|---|---|
| `FACE_DATA_DIR` | `{NA数据目录}/face`（存在时复用）→ 插件数据目录/`face` |
| `FACE_TOOLS_DIR` | 插件包内 `tools/` |
| `FACE_VENV_PYTHON` | 默认 `/opt/face_venv/bin/python`（需自备依赖环境） |
| `FACE_UPLOAD_ROOT` | `{NA数据目录}/uploads` |
| `FACE_SERVER_URL` | `http://127.0.0.1:8766` |

引擎子进程通过 `FACE_DATA_DIR` / `FACE_UPLOAD_ROOT` / `FACE_SERVER_PORT` 环境变量获知路径，
因此引擎脚本也能脱离插件单独运行（不带变量时按上述默认约定路径解析）。

## 六、设计说明

插件与识别引擎通过 `{DATA}/face/review/` 目录解耦：待审核队列的文件格式由
`face_review_queue` 定义，引擎负责写入、插件负责读取与呈现。两侧可分别部署与替换，
只需保持队列格式一致。
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import Field

from nekro_agent.api import i18n, schemas
from nekro_agent.api.plugin import ConfigBase, ExtraField, NekroPlugin, SandboxMethodType


plugin = NekroPlugin(
    name="人脸识别",
    module_name="nekro_face",
    description="让 AI 认出图片中的人脸是谁（动漫/真人），并把认不准的图收集起来供人工复核补图",
    version="1.0.0",
    author="NTidal",
    url="https://github.com/NTidal/nekro_face",
    i18n_name=i18n.i18n_text(zh_CN="人脸识别", en_US="Face Recognition"),
    i18n_description=i18n.i18n_text(
        zh_CN="让 AI 认出图片中的人脸是谁（动漫/真人），并把认不准的图收集起来供人工复核补图",
        en_US="Let the AI recognize faces (anime/real) and queue low-confidence results for human review",
    ),
    allow_sleep=False,
)


@plugin.mount_config()
class FaceRecognitionConfig(ConfigBase):
    """人脸识别 + 待审核队列配置。"""

    # ---- 识别阈值 ----
    THRESHOLD: float = Field(
        default=0.5,
        title="真人识别阈值",
        description="真人照片相似度阈值（0~1，越高越严格；认错人调高，认不出调低）",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="真人识别阈值", en_US="Real-Face Threshold"),
            i18n_description=i18n.i18n_text(
                zh_CN="真人照片相似度阈值（0~1，越高越严格；认错人调高，认不出调低）",
                en_US="Similarity threshold for real photos (higher = stricter)",
            ),
        ).model_dump(),
    )
    ANIME_THRESHOLD: float = Field(
        default=0.78,
        title="动漫识别阈值",
        description="动漫头像相似度阈值（0~1，越高越严格；CCIP 模型推荐 0.75~0.85）",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="动漫识别阈值", en_US="Anime Threshold"),
            i18n_description=i18n.i18n_text(
                zh_CN="动漫头像相似度阈值（0~1，越高越严格；CCIP 模型推荐 0.75~0.85）",
                en_US="Similarity threshold for anime avatars (CCIP recommends 0.75~0.85)",
            ),
        ).model_dump(),
    )

    # ---- 待审核队列 ----
    REVIEW_ENABLED: bool = Field(
        default=True,
        title="启用待审核收集",
        description="关闭后不再收集新的待审核项（已有队列仍可在审核页处理）",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="启用待审核收集", en_US="Enable Review Collection"),
            i18n_description=i18n.i18n_text(
                zh_CN="关闭后不再收集新的待审核项（已有队列仍可在审核页处理）",
                en_US="When off, no new items are collected (existing queue stays manageable)",
            ),
        ).model_dump(),
    )
    REVIEW_MAX_ITEMS: int = Field(
        default=500,
        title="队列上限",
        description="队列最多保留多少条；超出时自动丢弃最旧的（图片一并删除）",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="队列上限", en_US="Max Queue Size"),
            i18n_description=i18n.i18n_text(
                zh_CN="队列最多保留多少条；超出时自动丢弃最旧的（图片一并删除）",
                en_US="Maximum items kept; oldest are dropped along with their images",
            ),
        ).model_dump(),
    )
    # ---- 路径与引擎（留空 = 自动探测）----
    FACE_DATA_DIR: str = Field(
        default="",
        title="人脸数据目录",
        description="库文件/队列/代表图所在目录。留空自动：优先复用 {NA数据目录}/face（存在时），否则用插件数据目录下的 face/",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="人脸数据目录", en_US="Face Data Directory"),
            i18n_description=i18n.i18n_text(
                zh_CN="留空自动探测：优先复用 {NA数据目录}/face，否则用插件数据目录下的 face/",
                en_US="Empty = auto-detect: reuse {NA data}/face if present, else <plugin data>/face",
            ),
        ).model_dump(),
    )
    FACE_TOOLS_DIR: str = Field(
        default="",
        title="识别引擎目录",
        description="识别引擎脚本所在目录。留空用本插件包内的 tools/",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="识别引擎目录", en_US="Engine Tools Directory"),
            i18n_description=i18n.i18n_text(
                zh_CN="留空用插件包内 tools/",
                en_US="Empty = bundled tools/ inside the plugin package",
            ),
        ).model_dump(),
    )
    FACE_VENV_PYTHON: str = Field(
        default="/opt/face_venv/bin/python",
        title="识别引擎解释器",
        description="运行识别引擎的 Python（需已安装 onnxruntime/insightface/opencv 等依赖），指向独立 venv 的解释器",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="识别引擎解释器", en_US="Engine Python"),
            i18n_description=i18n.i18n_text(
                zh_CN="运行识别引擎的 Python 解释器（独立 venv）",
                en_US="Python interpreter used to run the recognition engine (separate venv)",
            ),
        ).model_dump(),
    )
    FACE_UPLOAD_ROOT: str = Field(
        default="",
        title="NA 上传目录",
        description="聊天图片的落盘根目录，引擎按文件名兜底找图时使用。留空自动探测",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="NA 上传目录", en_US="NA Upload Root"),
            i18n_description=i18n.i18n_text(
                zh_CN="聊天图片落盘根目录，留空自动探测",
                en_US="Root directory of chat uploads; empty = auto-detect",
            ),
        ).model_dump(),
    )
    FACE_SERVER_URL: str = Field(
        default="http://127.0.0.1:8766",
        title="识别服务地址",
        description="常驻识别服务地址（建议保持 127.0.0.1，仅本机访问）",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="识别服务地址", en_US="Face Server URL"),
            i18n_description=i18n.i18n_text(
                zh_CN="常驻识别服务地址，建议保持 127.0.0.1",
                en_US="Resident recognition service URL; keep 127.0.0.1 recommended",
            ),
        ).model_dump(),
    )
    FACE_SERVER_AUTOSTART: bool = Field(
        default=True,
        title="自动拉起识别服务",
        description="服务不可用时由插件自动拉起；关闭后需自行启动（不可用时回退子进程模式，每次都要重载模型）",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="自动拉起识别服务", en_US="Auto-start Face Server"),
            i18n_description=i18n.i18n_text(
                zh_CN="服务不可用时自动拉起；关闭则回退子进程模式",
                en_US="Auto-start the server when down; off = fall back to per-call subprocess",
            ),
        ).model_dump(),
    )
    REVIEW_DEDUP_SECONDS: int = Field(
        default=3600,
        title="相同图片去重窗口（秒）",
        description="同一张图在该时间内重复识别失败时不再重复入队，避免刷屏",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="去重窗口", en_US="Dedup Window"),
            i18n_description=i18n.i18n_text(
                zh_CN="同一张图在该时间内重复识别失败时不再重复入队，避免刷屏",
                en_US="Skip re-queueing the same image within this window",
            ),
        ).model_dump(),
    )


config = plugin.get_config(FaceRecognitionConfig)

# ---------------------------------------------------------------------------
# 路径解析（全部可由配置项覆盖；留空时按下列顺序自动探测）
#
# 识别引擎随插件包分发（tools/），数据目录 / 引擎解释器 / 服务地址均为配置项，
# 不依赖任何写死的部署路径。
# ---------------------------------------------------------------------------
_DEFAULT_FACE_DIR = "/var/lib/docker/nekro_agent_data/face"
_DEFAULT_UPLOAD_ROOT = "/var/lib/docker/nekro_agent_data/uploads"
_PLUGIN_DIR = Path(__file__).resolve().parent

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# 与 face_engine 保持一致的展示名剥离规则
WORK_SEP = "·"
WORK_PREFIXES = ("星铁", "鸣潮", "原神", "崩坏", "明日方舟", "终末地",
                 "蔚蓝档案", "异环", "NIKKE", "绝区零")


def _resolve_data_dir() -> str:
    """人脸数据目录：配置优先 → {NA数据目录}/face（存在则复用）→ 插件数据目录/face。"""
    if config.FACE_DATA_DIR.strip():
        return str(Path(config.FACE_DATA_DIR.strip()).expanduser())
    if os.path.isdir(_DEFAULT_FACE_DIR):
        return _DEFAULT_FACE_DIR
    return str(plugin.get_plugin_data_dir() / "face")


def _resolve_tools_dir() -> str:
    """识别引擎目录：配置优先 → 本插件包内 tools/。"""
    if config.FACE_TOOLS_DIR.strip():
        return str(Path(config.FACE_TOOLS_DIR.strip()).expanduser())
    return str(_PLUGIN_DIR / "tools")


def _resolve_upload_root() -> str:
    """NA 上传目录（引擎按文件名兜底找图用）：配置优先 → {NA数据目录}/uploads。"""
    if config.FACE_UPLOAD_ROOT.strip():
        return str(Path(config.FACE_UPLOAD_ROOT.strip()).expanduser())
    if os.path.isdir(_DEFAULT_UPLOAD_ROOT):
        return _DEFAULT_UPLOAD_ROOT
    # plugin_data_dir = {DATA}/plugin_data/{key} → 上两级即 {DATA}
    return str(Path(plugin.get_plugin_data_dir()).resolve().parent.parent / "uploads")


def _server_port() -> str:
    """常驻服务端口：取自 FACE_SERVER 配置。"""
    try:
        from urllib.parse import urlparse

        return str(urlparse(FACE_SERVER).port or 8766)
    except Exception:  # noqa: BLE001
        return "8766"


def _engine_env() -> dict:
    """注入给识别引擎子进程的环境变量（引擎侧按同名变量覆盖内置默认路径）。"""
    env = os.environ.copy()
    env["FACE_DATA_DIR"] = FACE_DIR
    env["FACE_UPLOAD_ROOT"] = FACE_UPLOAD_ROOT
    env["FACE_SERVER_PORT"] = _server_port()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = FACE_TOOLS_DIR + (os.pathsep + prev if prev else "")
    return env

FACE_DIR = _resolve_data_dir()
FACE_VENV_PY = config.FACE_VENV_PYTHON.strip() or "/opt/face_venv/bin/python"
FACE_TOOLS_DIR = _resolve_tools_dir()
FACE_UPLOAD_ROOT = _resolve_upload_root()
FACE_DB = f"{FACE_DIR}/face_db.json"
ANIME_DB = f"{FACE_DIR}/anime_db.json"
FACE_CONFIG = f"{FACE_DIR}/config.json"
WEB_UPLOAD_DIR = f"{FACE_DIR}/web_uploads"
REVIEW_DIR = f"{FACE_DIR}/review"
REVIEW_IMAGES = f"{REVIEW_DIR}/images"
QUEUE_FILE = f"{REVIEW_DIR}/queue.json"
FACE_SERVER = config.FACE_SERVER_URL.strip().rstrip("/") or "http://127.0.0.1:8766"



# ---------------------------------------------------------------------------
# 常驻识别服务：健康检查 / 拉起 / 调用
# ---------------------------------------------------------------------------
_server_proc: Optional[subprocess.Popen] = None
_server_lock = asyncio.Lock()


def _run_tool(script: str, *args: str, timeout: int = 180) -> tuple[int, str]:
    """兜底方案：直接起子进程执行脚本（慢，每次都要加载模型）。"""
    cmd = [FACE_VENV_PY, str(Path(FACE_TOOLS_DIR) / script), *args]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", env=_engine_env())
        out = (p.stdout or "").strip()
        err = (p.stderr or "").strip()
        if p.returncode != 0:
            return p.returncode, out or err or f"退出码 {p.returncode}"
        return p.returncode, out
    except Exception as e:  # noqa: BLE001
        plugin.logger.exception("人脸工具调用失败")
        return -1, f"调用失败: {e}"


async def _health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{FACE_SERVER}/health")
            return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


async def _ensure_server() -> bool:
    """确保常驻识别服务在跑（不在就拉起），返回是否可用。"""
    global _server_proc
    if await _health():
        return True
    if not config.FACE_SERVER_AUTOSTART:
        # 关闭自动拉起：不启动服务，由调用方回退子进程模式
        return False
    async with _server_lock:
        # 双检
        if await _health():
            return True
        try:
            log = open(f"{FACE_DIR}/server.log", "ab")
            _server_proc = subprocess.Popen(
                [FACE_VENV_PY, str(Path(FACE_TOOLS_DIR) / "face_server.py")],
                stdout=log, stderr=log, start_new_session=True, env=_engine_env(),
            )
            plugin.logger.info(f"已拉起人脸识别常驻服务 (pid={_server_proc.pid})，等待预热…")
        except Exception:  # noqa: BLE001
            plugin.logger.exception("拉起人脸识别服务失败")
            return False
        # 等预热完成（首次加载模型）
        for _ in range(60):
            await asyncio.sleep(0.5)
            if await _health():
                return True
        return False


def _review_params() -> dict:
    """把审核队列配置传给识别服务（否则 face_review_queue 只能用硬编码默认值）。"""
    return {
        "review_enabled": bool(config.REVIEW_ENABLED),
        "review_max_items": int(config.REVIEW_MAX_ITEMS),
        "review_dedup_seconds": int(config.REVIEW_DEDUP_SECONDS),
    }


async def _call_server(endpoint: str, payload: dict, timeout: float = 120) -> tuple[bool, str]:
    """调用常驻服务；服务不可用则回退子进程。返回 (是否成功, 文本)。"""
    if await _ensure_server():
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                r = await c.post(f"{FACE_SERVER}{endpoint}", json=payload)
            data = r.json()
            if r.status_code == 200 and data.get("ok"):
                return True, str(data.get("result", ""))
            return False, str(data.get("error") or data.get("result") or f"HTTP {r.status_code}")
        except Exception as e:  # noqa: BLE001
            plugin.logger.warning(f"常驻服务调用失败，回退子进程: {e}")

    # 兜底：子进程（不支持队列参数，用 face_review_queue 的默认值）
    chat_key = payload.get("chat_key", "")
    if endpoint == "/identify":
        args = [
            "face_identify.py", payload.get("image", ""),
            "--threshold", str(payload.get("threshold", 0.5)),
            "--anime-threshold", str(payload.get("anime_threshold", 0.78)),
        ]
        if chat_key:
            args += ["--chat-key", chat_key]
        if payload.get("detail"):
            args.append("--detail")
        code, out = await asyncio.to_thread(_run_tool, *args)
    else:
        args = ["face_register.py", payload["image"], payload["name"]]
        if payload.get("mode"):
            args += ["--mode", payload["mode"]]
        if chat_key:
            args += ["--chat-key", chat_key]
        code, out = await asyncio.to_thread(_run_tool, *args)
    return code == 0, out


async def _call_server_ex(endpoint: str, payload: dict, timeout: float = 120) -> tuple[bool, str, dict]:
    """同 _call_server，但额外返回服务端完整响应（含 meta / review_id）。"""
    if await _ensure_server():
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                r = await c.post(f"{FACE_SERVER}{endpoint}", json=payload)
            data = r.json()
            if r.status_code == 200 and data.get("ok"):
                return True, str(data.get("result", "")), data
            return False, str(data.get("error") or data.get("result") or f"HTTP {r.status_code}"), {}
        except Exception as e:  # noqa: BLE001
            plugin.logger.warning(f"常驻服务调用失败，回退子进程: {e}")

    ok, out = await _call_server(endpoint, payload, timeout)
    return ok, out, {}


# ---------------------------------------------------------------------------
# 库 / 阈值 / 队列 读写
# ---------------------------------------------------------------------------

def _write_json_atomic(path: str, obj) -> None:
    """原子写 JSON：先备份 .bak，再写临时文件 + fsync + os.replace。

    与 _save_queue 同款策略；配置类小文件额外留一份 .bak，避免写坏后无法回退。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            shutil.copy2(target, target.with_suffix(target.suffix + ".bak"))
        except Exception:  # noqa: BLE001
            pass
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target)

def _get_thresholds() -> tuple[float, float]:
    """返回 (真人阈值, 动漫阈值)。config.json 优先于插件配置。"""
    real_t, anime_t = float(config.THRESHOLD), float(config.ANIME_THRESHOLD)
    try:
        with open(FACE_CONFIG, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        real_t = float(cfg.get("threshold", real_t))
        anime_t = float(cfg.get("anime_threshold", anime_t))
    except Exception:  # noqa: BLE001
        pass
    return max(0.1, min(0.95, real_t)), max(0.1, min(0.95, anime_t))


# ---------------------------------------------------------------------------
# 库访问架构：插件进程【不直接读/写 63MB 库 JSON】。
# 一切经 face_server /library 单源供数（其内存持有解析后的库，mtime 缓存），
# 插件侧再做 5s TTL 二级缓存 —— 首查从 0.85s 降到 ~0.03s，进程内存省 ~150MB，
# 并消除「插件直写库」与 face_server 的并发隐患。
# ---------------------------------------------------------------------------

_LIB_CACHE: dict = {}
_Q_CACHE: dict = {}


async def _lib_entries(force: bool = False) -> list[dict]:
    """库概览（name/display/work/count/kind/thumbs/cover），5s TTL 缓存。"""
    now = time.time()
    c = _LIB_CACHE.get("v")
    if not force and c and now - c[0] < 5:
        return c[1]
    if not await _ensure_server():
        raise HTTPException(status_code=503, detail="识别服务不可用")
    async with httpx.AsyncClient(timeout=30) as cl:
        r = await cl.get(f"{FACE_SERVER}/library")
    entries = (r.json() or {}).get("entries") or []
    _LIB_CACHE["v"] = (now, entries)
    return entries


def _resolve_from_entries(name: str, entries: list[dict]) -> tuple[str, bool]:
    """展示名/全名 → 库内全名（防裸名重复条目），语义同旧 _resolve_db_key。

      1. 名字本身就是库内键 → 原样
      2. 匹配某键展示名（撞名取特征最多的键）→ 该全名
      3. 都不匹配 → (原名, 新建)
    """
    name = (name or "").strip()
    if not name:
        return "", True
    for e in entries:
        if e.get("name") == name:
            return name, False
    matches = [e for e in entries if e.get("display") == name]
    if len(matches) == 1:
        return matches[0]["name"], False
    if matches:
        matches.sort(key=lambda e: -e.get("count", 0))
        return matches[0]["name"], False
    return name, True


def _load_queue() -> list[dict]:
    try:
        mt = os.path.getmtime(QUEUE_FILE)
    except Exception:  # noqa: BLE001
        mt = 0
    c = _Q_CACHE.get("v")
    if c and c[0] == mt:
        return c[1]
    try:
        with open(QUEUE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        data = data if isinstance(data, list) else []
    except Exception:  # noqa: BLE001
        data = []
    _Q_CACHE["v"] = (mt, data)
    return data


def _save_queue(items: list[dict]) -> None:
    os.makedirs(REVIEW_DIR, exist_ok=True)
    tmp = QUEUE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    os.replace(tmp, QUEUE_FILE)  # 原子替换，避免读到半截文件
    try:
        _Q_CACHE["v"] = (os.path.getmtime(QUEUE_FILE), items)
    except Exception:  # noqa: BLE001
        pass


def _display_name(name: str) -> str:
    """库内全名 → 展示名（去掉已知作品前缀）。"""
    for p in WORK_PREFIXES:
        head = p + WORK_SEP
        if name.startswith(head) and len(name) > len(head):
            return name[len(head):]
    return name


@plugin.mount_init_method()
async def _on_init() -> None:
    """插件加载时预热常驻识别服务 + 准备队列目录。"""
    os.makedirs(REVIEW_IMAGES, exist_ok=True)
    ok = await _ensure_server()
    items = _load_queue()
    plugin.logger.info(
        f"[face] 数据目录 {FACE_DIR} | 引擎 {FACE_TOOLS_DIR} | 解释器 {FACE_VENV_PY} | "
        f"服务 {FACE_SERVER}",
    )
    plugin.logger.info(
        f"[face] 识别服务{'已就绪' if ok else '启动失败（将回退子进程）'} | "
        f"待审核 {len(items)} 条 | review_enabled={config.REVIEW_ENABLED}",
    )
    return None


# ---------------------------------------------------------------------------
# LLM 工具（沙盒方法）
#
# 设计原则：插件**不主动说话、不注册聊天指令**。
# 只提供工具，由 AI 自行判断何时调用；结果进入 AI 上下文，
# 再由 AI 按当前角色人格自然表达。人脸的增删管理走 WebUI。
# ---------------------------------------------------------------------------
@plugin.mount_sandbox_method(
    SandboxMethodType.AGENT,
    name="识别人脸",
    description=(
        "【认人专用】识别图片里的人脸是谁，返回一个姓名（基于本地已注册人脸库，覆盖动漫角色与真人）。"
        "调用时机：**你自己觉得需要知道图中的人是谁**时——比如用户发来图片并想确认身份，"
        "或你想称呼图中人物、需要准确的名字而不是靠猜。判断权在你，不需要等用户说出特定口令。"
        "**直接调用本方法即可，无需先用 view_image 看图**——本方法内部会自行检测人脸。"
        "参数说明：image_path 可传图片路径，也可直接传消息里的图片文件名（如 abc123.jpg）；"
        "留空则自动识别当前会话最近收到的图片。"
        "【重要】本方法返回的是**识别结论这一事实**，不是给你照抄的回复。"
        "拿到结果后，请用你自己的身份、语气和人设去组织语言，"
        "不要把内部的措辞、标点或「检测到X张脸」之类的过程信息原样复述或翻译给用户。"
        "如果返回中提到「不确定」或「很像另一个角色」，请也用你的人格自然地表达出这份犹疑，不要说得斩钉截铁。"
    ),
)
async def face_identify(
    _ctx: schemas.AgentCtx,
    image_path: str = "",
) -> str:
    """识别图片中的人脸是谁（动漫角色 / 真人）。

    Args:
        image_path (str): 图片路径或文件名，可选。留空时自动识别当前会话最近收到的图片。
    """
    path = image_path.strip() if image_path else ""
    real_t, anime_t = _get_thresholds()
    payload = {
        "image": path,
        "threshold": real_t,
        "anime_threshold": anime_t,
        "chat_key": getattr(_ctx, "chat_key", "") or "",
    }
    payload.update(_review_params())
    ok, out = await _call_server("/identify", payload)
    if not ok:
        return f"人脸识别失败：{out}"
    return out


@plugin.mount_sandbox_method(
    SandboxMethodType.AGENT,
    name="查询已注册人脸",
    description=(
        "查询当前人脸库里已注册了哪些人（分动漫库和真人库）。"
        "当你想知道能否认出某个人、或想向用户说明你认识哪些人时调用。"
    ),
)
async def face_registered_list(_ctx: schemas.AgentCtx) -> str:
    """查询已注册的人脸名单。"""
    try:
        entries = await _lib_entries()
    except Exception:  # noqa: BLE001
        return "人脸库暂时读不出来（识别服务未就绪），稍后再试一次。"
    if not entries:
        return "人脸库为空，还没有登记任何人。需要新增请到插件 WebUI 管理台上传照片登记。"
    anime = [f"{e['display']}（{e['count']} 张）" for e in entries if e["kind"] == "anime"]
    real = [f"{e['display']}（{e['count']} 张）" for e in entries if e["kind"] == "real"]
    parts = []
    if anime:
        parts.append("动漫头像库：" + "、".join(anime))
    if real:
        parts.append("真人照片库：" + "、".join(real))
    return "\n".join(parts) + "。"


# ---------------------------------------------------------------------------
# WebUI ①：管理台
# ---------------------------------------------------------------------------
MANAGE_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>人脸识别管理台</title>
<style>
:root{--bg:#0f1115;--card:#171a21;--border:#232733;--text:#e6e8ee;--muted:#8b93a5;--accent:#4f8cff;--danger:#e5484d;--ok:#46a758;--warn:#f5a524;--anime:#a855f7}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif;padding:24px;max-width:1100px;margin:0 auto}
h1{font-size:20px;margin-bottom:4px}
.sub{color:var(--muted);font-size:13px;margin-bottom:16px}
.nav{margin-bottom:14px}
.nav a{color:var(--accent);text-decoration:none;font-size:13px;border:1px solid var(--border);padding:6px 12px;border-radius:7px;display:inline-block}
.nav a:hover{border-color:var(--accent);color:#fff}
.usage{background:rgba(245,165,36,.08);border:1px solid rgba(245,165,36,.25);border-radius:10px;padding:14px 16px;font-size:13px;line-height:2;margin-bottom:16px;color:#e8d5a8}
.usage code{background:#0d0f14;padding:1px 6px;border-radius:4px;color:#ffd27f}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px}
.card h2{font-size:15px;margin-bottom:12px}
input[type=text],input[type=number],input[type=file]{width:100%;padding:8px 10px;border:1px solid var(--border);border-radius:6px;background:#0d0f14;color:var(--text);margin-bottom:10px;font-size:13px}
input[type=file]{padding:6px}
button{padding:8px 14px;border:none;border-radius:6px;background:var(--accent);color:#fff;font-size:13px;cursor:pointer}
button:hover{filter:brightness(1.12)}
button.danger{background:var(--danger)}
button:disabled{opacity:.5;cursor:not-allowed}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--border)}
th{color:var(--muted);font-weight:500}
.badge{display:inline-block;padding:1px 7px;border-radius:999px;font-size:11px;border:1px solid}
.badge.anime{color:var(--anime);border-color:rgba(168,85,247,.4);background:rgba(168,85,247,.12)}
.badge.real{color:var(--ok);border-color:rgba(70,167,92,.4);background:rgba(70,167,92,.12)}
.msg{margin-top:10px;font-size:13px;padding:8px 10px;border-radius:6px;display:none;white-space:pre-wrap;word-break:break-all}
.msg.ok{display:block;background:rgba(70,167,92,.12);color:var(--ok);border:1px solid rgba(70,167,92,.3)}
.msg.err{display:block;background:rgba(229,72,77,.12);color:var(--danger);border:1px solid rgba(229,72,77,.3)}
.msg.info{display:block;background:rgba(79,140,255,.1);color:var(--accent);border:1px solid rgba(79,140,255,.25)}
.empty{color:var(--muted);font-size:13px;padding:8px 0}
.hint{color:var(--muted);font-size:12px;margin-top:8px;line-height:1.7}
.row{display:flex;gap:10px}
.row>div{flex:1}
label{font-size:12px;color:var(--muted);display:block;margin-bottom:4px}
</style>
</head>
<body>
<h1>🙂 人脸识别管理台</h1>
<div class="sub">NA 插件 · 动漫头像 + 真人照片双模式 · 纯本地离线 · v1.0.0（分类管理 / 逐脸注册 / 待审核）</div>
<div class="nav"><a href="__REVIEW_URL__">📋 待审核队列</a> <a href="library">📚 角色库</a></div>
<div class="usage">
📌 本插件不给群聊加任何指令。人脸库的增删改都在这张管理台上完成：下面上传照片登记，AI 聊天时就能认出这个人<br>
🤖 AI 侧通过「识别人脸」工具主动调用，识别结果由 AI 按当前角色人设自然表达<br>
🎭 动漫头像走 CCIP 特征库，真人照片走 ArcFace 特征库，系统自动判断类型<br>
📋 识别<b>未确信</b>的图会自动进入「待审核队列」，确认角色后一键注册即可补图
</div>
<div class="grid">
  <div class="card">
    <h2>👥 已注册名单</h2>
    <div id="faceList"><div class="empty">加载中…</div></div>
    <div id="listMsg" class="msg"></div>
  </div>
  <div class="card">
    <h2>➕ 注册人脸</h2>
    <label style="font-size:12px;color:var(--muted)">分类（作品）</label>
    <select id="regWork" style="width:100%;margin-bottom:8px"></select>
    <input type="text" id="regWorkNew" placeholder="新分类名（如：碧蓝航线）" style="display:none;margin-bottom:8px">
    <label style="font-size:12px;color:var(--muted)">角色名</label>
    <input type="text" id="regName" placeholder="姓名（如：流萤）">
    <input type="file" id="regFile" accept="image/*">
    <button id="regBtn" onclick="doRegister()">上传注册</button>
    <div class="hint">动漫头像、真人照片都支持，系统自动识别类型入库；同一人可多次上传追加存档<br>
      ⚠️ 选好分类再填角色名：已有该角色自动<b>并入</b>，没有则新建；选「➕ 新建分类」可开新作品</div>
    <div id="regMsg" class="msg"></div>
  </div>
  <div class="card">
    <h2>📁 分类管理</h2>
    <div id="workList"><div class="empty">加载中…</div></div>
    <div class="hint">改名会把该分类下<b>所有角色</b>一并迁移（已存在的同名角色自动并入）</div>
    <div id="workMsg" class="msg"></div>
  </div>
  <div class="card">
    <h2>🔍 测试识别</h2>
    <input type="file" id="idFile" accept="image/*">
    <button id="idBtn" onclick="doIdentify()">开始识别</button>
    <div class="hint">未确信的图会自动加入「待审核队列」</div>
    <div id="idMsg" class="msg"></div>
  </div>
  <div class="card">
    <h2>⚙️ 识别阈值</h2>
    <div class="row">
      <div>
        <label>真人阈值（默认 0.50）</label>
        <input type="number" id="thresh" min="0.1" max="0.95" step="0.05" placeholder="0.50">
      </div>
      <div>
        <label>动漫阈值（默认 0.78）</label>
        <input type="number" id="animeThresh" min="0.1" max="0.95" step="0.01" placeholder="0.78">
      </div>
    </div>
    <button onclick="saveThreshold()">保存</button>
    <div class="hint">认错人 → 调高；认不出 → 调低。动漫用 CCIP 模型，推荐 0.75~0.85</div>
    <div id="cfgMsg" class="msg"></div>
  </div>
</div>
<script>
const API = location.pathname.replace(/\/$/, "");
function esc(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;")}
function show(id,text,cls){const el=document.getElementById(id);el.className="msg "+cls;el.textContent=text}
async function call(url,opts){
  const r=await fetch(url,opts);let d=null;
  try{d=await r.json()}catch(e){d={detail:await r.text()}}
  if(!r.ok)throw new Error(d.detail||("HTTP "+r.status));return d
}
async function loadFaces(){
  try{
    const d=await call(API+"/api/faces");
    const el=document.getElementById("faceList");
    if(!d.names.length){el.innerHTML='<div class="empty">暂无注册人脸，上传一张照片或头像注册吧</div>';return}
    let h='<table><tr><th>姓名</th><th>类型</th><th>照片数</th><th style="width:110px"></th></tr>';
    for(const n of d.names){
      const badge=n.kind==="anime"?'<span class="badge anime">动漫</span>':'<span class="badge real">真人</span>';
      h+='<tr><td>'+esc(n.name)+'</td><td>'+badge+'</td><td>'+n.count+'</td>'
        +'<td><button data-act="rn" data-name="'+esc(n.name)+'" data-kind="'+n.kind+'">改名</button> '
        +'<button class="danger" data-act="del" data-name="'+esc(n.name)+'" data-kind="'+n.kind+'">删除</button></td></tr>';
    }
    el.innerHTML=h+'</table>';
  }catch(e){document.getElementById("faceList").innerHTML='<div class="empty">加载失败</div>';show("listMsg",e.message,"err")}
}
document.getElementById("faceList").addEventListener("click",e=>{
  const btn=e.target.closest("button[data-name]");if(!btn)return;
  if(btn.dataset.act==="del")delFace(btn.dataset.name,btn.dataset.kind);
  else if(btn.dataset.act==="rn")renameFace(btn.dataset.name,btn.dataset.kind);
});
async function renameFace(name,kind){
  const nw=prompt("把「"+name+"」改名为（可含分类前缀，如 鸣潮·某某；若新名已存在则自动并入）：",name);
  if(!nw||nw===name)return;
  try{
    const fd=new FormData();fd.append("old",name);fd.append("new",nw);fd.append("kind",kind);
    const d=await call(API+"/api/entry/rename",{method:"POST",body:fd});
    show("listMsg","已改名："+name+" → "+nw+(d.merged?"（并入已有条目）":""),"ok");loadFaces()
  }catch(e){show("listMsg",e.message,"err")}
}
// ── 分类管理 ──
let WORKS=[];
async function loadWorks(){
  try{
    const d=await call(API+"/api/library");
    const by={};
    for(const e of d.entries||[]){
      const w=e.work||"未标注";
      if(!by[w])by[w]={chars:0,feats:0};
      by[w].chars++;by[w].feats+=e.count||0;
    }
    WORKS=Object.keys(by).sort().map(w=>({name:w,chars:by[w].chars,feats:by[w].feats}));
    const sel=document.getElementById("regWork");
    const prev=sel.value;
    sel.innerHTML='<option value="">未分类</option>'
      + WORKS.map(w=>'<option value="'+esc(w.name)+'">'+esc(w.name)+'（'+w.chars+'）</option>').join("")
      + '<option value="__new__">➕ 新建分类</option>';
    if(prev)sel.value=prev;
    const el=document.getElementById("workList");
    el.innerHTML='<table><tr><th>分类</th><th>角色</th><th>特征</th><th style="width:56px"></th></tr>'
      + WORKS.map(w=>'<tr><td>'+esc(w.name)+'</td><td>'+w.chars+'</td><td>'+w.feats+'</td>'
        +'<td><button data-w="'+esc(w.name)+'">改名</button></td></tr>').join("")
      + '</table>';
  }catch(e){document.getElementById("workList").innerHTML='<div class="empty">加载失败</div>'}
}
document.getElementById("workList").addEventListener("click",e=>{
  const btn=e.target.closest("button[data-w]");if(btn)renameWork(btn.dataset.w)
});
async function renameWork(w){
  const nw=prompt("把分类「"+w+"」改名为（该分类下所有角色将一并迁移）：",w);
  if(!nw||nw===w)return;
  try{
    const fd=new FormData();fd.append("old",w);fd.append("new",nw);
    const d=await call(API+"/api/work/rename",{method:"POST",body:fd});
    show("workMsg","已改名："+w+" → "+nw+"（"+d.renamed+"/"+d.total+" 角色）","ok");
    loadWorks();loadFaces()
  }catch(e){show("workMsg",e.message,"err")}
}
document.getElementById("regWork").addEventListener("change",()=>{
  const isnew=document.getElementById("regWork").value==="__new__";
  document.getElementById("regWorkNew").style.display=isnew?"block":"none";
});
function busy(on){for(const id of ["regBtn","idBtn"])document.getElementById(id).disabled=on}
async function doRegister(){
  const name=document.getElementById("regName").value.trim();
  const file=document.getElementById("regFile").files[0];
  if(!name)return show("regMsg","请输入角色名","err");
  if(!file)return show("regMsg","请选择图片","err");
  let w=document.getElementById("regWork").value;
  if(w==="__new__"){
    w=document.getElementById("regWorkNew").value.trim();
    if(!w)return show("regMsg","请填写新分类名","err");
    try{
      const fd=new FormData();fd.append("work",w);
      await call(API+"/api/prefix/add",{method:"POST",body:fd});
    }catch(e){return show("regMsg","新建分类失败："+e.message,"err")}
  }
  const full=w?(w+"·"+name):name;
  busy(true);show("regMsg","注册中，请稍候（首次加载模型约需几秒）…","info");
  try{
    const fd=new FormData();fd.append("name",full);fd.append("file",file);
    const d=await call(API+"/api/register",{method:"POST",body:fd});
    show("regMsg",(d.message||"注册成功")+(d.db_key?"（库内键："+d.db_key+"）":""),"ok");
    loadFaces();loadWorks();
  }catch(e){show("regMsg",e.message,"err")}finally{busy(false)}
}
async function doIdentify(){
  const file=document.getElementById("idFile").files[0];
  if(!file)return show("idMsg","请选择图片","err");
  busy(true);show("idMsg","识别中，请稍候…","info");
  try{
    const fd=new FormData();fd.append("file",file);
    const d=await call(API+"/api/identify",{method:"POST",body:fd});
    var msg=d.output||"完成";
    if(d.review_id){msg+="\n\n（未确信，已加入「待审核队列」，可到审核页确认角色后注册）"}
    show("idMsg",msg,"ok");
  }catch(e){show("idMsg",e.message,"err")}finally{busy(false)}
}
async function loadThreshold(){
  try{
    const d=await call(API+"/api/config");
    document.getElementById("thresh").value=d.threshold;
    document.getElementById("animeThresh").value=d.anime_threshold;
  }catch(e){}
}
async function saveThreshold(){
  const v=parseFloat(document.getElementById("thresh").value);
  const a=parseFloat(document.getElementById("animeThresh").value);
  if(isNaN(v)||v<0.1||v>0.95)return show("cfgMsg","真人阈值需在 0.1 ~ 0.95 之间","err");
  if(isNaN(a)||a<0.1||a>0.95)return show("cfgMsg","动漫阈值需在 0.1 ~ 0.95 之间","err");
  try{
    const fd=new FormData();fd.append("threshold",v);fd.append("anime_threshold",a);
    const d=await call(API+"/api/config",{method:"POST",body:fd});
    show("cfgMsg","已保存：真人 "+d.threshold+" / 动漫 "+d.anime_threshold,"ok");
  }catch(e){show("cfgMsg",e.message,"err")}
}
loadFaces();loadThreshold();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# WebUI 路由（单 router：三个 URL 由同一 SPA 承载，按路径切换视图）
#
# ⚠️ mount_router 每个插件只能挂一个路由函数，故多套 UI 合并到这里。
#     GET /  GET /review  GET /library → 均返回 face_webui.html（统一 SPA）
# ---------------------------------------------------------------------------
_FACE_WEBUI_PATH = Path(__file__).with_name("face_webui.html")
try:
    FACE_WEBUI_HTML = _FACE_WEBUI_PATH.read_text(encoding="utf-8")
except OSError as exc:
    import warnings
    warnings.warn(
        f"无法读取人脸识别 WebUI 页面 {_FACE_WEBUI_PATH}，回退到内置兜底页：{exc}",
        RuntimeWarning,
        stacklevel=1,
    )
    FACE_WEBUI_HTML = MANAGE_HTML


@plugin.mount_router()
def create_router() -> APIRouter:
    router = APIRouter()

    async def _save_upload(file: UploadFile) -> str:
        ext = Path(file.filename or "").suffix.lower() or ".jpg"
        if ext not in ALLOWED_EXT:
            raise HTTPException(status_code=400, detail="不支持的图片格式，仅支持 jpg/png/webp/bmp")
        content = await file.read()
        if len(content) > 15 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="图片过大（>15MB）")
        os.makedirs(WEB_UPLOAD_DIR, exist_ok=True)
        path = os.path.join(WEB_UPLOAD_DIR, uuid.uuid4().hex + ext)
        with open(path, "wb") as f:
            f.write(content)
        return path

    # ---------------- 页面 ----------------
    @router.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return FACE_WEBUI_HTML

    @router.get("/review", response_class=HTMLResponse)
    async def review_page() -> str:
        return FACE_WEBUI_HTML

    @router.get("/library", response_class=HTMLResponse)
    async def library_page() -> str:
        return FACE_WEBUI_HTML

    # ---------------- 角色库 API（代理 face_server，文件逻辑都在引擎侧）----------------
    @router.get("/api/library")
    async def api_library() -> dict:
        if not await _ensure_server():
            raise HTTPException(status_code=503, detail="识别服务不可用")
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.get(f"{FACE_SERVER}/library")
            return r.json()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"读取角色库失败：{e}")

    @router.get("/api/library/thumb")
    async def api_library_thumb(kind: str = "anime", name: str = "", idx: str = "0"):
        # idx 用字符串直通引擎：既支持历史数字索引（兼容旧链接，引擎会回落到代表图），
        # 也支持 idx=cover 取代表图。声明成 int 会让 idx=cover 触发参数校验 400。
        if not name:
            raise HTTPException(status_code=400, detail="name required")
        if not await _ensure_server():
            raise HTTPException(status_code=503, detail="识别服务不可用")
        try:
            async with httpx.AsyncClient(timeout=15) as c:
                r = await c.get(f"{FACE_SERVER}/thumb",
                                params={"kind": kind, "name": name, "idx": idx})
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"读取缩略图失败：{e}")
        if r.status_code != 200:
            raise HTTPException(status_code=404, detail="无图")
        return Response(content=r.content, media_type="image/jpeg")

    @router.post("/api/library/remove")
    async def api_library_remove(
        name: str = Form(...),
        idx: int = Form(...),
        kind: str = Form("anime"),
    ) -> dict:
        if not await _ensure_server():
            raise HTTPException(status_code=503, detail="识别服务不可用")
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.post(f"{FACE_SERVER}/remove_feature",
                                 json={"name": name, "idx": idx, "kind": kind})
            return r.json()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"删除失败：{e}")

    # ---------------- 管理台 API ----------------
    @router.post("/api/work/rename")
    async def work_rename(old: str = Form(...), new: str = Form(...)) -> dict:
        """整个分类改名（old·X → new·X），并登记新前缀白名单。"""
        if not await _ensure_server():
            raise HTTPException(status_code=503, detail="识别服务不可用")
        try:
            async with httpx.AsyncClient(timeout=120) as c:
                r = await c.post(f"{FACE_SERVER}/rename_work",
                                 json={"old": old, "new": new})
            _LIB_CACHE["v"] = (0, [])  # 强制刷新库缓存
            return r.json()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"分类改名失败：{e}")

    @router.post("/api/prefix/add")
    async def prefix_add(work: str = Form(...)) -> dict:
        """新建分类：登记前缀白名单（否则 LLM 会看到带前缀的全名）。"""
        if not await _ensure_server():
            raise HTTPException(status_code=503, detail="识别服务不可用")
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.post(f"{FACE_SERVER}/add_prefix", json={"work": work})
            _LIB_CACHE["v"] = (0, [])
            return r.json()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"新建分类失败：{e}")

    @router.post("/api/entry/rename")
    async def entry_rename(
        old: str = Form(...),
        new: str = Form(...),
        kind: str = Form("anime"),
    ) -> dict:
        """角色改名（新名已存在则并入），迁移特征与缩略图。"""
        if not await _ensure_server():
            raise HTTPException(status_code=503, detail="识别服务不可用")
        try:
            async with httpx.AsyncClient(timeout=120) as c:
                r = await c.post(f"{FACE_SERVER}/rename_entry",
                                 json={"old": old, "new": new, "kind": kind})
            _LIB_CACHE["v"] = (0, [])
            return r.json()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"改名失败：{e}")

    @router.get("/api/faces")
    async def list_faces() -> dict:
        entries = await _lib_entries()
        names = [{"name": e["name"], "display": e["display"], "count": e["count"], "kind": e["kind"]}
                 for e in entries]
        return {"names": names, "total": len(names)}

    @router.post("/api/register")
    async def api_register(name: str = Form(...), file: UploadFile = File(...)) -> dict:
        name = name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="名字不能为空")
        # 与审核页一致：把展示名解析成库内全名，避免产生无前缀的重复条目
        db_key, is_new = _resolve_from_entries(name, await _lib_entries())
        path = await _save_upload(file)
        try:
            ok, out = await _call_server("/register", {"image": path, "name": db_key, "mode": "auto"})
        finally:
            try:
                os.remove(path)
            except Exception:  # noqa: BLE001
                pass
        if not ok:
            raise HTTPException(status_code=500, detail=f"注册失败：{out}")
        return {"ok": True, "message": out, "db_key": db_key, "is_new": is_new}

    @router.post("/api/identify")
    async def api_identify(
        file: UploadFile = File(...),
        threshold: Optional[float] = Form(None),
        anime_threshold: Optional[float] = Form(None),
    ) -> dict:
        real_t, anime_t = _get_thresholds()
        if threshold is not None:
            real_t = threshold
        if anime_threshold is not None:
            anime_t = anime_threshold
        path = await _save_upload(file)
        review_id = ""
        try:
            payload = {
                "image": path, "threshold": real_t, "anime_threshold": anime_t,
                "detail": True,
                # 标记来源，便于在审核队列里区分测试图与真实聊天图
                "source": "webui_test",
            }
            payload.update(_review_params())
            ok, out, extra = await _call_server_ex("/identify", payload)
            review_id = str(extra.get("review_id") or "")
        finally:
            try:
                os.remove(path)
            except Exception:  # noqa: BLE001
                pass
        if not ok:
            raise HTTPException(status_code=500, detail=f"识别失败：{out}")
        return {
            "ok": True, "output": out, "threshold": real_t, "anime_threshold": anime_t,
            "review_id": review_id,
        }

    @router.delete("/api/faces/{name}")
    async def delete_face(name: str, kind: str = "all") -> dict:
        """删除整个条目 —— 经 face_server 执行（插件不再直写库文件）。"""
        deleted = []
        targets = []
        if kind in ("all", "real"):
            targets.append("real")
        if kind in ("all", "anime"):
            targets.append("anime")
        for kd in targets:
            ok, out, _extra = await _call_server_ex("/remove_entry", {"name": name, "kind": kd})
            if ok:
                deleted.append(kd)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"未找到「{name}」")
        return {"ok": True, "deleted": name, "kinds": deleted}

    @router.get("/api/config")
    async def get_cfg() -> dict:
        real_t, anime_t = _get_thresholds()
        return {
            "threshold": real_t, "anime_threshold": anime_t,
            "review_enabled": bool(config.REVIEW_ENABLED),
            "review_max_items": int(config.REVIEW_MAX_ITEMS),
            "review_dedup_seconds": int(config.REVIEW_DEDUP_SECONDS),
        }

    @router.post("/api/config")
    async def set_cfg(
        threshold: float = Form(...),
        anime_threshold: Optional[float] = Form(None),
    ) -> dict:
        real_t = max(0.1, min(0.95, threshold))
        _, cur_anime = _get_thresholds()
        anime_t = max(0.1, min(0.95, anime_threshold if anime_threshold is not None else cur_anime))
        _write_json_atomic(FACE_CONFIG, {"threshold": real_t, "anime_threshold": anime_t})
        return {"ok": True, "threshold": real_t, "anime_threshold": anime_t}

    # ---------------- 待审核队列 API ----------------
    @router.get("/api/queue")
    async def get_queue(confidence: str = "") -> dict:
        items = _load_queue()
        items.sort(key=lambda x: x.get("ts", 0), reverse=True)
        total = len(items)
        if confidence:
            items = [i for i in items if i.get("confidence") == confidence]
        # 补上可读时间 + 库内全名（经 face_server 单源查表，插件不读库文件）
        entries = await _lib_entries()
        by_name = {e["name"]: e for e in entries}
        by_disp = {}
        for e in entries:
            d = e.get("display") or ""
            cur = by_disp.get(d)
            if cur is None or e.get("count", 0) > cur.get("count", 0):
                by_disp[d] = e
        for i in items:
            try:
                i["ts_str"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(i.get("ts", 0)))
            except Exception:  # noqa: BLE001
                i["ts_str"] = ""
            for f in i.get("faces") or []:
                for c in f.get("candidates") or []:
                    e = by_name.get(c.get("name", "")) or by_disp.get(c.get("name", ""))
                    if e is not None:
                        c["db_key"], c["feats"], in_lib = e["name"], e.get("count", 0), True
                    else:
                        c["db_key"], c["feats"], in_lib = c.get("name", ""), 0, False
                    c["in_library"] = in_lib
        return {"ok": True, "total": total, "count": len(items), "items": items}

    @router.get("/api/image/{item_id}")
    async def get_image(item_id: str):
        for i in _load_queue():
            if i.get("id") == item_id:
                p = i.get("image_path") or ""
                if p and os.path.exists(p):
                    return FileResponse(p)
        raise HTTPException(status_code=404, detail="图片不存在")

    @router.get("/api/image/{item_id}/t")
    async def get_image_thumb(item_id: str):
        """卡片缩略图（最长边 480px，磁盘缓存 t_{id}.jpg）。

        队列原图中位 2.5MB、最大 28MB，227 张合计 1.1GB ——
        卡片直接加载原图是审核页卡顿的主因。
        """
        from PIL import Image

        for i in _load_queue():
            if i.get("id") == item_id:
                p = i.get("image_path") or ""
                if not p or not os.path.exists(p):
                    break
                thumb = os.path.join(os.path.dirname(p), f"t_{item_id}.jpg")
                if not os.path.exists(thumb):
                    try:
                        with Image.open(p) as im:
                            im = im.convert("RGB")
                            im.thumbnail((480, 480))
                            im.save(thumb, "JPEG", quality=82)
                    except Exception as e:  # noqa: BLE001
                        raise HTTPException(status_code=500, detail=f"缩略图生成失败：{e}")
                return FileResponse(thumb, media_type="image/jpeg")
        raise HTTPException(status_code=404, detail="图片不存在")

    @router.post("/api/item/{item_id}/register")
    async def register_item(
        item_id: str,
        name: str = Form(...),
        face: Optional[int] = Form(None),
    ) -> dict:
        """注册一条待审核项。

        - 带 face=索引：注册【那张脸】（合照逐脸注册）。条目保留在队列，
          并把该脸记入 registered，供一张合照注册多个角色。
        - 不带 face：旧行为，注册最大脸并移出队列。
        """
        name = name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="名字不能为空")
        items = _load_queue()
        target = next((i for i in items if i.get("id") == item_id), None)
        if not target:
            raise HTTPException(status_code=404, detail="该条目不存在")
        img = target.get("image_path") or ""
        if not img or not os.path.exists(img):
            raise HTTPException(status_code=404, detail="原图已丢失，无法注册")

        # ⚠️ 把展示名解析成库内全名，避免新建无前缀的重复条目
        db_key, is_new = _resolve_from_entries(name, await _lib_entries())

        payload = {"image": img, "name": db_key, "mode": "anime",
                   # 审核注册不允许静默落入真人库（动漫图检出失败时宁可直接报错）
                   "no_fallback": True}
        keep = False
        if face is not None:
            faces = target.get("faces") or []
            if face < 0 or face >= len(faces):
                raise HTTPException(status_code=400, detail="脸序号越界")
            bbox = faces[face].get("bbox") or []
            if len(bbox) != 4:
                raise HTTPException(status_code=400,
                                    detail="该脸没有坐标（旧数据），请用整图注册")
            payload["bbox"] = bbox
            keep = True

        ok, out = await _call_server("/register", payload)
        if not ok:
            return {"ok": False, "error": out}

        if keep:
            # 合照逐脸：条目保留，记录已注册的脸；
            # 【全部脸都注册完 → 自动移出队列】（单脸图即注册即移除）
            reg = target.setdefault("registered", [])
            if face not in reg:
                reg.append(face)
            faces_all = target.get("faces") or []
            all_done = bool(faces_all) and all(i in reg for i in range(len(faces_all)))
            if all_done:
                items = [i for i in items if i.get("id") != item_id]
                _save_queue(items)
                try:
                    os.remove(img)
                except Exception:  # noqa: BLE001
                    pass
                plugin.logger.info(
                    f"[face] 逐脸注册完成（全部 {len(faces_all)} 脸），条目 {item_id} 自动移出队列",
                )
                return {"ok": True, "message": out, "db_key": db_key,
                        "is_new": is_new, "kept": False, "auto_done": True}
            _save_queue(items)
            note = "（新建条目）" if is_new else "（并入已有条目）"
            plugin.logger.info(
                f"[face] 逐脸注册 {db_key!r}{note}（待审核 {item_id} 脸{face}，填写 {name!r}）: {out}",
            )
            return {"ok": True, "message": out, "db_key": db_key,
                    "is_new": is_new, "kept": True, "registered": reg}

        # 注册成功后移出队列并删图（旧行为）
        items = [i for i in items if i.get("id") != item_id]
        _save_queue(items)
        try:
            os.remove(img)
        except Exception:  # noqa: BLE001
            pass
        note = "（新建条目）" if is_new else "（并入已有条目）"
        plugin.logger.info(
            f"[face] 已注册 {db_key!r}{note}（来自待审核 {item_id}，用户填写 {name!r}）: {out}",
        )
        return {"ok": True, "message": out, "db_key": db_key, "is_new": is_new}

    @router.delete("/api/item/{item_id}")
    async def drop_item(item_id: str) -> dict:
        items = _load_queue()
        target = next((i for i in items if i.get("id") == item_id), None)
        if not target:
            raise HTTPException(status_code=404, detail="该条目不存在")
        items = [i for i in items if i.get("id") != item_id]
        _save_queue(items)
        try:
            p = target.get("image_path") or ""
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:  # noqa: BLE001
            pass
        return {"ok": True}

    @router.delete("/api/queue")
    async def clear_queue() -> dict:
        items = _load_queue()
        for i in items:
            try:
                p = i.get("image_path") or ""
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:  # noqa: BLE001
                pass
        _save_queue([])
        return {"ok": True, "removed": len(items)}

    @router.get("/api/health")
    async def api_health() -> dict:
        return {"ok": True, "service": await _health(), "face_server": FACE_SERVER,
                "plugin_version": plugin.version}

    @router.get("/api/stats")
    async def stats() -> dict:
        items = _load_queue()
        by_conf: dict[str, int] = {}
        for i in items:
            c = i.get("confidence", "?")
            by_conf[c] = by_conf.get(c, 0) + 1
        entries = await _lib_entries()
        lib = {"anime_characters": 0, "anime_features": 0,
               "real_people": 0, "real_features": 0}
        for e in entries:
            if e["kind"] == "anime":
                lib["anime_characters"] += 1
                lib["anime_features"] += e.get("count", 0)
            else:
                lib["real_people"] += 1
                lib["real_features"] += e.get("count", 0)
        return {
            "ok": True,
            "queue": len(items),
            "by_confidence": by_conf,
            "review_enabled": bool(config.REVIEW_ENABLED),
            "library": lib,
            "plugin_version": plugin.version,
            "data_dir": FACE_DIR,
        }

    return router

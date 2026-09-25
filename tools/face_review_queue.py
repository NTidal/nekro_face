"""待审核队列写入器（由 face_server 调用）。

职责：把「未确信」的识别结果落盘，供 nekro_face_review 插件的 WebUI 审核补图。

设计要点
--------
- **零依赖**：本模块不 import 任何 NA 插件，只读写文件。
  识别服务（face_venv）与插件（NA 主进程）之间**只通过文件解耦**，
  任一方故障都不影响另一方。
- **只记录不确定的**：confidence == "certain" 的不入队（已认对，无需人工看）。
- **去重**：同一张图（按文件内容 sha1）在窗口期内不重复入队，避免刷屏。
- **上限**：超出 MAX_ITEMS 时丢最旧的，并删除其图片，避免磁盘无上限增长。
- **原子写**：queue.json 用「临时文件 + os.replace」替换，避免读到半截 JSON。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid

FACE_DIR = os.environ.get("FACE_DATA_DIR", "/var/lib/docker/nekro_agent_data/face")
REVIEW_DIR = f"{FACE_DIR}/review"
REVIEW_IMAGES = f"{REVIEW_DIR}/images"
QUEUE_FILE = f"{REVIEW_DIR}/queue.json"

MAX_ITEMS = 500
DEDUP_SECONDS = 3600
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")

# 只记录这些置信度（certain 表示已认对，不必人工看）
RECORD_CONFIDENCE = {"mid", "low", "unknown", "noface"}

# 运行时可覆盖的参数（由 face_server 从插件请求里透传）
# 背景：本模块跑在 face_venv 的独立进程里，**读不到 NA 的插件配置**，
# 本模块因此读不到 NA 的插件配置，这些参数需由调用方随请求透传覆盖。
# 现在改为：请求体里带 review_* 参数时覆盖，不带则用下面的默认值。
_runtime = {
    "enabled": True,
    "max_items": MAX_ITEMS,
    "dedup_seconds": DEDUP_SECONDS,
}


def configure(*, enabled: bool | None = None, max_items: int | None = None,
              dedup_seconds: int | None = None) -> None:
    """由 face_server 在每次识别前按请求参数刷新运行时配置。"""
    if enabled is not None:
        _runtime["enabled"] = bool(enabled)
    if max_items is not None:
        try:
            _runtime["max_items"] = max(1, int(max_items))
        except Exception:
            pass
    if dedup_seconds is not None:
        try:
            _runtime["dedup_seconds"] = max(0, int(dedup_seconds))
        except Exception:
            pass


def _load() -> list:
    try:
        with open(QUEUE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save(items: list) -> None:
    os.makedirs(REVIEW_DIR, exist_ok=True)
    tmp = QUEUE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    os.replace(tmp, QUEUE_FILE)


def _sha1_of(path: str) -> str:
    h = hashlib.sha1()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def record(meta: dict, *, source: str = "face_identify") -> str | None:
    """按识别 meta 决定是否入队。返回入队 id，或 None（未入队）。

    Args:
        meta: face_engine.identify_ex() 返回的结构化结果
        source: 来源标记，便于区分是聊天调用还是 WebUI 测试
    """
    try:
        # 总开关（由插件配置透传；默认开启）
        if not _runtime["enabled"]:
            return None
        conf = str(meta.get("confidence") or "")
        if conf not in RECORD_CONFIDENCE:
            return None
        src_path = str(meta.get("path") or "")
        if not src_path or not os.path.exists(src_path):
            return None
        if not src_path.lower().endswith(IMG_EXT):
            return None

        digest = _sha1_of(src_path)
        if not digest:
            return None

        now = time.time()
        items = _load()
        dedup_seconds = int(_runtime["dedup_seconds"])

        # 去重：同一张图在窗口期内不重复入队
        for it in items:
            if it.get("sha1") == digest and (now - float(it.get("ts", 0))) < dedup_seconds:
                return None

        # 复制原图（uploads 可能被清理，复制一份保险）
        os.makedirs(REVIEW_IMAGES, exist_ok=True)
        item_id = uuid.uuid4().hex[:12]
        ext = os.path.splitext(src_path)[1].lower() or ".jpg"
        dst = os.path.join(REVIEW_IMAGES, item_id + ext)
        shutil.copy2(src_path, dst)

        items.append({
            "id": item_id,
            "ts": now,
            "sha1": digest,
            "confidence": conf,
            "recognized": list(meta.get("recognized") or []),
            "faces": list(meta.get("faces") or []),
            "chat_key": str(meta.get("chat_key") or ""),
            "kind": str(meta.get("kind") or ""),
            "anime_score": meta.get("anime_score"),
            "real_score": meta.get("real_score"),
            "thresholds": meta.get("thresholds") or {},
            "src_path": src_path,
            "image_path": dst,
            "source": source,
        })

        # 超上限：丢最旧的（连同图片）
        max_items = int(_runtime["max_items"])
        if len(items) > max_items:
            items.sort(key=lambda x: float(x.get("ts", 0)))
            dropped = items[: len(items) - max_items]
            items = items[len(items) - max_items :]
            for d in dropped:
                try:
                    p = d.get("image_path") or ""
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

        _save(items)
        return item_id
    except Exception:
        # 记录失败绝不能影响识别主流程
        return None

"""人脸识别常驻服务：模型只加载一次，通过 HTTP 处理识别/注册请求。

用法（后台）：/opt/face_venv/bin/python face_server.py &
接口：
  POST /identify  {"image": "<path>", "threshold": 0.5, "anime_threshold": 0.78}
  POST /register  {"image": "<path>", "name": "xx", "mode": "anime|auto|real"}
  GET  /health
  GET  /library                      已注册条目概览（含缩略图索引）
  GET  /thumb?kind=&name=&idx=       某特征的裁剪图（JPEG）
  POST /remove_feature {"name","idx","kind"}  删除单张特征
"""
import json
import os
import sys
import threading
import time
from urllib.parse import parse_qs, urlparse  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402

import face_engine  # noqa: E402

try:
    import face_review_queue  # noqa: E402  待审核队列写入器（可选依赖）
except Exception:  # noqa: BLE001
    face_review_queue = None

PORT = int(os.environ.get("FACE_SERVER_PORT", "8766"))
_lock = threading.Lock()  # 串行化推理，避免 CPU 争抢
_started = time.time()
_count = {"identify": 0, "register": 0}


def warmup():
    """预热：加载全部模型（首次请求不再慢）。"""
    try:
        import numpy as np

        import cv2

        blank = (np.ones((320, 320, 3), dtype=np.uint8) * 128)
        face_engine.classify_anime_real(blank)
        face_engine.detect_anime(blank)
        face_engine.anime_embed(blank)
        face_engine.detect_real(blank)
        # 预热库缓存：63MB JSON 的解析挪到启动期，首次识别/审核不再付费
        face_engine.load_db(face_engine.ANIME_DB)
        face_engine.load_db(face_engine.REAL_DB)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[warmup] 失败: {e}", flush=True)
        return False


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # 静默
        pass

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _bytes(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/health":
            self._json(200, {"ok": True, "uptime": round(time.time() - _started, 1), "counts": _count})
        elif u.path == "/library":
            try:
                self._json(200, {"ok": True, "entries": face_engine.library_overview()})
            except Exception as e:  # noqa: BLE001
                self._json(500, {"ok": False, "error": str(e)})
        elif u.path == "/thumb":
            q = parse_qs(u.query)
            kind = (q.get("kind") or ["anime"])[0]
            name = (q.get("name") or [""])[0]
            idx = (q.get("idx") or ["0"])[0]
            if not name:
                self._json(400, {"error": "name required"})
                return
            data = face_engine.read_thumb(kind, name, idx)
            if data is None:
                self._json(404, {"error": "no thumb"})
                return
            self._bytes(200, data, "image/jpeg")
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        except Exception as e:  # noqa: BLE001
            self._json(400, {"error": f"bad request: {e}"})
            return

        t0 = time.time()
        try:
            meta = None
            with _lock:
                if self.path == "/identify":
                    _count["identify"] += 1
                    out, meta = face_engine.identify_ex(
                        req.get("image", ""),
                        float(req.get("threshold", face_engine.REAL_THRESHOLD)),
                        float(req.get("anime_threshold", face_engine.ANIME_THRESHOLD)),
                        req.get("chat_key", ""),
                        bool(req.get("detail", False)),
                    )
                elif self.path == "/register":
                    _count["register"] += 1
                    out = face_engine.register(
                        req.get("image", ""), req.get("name", ""), req.get("mode", "auto"),
                        req.get("chat_key", ""), req.get("bbox"),
                        bool(req.get("no_fallback", False)),
                    )
                elif self.path == "/remove_feature":
                    out = face_engine.remove_feature(
                        req.get("name", ""), int(req.get("idx", -1)), req.get("kind", "anime"),
                    )
                    payload = {"ok": True, "result": out, "elapsed": round(time.time() - t0, 3)}
                    self._json(200, payload)
                    return
                elif self.path == "/remove_entry":
                    out = face_engine.remove_entry(
                        req.get("name", ""), req.get("kind", "anime"),
                    )
                    payload = {"ok": True, "result": out, "elapsed": round(time.time() - t0, 3)}
                    self._json(200, payload)
                    return
                elif self.path == "/rename_entry":
                    out = face_engine.rename_entry(
                        req.get("old", ""), req.get("new", ""), req.get("kind", "anime"),
                    )
                    payload = {"ok": True, "result": out, "elapsed": round(time.time() - t0, 3)}
                    self._json(200, payload)
                    return
                elif self.path == "/rename_work":
                    out = face_engine.rename_work(
                        req.get("old", ""), req.get("new", ""), req.get("kind", "anime"),
                    )
                    payload = {"ok": True, "result": out, "elapsed": round(time.time() - t0, 3)}
                    self._json(200, payload)
                    return
                elif self.path == "/add_prefix":
                    out = face_engine.add_work_prefix(req.get("work", ""))
                    payload = {"ok": True, "result": out, "elapsed": round(time.time() - t0, 3)}
                    self._json(200, payload)
                    return
                else:
                    self._json(404, {"error": "not found"})
                    return
            payload = {"ok": True, "result": out, "elapsed": round(time.time() - t0, 3)}
            if meta is not None:
                payload["meta"] = meta
                # 未确信的图入待审核队列（失败绝不影响识别结果返回）
                if face_review_queue is not None:
                    try:
                        # 队列参数由调用方（插件）透传，使插件配置真正生效；
                        # 未透传时 face_review_queue 用自身默认值。
                        face_review_queue.configure(
                            enabled=req.get("review_enabled"),
                            max_items=req.get("review_max_items"),
                            dedup_seconds=req.get("review_dedup_seconds"),
                        )
                        # source 由调用方指定：聊天调用为 face_identify，WebUI 测试为 webui_test
                        item_id = face_review_queue.record(
                            meta, source=str(req.get("source") or "face_identify"),
                        )
                        if item_id:
                            payload["review_id"] = item_id
                    except Exception:  # noqa: BLE001
                        pass
            self._json(200, payload)
        except Exception as e:  # noqa: BLE001
            self._json(500, {"error": str(e)})


if __name__ == "__main__":
    t = time.time()
    ok = warmup()
    print(f"[face_server] 预热{'成功' if ok else '失败'}，耗时 {time.time() - t:.2f}s | 端口 {PORT}", flush=True)
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[face_server] 已启动，监听 127.0.0.1:{PORT}", flush=True)
    srv.serve_forever()

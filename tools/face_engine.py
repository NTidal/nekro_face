"""双模式人脸识别引擎。

- 动漫头像：anime_face_detection (YOLOv8) 检测 + CCIP 特征 (768 维, 阈值 ~0.82)
- 真人照片：insightface buffalo_l (SCRFD + ArcFace, 512 维, 阈值 ~0.5)
- 自动分流：anime_real_cls 判断图片是动漫还是真人
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
import onnxruntime as ort

import cv2
from typing import Optional

# 数据根目录：由插件通过 FACE_DATA_DIR 环境变量注入；
# 未注入时按下面的默认约定路径解析，便于脱离插件单独运行引擎脚本。
BASE = os.environ.get("FACE_DATA_DIR", "/var/lib/docker/nekro_agent_data/face")
REAL_DB = f"{BASE}/face_db.json"
ANIME_DB = f"{BASE}/anime_db.json"
ANIME_MODELS = f"{BASE}/anime_models"
MODELS_ROOT = BASE
THUMBS_DIR = f"{BASE}/thumbs"   # 注册特征时保存的人脸裁剪图（供 WebUI 可视化管理）

ANIME_CONF_THR = 0.6
ANIME_IOU_THR = 0.5
REAL_THRESHOLD = 0.5
ANIME_THRESHOLD = 0.78
REAL_DET_THR = 0.25
# 跨作品入库时的名字前缀分隔符（如 "星铁·姬子"）。
# 库内用带前缀的全名避免撞名，对外展示/AI 输出时去掉前缀。
WORK_SEP = "·"
# 内置作品前缀。注意：库里本就有「陆·赫斯」「阮·梅」这类自带「·」的名字，
# 因此剥离时必须**只匹配已知前缀**，绝不能盲目 split("·")。
_DEFAULT_WORK_PREFIXES = ("星铁", "鸣潮", "原神", "崩坏", "明日方舟", "终末地",
                          "蔚蓝档案", "异环", "NIKKE", "绝区零")
# 运行时可扩展：WebUI「新建分类」会追加写入 work_prefixes.json
WORK_PREFIXES_FILE = f"{BASE}/work_prefixes.json"
_PREFIX_CACHE = {"mt": None, "list": _DEFAULT_WORK_PREFIXES}


def work_prefixes() -> tuple:
    """作品前缀白名单（内置 + 数据文件扩展，mtime 缓存）。"""
    try:
        mt = os.path.getmtime(WORK_PREFIXES_FILE)
    except Exception:
        mt = None
    if _PREFIX_CACHE["mt"] != mt:
        lst = _DEFAULT_WORK_PREFIXES
        try:
            with open(WORK_PREFIXES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                lst = tuple(dict.fromkeys(list(_DEFAULT_WORK_PREFIXES) + [str(x) for x in data]))
        except Exception:
            pass
        _PREFIX_CACHE["mt"] = mt
        _PREFIX_CACHE["list"] = lst
    return _PREFIX_CACHE["list"]


def add_work_prefix(work: str) -> dict:
    """新增作品前缀到白名单文件（WebUI 新建分类时调用）。"""
    work = (work or "").strip().strip(WORK_SEP)
    if not work:
        return {"ok": False, "error": "分类名不能为空"}
    if WORK_SEP in work:
        return {"ok": False, "error": "分类名不能包含分隔符「·」"}
    cur = list(work_prefixes())
    if work in cur:
        return {"ok": True, "added": False, "prefixes": list(cur)}
    cur.append(work)
    os.makedirs(os.path.dirname(WORK_PREFIXES_FILE), exist_ok=True)
    with open(WORK_PREFIXES_FILE, "w", encoding="utf-8") as f:
        json.dump(cur, f, ensure_ascii=False, indent=2)
    _PREFIX_CACHE["mt"] = None  # 失效，下次读取重建
    return {"ok": True, "added": True, "prefixes": list(cur)}


def display_name(name: str) -> str:
    """把库内全名转成对外展示名（去掉作品前缀）。

    "星铁·姬子" → "姬子"；"陆·赫斯" → "陆·赫斯"（无已知前缀，原样返回）
    """
    for p in work_prefixes():
        head = p + WORK_SEP
        if name.startswith(head) and len(name) > len(head):
            return name[len(head):]
    return name


def _assign_pos(entries: list, shape: tuple) -> str:
    """按整组脸的排布生成自然方位描述（写回每个 entry 的 "pos"）。

    排布是【组属性】——"左边"只有参照整组才有意义，所以必须整组一起算：
      · 一排   → pos = 左1/左2/…（从左到右），文本用「左边/右边」「从左到右」
      · 一列   → pos = 上1/上2/…（从上到下），文本用「上面/下面」「从上到下」
      · 散布   → 九宫格自然方位（左上/上方/右上/左侧/中间/右侧/左下/下方/右下），
                 同区多脸加「靠上/靠下/靠左/靠右」消歧
    单张脸不标方位（无参照）。

    Returns:
        排布模式 "row" | "col" | "grid"
    """
    if not entries:
        return "grid"
    h_img, w_img = shape[0], shape[1]
    pts = []
    for e in entries:
        b = e.get("bbox") or []
        if len(b) == 4:
            pts.append((e, (b[0] + b[2]) / 2 / max(1, w_img),
                        (b[1] + b[3]) / 2 / max(1, h_img)))
        else:
            e["pos"] = ""
    if not pts:
        return "grid"
    if len(pts) == 1:
        pts[0][0]["pos"] = ""
        return "grid"

    xs = sorted(p[1] for p in pts)
    ys = sorted(p[2] for p in pts)
    spread_x, spread_y = xs[-1] - xs[0], ys[-1] - ys[0]

    if spread_y <= 0.12 and spread_x > 0.02:
        order = sorted(pts, key=lambda t: t[1])
        for k, (e, _x, _y) in enumerate(order, 1):
            e["pos"] = f"左{k}"
        return "row"
    if spread_x <= 0.12 and spread_y > 0.02:
        order = sorted(pts, key=lambda t: t[2])
        for k, (e, _x, _y) in enumerate(order, 1):
            e["pos"] = f"上{k}"
        return "col"

    # 散布：九宫格自然方位
    zones: dict = {}
    for e, cx, cy in pts:
        v = "上" if cy < 1 / 3 else ("中" if cy < 2 / 3 else "下")
        h = "左" if cx < 1 / 3 else ("中" if cx < 2 / 3 else "右")
        if h == "中" and v == "中":
            name = "中间"
        elif h == "中":
            name = v + "方"
        elif v == "中":
            name = h + "侧"
        else:
            name = h + v
        zones.setdefault(name, []).append((e, cx, cy))
    for name, grp in zones.items():
        if len(grp) == 1:
            grp[0][0]["pos"] = name
            continue
        ys2 = [g[2] for g in grp]
        xs2 = [g[1] for g in grp]
        if max(ys2) - min(ys2) > 0.05:
            order, tags = sorted(grp, key=lambda t: t[2]), ["靠上", "居中", "靠下"]
        elif max(xs2) - min(xs2) > 0.05:
            order, tags = sorted(grp, key=lambda t: t[1]), ["靠左", "居中", "靠右"]
        else:
            order, tags = sorted(grp, key=lambda t: (t[2], t[1])), None
        n = len(order)
        for k, (e, _x, _y) in enumerate(order):
            if tags is None:
                e["pos"] = f"{name}·{k + 1}"
            else:
                idx = round(k * (len(tags) - 1) / max(1, n - 1))
                e["pos"] = name + tags[idx]
    return "grid"


def _certain_of(name: str, cands: list, thr: float) -> bool:
    """单张脸是否「笃定认出」：达阈值 且 与次名差距达标（gap 规则）。

    多脸合照的 confidence 需要对每张脸分别判定 —— 只要有一张不笃定，
    整图就不能叫 certain。
    """
    if not cands or name == "未知":
        return False
    s1 = float(cands[0][0])
    if s1 < thr:
        return False
    gap = (s1 - float(cands[1][0])) if len(cands) > 1 else s1
    return _is_certain(s1, gap)


# ---------------------------------------------------------------------------
# 条目代表图：注册时保存最新一次的人脸裁剪图，供 WebUI 标识该条目。
# 布局：{THUMBS_DIR}/{kind}/{safe_name}/cover.jpg —— 每条目【仅一张】，随注册覆盖。
#
# 不采用「每特征一张 {idx}.jpg」的原因有二：
#   1) 存储与探测随特征数线性增长（每次 /library 逐特征 os.path.exists）；
#   2) 索引与库内特征顺序强绑定，改名并入/删除特征都要重排索引。
# 现改为单一代表图：与索引无关，改名/合并/删特征都不再需要重排。
# 遗留的 {idx}.jpg 会在该条目下次注册时自动清理。
# ---------------------------------------------------------------------------
_SAFE_OK = set("·-_")


def _safe_name(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch in _SAFE_OK:
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).strip() or "_"
    return s[:80]  # 防超长文件名


def _thumb_file(kind: str, name: str, idx: int) -> str:
    return f"{THUMBS_DIR}/{kind}/{_safe_name(name)}/{idx}.jpg"


def _cover_file(kind: str, name: str) -> str:
    """该条目的代表图路径（每条目唯一一张）。"""
    return f"{THUMBS_DIR}/{kind}/{_safe_name(name)}/cover.jpg"


def _save_cover(kind: str, name: str, crop) -> None:
    """注册成功后写入/覆盖该条目的代表图（任何失败都不影响注册本身）。

    只保留最新一张：写入前先落临时文件再 os.replace，避免半截图被读到；
    同时顺手清理历史遗留的逐特征缩略图 {idx}.jpg。
    """
    try:
        d = os.path.dirname(_cover_file(kind, name))
        os.makedirs(d, exist_ok=True)
        ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            return
        tmp = os.path.join(d, "cover.jpg.tmp")
        with open(tmp, "wb") as fh:
            fh.write(buf.tobytes())
        os.replace(tmp, _cover_file(kind, name))
        for f in os.listdir(d):
            if f.endswith(".jpg") and f[:-4].isdigit():
                try:
                    os.remove(os.path.join(d, f))
                except Exception:
                    pass
    except Exception:
        pass


def _work_of(name: str) -> str:
    """从库内全名解析作品前缀（星铁·流萤 → 星铁；无前缀 → ""）。"""
    for p in work_prefixes():
        if name.startswith(p + WORK_SEP):
            return p
    return ""


def library_overview() -> list[dict]:
    """全部已注册条目概览（供 WebUI 角色库页）。

    附带 work 字段（作品前缀，无则空串）—— 前端按作品分组/筛选。
    """
    out: list[dict] = []
    for kind, path in (("anime", ANIME_DB), ("real", REAL_DB)):
        db = load_db(path)
        for name, lst in db.items():
            n = len(lst)
            # 只探测代表图一处（原先逐特征探测，3835 特征 = 3835 次 stat/次请求）
            # thumbs 字段保留为空列表，兼容仍缓存着旧页面的浏览器。
            cover = os.path.exists(_cover_file(kind, name))
            out.append({"kind": kind, "name": name, "display": display_name(name),
                        "work": _work_of(name),
                        "count": n, "thumbs": [], "cover": cover})
    out.sort(key=lambda e: (-e["count"], e["display"]))
    return out


def _move_thumbs(kind: str, old: str, new: str, merged_offset: int | None) -> None:
    """代表图目录随改名/并入搬移：整目录改名；并入时只留一张（与索引无关）。

    merged_offset 参数保留仅为兼容既有调用点，单一代表图下不再使用。
    """
    old_dir = os.path.dirname(_thumb_file(kind, old, 0))
    new_dir = os.path.dirname(_thumb_file(kind, new, 0))
    if not os.path.isdir(old_dir):
        return
    if merged_offset is None:  # 纯改名：整目录搬走
        os.makedirs(os.path.dirname(new_dir), exist_ok=True)
        os.replace(old_dir, new_dir)
        return
    # 并入：新条目已有代表图则保留新的，否则沿用旧的
    os.makedirs(new_dir, exist_ok=True)
    src_cover = os.path.join(old_dir, "cover.jpg")
    dst_cover = os.path.join(new_dir, "cover.jpg")
    if os.path.exists(src_cover) and not os.path.exists(dst_cover):
        shutil.move(src_cover, dst_cover)
    shutil.rmtree(old_dir, ignore_errors=True)


def rename_entry(old: str, new: str, kind: str = "anime") -> dict:
    """重命名条目；新名已存在则【并入】（特征追加 + 缩略图索引顺延）。"""
    old = (old or "").strip()
    new = (new or "").strip()
    if not old or not new:
        return {"ok": False, "error": "名字不能为空"}
    if old == new:
        db = load_db(REAL_DB if kind == "real" else ANIME_DB)
        return {"ok": True, "merged": False, "count": len(db.get(new, []))}
    path = REAL_DB if kind == "real" else ANIME_DB
    db = load_db(path)
    if old not in db:
        return {"ok": False, "error": f"条目不存在：{old}"}
    lst = db.pop(old)
    merged = new in db
    if merged:
        db[new].extend(lst)
    else:
        db[new] = lst
    save_db(path, db)
    _move_thumbs(kind, old, new, len(db[new]) if merged else None)
    return {"ok": True, "merged": merged, "count": len(db[new])}


def rename_work(old: str, new: str, kind: str = "anime") -> dict:
    """整个作品分类改名：`old·X` → `new·X`，并自动登记新前缀白名单。"""
    old = (old or "").strip().strip(WORK_SEP)
    new = (new or "").strip().strip(WORK_SEP)
    if not old or not new:
        return {"ok": False, "error": "分类名不能为空"}
    if WORK_SEP in new:
        return {"ok": False, "error": "分类名不能包含分隔符「·」"}
    if old == new:
        return {"ok": False, "error": "新旧分类名相同"}
    path = REAL_DB if kind == "real" else ANIME_DB
    db = load_db(path)
    head_old, head_new = old + WORK_SEP, new + WORK_SEP
    targets = sorted(k for k in db if k.startswith(head_old))
    if not targets:
        return {"ok": False, "error": f"没有以「{head_old}」开头的条目"}
    add_work_prefix(new)  # 先登记白名单，改名后 display 才能正确剥离
    ok = 0
    for k in targets:
        r = rename_entry(k, head_new + k[len(head_old):], kind)
        if r.get("ok"):
            ok += 1
    return {"ok": True, "renamed": ok, "total": len(targets)}


def remove_entry(name: str, kind: str = "anime") -> dict:
    path = REAL_DB if kind == "real" else ANIME_DB
    db = load_db(path)
    if name not in db:
        return {"ok": False, "error": f"条目不存在：{name}"}
    n = len(db[name])
    del db[name]
    save_db(path, db)
    import shutil
    d = os.path.dirname(_thumb_file(kind, name, 0))
    shutil.rmtree(d, ignore_errors=True)
    return {"ok": True, "removed_features": n}


def read_thumb(kind: str, name: str, idx):
    """读取某特征的缩略图字节；idx 传 "cover" 读代表图；无图返回 None。"""
    if str(idx) == "cover":
        p = _cover_file(kind, name)
    else:
        # 兼容旧链接：优先历史逐特征图，缺失则回落到代表图
        p = _thumb_file(kind, name, int(idx))
        if not os.path.exists(p):
            p = _cover_file(kind, name)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "rb") as fh:
            return fh.read()
    except Exception:
        return None


def remove_feature(name: str, idx: int, kind: str = "anime") -> dict:
    """删除某条目的第 idx 张特征（连同缩略图），其余特征前移补位。

    用于清理误注册/低质特征 —— 类内过散（如最小相似度 0.388 的条目）
    往往就是一两张错图拖累的。
    """
    path = REAL_DB if kind == "real" else ANIME_DB
    db = load_db(path)
    lst = db.get(name)
    if lst is None:
        return {"ok": False, "error": f"条目不存在：{name}"}
    idx = int(idx)
    if idx < 0 or idx >= len(lst):
        return {"ok": False, "error": f"索引越界：{idx}（共 {len(lst)} 张）"}
    lst.pop(idx)
    removed_entry = False
    if not lst:
        del db[name]
        removed_entry = True
    save_db(path, db)
    # 代表图与索引无关，无需重排：只清掉可能残留的历史逐特征图
    try:
        d = os.path.dirname(_cover_file(kind, name))
        gone = _thumb_file(kind, name, idx)
        if os.path.exists(gone):
            os.remove(gone)
        if removed_entry and os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
    except Exception:
        pass
    return {"ok": True, "count": len(lst), "removed_entry": removed_entry}

IN_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IN_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

_sessions: dict = {}
_face_app = None


def _ort(path: str) -> ort.InferenceSession:
    if path not in _sessions:
        _sessions[path] = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return _sessions[path]


def _face_analysis():
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name="buffalo_l",
            root=MODELS_ROOT,
            allowed_modules=["detection", "recognition"],
            det_size=(640, 640),
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_thresh=REAL_DET_THR)
        _face_app = app
    return _face_app


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _nms(boxes, scores, thr=ANIME_IOU_THR):
    idx = np.argsort(scores)[::-1]
    keep = []
    while len(idx) > 0:
        i = idx[0]
        keep.append(i)
        if len(idx) == 1:
            break
        rest = idx[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        a1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a2 = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (a1 + a2 - inter + 1e-9)
        idx = rest[iou < thr]
    return keep


def classify_anime_real(img) -> tuple[str, float, float]:
    """判断图片类型，返回 (anime|real, anime_score, real_score)。小图先放大会更准。"""
    sess = _ort(f"{ANIME_MODELS}/anime_real_cls.onnx")
    if max(img.shape[:2]) < 512:
        s = 512.0 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    im = cv2.resize(img, (384, 384), interpolation=cv2.INTER_CUBIC)[:, :, ::-1].astype(np.float32) / 255.0
    im = (im - IN_MEAN) / IN_STD
    out = sess.run(None, {"input": np.transpose(im, (2, 0, 1))[None]})[0][0]
    a, r = float(out[0]), float(out[1])
    return ("anime" if a >= r else "real"), a, r


def detect_anime(img) -> list[tuple[tuple[int, int, int, int], float]]:
    """动漫脸检测，返回 [((x1,y1,x2,y2), score), ...]。小图会自动放大以提高小脸检出率。"""
    sess = _ort(f"{ANIME_MODELS}/anime_face_detect.onnx")
    h0, w0 = img.shape[:2]
    up = 1.0
    if max(h0, w0) < 800:
        up = 800.0 / max(h0, w0)
        img = cv2.resize(img, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    h, w = img.shape[:2]
    im = cv2.resize(img, (640, 640))[:, :, ::-1].astype(np.float32) / 255.0
    out = sess.run(None, {"images": np.transpose(im, (2, 0, 1))[None]})[0]
    pred = out[0].T  # (8400, 5): cx, cy, w, h, logit
    conf = _sigmoid(pred[:, 4])
    m = conf > ANIME_CONF_THR
    if not m.any():
        return []
    xywh, sc = pred[m, :4], conf[m]
    boxes = np.stack(
        [xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
         xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2],
        axis=1,
    )
    keep = _nms(boxes, sc)
    scale = np.array([w, h, w, h]) / up
    res = []
    for i in keep:
        b = boxes[i] / 640.0 * scale
        x1, y1, x2, y2 = (max(0, int(b[0])), max(0, int(b[1])), min(w0, int(b[2])), min(h0, int(b[3])))
        if x2 > x1 and y2 > y1:
            res.append(((x1, y1, x2, y2), float(sc[i])))
    return res


def _crop(img, box, pad=0.25):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return img[max(0, int(y1 - h * pad)):min(img.shape[0], int(y2 + h * pad)),
               max(0, int(x1 - w * pad)):min(img.shape[1], int(x2 + w * pad))]


def anime_embed(face_img) -> np.ndarray:
    """CCIP 特征（768 维，已 L2 归一化）。"""
    sess = _ort(f"{ANIME_MODELS}/ccip_feat.onnx")
    im = cv2.resize(face_img, (384, 384), interpolation=cv2.INTER_CUBIC)[:, :, ::-1].astype(np.float32) / 255.0
    im = (im - CLIP_MEAN) / CLIP_STD
    e = sess.run(None, {"input": np.transpose(im, (2, 0, 1))[None]})[0][0]
    return e / (np.linalg.norm(e) + 1e-12)


def detect_real(img) -> list[tuple[tuple[int, int, int, int], float, np.ndarray]]:
    """真人脸检测 + ArcFace 特征，返回 [((x1,y1,x2,y2), score, emb), ...]。"""
    app = _face_analysis()
    faces = app.get(img)
    res = []
    for f in faces:
        x1, y1, x2, y2 = (max(0, int(v)) for v in f.bbox)
        emb = np.asarray(f.embedding, dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        res.append(((x1, y1, x2, y2), float(f.det_score), emb))
    return res


# 库 JSON 达 63MB，若每次识别/注册都完整解析，实测有约 1s 的固定开销。
# 改为 mtime 缓存：face_server 是唯一写入方，写后同步缓存。
_DB_CACHE: dict = {}


def load_db(path: str) -> dict:
    try:
        mt = os.path.getmtime(path)
    except Exception:
        mt = 0
    c = _DB_CACHE.get(path)
    if c and c[0] == mt:
        return c[1]
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data = data if isinstance(data, dict) else {}
    except Exception:
        data = {}
    _DB_CACHE[path] = (mt, data)
    return data


def save_db(path: str, db: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False)
    try:
        _DB_CACHE[path] = (os.path.getmtime(path), db)
    except Exception:
        pass


UPLOAD_ROOT = os.environ.get("FACE_UPLOAD_ROOT", "/var/lib/docker/nekro_agent_data/uploads")
_IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def latest_upload_image() -> Optional[str]:
    """uploads 目录下最近收到的图片。"""
    best: Optional[str] = None
    best_m = 0.0
    try:
        for dirpath, _dirs, files in os.walk(UPLOAD_ROOT):
            for f in files:
                if f.lower().endswith(_IMG_EXT):
                    p = os.path.join(dirpath, f)
                    m = os.path.getmtime(p)
                    if m > best_m:
                        best, best_m = p, m
    except Exception:  # noqa: BLE001
        return None
    return best


def _latest_in_dir(d: str) -> Optional[str]:
    """指定目录下最近修改的图片。"""
    best, best_m = None, 0.0
    try:
        for f in os.listdir(d):
            if f.lower().endswith(_IMG_EXT):
                fp = os.path.join(d, f)
                if not os.path.isfile(fp):
                    continue
                m = os.path.getmtime(fp)
                if m > best_m:
                    best, best_m = fp, m
    except Exception:  # noqa: BLE001
        return None
    return best


def resolve_image_path(raw: str, chat_key: str = "") -> Optional[str]:
    """把各种视角的图片路径解析为主容器内的真实路径。

    解析顺序：
    1. 主容器真实路径（直接存在）
    2. 纯文件名 / 沙盒视角路径（如 /app/uploads/xxx.jpg）→ 按文件名在 uploads 下查找，
       **优先该 chat_key 目录**，避免多群同名文件取错
    3. 兜底：该 chat_key 目录下最近的图片 → 全局最近的图片
    """
    p = (raw or "").strip()
    if p and os.path.exists(p):
        return p

    name = os.path.basename(p) if p else ""

    # 2a. 该聊天目录下直接命中
    if name and chat_key:
        cand = os.path.join(UPLOAD_ROOT, chat_key, name)
        if os.path.exists(cand):
            return cand

    # 2b. 全库按文件名找（同群优先）
    if name:
        hits: list[str] = []
        try:
            for dirpath, _dirs, files in os.walk(UPLOAD_ROOT):
                if name in files:
                    hits.append(os.path.join(dirpath, name))
        except Exception:  # noqa: BLE001
            hits = []
        if hits:
            if chat_key:
                for h in hits:
                    if os.sep + chat_key + os.sep in h:
                        return h
            return max(hits, key=os.path.getmtime)

    # 3. 兜底：该群最近图 → 全局最近图
    if chat_key:
        local = _latest_in_dir(os.path.join(UPLOAD_ROOT, chat_key))
        if local:
            return local
    return latest_upload_image()


# 打分聚合权重：score = BLEND_W * max + (1-BLEND_W) * 质心。
# 经留出法 + 嵌套交叉验证（阈值也在验证集上选）证实：
# 纯 max 在「误判<=1%」约束下只能认出 13.5%，0.6 权重可到 32%（约 2.4 倍），
# 且高置信区间误判不增加。置 1.0 可退回纯 max 的旧行为。
BLEND_W = 0.6

# 分级置信度：与其追求 100% 精准（实测不可达），不如如实分层，把不确定性交给 LLM。
#
# 笃定阈值 CONF_CERTAIN 的取值依据（生产库 520 角色 / 220 次真实查询）：
#   分数区间      次数    top-1 正确率
#   [0.78,0.82)   27      77.8%   ← 明显不该笃定
#   [0.82,0.86)   34     100.0%   ← 从 0.82 起就可靠了
#   [0.86,0.90)   33      97.0%
#   [0.90,0.94)   45     100.0%
#   累积 ≥0.82: 155 次(70%覆盖) 错误仅 1 次(0.6%)
#   累积 ≥0.86: 121 次(55%覆盖) 错误仅 1 次(0.8%)
# 且多随机划分验证 5 轮，≥0.82 错误率 0.0%~1.4%，≥0.86 恒为 0%。
# → 取 0.82：笃定覆盖率明显更高(70% vs 55%)，错误率同样极低。
CONF_CERTAIN = 0.82  # 单独的绝对分线（仅作参考；实际笃定还要求 gap 达标，见 GAP_CERTAIN）
CONF_MID = 0.78      # ≥ 此分：主候选 + 次候选（"大概率是X，也可能是Y"）
CONF_LOW = 0.68      # ≥ 此分：给 3 个候选并声明不确定
MAX_CANDIDATES = 3   # 最多列几个候选（再多收益递减且干扰模型）

# 「与次名的差距」门控 —— 光看绝对分数不足以判断该不该笃定。
#
# 实测依据（21 条真实人工标注样本，使用本文件的真实打分 _match_topk = 0.6*max+0.4*质心）：
#   规则                          笃定数  正确  错误   笃定正确率  覆盖率
#   仅 s1 ≥ 0.82                    10     7     3      70%       48%
#   s1 ≥ 0.82 且 gap ≥ 0.06          6     6     0     100%       29%
#   s1 ≥ 0.78 且 gap ≥ 0.06          9     9     0     100%       43%   ← 采用
#   s1 ≥ 0.78 且 gap ≥ 0.08          8     8     0     100%       38%
#
# 三个「自信认错」样本的 gap 分别只有 0.004 / 0.023 / 0.026 —— 都是两名几乎并列，
# 加 gap 约束后全部被降级为「不确定」；而覆盖率仅从 48% 降到 43%，还多认出 2 条。
# 代价：1 条正确样本（0.853 gap=0.038 的达妮娅）会被降级为 mid，可接受。
#
# 注意 gap 用【原始 topk 的第 2 名】，不要求次名达到阈值 ——
# 否则「0.803 vs 0.721」这类（次名没过阈值但差距明确）会被误判为不安全。
GAP_CERTAIN = 0.06


def _is_certain(top_s: float, gap: float) -> bool:
    """是否可判定为「笃定」。

    两个条件同时满足：
      - 绝对相似度 ≥ CONF_MID（0.78）—— 太低的分数谈不上笃定
      - 与次名的差距 ≥ GAP_CERTAIN（0.06）—— 两名接近时即使分高也不该笃定

    实测（21 条标注样本）：该规则把笃定正确率从 70% 提到 100%，
    覆盖率仅从 48% 降到 43%，且多认出 2 条。
    """
    return top_s >= CONF_MID and gap >= GAP_CERTAIN


def _match(emb, db: dict, blend_w: float = BLEND_W) -> tuple[str, float, str, float]:
    """返回库中最接近的两名 (名字, 相似度, 第二名名字, 第二名相似度)。

    按角色聚合：score = blend_w * 最高单特征相似度 + (1-blend_w) * 与质心的相似度。
    质心项能稳定「特征数多、质量参差」的角色（避免单张噪声特征拉高或漏判）。
    """
    scores = _match_topk(emb, db, k=2, blend_w=blend_w)
    if not scores:
        return "未知", 0.0, "", 0.0
    s1, n1 = scores[0]
    s2, n2 = scores[1] if len(scores) > 1 else (0.0, "")
    return n1, s1, n2, s2


def _match_topk(emb, db: dict, k: int = 3,
                blend_w: float = BLEND_W) -> list[tuple[float, str]]:
    """返回库中得分最高的 k 个 (分数, 名字)，已按分数降序。"""
    scores: list[tuple[float, str]] = []
    w = float(blend_w)
    for name, embs in db.items():
        if not embs:
            continue
        arr = np.asarray(embs, dtype=np.float32)
        best = float(np.max(emb @ arr.T))
        if w < 1.0 and len(arr) > 1:
            cent = arr.mean(axis=0)
            cent = cent / (np.linalg.norm(cent) + 1e-12)
            best = w * best + (1.0 - w) * float(emb @ cent)
        scores.append((best, name))
    scores.sort(reverse=True)
    return scores[:max(1, k)]
    if not scores:
        return "未知", 0.0, "", 0.0
    s1, n1 = scores[0]
    s2, n2 = scores[1] if len(scores) > 1 else (0.0, "")
    return n1, s1, n2, s2


def register(image_path: str, name: str, mode: str = "auto", chat_key: str = "",
             bbox=None, no_fallback: bool = False) -> str:
    """注册人脸。bbox=(x1,y1,x2,y2) 时注册【指定的那张脸】（合照逐脸注册用），
    未指定则取最大脸（旧行为）。

    no_fallback=True：anime 模式检不出动漫脸时【不】兜底注册进真人库
    （审核管线用 —— 否则动漫图会被静默塞进真人库，产出无识别价值的脏特征）。"""
    path = resolve_image_path(image_path, chat_key)
    if not path:
        return f"找不到图片：{image_path}"
    img = cv2.imread(path)
    if img is None:
        return f"无法读取图片：{path}"

    kind = mode
    if mode == "auto":
        kind, a, r = classify_anime_real(img)
        detail = f"，判别 动漫{a:.2f}/真人{r:.2f}"
    else:
        detail = ""

    # 统一的收尾：写入特征 + 保存裁剪缩略图
    # ⚠️ kind 必须是机器名（anime/real），缩略图目录、library_overview、
    #    read_thumb 都按机器名寻址；kdisp 仅用于给人看的返回文案。
    def _commit(db_path: str, emb_vec, crop_img, kind: str, kdisp: str, dim_note: str) -> str:
        db = load_db(db_path)
        lst = db.setdefault(name, [])
        lst.append([float(x) for x in emb_vec])
        save_db(db_path, db)
        _save_cover(kind, name, crop_img)
        return f"已注册: {name}（{kdisp}库, 第 {len(lst)} 张特征, 维度 {dim_note}{detail}）"

    def _nearest(dets, bbox):
        """取中心离 bbox 最近的一个检测框 —— 指定脸注册的选脸依据。"""
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

        def dist(d):
            b = d[0]
            return ((b[0] + b[2]) / 2 - cx) ** 2 + ((b[1] + b[3]) / 2 - cy) ** 2
        return min(dets, key=dist)

    # ── 指定脸注册（合照逐脸）：只认 bbox，不做跨类型兜底 ──
    if bbox and len(bbox) == 4:
        if kind == "anime":
            dets = detect_anime(img)
            box = _nearest(dets, bbox)[0] if dets else list(bbox)
            crop = _crop(img, box)
            emb = anime_embed(crop)
            return _commit(ANIME_DB, emb, crop, "anime", "动漫", str(len(emb)))
        real = detect_real(img)
        if not real:
            return "未检测到人脸（指定区域附近无人脸，无法注册）"
        box, score, emb = _nearest(real, bbox)
        return _commit(REAL_DB, emb, _crop(img, box), "real", "真人", "512")

    if kind == "anime":
        dets = detect_anime(img)
        if not dets:
            # 兜底：尝试真人（no_fallback=True 时直接拒绝，见 docstring）
            real = [] if no_fallback else detect_real(img)
            if real:
                box, score, emb = max(real, key=lambda d: (d[0][2] - d[0][0]) * (d[0][3] - d[0][1]))
                return _commit(REAL_DB, emb, _crop(img, box), "real", "真人", "512")
            if no_fallback:
                return "未检测到动漫人脸（已按你的要求不落入真人库）"
            return "未检测到人脸。请上传清晰的正脸图片（动漫头像或真人照片均可）"
        (box, score) = max(dets, key=lambda d: (d[0][2] - d[0][0]) * (d[0][3] - d[0][1]))
        crop = _crop(img, box)
        emb = anime_embed(crop)
        return _commit(ANIME_DB, emb, crop, "anime", "动漫", str(len(emb)))

    real = detect_real(img)
    if not real:
        dets = detect_anime(img)
        if dets:
            (box, score) = max(dets, key=lambda d: (d[0][2] - d[0][0]) * (d[0][3] - d[0][1]))
            crop = _crop(img, box)
            emb = anime_embed(crop)
            return _commit(ANIME_DB, emb, crop, "anime", "动漫", str(len(emb)))
        return "未检测到人脸。请上传清晰的正脸图片（动漫头像或真人照片均可）"
    box, score, emb = max(real, key=lambda d: (d[0][2] - d[0][0]) * (d[0][3] - d[0][1]))
    return _commit(REAL_DB, emb, _crop(img, box), "real", "真人", "512")


def identify(
    image_path: str,
    real_thr: float = REAL_THRESHOLD,
    anime_thr: float = ANIME_THRESHOLD,
    chat_key: str = "",
    detail: bool = False,
) -> str:
    """识别图片中的人脸（只返回文本，向后兼容）。

    输出分两种模式：
    - detail=False（默认，给 LLM 用）：只返回「事实」——认出了谁、有多确定、是否易混。
      不带阈值、不指导调参、不写"建议"，把怎么说话完全留给角色人格去组织。
    - detail=True（给 WebUI / 命令行排查用）：附阈值、检测统计、调参建议等完整信息。
    """
    text, _meta = identify_ex(image_path, real_thr, anime_thr, chat_key, detail)
    return text


def identify_ex(
    image_path: str,
    real_thr: float = REAL_THRESHOLD,
    anime_thr: float = ANIME_THRESHOLD,
    chat_key: str = "",
    detail: bool = False,
) -> tuple[str, dict]:
    """识别图片中的人脸，并额外返回结构化 meta。

    meta 供「待审核队列」使用（记录候选与分数，便于人工复核补图）：
        {
          "path": 解析后的真实路径, "chat_key": 来源会话,
          "confidence": certain|mid|low|unknown|noface|error,
          "recognized": [展示名, ...],
          "faces": [{name, raw_name, sim, close, candidates:[{name, raw_name, sim}]}],
          "thresholds": {"anime":..., "real":...},
        }
    文本部分与 identify() 完全一致（detail=False 时供 LLM 使用）。
    """
    path = resolve_image_path(image_path, chat_key)
    if not path:
        return f"找不到图片：{image_path}", {
            "confidence": "error", "error": "not_found", "image_arg": image_path,
            "chat_key": chat_key, "recognized": [], "faces": [],
        }
    img = cv2.imread(path)
    if img is None:
        return f"无法读取图片：{path}", {
            "confidence": "error", "error": "unreadable", "path": path,
            "chat_key": chat_key, "recognized": [], "faces": [],
        }

    kind, a_score, r_score = classify_anime_real(img)
    # 每块: (命中数, 标题, [entry], 库是否非空)
    # entry = dict(name, sim, cands=[(分数, 名字), ...], close, cand2, sim2)
    blocks: list = []
    CLOSE_GAP = 0.06  # 第一、二名差距小于此值 → 视为擦边易混

    # ── 路径选择 ─────────────────────────────────────────────
    # 优先走判定的那条；若该路径「颗粒无收」（0 命中且全未知），
    # 再跑另一条兜底 —— 分类器实测会把部分动漫图误判为 real
    # （案例：双人动漫图 a=0.000/r=1.000，走真人路径只得 sim=0.058 的垃圾框，
    #   而动漫路径本可 0.94/0.82 完美认出）。两条都空才算真的没检出。
    _primary = "anime" if a_score >= r_score else "real"
    _fallback = "real" if _primary == "anime" else "anime"

    def _run_anime() -> None:
        dets = detect_anime(img)
        if not dets:
            return
        db = load_db(ANIME_DB)
        entries, hits = [], 0
        for box, _score in sorted(dets, key=lambda d: d[1], reverse=True):
            emb = anime_embed(_crop(img, box))
            topk = _match_topk(emb, db, k=MAX_CANDIDATES) if db else []
            cand, sim = (topk[0][1], topk[0][0]) if topk else ("未知", 0.0)
            cand2, sim2 = (topk[1][1], topk[1][0]) if len(topk) > 1 else ("", 0.0)
            name = cand if sim >= anime_thr else "未知"
            if name != "未知":
                hits += 1
            close = bool(name != "未知" and sim2 >= anime_thr and (sim - sim2) < CLOSE_GAP)
            entries.append({"name": name, "sim": sim, "cands": topk,
                            "cand2": cand2, "sim2": sim2, "close": close,
                            # top1 = 原始最高分候选（无论是否达阈值）。
                            # name 会被阈值压平成「未知」，若展示层直接用 name，
                            # 就会出现「[1] 未知 (相似度 0.778)」这种丢信息的输出。
                            "top1": display_name(cand) if cand != "未知" else "未知",
                            "bbox": [int(v) for v in box],
                            # 该脸自身是否笃定（多脸时 confidence 要看全部脸）
                            "_certain": _certain_of(name, topk, anime_thr)})
        blocks.append((hits, f"动漫头像（判别 动漫{a_score:.2f}/真人{r_score:.2f}）：检测到 {len(dets)} 张脸", entries, bool(db)))

    def _run_real() -> None:
        real = detect_real(img)
        if not real:
            return
        db = load_db(REAL_DB)
        entries, hits = [], 0
        for box, _score, emb in sorted(real, key=lambda d: d[1], reverse=True):
            topk = _match_topk(emb, db, k=MAX_CANDIDATES) if db else []
            cand, sim = (topk[0][1], topk[0][0]) if topk else ("未知", 0.0)
            cand2, sim2 = (topk[1][1], topk[1][0]) if len(topk) > 1 else ("", 0.0)
            name = cand if sim >= real_thr else "未知"
            if name != "未知":
                hits += 1
            close = bool(name != "未知" and sim2 >= real_thr and (sim - sim2) < CLOSE_GAP)
            entries.append({"name": name, "sim": sim, "cands": topk,
                            "cand2": cand2, "sim2": sim2, "close": close,
                            "top1": display_name(cand) if cand != "未知" else "未知",
                            "bbox": [int(v) for v in box],
                            "_certain": _certain_of(name, topk, real_thr)})
        blocks.append((hits, f"真人照片（判别 动漫{a_score:.2f}/真人{r_score:.2f}）：检测到 {len(real)} 张人脸", entries, bool(db)))

    # 主路径 + 兜底：主路径颗粒无收时才跑另一条。
    # 兜底的意义：分类器误判时（动漫图被判 real），真人路径只会给出
    # sim≈0.06 的噪音框 —— 此时必须【丢弃】这些噪音，改用另一条的结果，
    # 否则垃圾 entry 会与正确结果混在同一份 faces 里，
    # 并把「全部脸笃定才 certain」的判定拖垮。
    def _blocks_useful() -> bool:
        """blocks 里是否存在至少一张「达阈值」的脸。"""
        return any(h > 0 for h, _t, _e, _d in blocks)

    if _primary == "anime":
        _run_anime()
        if not _blocks_useful():
            _saved = list(blocks)
            blocks.clear()
            _run_real()
            if not _blocks_useful():
                blocks.clear()
                blocks.extend(_saved)  # 双路径都没匹配上 → 保留主路径结果（脸是真实存在的，别谎报 noface）
    else:
        _run_real()
        if not _blocks_useful():
            _saved = list(blocks)
            blocks.clear()
            _run_anime()
            if not _blocks_useful():
                blocks.clear()
                blocks.extend(_saved)

    # ── 方位标注（整组一起算）────────────────────────────────
    # 在主/兜底路径定型后执行：排布是组属性，必须等最终的脸集合确定
    _pos_mode = _assign_pos(
        [e for _h, _t, _es, _d in blocks for e in _es], img.shape)

    if not blocks:
        meta_noface = {
            "path": path, "chat_key": chat_key, "confidence": "noface",
            "recognized": [], "faces": [], "kind": kind,
            "anime_score": round(a_score, 4), "real_score": round(r_score, 4),
            "thresholds": {"anime": anime_thr, "real": real_thr},
        }
        if detail:
            h, w = img.shape[:2]
            short = min(h, w)
            tips = ["换一张清晰的正脸 / 半侧脸图（避免极端侧脸）"]
            if short < 200:
                tips.append(f"图片较小（{w}x{h}），可换分辨率更高的原图")
            if max(h, w) / max(1, short) > 2.2:
                tips.append("画面过于狭长，可能是局部裁剪，建议发完整头像")
            return "图中未检测到人脸（真人、动漫都没检出）。当前图片可能太小、过于模糊，或面部被头发/遮挡物大面积覆盖。建议：" + "；".join(tips) + "。", meta_noface
        return "没有检测到人脸。", meta_noface

    hit_blocks = [b for b in blocks if b[0] > 0]

    # 汇总结构化人脸信息（供待审核队列用）
    _faces_meta = []
    for _h, _header, entries, _has_db in blocks:
        for e in entries:
            _faces_meta.append({
                "name": display_name(e["name"]),
                "raw_name": e["name"],
                # top1 = 原始最高分候选（不论是否达阈值）。
                # name 会被阈值压平成「未知」；展示层应优先用 top1，
                # 否则会出现「最像的候选是未知」这种无用信息。
                "top1": e.get("top1") or display_name(e["name"]),
                "matched": e["name"] != "未知",
                "sim": round(float(e["sim"]), 4),
                "close": bool(e.get("close")),
                # 位置：合照里用于说清「谁在哪儿」
                "pos": e.get("pos") or "",
                "bbox": e.get("bbox") or [],
                # 该脸自身是否笃定（多脸时 confidence 需要看全部脸）
                "certain": bool(e.get("_certain")),
                "candidates": [
                    {"name": display_name(n), "raw_name": n, "sim": round(float(s), 4)}
                    for s, n in (e.get("cands") or [])
                ],
            })

    def _meta(conf: str, recognized: list[str]) -> dict:
        return {
            "path": path, "chat_key": chat_key, "confidence": conf,
            "recognized": recognized, "faces": _faces_meta, "kind": kind,
            "anime_score": round(a_score, 4), "real_score": round(r_score, 4),
            "thresholds": {"anime": anime_thr, "real": real_thr},
        }

    # ---------- 简洁模式：只给事实，措辞交给角色人格 ----------
    if not detail:
        # 收集所有检测块的候选（取每块主候选，按分数排序）
        all_entries: list[dict] = []
        for _h, _header, entries, _has_db in blocks:
            all_entries.extend(entries)
        if not all_entries:
            return "没有检测到人脸。", _meta("noface", [])

        # 多脸场景：收集所有已认出的脸（不同角色），而不仅取最高分的一张
        recognized: list[dict] = []    # [{dn, sim, entry}]
        for e in all_entries:
            cands: list[tuple[float, str]] = e.get("cands") or []
            if not cands:
                continue
            s, n = cands[0]
            if s >= anime_thr:
                dn = display_name(n)
                # 去重（同人可能出现在多个检测块中）
                if dn not in [r["dn"] for r in recognized]:
                    recognized.append({"dn": dn, "sim": s, "entry": e})
        if not recognized:
            # 没有任何一张脸被认出 → 用最高分的那张给候选建议
            best = max(all_entries, key=lambda e2: e2["sim"])
            cands: list[tuple[float, str]] = best.get("cands") or []
            if not cands:
                return "没有检测到人脸。", _meta("noface", [])
            top_s, top_n = cands[0]
            dn = display_name(top_n)
            if top_s >= CONF_LOW:
                others = [display_name(n) for s, n in cands[1:]
                          if s >= CONF_LOW - 0.06 and n != top_n][:MAX_CANDIDATES - 1]
                if others:
                    joined = "」、「".join(others)
                    return (
                        f"没能确定是谁。看着最像「{dn}」，但也可能是「{joined}」。"
                        f"请如实说你不确定，可以猜一两个名字，别说得斩钉截铁。"
                    ), _meta("low", [])
            return (
                f"没能认出是谁（最像的「{dn}」也只有很低的相似度，不可信）。"
                f"请如实说认不出来，不要硬猜名字。"
            ), _meta("unknown", [])

        # ── 按位置排（上→下、左→右）───────────────────────────────
        # 为什么不能按相似度排：合照里 LLM 只拿到一串名字，无法对应到图上位置，
        # 角色想说「左边那位是谁」就无从判断。按位置排 + 标注方位后，
        # 才可能讲清「上方是 A，下方是 B」。
        recognized.sort(key=lambda r: (
            (r["entry"].get("bbox") or [0, 10 ** 9])[1],   # y1（上→下）
            (r["entry"].get("bbox") or [10 ** 9, 0])[0],   # x1（左→右）
        ))
        ordered = [r["dn"] for r in recognized]
        labels = {r["dn"]: (r["entry"].get("pos") or "") for r in recognized}

        # 自然方位文本：一排/一列用「从左到右/从上到下」，散布用九宫格方位
        def _seq(names):
            return "、".join(f"「{x}」" for x in names)

        def _pair_txt(a, b):
            if _pos_mode == "row":
                return f"左边是「{a}」，右边是「{b}」"
            if _pos_mode == "col":
                return f"上面是「{a}」，下面是「{b}」"
            la, lb = labels.get(a, ""), labels.get(b, "")
            return f"{la}是「{a}」，{lb}是「{b}」"

        def _members(names):
            if _pos_mode == "row":
                return f"从左到右依次是：{_seq(names)}"
            if _pos_mode == "col":
                return f"从上到下依次是：{_seq(names)}"
            return "、".join(f"{labels.get(x, '')}是「{x}」" for x in names)

        # ── confidence：必须【全部脸都笃定】才算 certain ────────────
        # （不再只看「已认出」的脸 —— 图里没认出的脸同样算数）
        _uncertain = [e for e in all_entries if not e.get("_certain")]
        _n_uncertain = len(_uncertain)
        if _n_uncertain == 0:
            _conf = "certain"
        elif ordered:
            _conf = "mid" if max(r["sim"] for r in recognized) >= CONF_MID else "low"
        else:
            _conf = "low"

        def _uncertain_hint() -> str:
            """把未笃定的脸归纳成一句提示（带方位，便于人工去审核队列处理）。"""
            if not _uncertain:
                return ""
            us = []
            for e in _uncertain[:3]:
                t1 = e.get("top1") or ""
                us.append(f"{e.get('pos') or '某处'}最像「{t1}」({float(e['sim']):.2f})")
            more = f" 等 {len(_uncertain)} 张" if len(_uncertain) > 3 else ""
            return f"另有 {_n_uncertain} 张脸未确信：{'、'.join(us)}{more}"

        # 超过 MAX_GROUP_REPORT 位时不逐个列名 ——
        # 实测 36 脸合照列出 34 个名字，文本近千字，LLM 根本无法消化。
        MAX_GROUP_REPORT = 6

        # ── 输出 ────────────────────────────────────────────────
        if _conf == "certain":
            if len(ordered) == 1:
                return f"认出来了，是「{ordered[0]}」。", _meta("certain", ordered)
            if len(ordered) == 2:
                return f"认出来了：{_pair_txt(ordered[0], ordered[1])}。", _meta("certain", ordered)
            if len(ordered) <= MAX_GROUP_REPORT:
                return f"认出来了，共 {len(ordered)} 位，{_members(ordered)}。", _meta("certain", ordered)
            return (
                f"认出来了，共 {len(ordered)} 位（{_members(ordered[:MAX_GROUP_REPORT])} 等）。"
                f"人数较多，不必逐个指认；若要提某位，按上述方位/顺序说即可。"
            ), _meta("certain", ordered)

        # 有脸未笃定 → 如实交代，不说得斩钉截铁
        hint = _uncertain_hint()
        parts = [f"{labels.get(x, '')}是「{x}」" if labels.get(x, "") else f"「{x}」"
                 for x in ordered]
        if len(ordered) > MAX_GROUP_REPORT:
            if _pos_mode in ("row", "col"):
                shown = _seq(ordered[:MAX_GROUP_REPORT])
                return (
                    f"能认出的有 {shown} 等 {len(ordered)} 位（按画面位置排序）；{hint}。"
                    f"请如实区分：确定的说确定，没把握的就说没把握，不要混为一谈。"
                ), _meta(_conf, ordered)
            shown = "、".join(parts[:MAX_GROUP_REPORT])
            return (
                f"能认出的有 {shown} 等 {len(ordered)} 位；{hint}。"
                f"请如实区分：确定的说确定，没把握的就说没把握，不要混为一谈。"
            ), _meta(_conf, ordered)
        if _pos_mode in ("row", "col"):
            return (
                f"能认出的有（按画面位置排序）{_seq(ordered)}；{hint}。"
                f"请如实区分：确定的说确定，没把握的就说没把握，不要混为一谈。"
            ), _meta(_conf, ordered)
        return (
            f"能认出的有 {'、'.join(parts)}；{hint}。"
            f"请如实区分：确定的说确定，没把握的就说没把握，不要混为一谈。"
        ), _meta(_conf, ordered)


    # ---------- 详细模式：附阈值与调参建议（WebUI / 排查） ----------
    shown = hit_blocks or blocks
    lines = []
    for _hits, header, entries, has_db in shown:
        if lines:
            lines.append("")
        lines.append(header)
        if not has_db:
            lines.append("  （该库为空，可让用户发「注册人脸 名字」+ 图片来注册）")
        for i, e in enumerate(entries, 1):
            # 用 top1（原始最高分候选），而不是被阈值压平的 name ——
            # 否则未达阈值时会显示「[1] 未知 (相似度 0.778)」，丢掉「最像谁」
            if e["name"] == "未知" and e.get("top1") not in ("未知", ""):
                lines.append(f"  [{i}] 未达阈值（最像 {e['top1']}，相似度 {e['sim']:.3f}）")
            else:
                lines.append(f"  [{i}] {display_name(e['name'])} (相似度 {e['sim']:.3f})")
            cands = e.get("cands") or []
            if len(cands) > 1:
                rest = "、".join(f"{display_name(n)}({s:.3f})" for s, n in cands[1:])
                lines.append(f"      次候选: {rest}")
            if e["close"]:
                lines.append(f"      ⚠️ 与「{display_name(e['cand2'])}」({e['sim2']:.3f}) 非常接近，可能混淆——建议给这两个角色各补几张图区分")
    if not hit_blocks:
        # 给出最接近的候选，便于判断阈值是否需要调
        # 注意：不能用 e["name"]（会被阈值压平成「未知」），要用原始 top1
        best_name, best_sim = "无", 0.0
        for _h, _header, entries, _has_db in blocks:
            for e in entries:
                if e["sim"] > best_sim:
                    raw = e.get("top1") or e["name"]
                    best_name, best_sim = raw, e["sim"]
        lines.append(
            f"\n（都没认出来。最接近的是「{display_name(best_name)}」相似度 {best_sim:.3f}；"
            f"当前阈值 动漫{anime_thr:.2f} / 真人{real_thr:.2f}，"
            f"若图确实是该角色可在 WebUI 把动漫阈值调低到 {max(0.1, round(best_sim - 0.02, 2)):.2f} 以下）"
        )
        # 按分数分级（与简洁模式一致），而不是一律 unknown ——
        # 否则 WebUI「测试识别」的 low/mid 档会被错记成 unknown，甚至不入待审核队列
        if best_sim >= CONF_LOW:
            return "\n".join(lines), _meta("low", [])
        return "\n".join(lines), _meta("unknown", [])

    # 已有一张以上被认出 → 按主候选分数 + gap 分级（与简洁模式同规则）。
    # ★ 多脸时必须【全部脸都笃定】才算 certain —— 只要有一张没把握，
    #   整图就降级，让未确信的脸能进入待审核队列。
    _all_entries_d = [e for _h, _hd, entries, _db in blocks for e in entries]
    _best = max((e for _hb in hit_blocks for e in _hb[2] if e["name"] != "未知"),
                key=lambda e: e["sim"], default=None)
    _rec = [display_name(e["name"]) for _hb in hit_blocks for e in _hb[2] if e["name"] != "未知"]
    if _best is None:
        return "\n".join(lines), _meta("unknown", [])
    _c = _best.get("cands") or []
    _gap = (_c[0][0] - _c[1][0]) if len(_c) > 1 else _c[0][0] if _c else 0.0
    # 各 entry 的 _certain 已在检测循环里用各自路径的正确阈值算好
    _all_certain = bool(_all_entries_d) and all(bool(e.get("_certain")) for e in _all_entries_d)
    if _all_certain and _is_certain(_best["sim"], _gap):
        return "\n".join(lines), _meta("certain", _rec)
    if _best["sim"] >= CONF_MID:
        return "\n".join(lines), _meta("mid", _rec)
    if _best["sim"] >= CONF_LOW:
        return "\n".join(lines), _meta("low", _rec)
    return "\n".join(lines), _meta("unknown", [])

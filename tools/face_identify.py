#!/usr/bin/env python3
"""识别人脸：face_identify.py <图片路径> [--threshold 0.5] [--anime-threshold 0.78] [--chat-key onebot_v11-group_xxx]

图片路径可省略：此时按 chat_key 找该会话最近收到的图片。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import face_engine  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("image", nargs="?", default="", help="图片路径或文件名（可省略）")
ap.add_argument("--threshold", type=float, default=None, help="真人相似度阈值（默认 0.5）")
ap.add_argument("--anime-threshold", type=float, default=None, help="动漫相似度阈值（默认 0.78）")
ap.add_argument("--chat-key", default="", help="会话标识，用于定位该会话最近图片、避免多群串图")
ap.add_argument("--detail", action="store_true", help="输出完整技术细节（阈值/检测统计/调参建议）")
args = ap.parse_args()

real_thr = args.threshold if args.threshold is not None else face_engine.REAL_THRESHOLD
anime_thr = args.anime_threshold if args.anime_threshold is not None else face_engine.ANIME_THRESHOLD
print(face_engine.identify(args.image, real_thr, anime_thr, args.chat_key, detail=args.detail))

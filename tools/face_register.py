#!/usr/bin/env python3
"""注册人脸：face_register.py <图片路径> <姓名> [--mode auto|anime|real] [--chat-key onebot_v11-group_xxx]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import face_engine  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("name")
ap.add_argument("--mode", default="auto", choices=["auto", "anime", "real"], help="auto=自动判断动漫/真人")
ap.add_argument("--chat-key", default="", help="会话标识，用于定位该会话的图片")
args = ap.parse_args()
print(face_engine.register(args.image, args.name, args.mode, args.chat_key))

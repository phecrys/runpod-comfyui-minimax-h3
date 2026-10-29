#!/usr/bin/env python3
"""ComfyUI-Manager 보안 정책 설정.
--listen 0.0.0.0 (RunPod 프록시) 환경에서는 Manager 가 기본적으로 노드 설치를 막으므로
network_mode=personal_cloud, security_level=normal 로 설정합니다.
환경변수 MANAGER_SECURITY_LEVEL / MANAGER_NETWORK_MODE 로 바꿀 수 있습니다."""
import configparser
import os
import sys
from pathlib import Path

comfy = Path(os.environ.get("COMFYUI_DIR", "/opt/ComfyUI"))
paths = [comfy / "user" / "__manager" / "config.ini",
         comfy / "user" / "default" / "ComfyUI-Manager" / "config.ini"]
want = {"security_level": os.environ.get("MANAGER_SECURITY_LEVEL", "normal"),
        "network_mode": os.environ.get("MANAGER_NETWORK_MODE", "personal_cloud")}
for p in paths:
    if p.parent.name == "ComfyUI-Manager" and not p.parent.exists():
        continue  # 구버전 경로는 이미 있을 때만
    p.parent.mkdir(parents=True, exist_ok=True)
    cfg = configparser.ConfigParser()
    if p.exists():
        cfg.read(p, encoding="utf-8")
    if not cfg.has_section("default"):
        cfg.add_section("default")
    for k, v in want.items():
        cfg.set("default", k, v)
    with open(p, "w", encoding="utf-8") as fh:
        cfg.write(fh)
    print(f"[manager] {p}: " + ", ".join(f"{k}={v}" for k, v in want.items()))
sys.exit(0)

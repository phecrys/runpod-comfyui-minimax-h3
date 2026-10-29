"""Comfy-Org 공식 MiniMax H3 템플릿(MIT)을 workflows/ 로 가져옵니다.

  python3 tools/vendor_workflows.py <workflow_templates 저장소의 templates 폴더>

- 노드 properties 의 "models"(다운로드 링크 메타데이터)는 지웁니다.
  모델은 provision.py 가 받고, 파일명은 옵션(INT6/INT4 등)에 맞게 치환되므로
  링크 메타데이터가 남아 있으면 프런트엔드가 엉뚱한 파일을 '누락'으로 표시할 수 있습니다.
"""
import json
import sys
from pathlib import Path

SRC = {
    "video_minimax_h3_t2v.json": "minimax_h3_t2v.json",
    "video_minimax_h3_i2v.json": "minimax_h3_i2v.json",
    "video_minimax_h3_r2v.json": "minimax_h3_r2v.json",
}


def strip_models(obj):
    if isinstance(obj, dict):
        props = obj.get("properties")
        if isinstance(props, dict):
            props.pop("models", None)
        for v in obj.values():
            strip_models(v)
    elif isinstance(obj, list):
        for v in obj:
            strip_models(v)


def main(src_dir: str) -> None:
    out = Path(__file__).resolve().parents[1] / "workflows"
    out.mkdir(exist_ok=True)
    for src, dst in SRC.items():
        wf = json.loads((Path(src_dir) / src).read_text(encoding="utf-8"))
        wf.pop("models", None)
        strip_models(wf)
        (out / dst).write_text(json.dumps(wf, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{src} → workflows/{dst}")


if __name__ == "__main__":
    main(sys.argv[1])

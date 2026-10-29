import json, os, struct, sys, tempfile, importlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
tmp = Path(tempfile.mkdtemp())
os.environ["WORKSPACE_DIR"] = str(tmp / "ws")
os.environ["COMFYUI_DIR"] = str(tmp / "comfy")
os.environ["CIVITAI_TOKEN"] = "SECRET123"
sys.path.insert(0, str(ROOT / "scripts"))
import provision as P

def st(path, keys):
    hdr = {k: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for k in keys}
    hdr["__metadata__"] = {}
    b = json.dumps(hdr).encode()
    path.write_bytes(struct.pack("<Q", len(b)) + b + b"\0" * 4096)

# refs
assert P.parse_civitai_ref(123) == ("version", 123)
assert P.parse_civitai_ref("https://civitai.com/models/111?modelVersionId=222") == ("version", 222)
assert P.parse_civitai_ref("https://civitai.com/models/111/some-name") == ("model", 111)
assert P.parse_civitai_ref("https://civitai.com/api/download/models/333?type=Model") == ("version", 333)
assert P.parse_civitai_ref("model:5") == ("model", 5)
# AIR
air = "urn:air:krea2:checkpoint:civitai:573152@3112728+2997637"
assert P.parse_civitai_ref(air) == ("version", 3112728)
assert P.parse_air_file_id(air) == 2997637
assert P.parse_civitai_ref("urn:air:krea2:lora:civitai:573152") == ("model", 573152)
assert P.parse_air_file_id("urn:air:krea2:lora:civitai:1@2") is None
assert P.parse_env_models(air + " urn:air:krea2:lora:civitai:5@6 urn:air:krea2:lora:civitai:5@6@loras/k") == [
    {"civitai": air}, {"civitai": "urn:air:krea2:lora:civitai:5@6"},
    {"civitai": "urn:air:krea2:lora:civitai:5@6", "dir": "loras/k"}]
fl = [{"id": 1, "name": "a.safetensors", "type": "Model", "primary": True}, {"id": 2997637, "name": "b.safetensors", "type": "Model"}]
assert P.pick_civitai_file(fl, None, 2997637)["name"] == "b.safetensors"
assert P.pick_civitai_file(fl, None, 999)["name"] == "a.safetensors"
# env models
e = P.parse_env_models("civitai:123@loras https://huggingface.co/a/b/blob/main/x/y.safetensors@vae, 999 https://civitai.com/models/1?modelVersionId=2@loras/krea2")
assert e == [{"civitai": "123", "dir": "loras"},
             {"url": "https://huggingface.co/a/b/blob/main/x/y.safetensors", "dir": "vae"},
             {"civitai": "999"},
             {"civitai": "https://civitai.com/models/1?modelVersionId=2", "dir": "loras/krea2"}], e
# HF url
r = P.resolve_url({"url": "https://huggingface.co/Comfy-Org/Krea-2/blob/main/vae/qwen_image_vae.safetensors", "dir": "vae"})
assert r["hf"] == ("Comfy-Org/Krea-2", "main", "vae/qwen_image_vae.safetensors") and r["name"] == "qwen_image_vae.safetensors"
# file pick
files = [{"name": "a.pt", "type": "Model", "metadata": {"format": "PickleTensor"}},
         {"name": "b_fp8.safetensors", "type": "Model", "primary": True, "metadata": {"format": "SafeTensor"}},
         {"name": "b_bf16.safetensors", "type": "Model", "metadata": {"format": "SafeTensor"}},
         {"name": "train.zip", "type": "Training Data"}]
assert P.pick_civitai_file(files)["name"] == "b_fp8.safetensors"
assert P.pick_civitai_file(files, "bf16")["name"] == "b_bf16.safetensors"
# civitai resolve with mocked API
def fake_get(path):
    if path == "/models/111": return {"modelVersions": [{"id": 777, "name": "v3"}]}
    if path == "/model-versions/777": return {"name": "v3", "baseModel": "Krea 2", "model": {"name": "MyKrea", "type": "Checkpoint"},
        "files": [{"name": "mykrea.safetensors", "type": "Model", "primary": True, "sizeKB": 1000,
                   "downloadUrl": "https://civitai.com/api/download/models/777"}]}
    if path == "/model-versions/888": return {"name": "v1", "model": {"name": "Sty", "type": "LORA"},
        "files": [{"name": "sty.safetensors", "type": "Model", "downloadUrl": "https://civitai.com/api/download/models/888?type=Model&format=SafeTensor"}]}
    raise AssertionError(path)
P.civitai_get = fake_get
it = P.resolve({"civitai": "https://civitai.com/models/111"})
assert it["dir"] == "@auto_checkpoint" and it["url"].endswith("?token=SECRET123") and it["key"] == "civitai:777:"
it2 = P.resolve({"civitai": 888})
assert it2["dir"] == "loras" and it2["url"].endswith("&token=SECRET123")
assert "SECRET123" not in P.redact("url " + it2["url"])
# auto checkpoint
d = tmp / "x"; d.mkdir()
st(d / "dit.safetensors", ["model.diffusion_model.blocks.0.w", "img_in.weight"])
st(d / "full.safetensors", ["model.diffusion_model.a", "first_stage_model.decoder.w", "conditioner.x"])
assert P.auto_checkpoint_dir(d / "dit.safetensors") == "diffusion_models"
assert P.auto_checkpoint_dir(d / "full.safetensors") == "checkpoints"
(d / "bad.safetensors").write_bytes(b"<html>login</html>" * 100)
assert P.safetensors_keys(d / "bad.safetensors") is None

# end-to-end download with fake transport
src = tmp / "srcfile.safetensors"; st(src, ["img_in.weight"])
def fake_aria(url, out_dir, out_name):
    import shutil; shutil.copy(src, out_dir / out_name); return True
P._run_aria2 = fake_aria
it["size"] = src.stat().st_size
dest = P.download(it)
assert dest == P.MODELS_DIR / "diffusion_models" / "mykrea.safetensors", dest
assert json.loads(P.MANIFEST.read_text())["civitai:777:"] == str(dest)
# second time skipped
assert P.download(it) == dest
# provision_models skip-by-manifest + failure counting
calls = []
P.resolve = lambda e: (calls.append(e), (_ for _ in ()).throw(RuntimeError("boom")))[1]
f = P.provision_models([{"civitai": 777}, {"url": "https://x/y.safetensors", "dir": "vae"}])
assert f == 1 and len(calls) == 1, (f, calls)   # 777 skipped via manifest, url failed
# ---------------- MiniMax H3 템플릿 전용 ----------------
importlib.reload(P)
CFG = ROOT / "config" / "models.yaml"

def names(cfg):
    return sorted(P.entry_filename(m) for m in cfg["models"])

def with_env(**kw):
    for k in ("MINIMAX_VARIANT", "MINIMAX_MODES", "MINIMAX_TE", "MINIMAX_VAE", "MINIMAX_TURBO", "MODELS"):
        os.environ.pop(k, None)
    os.environ.update(kw)
    return P.load_config(CFG)

# 기본값: INT8 FL2VA + nvfp4 TE + int8 VAE + audio VAE + 8step LoRA
cfg = with_env()
assert names(cfg) == sorted([
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "minimax_h3_video_vae_int8_convrot.safetensors",
    "minimax_h3_audio_vae_fp32.safetensors",
    "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"]), names(cfg)
assert P.workflow_replacements(cfg["models"]) == {}
assert len(cfg["custom_nodes"]) == 1 and len(cfg["inputs"]) == 3 and cfg["workflows"] == []
fl8 = [m for m in cfg["models"] if "civitai" in m][0]
assert fl8["civitai"] == 3193337 and fl8["file_id"] == 3074134 and fl8["dir"] == "diffusion_models"

# INT4 + 둘 다 + int8 TE + turbo 끔
cfg = with_env(MINIMAX_VARIANT="int4", MINIMAX_MODES="fl2va, ref2va", MINIMAX_TE="int8", MINIMAX_TURBO="0")
n = names(cfg)
assert "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors" in n and "minimax_h3_ref2va_pruned_w4a8_mixed.safetensors" in n
assert not any("int8_convrot" in x and "pruned" in x for x in n) and not any("turbo" in x for x in n), n
assert "qwen3vl_32b_minimax_h3_int8_convrot.safetensors" in n and len(n) == 5, n
rep = P.workflow_replacements(cfg["models"])
assert rep == {"minimax_h3_fl2va_pruned_int8_convrot.safetensors": "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
               "minimax_h3_ref2va_pruned_int8_convrot.safetensors": "minimax_h3_ref2va_pruned_w4a8_mixed.safetensors",
               "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"}, rep
ids = {m["file_id"] for m in cfg["models"] if "civitai" in m}
assert ids == {3089623, 3091888}, ids

# INT6 → 같은 버전의 두 파일을 file_id 로 구분
cfg = with_env(MINIMAX_VARIANT="INT6", MINIMAX_MODES="all")
civ = [m for m in cfg["models"] if "civitai" in m]
assert [(m["civitai"], m["file_id"]) for m in civ] == [(3369568, 3257813), (3369568, 3258013)]
assert P.civitai_key(3369568, civ[0]) != P.civitai_key(3369568, civ[1])

# 워크플로 파일명 치환: 따옴표 값 / 하위폴더 접두만 바꾸고 URL 은 건드리지 않음
txt = ('{"w": ["minimax_h3_fl2va_pruned_int8_convrot.safetensors", "default"], '
       '"x": "H3\\\\minimax_h3_fl2va_pruned_int8_convrot.safetensors", '
       '"u": "https://huggingface.co/a/b/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors", '
       '"y": "minimax_h3_fl2va_pruned_int8_convrot.safetensors.bak"}')
out = P.rewrite_workflow_text(txt, {"minimax_h3_fl2va_pruned_int8_convrot.safetensors": "NEW.safetensors"})
j = json.loads(out)
assert j["w"][0] == "NEW.safetensors" and j["x"] == "H3\\NEW.safetensors", j
assert j["u"].endswith("/minimax_h3_fl2va_pruned_int8_convrot.safetensors") and j["y"].endswith(".bak"), j

# 번들 워크플로 3종이 유효한 JSON 이고 치환 후에도 유효
P.APP = ROOT / "scripts"   # 번들 폴더 = ROOT/workflows 를 쓰도록
import shutil
(ROOT / "scripts" / "workflows").exists() and shutil.rmtree(ROOT / "scripts" / "workflows")
shutil.copytree(ROOT / "workflows", ROOT / "scripts" / "workflows")
try:
    assert P.provision_workflows([], rep) == 0
    wf_dir = P.WS / "user" / "default" / "workflows"
    got = sorted(p.name for p in wf_dir.glob("minimax_h3_*.json"))
    assert got == ["minimax_h3_i2v.json", "minimax_h3_r2v.json", "minimax_h3_t2v.json"], got
    for p in wf_dir.glob("minimax_h3_*.json"):
        t2 = p.read_text(encoding="utf-8"); json.loads(t2)
        assert "minimax_h3_fl2va_pruned_int8_convrot.safetensors\"" not in t2 or "w4a8" in t2
        assert '"qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"' not in t2, p.name
    t2v = (wf_dir / "minimax_h3_t2v.json").read_text(encoding="utf-8")
    assert '"minimax_h3_fl2va_pruned_w4a8_mixed.safetensors"' in t2v
    assert '"minimax_h3_fl2va_pruned_int8_convrot.safetensors"' not in t2v
finally:
    shutil.rmtree(ROOT / "scripts" / "workflows")

for k in ("MINIMAX_VARIANT", "MINIMAX_MODES", "MINIMAX_TE", "MINIMAX_TURBO"):
    os.environ.pop(k, None)
print("ALL TESTS PASSED")

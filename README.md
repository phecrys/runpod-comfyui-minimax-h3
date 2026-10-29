# RunPod 커스텀 템플릿: ComfyUI + MiniMax H3 (INT8 / INT6 / INT4 ConvRot)

Civitai **[Minimax H3 INT8/INT6/INT4 ConvRot](https://civitai.red/models/2830065/minimax-h3-int8int6int4-convrot)** (작성자 tsolful) 모델로
**텍스트/이미지 → 영상 + 오디오**를 만드는 RunPod 템플릿입니다. Krea2 템플릿과 같은 구조라 사용법이 같습니다.

파드를 켜면 아래 순서로 **자동 준비**됩니다.

1. 최신 ComfyUI(+ Manager, KJNodes) 실행 → `8188` 포트
2. 환경변수로 고른 정밀도(INT8/INT6/INT4/BF16)의 MiniMax H3 + 텍스트인코더 + VAE + Turbo LoRA 를 내려받음 (이미 받은 건 건너뜀)
3. 워크플로 3종(`minimax_h3_t2v` / `minimax_h3_i2v` / `minimax_h3_r2v`)이 워크플로 목록에 들어가며,
   **고른 정밀도의 파일명으로 자동 치환**되어 있어 바로 *Run* 가능

실행 중에도 `add-model <Civitai 링크>` 한 줄로 LoRA 등을 추가할 수 있습니다 (재시작 불필요).

```
├─ Dockerfile                    # CUDA 12.8 + PyTorch + 최신 ComfyUI(master) 이미지
├─ .github/workflows/build.yml   # main 에 push 하면 자동으로 이미지 빌드 (로컬 Docker 불필요)
├─ config/models.yaml            # ★ 받을 모델 목록 + 옵션 (볼륨에는 models-h3.yaml 로 복사됨)
├─ scripts/start.sh              # 컨테이너 시작 스크립트
├─ scripts/provision.py          # Civitai / HF 다운로더 (파일 ID 지정, 옵션별 선택, 워크플로 파일명 치환)
├─ scripts/add-model             # 실행 중 모델 추가 명령
├─ workflows/minimax_h3_*.json   # Comfy-Org 공식 템플릿(MIT) 기반 T2V / I2V(첫·끝 프레임) / R2V
├─ tools/vendor_workflows.py     # 공식 템플릿 갱신용
└─ tests/                        # 다운로더 단위 테스트
```

---

## 0. 이 모델에 대해

MiniMax H3 는 **영상과 스테레오 오디오(대사·효과음·음악)를 한 번에 생성**하는 모델입니다 (24fps, 최대 약 15초).

| 종류 | 용도 | 번들 워크플로 |
|---|---|---|
| **FL2VA** | 텍스트 → 영상, 첫 프레임/끝 프레임 → 영상 | `minimax_h3_t2v`, `minimax_h3_i2v` |
| **REF2VA** | 레퍼런스 이미지(최대 9)·영상(3)·오디오(3) → 영상 | `minimax_h3_r2v` |

Civitai 페이지의 버전과 이 템플릿 옵션의 대응 (파일 SHA256 이 Comfy-Org/Kijai 배포본과 같은 것을 확인했고, 공식 파일명으로 저장합니다):

| `MINIMAX_VARIANT` | Civitai 버전 (파일 ID) | 크기 | 메모 |
|---|---|---|---|
| `int8` (기본) | FL2VA INT8 Pruned `3193337` (3074134) / REF2VA INT8 Pruned `3193385` (3074237) | 19.5 GB | W8A8, 거의 무손실. 기본 권장 |
| `int6` | FL/REF2VA Pruned INT6W6A8 `3369568` (FL 3257813 / REF 3258013) | 14.9 GB | INT8 이 살짝 안 들어갈 때 |
| `int4` | FL/REF2VA Pruned AsymW4A8 `3207975` (FL 3089623 / REF 3091888) | 11.7 / 11.0 GB | 메모리 절약용(속도는 INT8 과 같음). ComfyUI v0.31.0+ |
| `bf16` | (Civitai 에 없음 → Comfy-Org HF) | 37.5 GB | 비교용 원본 |

`DEPRECATED` 표시된 INT4Q 버전은 받지 않습니다.

---

## 1. 이미지 빌드 (GitHub Actions)

이 저장소(`phecrys/runpod-comfyui-minimax-h3`)의 `main` 에 push 하면 **Actions → build-image** 가 자동으로 이미지를 빌드합니다 (약 15~25분).

1. **Actions** 탭에서 `build-image` 가 초록색이 될 때까지 대기
2. 첫 빌드 후 한 번만: 프로필 → **Packages** → `runpod-comfyui-minimax-h3` → *Package settings* → **Change visibility → Public**
3. ComfyUI 를 최신으로 다시 받고 싶을 때: Actions → build-image → *Run workflow* (원하면 `comfyui_ref` 에 태그 지정)

결과 이미지 주소: `ghcr.io/phecrys/runpod-comfyui-minimax-h3:latest`

---

## 2. RunPod Secret

Krea2 때 만든 `civitai_token` Secret 을 그대로 쓰면 됩니다. (없으면 RunPod 콘솔 → **Secrets** → `civitai_token` 생성)

> 이 모델의 INT8/INT6/INT4 파일은 Civitai 에서 받으므로 **CIVITAI_TOKEN 이 필수**입니다.
> 텍스트 인코더·VAE·LoRA·BF16 은 Hugging Face 공개 저장소라 토큰이 필요 없습니다.

---

## 3. 템플릿 만들기

RunPod 콘솔 → **My Templates → New Template**

| 항목 | 값 |
|---|---|
| Template Name | `ComfyUI MiniMax H3` |
| Template Type | Pod |
| Container Image | `ghcr.io/phecrys/runpod-comfyui-minimax-h3:latest` |
| Container Start Command | (비워 두기) |
| Container Disk | `30 GB` |
| Volume Disk | `150 GB` (Network Volume 을 쓸 거면 0) |
| Volume Mount Path | `/workspace` |
| Expose HTTP Ports | `8188,8888` |
| Expose TCP Ports | `22` |

**Environment Variables**

| Key | Value | 설명 |
|---|---|---|
| `CIVITAI_TOKEN` | `{{ RUNPOD_SECRET_civitai_token }}` | 필수 |
| `MINIMAX_VARIANT` | `int8` | `int8` / `int6` / `int4` / `bf16` |
| `MINIMAX_MODES` | `fl2va` | `fl2va` / `ref2va` / `fl2va,ref2va` (둘 다 = `all`) |
| `MINIMAX_TE` | `nvfp4` | 텍스트 인코더: `nvfp4`(14.6GB) / `int8`(25.3GB) / `bf16`(48GB) |
| `MINIMAX_VAE` | `int8` | 비디오 VAE: `int8`(2.6GB) / `fp16`(4.9GB) |
| `MINIMAX_TURBO` | `1` | Turbo LoRA 받기 (`0` 이면 안 받음) |
| `JUPYTER_PASSWORD` | 원하는 비밀번호 | 설정 시 8888 포트에 JupyterLab |
| `HF_TOKEN` | `{{ RUNPOD_SECRET_hf_token }}` | 선택 |
| `MODELS` | 예) `civitai:1234567@loras` | 선택 — 파드마다 추가로 받을 모델 |
| `PROVISION_WAIT` | `0` | `1` 이면 모델 다 받은 뒤 ComfyUI 시작 |
| `DOWNLOAD_CONCURRENCY` | `3` | 동시 다운로드 수 |
| `COMFY_ARGS` | (비움) | 예: `--reserve-vram 2` (OOM 이 나면) |

옵션은 **파드를 만들 때마다 바꿀 수 있습니다** (템플릿은 기본값, Deploy 화면 *Edit Template* 에서 덮어쓰기).
같은 볼륨에서 옵션을 바꿔 재시작하면 새 파일만 추가로 받고, 워크플로 파일명도 새 옵션으로 다시 치환됩니다.

**필요 용량 (대략)**

| 조합 | 다운로드 |
|---|---|
| INT8 + FL2VA (기본) | 약 39 GB |
| INT8 + 둘 다 | 약 60 GB |
| INT4 + FL2VA | 약 31 GB |
| BF16 + 둘 다 + TE bf16 | 약 130 GB |

> **Network Volume 권장**: Krea2 템플릿과 **같은 Network Volume 을 같이 써도 됩니다.**
> 이 템플릿은 설정 파일을 `/workspace/models-h3.yaml` 로 따로 두기 때문에 Krea2 의 `models.yaml` 과 섞이지 않고,
> `models/` 폴더는 공유되므로 한 번 받은 파일은 재사용됩니다. 용량만 넉넉히(200GB 이상) 잡으세요.

---

## 4. 파드 실행

1. *Deploy* → GPU 선택 → 템플릿 선택. INT8 계열 가속은 **RTX 30 시리즈 이상(Ampere+)** 에서 동작합니다.

   | GPU (VRAM) | 추천 `MINIMAX_VARIANT` |
   |---|---|
   | RTX 4090 / L4 / A5000 (24GB) | `int6` 또는 `int4` (INT8 도 로드는 되지만 여유가 1~2GB 뿐) |
   | RTX 5090 (32GB) | `int8` |
   | L40S / A6000 / RTX 6000 Ada (48GB) | `int8` (여유롭게 긴 영상·고해상도) |
   | A100 / H100 (80GB) | `int8` 또는 `bf16` |

   텍스트 인코더(32B)는 VRAM 이 모자라면 ComfyUI 가 RAM 으로 내리므로 **시스템 RAM 64GB 이상** 파드가 안정적입니다.
2. *Connect* → **HTTP 8188** → ComfyUI
3. 처음엔 모델이 백그라운드로 받아집니다. 진행상황 (JupyterLab 터미널 / SSH):
   ```bash
   tail -f /workspace/logs/provision.log
   ```
   “모델 프로비저닝 끝” 이 뜨면 ComfyUI 화면에서 **R** 키 → 모델 목록 새로고침
4. 왼쪽 *Workflows* → `minimax_h3_t2v.json` 열기 → *Run*
   - `minimax_h3_i2v.json`: 첫 프레임(필수)·끝 프레임(선택) 이미지를 넣어 영상 생성
   - `minimax_h3_r2v.json`: `MINIMAX_MODES` 에 `ref2va` 를 넣었을 때만 사용 가능

> 번들 워크플로는 파드가 켜질 때마다 최신본(+옵션 파일명)으로 **덮어씁니다.** 직접 고친 워크플로는
> *Save As* 로 다른 이름으로 저장하세요.

---

## 5. 생성 설정 팁

- **프롬프트**: 장면·카메라·오디오(대사·효과음·음악)를 한 블록에 같이 씁니다. 대사는 `<d>[English] ...</d>`,
  레퍼런스는 `<Picture 1>` / `<Video 1>` / `<Audio 1>` 처럼 지칭합니다 (워크플로 예제 프롬프트 참고).
- **해상도**: 기본 캔버스는 짧은 변 768px (최대 1344×768), 32의 배수. *Resolution Selector* 의 megapixels 로 조절.
  처음엔 0.4MP(864×480)로 빠르게 확인 → 0.98MP(1344×768)로 본 생성.
- **길이**: *duration(초)* 를 넣으면 17k+5 프레임 격자(24fps)로 자동 맞춤. 학습 범위는 약 5~15초.
- **Turbo**: 워크플로의 **Enable Lightning LoRA** 를 켜면 Turbo LoRA + 적은 스텝(T2V/I2V 6, R2V 4)으로 몇 배 빨라집니다.
  끄면 20 스텝 기본 모델.
- **샘플러**: 공식 템플릿은 `res_multistep / simple`. Civitai 작성자는 `er_sde` 또는 `res_multistep` + `simple` 을 사용.
- **VRAM 이 빠듯할 때 (24GB)**: KJNodes 가 설치돼 있으므로 모델 노드와 샘플러 사이에
  `MiniMax H3 Chunk FeedForward`(chunks 2) 와 `MiniMax H3 Low VRAM Attention` 노드를 끼워 넣으면 피크 VRAM 이 줄어듭니다.
  해상도·길이를 줄이는 것도 효과가 큽니다.

---

## 6. 원하는 Civitai 모델 넣는 법

Krea2 템플릿과 같습니다. 설정 파일만 `/workspace/models-h3.yaml` 입니다.

```bash
add-model "https://civitai.com/models/111111?modelVersionId=2222222"   # LoRA → 자동으로 loras/
add-model 3333333 --dir loras/h3 --name my_style.safetensors
# 한 버전에 파일이 여러 개면 파일 ID 로 정확히 지정 (Civitai 다운로드 링크의 ?fileId= 값)
add-model 3207975 --file-id 3089623 --dir diffusion_models --name minimax_h3_fl2va_pruned_w4a8_mixed.safetensors
```

`models-h3.yaml` 에 직접 쓸 때:
```yaml
  - civitai: 2222222
    file_id: 7654321        # (선택) 파일이 여러 개인 버전
    dir: loras/h3
```

> Civitai 원작자 워크플로(JSON)를 그대로 쓰고 싶다면 `models-h3.yaml` 의 `custom_nodes` 에서
> VideoHelperSuite / rgthree 주석을 풀고 재시작하세요. 그 워크플로는 SageAttention 패치 노드를 쓰는데,
> 이 이미지에는 SageAttention 이 없으므로 해당 노드 2개(Patch Sage Attention KJ / MiniMax H3 Mem Eff Sage Attention Patch)는
> **Ctrl+B 로 우회**하고, LoRA 파일명은 받은 Turbo LoRA 로 바꿔 선택하세요.

---

## 7. 문제 해결

| 증상 | 해결 |
|---|---|
| `MiniMaxH3...` 노드가 없음 / `type: 'minimax'` 오류 | ComfyUI 구버전 → Actions 에서 이미지 재빌드 후 파드 재생성 |
| INT4 로드 오류 (`AsymW4A8` 관련) | ComfyUI v0.31.0 이상 필요 → 재빌드 |
| Civitai 다운로드 실패 / “safetensors 형식이 아닙니다” | `CIVITAI_TOKEN` Secret 확인 |
| CUDA OOM | `MINIMAX_VARIANT=int6`/`int4`, 해상도·길이 줄이기, `COMFY_ARGS=--reserve-vram 2`, 5장의 저VRAM 노드 |
| 첫 실행이 매우 느림 | 32B 텍스트 인코더 로드 중(수십 초~수 분). 두 번째부터는 캐시됨 |
| 새 모델이 목록에 없음 | ComfyUI 화면에서 **R**, 로그 `/workspace/logs/provision.log` 확인 |
| 디스크 부족 | Volume 늘리기, `/workspace/models/.downloads` 의 잔여 `.part` 삭제 |
| 워크플로가 `int8_convrot` 파일을 찾음 | 옵션을 바꾼 뒤 재시작했는지 확인 (치환은 파드 시작 시 적용) |

로그 위치: `/workspace/logs/{provision,comfyui,jupyter}.log`

---

## 개발 메모

- 다운로더 테스트: `python3 tests/test_provision.py`
- 공식 템플릿 갱신: `git clone https://github.com/Comfy-Org/workflow_templates` 후
  `python3 tools/vendor_workflows.py workflow_templates/templates`
- 라이선스: 모델은 MiniMax-H3 Community License, 번들 워크플로는 Comfy-Org workflow_templates (MIT)

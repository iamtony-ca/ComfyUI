# Qwen-Image-Edit 2509 기반 워크플로우 기술 문서

> **문서 목적**: ComfyUI에서 자체 설계·운영 중인 2개 워크플로우의 구조와 동작 원리를 정리하여 부서 내 공유 및 유지보수 기준으로 활용한다.
> **대상 워크플로우**
> - `user/default/workflows/1_click_multiple_scene_angles-v1.0.json`
> - `user/default/workflows/image_qwen_image_edit_2509.json`
> **공통 기반 모델**: Qwen-Image-Edit 2509 (fp8) + Lightning 4-step LoRA
> **작성 기준일**: 2026-06-06

---

## 0. 요약 (Executive Summary)

| 구분 | Workflow 1: Multi-Angle | Workflow 2: Qwen Edit (Dynamic) |
|------|-------------------------|----------------------------------|
| 한 줄 정의 | 이미지 1장 → 카메라 앵글 8종 일괄 생성 | 와일드카드 기반 무작위 편집 배리에이션 대량 생성 |
| 베이스 | 자체 설계 (원클릭) | ComfyUI 공식 템플릿 개조 |
| 핵심 기술 | 3단계 중첩 서브그래프(subgraph) | 2-variant + 불리언 토글 + Dynamic Prompts |
| 변동 요소 | 8개 고정 앵글 프롬프트 | DPRandomGenerator 와일드카드 |
| 적용 LoRA | Multi-angle + Lightning (2개 스택) | Lightning (토글 방식) |
| 주 용도 | 제품/씬 멀티앵글 촬영 자동화 | 데이터셋·배리에이션 대량 생산 |

두 워크플로우 모두 **ModelSamplingAuraFlow + CFGNorm + TextEncodeQwenImageEditPlus + Lightning 4-step LoRA** 라는 동일한 Qwen-Edit 추론 코어를 공유하며, 그 위에 서로 다른 운영 레이어(멀티앵글 병렬화 vs 와일드카드 대량생산)를 얹은 구조다.

---

## 1. 공통 추론 코어 (Common Inference Core)

두 워크플로우의 실제 이미지 생성부는 동일한 노드 체인을 사용한다. 이 코어를 먼저 이해하면 두 워크플로우 모두 빠르게 파악할 수 있다.

```
입력 이미지
  └─ (해상도 정규화: ImageScaleToTotalPixels 또는 FluxKontextImageScale)
        ├─→ TextEncodeQwenImageEditPlus (positive: 프롬프트 + 참조이미지 + VAE)
        ├─→ TextEncodeQwenImageEditPlus (negative: 빈 프롬프트)
        └─→ VAEEncode → latent
  ModelSamplingAuraFlow (shift=3)
        └─ CFGNorm (strength=1)
              └─ KSampler (euler / simple / denoise=1)
                    └─ VAEDecode → 결과 이미지
```

### 핵심 기술 포인트

| 노드 | 역할 | 기술적 의미 |
|------|------|-------------|
| `TextEncodeQwenImageEditPlus` | 텍스트 + 참조이미지를 함께 인코딩 | 편집 조건(reference)을 **conditioning에 임베드**. 일반 img2img와 달리 `denoise=1`(완전 재생성)에서도 원본 구도·피사체가 유지되는 이유. |
| `ModelSamplingAuraFlow (shift=3)` | flow-matching 스케줄 설정 | Qwen-Image 계열은 AuraFlow식 샘플링을 사용하므로 필수. |
| `CFGNorm (strength=1)` | CFG 정규화 | 저 step 환경에서 과포화/색 깨짐 억제. |
| Lightning 4-step LoRA | 추론 가속 | 4 step / cfg 1.0으로 고속 생성 가능. |

### 모델 구성 (공통)

| 종류 | 파일 | 저장 위치 |
|------|------|-----------|
| Diffusion (UNET) | `qwen_image_edit_2509_fp8_e4m3fn.safetensors` | `models/diffusion_models/` |
| Text Encoder (CLIP) | `qwen_2.5_vl_7b_fp8_scaled.safetensors` (type: `qwen_image`) | `models/text_encoders/` |
| VAE | `qwen_image_vae.safetensors` | `models/vae/` |
| LoRA (가속) | `Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors` | `models/loras/` |
| LoRA (앵글, WF1 전용) | `Qwen-Edit-2509-Multiple-angles.safetensors` | `models/loras/` |

### KSampler 설정 기준표

| 설정 | Qwen 공식 | Comfy 기본 | Lightning 4-step (본 워크플로우) |
|------|-----------|-----------|----------------------------------|
| Steps | 50 | 20 | **4** |
| CFG | 4.0 | 2.5 | **1.0** |
| Sampler / Scheduler | — | — | euler / simple |

---

## 2. Workflow 1 — `1_click_multiple_scene_angles-v1.0.json`

### 2.1 목적
입력 이미지 1장을 받아 **버튼 한 번에 8개 카메라 앵글**로 동일 피사체를 재생성한다. 제품/씬 촬영을 자동화하는 용도. (기본 입력: `Ergonomic chair in modern laboratory.png`)

### 2.2 전체 구조 — 3단계 중첩 서브그래프

이 워크플로우의 핵심은 **서브그래프를 3층으로 중첩**한 모듈식 설계다.

```
[최상위 캔버스]  (노드 28개)
 ├─ Loader 서브그래프 (id 48)           → VAE / CLIP / MODEL 출력
 ├─ LoadImage "Load Scene Image" (id 25)
 ├─ 8× PrimitiveStringMultiline (id 66~73)  → 앵글별 프롬프트 8종
 ├─ "Generate" 마스터 서브그래프 (id 65)     ← 오케스트레이터
 │     ├─ 8× 앵글별 서브그래프 인스턴스
 │     │     ├─ Generate Close-up
 │     │     ├─ Generate wide angle
 │     │     ├─ Generate 45deg right / 90deg right
 │     │     ├─ Generate 45deg left  / 90deg left
 │     │     ├─ Generate aerial view
 │     │     └─ Generate low angle view
 │     └─ 15× Reroute (공통 리소스 분배)
 └─ 8× (ImageScale 800×600 → Image Save) 출력 파이프라인
```

### 2.3 계층별 상세

**① Loader 서브그래프 (`Qwen Image Edit Model and Loras Loader`)**
모델 로딩을 캡슐화하고 `proxyWidgets`로 상위 캔버스에 모델/LoRA 선택 위젯을 노출한다.
- `UNETLoader` / `CLIPLoader` / `VAELoader` (위 공통 모델 표 참조)
- **LoRA 2개 직렬 스택** (핵심):
  - `Qwen-Edit-2509-Multiple-angles.safetensors` (strength 1.0) — 카메라 회전 능력 부여
  - `Qwen-Image-Edit-2509-Lightning-4steps...` (strength 1.0) — 4-step 가속

**② "Generate" 마스터 서브그래프 (오케스트레이터)**
입력으로 `vae / clip / input_image / model` + `prompt_1~8`을 받는다. 내부 15개 Reroute가 **공통 리소스(모델·VAE·CLIP·이미지)를 8개 앵글 서브그래프에 분배**하고, 각 서브그래프에는 해당 앵글 프롬프트만 다르게 연결한다. 출력은 `IMAGE ~ IMAGE_7` (8개).

**③ 앵글별 서브그래프 (8개 모두 동일, 각 9노드)**
구조는 [공통 추론 코어](#1-공통-추론-코어-common-inference-core)와 동일하며 **프롬프트만 다르다**. 입력 정규화는 `ImageScaleToTotalPixels` (1MP, nearest-exact) 사용, KSampler는 steps=4 / cfg=1.0 / euler / simple / denoise=1 / seed=randomize, 끝에 PreviewImage 부착.

### 2.4 앵글 프롬프트 8종

| # | 노드 제목 | 프롬프트 (영문) |
|---|-----------|------------------|
| 1 | Close up | Turn the camera to a close-up. |
| 2 | Wide angle | Turn the camera to a wide-angle lens. |
| 3 | 45deg right | Rotate the camera 45 degrees to the right. |
| 4 | 90deg right | Rotate the camera 90 degrees to the right. |
| 5 | Aerial view | Turn the camera to an aerial view. |
| 6 | Low angle | Turn the camera to a low-angle view. |
| 7 | Left 45deg | Rotate the camera 45 degrees to the left. |
| 8 | 90deg | Rotate the camera 90 degrees to the left. |

### 2.5 출력
8개 결과 모두 `ImageScale`(bilinear, 800×600) → `Image Save`(jpg, quality 95, dpi 300, 경로 `fab_backgrounds`, prefix `fab_bg`).

---

## 3. Workflow 2 — `image_qwen_image_edit_2509.json`

### 3.1 목적
ComfyUI 공식 Qwen-Edit 2509 템플릿을 베이스로, **Dynamic Prompts 와일드카드로 무작위 편집 배리에이션을 대량 생성**하도록 개조했다. (의자 씬에서 사람 제거 + 등받이에 옷 드레이프 + 의상/배경/회전각도 랜덤화)

### 3.2 전체 구조 — 2-variant 토글

```
[최상위 캔버스]  (노드 8개)
 ├─ LoadImage (Nano_Banana2_00005_1.png)   → 두 서브그래프에 동시 공급
 ├─ DPRandomGenerator (id 469)             ← 커스텀 노드 (Dynamic Prompts)
 │     └─ 와일드카드 프롬프트 → 두 서브그래프 prompt에 동시 연결
 ├─ 서브그래프 "Image Edit (Qwen 2509)"           (id 433, mode 0 = 활성)
 ├─ 서브그래프 "Image Edit (Qwen 2509 Raw Latent)" (id 466, mode 4 = 뮤트)
 ├─ 활성 경로: ImageScale(800×600) → Image Save (fab_backgrounds / fab_bg)
 └─ 뮤트 경로: SaveImage (mode 4)
```

### 3.3 DPRandomGenerator (Dynamic Prompts) — 핵심 변동 엔진

`{옵션A|옵션B|...}` 문법으로 매 실행마다 조합을 무작위 선택한다.

| 변수 | 와일드카드 내용 |
|------|------------------|
| 회전 각도 | `{15\|20\|25\|30}` 도 |
| 의상 | 35종 이상 (cleanroom smock, hi-vis vest, leather jacket, 다양한 fleece/vest 등) |
| 배경 | 16종 무인 장소 (반도체 팹, 오픈오피스, 로봇 창고, 도서관, 데이터센터 등) |
| 고정 규칙 | `[CRITICAL RULE]`: 사람 완전 제거 + 옷은 등받이 위에 자연스럽게 드레이프 |

> ⚠️ **의존성**: `DPRandomGenerator`는 `custom_nodes`의 **comfyui-dynamicprompts** 노드가 설치되어 있어야 동작한다. (설치 안내: `readme-tony.md`)

### 3.4 서브그래프 내부 — Lightning 토글 메커니즘 (핵심 설계)

불리언 1개로 "터보 모드(품질↔속도)"를 통째로 전환한다.

```
PrimitiveBoolean "Enable Lightning LoRA" (기본 true)
   └─→ 3× ComfySwitchNode 동시 분기:
        ├─ Switch(Model): Lightning LoRA 적용 ↔ 원본 모델
        ├─ Switch(Steps): 4 ↔ 20
        └─ Switch(CFG):   1.0 ↔ 4.0
```

이 외에 입력 정규화는 `FluxKontextImageScale`(지원 해상도로 맞춤)을 사용하며, 본 파이프라인은 [공통 추론 코어](#1-공통-추론-코어-common-inference-core)와 동일하다.

### 3.5 두 변형(variant) 비교

| 항목 | Image Edit (Qwen 2509) | Image Edit (Qwen 2509 Raw Latent) |
|------|------------------------|-----------------------------------|
| 노드 수 | 21 | 23 |
| 차이점 | 표준 VAEEncode 라텐트 | **`ReferenceLatent` 2개 추가** (최신 참조 라텐트 주입 방식) |
| 현재 상태 | **활성 (mode 0)** | 뮤트 (mode 4) — 실험용 대안 경로로 보존 |

---

## 4. 운영 가이드 (실무 참고)

### 실행 전 체크리스트
1. `models/` 하위에 [공통 모델 표](#모델-구성-공통)의 파일이 모두 존재하는지 확인
2. Workflow 2 사용 시 `comfyui-dynamicprompts` 커스텀 노드 설치 여부 확인
3. 출력 경로(`output/fab_backgrounds/`) 존재 및 쓰기 권한 확인

### 품질/속도 튜닝
- **빠른 미리보기**: Lightning LoRA ON (4 step / cfg 1.0) — 기본값
- **고품질 최종본**: Workflow 2는 "Enable Lightning LoRA" OFF로 전환 (20 step / cfg 4.0)
- 앵글이 약하게 나오는 경우(WF1): Lightning LoRA strength를 0.8~1.0 사이에서 조정

---

## 5. 알려진 개선 포인트 (Known Improvements)

| # | 워크플로우 | 내용 | 권장 조치 |
|---|-----------|------|-----------|
| 1 | WF1 | 8개 `Image Save`가 동일 prefix(`fab_bg`)·동일 폴더 사용 → 앵글 구분 없이 일련번호로만 누적 | 앵글별 prefix 부여 (예: `fab_bg_closeup`) |
| 2 | WF1 | Multi-angle(1.0) + Lightning(1.0) 동시 1.0은 LoRA 간 간섭 가능성 | 결과 보고 Lightning strength 미세조정 |
| 3 | WF2 | Raw Latent 분기가 뮤트(mode 4) 상태 — DPRandomGenerator는 양쪽 연결이나 한쪽만 실행 | A/B 동시 비교 필요 시 양쪽에 Save 노드 연결 |

---

## 6. 용어 정리 (Glossary)

| 용어 | 설명 |
|------|------|
| 서브그래프(Subgraph) | 여러 노드를 하나의 노드처럼 캡슐화하는 ComfyUI 기능. 재사용·중첩 가능. |
| proxyWidgets | 서브그래프 내부 위젯을 상위 캔버스로 노출하는 설정. |
| ComfySwitchNode | 불리언 입력에 따라 두 입력 중 하나를 통과시키는 분기 노드. |
| Lightning LoRA | 적은 step(4)으로 생성 가능하게 하는 가속 LoRA. |
| Dynamic Prompts | `{a\|b\|c}` 와일드카드로 프롬프트를 무작위 조합하는 커스텀 노드. |
| denoise=1 | 라텐트를 완전히 새로 생성. Qwen-Edit는 조건에 참조이미지가 포함되어 구도가 유지됨. |

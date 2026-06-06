
# 🚀 프로젝트: GenAI 기반 AMR 비전 인식용 합성 데이터 파이프라인

> **개정 이력**: v2 — 실제 ComfyUI 구현(소스) 대조 기반 기술 표현 정정판
> 주요 수정: U-Net→DiT, CLIP→Qwen2.5-VL 인코더, ADD→Step Distillation, Reference Latent(편집 메커니즘) 보강
> **기준일**: 2026-06-07

**[프로젝트 요약]**
자율이동로봇(AMR)의 객체 인식 모델(YOLO)이 겪는 OOD(Out-of-Distribution) 한계 및 Sim2Real Gap을 극복하기 위해, **최신 VLM(비전-언어 모델)과 잠재 확산 모델(Latent Diffusion Model) 기반의 데이터 합성 파이프라인**을 구축했습니다. 특히 추론 가속화 기법(Step Distillation)과 조합적 도메인 무작위화(Domain Randomization)를 적용하여, 모델의 강건성(Robustness)을 극대화하는 **Hard Negative 데이터셋 대량 생성 시스템**을 완성했습니다.

---

### 💡 1. Core AI Architecture & Inference Acceleration (핵심 모델 및 추론 가속)

* **VLM 기반 Semantic Context Control (의미론적 맥락 제어):**
* 텍스트와 이미지를 동시에 이해하는 최신 **Qwen 2.5-VL (Vision-Language Model)** 을 텍스트 인코더로 도입하여, 단순 키워드 매칭을 넘어 "사람 없이 등받이에 걸쳐진 옷"이라는 복잡한 공간적·인과적 제약 조건(Critical Rule)을 **Conditioning 벡터**로 정교하게 제어했습니다.
  * ※ ComfyUI가 텍스트 인코더 슬롯을 관례상 "CLIP"으로 표기하나, 본 파이프라인의 실제 인코더는 contrastive CLIP이 아닌 **LLM 기반 Qwen2.5-VL**입니다.

* **In-Context Editing (참조 라텐트 기반 편집):**
* 일반 txt2img가 아닌, 원본 이미지를 VAE로 인코딩한 **Reference Latent를 디노이징 토큰 시퀀스에 주입(concat)** 하는 In-Context Editing 방식을 활용했습니다. 이를 통해 의자의 구조적 본질과 구도는 보존한 채, 의류·배경 등 의미론적 요소만 선택적으로 합성하도록 제어했습니다.

* **Diffusion Distillation (확산 증류) 기반 추론 가속:**
* 대량 데이터 양산 시 발생하는 Diffusion 모델의 추론 속도 병목(50+ Steps)을 해결하기 위해, **Lightning LoRA (4-steps, few-step distillation)** 가중치를 Diffusion Transformer 백본에 패치(Patch)했습니다. 여기에 **fp8 양자화**를 결합하여 VRAM 효율을 높이고, **단 4회의 노이즈 제거(Denoising) 연산만으로 Teacher 모델 수준의 고해상도 이미지를 초고속 생성**하도록 파이프라인을 최적화했습니다.

* **Flow-Matching 스케줄 & CFG 안정화:**
* `ModelSamplingAuraFlow(shift=3)`로 flow-matching 시그마 스케줄을 편집 작업에 맞게 조정하고, `CFGNorm`으로 저(低)-step·저-CFG(1.0) 환경에서 발생하는 과포화·색 왜곡을 억제하여 4-step 생성의 품질을 안정화했습니다.

### 🎯 2. Data-Centric AI Strategy (데이터 중심 AI 전략)

* **Adversarial Hard Negative Mining (적대적 하드 네거티브 생성):**
* 비전 모델의 오탐지(False Positive)를 유발하는 엣지 케이스(Edge-case)를 생성형 AI로 역공학(Reverse-engineering)하여 합성했습니다. 의자의 구조적 본질은 유지한 채 인간의 실루엣을 띠는 방진복/작업복을 등받이에 자연스럽게 드레이프하여, '사람'으로 오인될 만한 시각적 노이즈를 추가함으로써 모델의 식별력을 극대화했습니다.

* **Combinatorial Domain Randomization (조합적 도메인 무작위화):**
* 정적 시뮬레이터(Isaac Sim 등) 데이터의 한계를 넘어, 동적 매개변수 주입(Parameterized Injection)을 통해 `{의류 35여 종} x {산업 환경 16종} x {회전각 4종}`의 방대한 변수 조합을 런타임에 렌더링하여 폭발적인 데이터 다양성을 확보했습니다.
  * 보조 워크플로우(1-Click Multi-Angle)에서는 동일 피사체를 **8종 카메라 앵글(close-up/wide/aerial/low/회전 등)** 로 일괄 재생성하여 시점 다양성까지 보강했습니다.

---

### 🛠 [Tech Stack & Tools]

* **AI Architecture:** Latent Diffusion Models (LDM), **Diffusion Transformer (MMDiT)**, Vision-Language Model (Qwen 2.5-VL), VAE
* **Editing Mechanism:** Reference Latent (In-Context Editing), Flow-Matching (AuraFlow scheduling)
* **Acceleration & Tuning:** Step Distillation (Lightning 4-step LoRA), fp8 Quantization, CFG Normalization (CFGNorm)
* **Pipeline & Tools:** ComfyUI, Dynamic Prompts (DPRandomGenerator), WAS Node Suite, Isaac Sim

---

### 🌊 전체 파이프라인 아키텍처 (Text Diagram)

```text
[ Phase 1: Data Injection (데이터 주입부) ]
  ├── 🖼️ Source Image (LoadImage): 빈 의자가 있는 베이스 씬(Scene) 이미지
  └── 📝 Dynamic Prompt (DPRandomGenerator): 실행 시마다 {의류 35여 종 | 배경 16종 | 회전각 4종}을
                                              무작위 조합한 텍스트 생성 (+ Critical Rule 제약)
          │
          ▼ (데이터 변환)

[ Phase 2: Multimodal Encoding (다중 양식 인코딩) ]
  ├── 🗜️ VAE Encode: 픽셀 이미지를 연산이 빠른 고차원 'Latent Tensor(잠재 텐서)'로 압축
  ├── 🧩 Reference Latent: 원본 이미지를 VAE 인코딩하여 '참조 라텐트'로 보존(편집 기준점)
  └── 🔠 Text Encode (Qwen2.5-VL): 프롬프트를 방향성을 지시하는 'Conditioning Vector'로 변환
          │                         (참조 라텐트가 conditioning에 함께 임베드됨)
          ▼ (잠재 공간 진입)

[ Phase 3: Core Generative Engine (초고속 디노이징 엔진) ]
  ⚙️ KSampler (확산 연산 심장부) — euler / simple / 4-Steps / CFG 1.0
  ├── 🧠 Backbone: Qwen-Image-Edit-2509 (Diffusion Transformer / MMDiT, fp8)
  ├── ⚡ Accelerator: Lightning LoRA (4-Steps step-distillation 가중치 패치)
  ├── 🌊 ModelSamplingAuraFlow (shift=3): flow-matching 시그마 스케줄
  └── 🛡️ CFGNorm: 저-step 과포화/색 왜곡 억제
      ▶ 동작: 노이즈 토큰 ⊕ 참조 토큰을 joint-attention으로 함께 처리하여,
              원본 구조를 유지한 채 50-step 분량의 합성을 단 4-Step에 완료
          │
          ▼ (픽셀 공간 복귀)

[ Phase 4: Decoding & Post-Processing (복원 및 후처리) ]
  ├── 🔓 VAE Decode: 합성 완료된 Latent Tensor를 RGB 픽셀 이미지로 복원
  ├── 📏 ImageScale: YOLO 데이터셋 규격/용량에 최적화된 800x600으로 리사이징 (Bilinear)
  └── 💾 Image Save (WAS Node): 워크플로우 메타데이터 미삽입(embed_workflow=false),
                                'fab_backgrounds' 디렉토리에 고효율 JPG(Quality 95%)로 일괄 자동 저장
```

---

### 📌 부록: v1 대비 주요 정정 내역

| # | v1 (기존) | v2 (정정) | 근거 |
|---|-----------|-----------|------|
| 1 | UNET 백본 | **Diffusion Transformer (MMDiT)** | `comfy/ldm/qwen_image/model.py` (`QwenImageTransformer2DModel`). `UNETLoader`는 레거시 노드명 |
| 2 | CLIP 임베딩 벡터 | **Qwen2.5-VL Conditioning 벡터** | 텍스트 인코더 = `qwen_2.5_vl_7b` (contrastive CLIP 아님) |
| 3 | Adversarial Diffusion Distillation | **Step Distillation (few-step)** | Qwen-Image-Lightning은 ADD가 아닌 few-step 증류 LoRA |
| 4 | (없음) | **Reference Latent / In-Context Editing 추가** | `nodes_qwen.py`, `nodes_edit_model.py`, `model_base.py` |
| 5 | (없음) | **Flow-Matching·CFGNorm 명시** | `ModelSamplingAuraFlow`, `CFGNorm` |
| 6 | 의류 30종 / 화각 4종 | **의류 35여 종 / 회전각 4종**, 8앵글은 보조 WF로 분리 | DPRandomGenerator 와일드카드 / WF1 분리 |
| 7 | VRAM 효율 극대화(LoRA) | **fp8 양자화 + 4-step** 병기 | VRAM 절감 주체는 fp8 |

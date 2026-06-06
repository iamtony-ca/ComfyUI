
# 🚀 프로젝트: GenAI 기반 AMR 비전 인식용 합성 데이터 파이프라인

**[프로젝트 요약]**
자율이동로봇(AMR)의 객체 인식 모델(YOLO)이 겪는 OOD(Out-of-Distribution) 한계 및 Sim2Real Gap을 극복하기 위해, **최신 VLM(비전-언어 모델)과 잠재 확산 모델(Latent Diffusion Model) 기반의 데이터 합성 파이프라인**을 구축했습니다. 특히 추론 가속화 기법(Distillation)과 조합적 도메인 무작위화(Domain Randomization)를 적용하여, 모델의 강건성(Robustness)을 극대화하는 **Hard Negative 데이터셋 대량 생성 시스템**을 완성했습니다.

---

### 💡 1. Core AI Architecture & Inference Acceleration (핵심 모델 및 추론 가속)

* **VLM 기반 Semantic Context Control (의미론적 맥락 제어):**
* 텍스트와 이미지를 동시에 이해하는 최신 **Qwen 2.5 (Vision-Language Model)** 아키텍처를 도입하여, 단순 키워드 매칭을 넘어 "사람 없이 등받이에 걸쳐진 옷"이라는 복잡한 공간적·인과적 제약 조건(Critical Rule)을 CLIP 임베딩 벡터로 정교하게 제어했습니다.


* **Adversarial Diffusion Distillation (적대적 확산 증류) 적용:**
* 대량의 데이터 양산 시 발생하는 Diffusion 모델의 추론 속도 병목(50+ Steps)을 해결하기 위해, **Lightning LoRA (4-steps)** 가중치를 UNET 뼈대에 병합(Patch)했습니다. 이를 통해 VRAM 효율을 극대화하고 **단 4회의 노이즈 제거(Denoising) 연산만으로 Teacher 모델 수준의 고해상도 이미지를 초고속 생성**하도록 파이프라인을 최적화했습니다.



### 🎯 2. Data-Centric AI Strategy (데이터 중심 AI 전략)

* **Adversarial Hard Negative Mining (적대적 하드 네거티브 생성):**
* 비전 모델의 오탐지(False Positive)를 유발하는 엣지 케이스(Edge-case)를 생성형 AI로 역공학(Reverse-engineering)하여 합성했습니다. 의자의 구조적 본질은 유지한 채 인간의 실루엣을 띠는 방진복/작업복 노이즈를 추가하여 모델의 식별력을 극대화했습니다.


* **Combinatorial Domain Randomization (조합적 도메인 무작위화):**
* 정적 시뮬레이터(Isaac Sim 등) 데이터의 한계를 넘어, 동적 매개변수 주입(Parameterized Injection)을 통해 `{30종 의류} x {16종 산업 환경} x {카메라 각도/화각}`의 무한한 변수 조합을 런타임에 렌더링하여 폭발적인 데이터 다양성을 확보했습니다.




---

### 🛠 [Tech Stack & Tools]

* **AI Architecture:** Latent Diffusion Models (LDM), Vision-Language Models (VLM, Qwen 2.5), CLIP, VAE
* **Acceleration & Tuning:** Adversarial Diffusion Distillation (Lightning LoRA), Classifier-Free Guidance (CFG) Tuning

---


---

### 🌊 전체 파이프라인 아키텍처 (Text Diagram)

```text
[ Phase 1: Data Injection (데이터 주입부) ]
  ├── 🖼️ Source Image (LoadImage): 빈 의자가 있는 베이스 씬(Scene) 이미지
  └── 📝 Dynamic Prompt (DPRandomGenerator): 실행 시마다 {의류 30종 | 배경 16종 | 화각 4종}을 무작위 조합한 텍스트 생성
          │
          ▼ (데이터 변환)

[ Phase 2: Multimodal Encoding (다중 양식 인코딩) ]
  ├── 🗜️ VAE Encode: 무거운 픽셀 이미지를 연산이 빠른 고차원의 'Latent Tensor(잠재 텐서)'로 압축
  └── 🔠 CLIP Encode: Qwen 2.5 텍스트 인코더를 통해 프롬프트를 방향성을 지시하는 'Conditioning Vector'로 변환
          │
          ▼ (잠재 공간 진입)

[ Phase 3: Core Generative Engine (초고속 디노이징 엔진) ]
  ⚙️ KSampler (확산 연산 심장부)
  ├── 🧠 Backbone LDM: Qwen-Image-Edit-2509 (UNET)
  └── ⚡ Accelerator: Lightning LoRA (4-Steps 가중치 패치)
      ▶ 결과: CLIP의 조종을 받으며 Latent 공간 내에서 50번의 노이즈 제거 연산을 단 4번(4-Steps) 만에 초고속으로 완료하여 의류/배경 합성
          │
          ▼ (픽셀 공간 복귀)

[ Phase 4: Decoding & Post-Processing (복원 및 후처리) ]
  ├── 🔓 VAE Decode: 합성이 완료된 Latent Tensor를 우리가 볼 수 있는 RGB 픽셀 이미지로 압축 해제
  ├── 📏 ImageScale: YOLO 데이터셋 규격 및 용량 관리에 최적화된 800x600 해상도로 리사이징 (Bilinear)
  └── 💾 Image Save (WAS Node): 메타데이터를 제거하고 'fab_backgrounds' 디렉토리에 고효율 JPG (Quality 95%) 형식으로 일괄 자동 저장

```


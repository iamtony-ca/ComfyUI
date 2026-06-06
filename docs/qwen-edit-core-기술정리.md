# Qwen-Image-Edit 워크플로우 핵심 기술 정리 (Core Technologies)

> **문서 목적**: 앞선 「Qwen-Image-Edit 2509 기반 워크플로우 기술 문서」에서 사용된 핵심 노드·모델 기술의 **내부 동작 원리**를 ComfyUI 실제 소스 구현 기준으로 정리한다.
> **대상 독자**: 워크플로우를 깊이 이해하거나 직접 튜닝·확장하려는 엔지니어
> **근거 소스**: `comfy/ldm/qwen_image/model.py`, `comfy_extras/nodes_qwen.py`, `nodes_cfg.py`, `nodes_edit_model.py`, `nodes_model_advanced.py`, `nodes_flux.py`, `comfy/model_base.py`
> **작성 기준일**: 2026-06-07

---

## 0. 큰 그림 (Big Picture)

Qwen-Image-Edit는 "텍스트 + 참조 이미지 → 편집된 이미지"를 생성하는 **Flow-Matching 기반 Diffusion Transformer(DiT)** 다. 워크플로우의 각 노드는 이 모델에 다음 4가지를 공급하는 역할로 나뉜다.

```
        ┌─ (1) 조건 생성: 텍스트 + 참조이미지를 토큰/라텐트로 변환
        │       → TextEncodeQwenImageEditPlus  (Qwen2.5-VL 텍스트 인코더)
        │
입력 ───┼─ (2) 노이즈 라텐트: 편집 대상의 시작 latent
        │       → VAEEncode / FluxKontextImageScale
        │
        ├─ (3) 샘플링 스케줄: flow-matching 시그마 설정
        │       → ModelSamplingAuraFlow
        │
        └─ (4) 가이던스 보정 + 가속
                → CFGNorm, Lightning 4-step LoRA
                          ↓
                   KSampler → VAEDecode → 결과
```

핵심은 **"편집(edit)"이 어떻게 구현되는가**인데, 이는 아래 *Reference Latent (In-Context Editing)* 메커니즘이 담당한다.

---

## 1. Qwen-Image 모델 아키텍처 — DiT (Diffusion Transformer)

`comfy/ldm/qwen_image/model.py`의 `QwenImageTransformer2DModel`.

### 1.1 기본 제원

| 항목 | 값 | 의미 |
|------|----|----|
| 구조 | MMDiT 계열 (joint attention) | 이미지·텍스트 토큰을 **하나의 어텐션**에서 함께 처리 |
| Attention heads | 24 × head_dim 128 = inner_dim 3072 | 트랜스포머 폭 |
| Text 조건 차원 | `joint_attention_dim = 3584` | Qwen2.5-VL hidden size |
| 정규화 | `RMSNorm` (Q/K norm 포함) | LayerNorm 대비 경량·안정 |
| 위치 인코딩 | `RoPE`, `axes_dims_rope=(16,56,56)` | **3축(시간/H/W) 회전 위치 인코딩** |
| 모델 타입 | `ModelType.FLOW` | Flow-Matching (노이즈→데이터 velocity 예측) |

### 1.2 MMDiT joint attention
이미지 latent를 패치화한 토큰(`img`)과 텍스트 토큰(`txt`)을 **연결(concat)하여 동일한 self-attention 블록**에 통과시킨다 (`QwenImageTransformerBlock`). 덕분에 텍스트의 의미가 이미지 패치에 직접 attend되어, 프롬프트 지시가 픽셀 단위로 반영된다.

```
img tokens ─┐
            ├─→ [RMSNorm Q/K] → RoPE → joint self-attention → ...×N blocks
txt tokens ─┘
```

---

## 2. Reference Latent — "편집"의 핵심 메커니즘 ⭐

**가장 중요한 개념.** Qwen-Edit가 일반 text-to-image와 다른 이유가 여기 있다.

### 2.1 동작 원리 (In-Context / "Kontext" 방식)
참조 이미지를 VAE로 인코딩한 latent를 **노이즈 latent와 같은 토큰 시퀀스에 이어붙인다**.

`comfy/ldm/qwen_image/model.py` `_forward()` 핵심:
```python
kontext, kontext_ids, _ = self.process_img(ref, index=index, ...)
hidden_states = torch.cat([hidden_states, kontext], dim=1)   # 참조 토큰을 시퀀스에 추가
img_ids      = torch.cat([img_ids,      kontext_ids], dim=1) # RoPE 위치 ID도 함께
```

즉, 참조 이미지는 "또 다른 이미지 토큰 묶음"으로 트랜스포머에 들어가고, denoising 중인 노이즈 latent가 **어텐션을 통해 참조 토큰을 직접 참고**한다. 이것이 구도·피사체를 유지하면서 지시대로 편집되는 원리다.

### 2.2 조건 전달 경로
```
TextEncodeQwenImageEditPlus
  └─ vae.encode(참조이미지) → ref_latent
  └─ conditioning_set_values(cond, {"reference_latents":[ref_latent]}, append=True)
        ↓ (conditioning에 실려 전달)
model_base.QwenImage.extra_conds()
  └─ out['reference_latent'] = process_latent_in(reference_latents[-1])
        ↓
모델 _forward(ref_latents=...) → 토큰 시퀀스에 concat
```

### 2.3 두 가지 주입 방식 (워크플로우의 두 variant와 직결)
| 방식 | 노드 | 특징 |
|------|------|------|
| **인코더 내장** | `TextEncodeQwenImageEditPlus` 내부에서 `vae.encode` | 텍스트 인코딩과 동시에 ref latent 생성 (WF2의 활성 경로) |
| **명시적 주입** | `ReferenceLatent` 노드로 latent를 conditioning에 chain | 여러 참조를 직접 제어 (WF2의 "Raw Latent" variant) |

> `ReferenceLatent`는 체인하여 **다중 참조 이미지**를 줄 수 있다(`append=True`). `ref_latents_method`(index / negative_index / index_timestep_zero)로 참조 토큰의 위치 배치 전략을 바꾼다.

---

## 3. TextEncodeQwenImageEditPlus — VL 텍스트 인코더

`comfy_extras/nodes_qwen.py`. Qwen2.5-VL(비전-언어 모델)을 텍스트 인코더로 사용하는 점이 특징.

### 3.1 시스템 프롬프트(llama_template) 내장
인코더가 단순 텍스트 임베딩이 아니라, **"입력 이미지의 특징을 묘사한 뒤 사용자 지시대로 변형하라"**는 시스템 프롬프트를 자동으로 감싼다.
```
system: Describe the key features of the input image (color, shape, size,
texture, objects, background), then explain how the user's text instruction
should alter or modify the image. Generate a new image that meets the user's
requirements while maintaining consistency with the original ...
user: {프롬프트}
```
이 덕분에 "편집 의도 이해 → 변형"이라는 추론이 모델 내부에서 일어난다.

### 3.2 이중 해상도 처리 (중요)
입력 이미지를 **두 갈래로 다른 해상도**로 가공한다.

| 용도 | 해상도 | 목적 |
|------|--------|------|
| **VL 비전 토큰** (`images_vl`) | 384×384 기준 면적 | 텍스트 인코더가 이미지 *의미*를 이해 |
| **VAE ref latent** (`ref_latents`) | 1024×1024 기준, **8의 배수 정렬** | 실제 픽셀 복원용 참조 latent |

또한 `Picture N: <|vision_start|><|image_pad|><|vision_end|>` 토큰을 프롬프트 앞에 붙여 멀티 이미지(image1~3)를 구분한다.

> 💡 **시사점**: `TextEncodeQwenImageEditPlus`는 image1~3 입력으로 **최대 3장의 참조**를 동시에 받는다. (WF1·WF2는 1장만 사용)

---

## 4. ModelSamplingAuraFlow — Flow-Matching 스케줄

`comfy_extras/nodes_model_advanced.py`. `ModelSamplingSD3`를 상속하며, `shift` 파라미터로 시그마(노이즈) 스케줄을 조정한다.

### 4.1 Flow Matching이란
DDPM식 단계적 디노이징이 아니라, **노이즈→데이터를 잇는 직선 경로의 velocity를 예측**하는 방식. 적은 step으로도 안정적이라 Lightning 가속과 궁합이 좋다.

### 4.2 shift의 의미
- `shift`는 시그마 분포를 고노이즈/저노이즈 쪽으로 치우치게 한다.
- 본 워크플로우는 **shift=3** 사용 (기본값 1.73보다 높음) → 고해상도·편집 작업에서 구조 보존에 유리한 설정.
- `multiplier=1.0` (AuraFlow/Qwen 계열 표준).

---

## 5. CFGNorm — 저(低) step 가이던스 안정화

`comfy_extras/nodes_cfg.py`. **post-CFG 보정 함수**를 모델에 패치한다(`set_model_sampler_post_cfg_function`).

### 5.1 동작
CFG 결과(`denoised`)의 norm이 조건부 예측(`cond_denoised`)의 norm을 넘지 않도록 스케일을 클램프한다.
```python
scale = (‖cond‖ / (‖pred‖ + 1e-8)).clamp(0.0, 1.0)
return pred_text * scale * strength
```
- **효과**: 적은 step(4) + 낮은 CFG(1.0) 환경에서 발생하는 **과포화·색 깨짐·번짐**을 억제.
- `strength`(기본 1.0)로 보정 강도 조절.

> 같은 파일의 `CFGZeroStar`는 관련 기법(CFG-Zero★)으로, uncond 방향 성분을 최적 스케일로 보정한다. 본 워크플로우는 `CFGNorm`을 사용.

---

## 6. FluxKontextImageScale — 해상도 정규화

`comfy_extras/nodes_flux.py`. 입력 이미지를 **모델이 학습한 선호 해상도 집합** 중 종횡비가 가장 가까운 것으로 스냅(lanczos)한다.

선호 해상도(모두 약 1MP, 종횡비만 다름):
```
(672,1568) (688,1504) (720,1456) (752,1392) (800,1328) (832,1248)
(880,1184) (944,1104) (1024,1024) ... (1568,672)
```
- **목적**: 학습 분포에서 벗어난 임의 해상도로 인한 품질 저하/아티팩트 방지.
- WF1은 대신 `ImageScaleToTotalPixels`(1MP, 8 정렬)로 동일 목적을 달성.

---

## 7. Lightning 4-step LoRA — 추론 가속

`LoraLoaderModelOnly` 노드로 모델 가중치에만 LoRA를 적용.

### 7.1 원리
- **Distillation(증류) LoRA**: 50 step 분량의 생성 궤적을 4 step으로 압축하도록 학습된 보조 가중치.
- 가이던스가 가중치에 내재화되어 **CFG=1.0**(사실상 guidance off)에서 동작 → 추가 forward(uncond) 없이 빠름.

### 7.2 모드별 설정 대응
| 모드 | LoRA | Steps | CFG |
|------|------|-------|-----|
| 터보(기본) | ON | 4 | 1.0 |
| 고품질 | OFF | 20 | 4.0 |

> WF2의 `ComfySwitchNode` 3개 + `PrimitiveBoolean`이 이 표를 통째로 전환하는 토글이다.

---

## 8. 전체 데이터 흐름 종합도

```
[참조 이미지]
   │
   ├─(384px)─→ Qwen2.5-VL 비전 토큰 ┐
   │                                ├─→ TextEncodeQwenImageEditPlus
[프롬프트]──────────────────────────┘        │  (llama_template 적용)
   │                                          ↓
   └─(1024px, /8)─→ VAE.encode → ref_latent ─→ conditioning{reference_latents}
                                                  │
[편집 대상 이미지] → FluxKontextImageScale → VAEEncode → 노이즈 latent
                                                  │
   모델: UNETLoader → (Lightning LoRA) → ModelSamplingAuraFlow(shift=3) → CFGNorm
                                                  ↓
   KSampler(euler/simple, 4step, cfg1.0)
     └─ DiT forward: [노이즈 토큰 ⊕ ref 토큰] joint-attention × N → velocity 예측
                                                  ↓
                                          VAEDecode → 결과 이미지
```

---

## 9. 핵심 개념 한 줄 요약

| 기술 | 한 줄 정의 |
|------|-----------|
| **DiT (MMDiT)** | 이미지·텍스트 토큰을 한 어텐션에서 함께 처리하는 트랜스포머 확산 모델 |
| **Reference Latent** | 참조 이미지를 토큰 시퀀스에 이어붙여 어텐션으로 참고 → "편집"의 핵심 |
| **Qwen2.5-VL 인코더** | 이미지를 이해하는 비전-언어 텍스트 인코더 (시스템 프롬프트 내장) |
| **Flow Matching / shift** | 직선 경로 velocity 예측 + 시그마 스케줄 조정 (적은 step에 강함) |
| **CFGNorm** | 저 step·저 CFG에서 과포화 억제하는 가이던스 정규화 |
| **Kontext ImageScale** | 학습된 선호 해상도로 스냅하여 아티팩트 방지 |
| **Lightning LoRA** | 4 step·CFG 1.0 생성을 가능케 하는 증류 가속 LoRA |

---

## 10. 더 알아보기 (소스 위치)

| 기술 | 구현 파일 |
|------|-----------|
| Qwen DiT 모델 | `comfy/ldm/qwen_image/model.py` |
| ref_latent 조건 처리 | `comfy/model_base.py` (`class QwenImage`) |
| Qwen 텍스트 인코드 노드 | `comfy_extras/nodes_qwen.py` |
| ReferenceLatent | `comfy_extras/nodes_edit_model.py` |
| CFGNorm / CFGZeroStar | `comfy_extras/nodes_cfg.py` |
| ModelSamplingAuraFlow | `comfy_extras/nodes_model_advanced.py` |
| FluxKontextImageScale | `comfy_extras/nodes_flux.py` |
| Qwen2.5-VL 텍스트 인코더 | `comfy/text_encoders/qwen_image.py`, `qwen_vl.py` |

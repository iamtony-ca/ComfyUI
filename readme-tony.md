### 🛠️ 방법. 터미널에서 수동 설치 (매니저에 안 뜰 경우)

만약 매니저에서 검색이 안 되거나 에러가 난다면, 터미널에서 직접 설치하는 것이 가장 빠릅니다. (이전에 WAS Node Suite를 설치하셨던 방식과 동일합니다.)

터미널에서 `ComfyUI/custom_nodes` 폴더로 이동한 뒤 아래 명령어들을 복사해서 붙여넣으세요.

**1. Dynamic Prompts 설치:**
```bash
git clone https://github.com/adieyal/comfyui-dynamicprompts.git
```

**2. WAS Node Suite 설치 (혹시 설치가 안 되어 있다면):**
```bash
git clone https://github.com/WASasquatch/was-node-suite-comfyui.git
cd was-node-suite-comfyui
python -m pip install -r requirements.txt
cd ..
```

설치가 끝난 후 터미널에서 ComfyUI를 재시작(`python main.py`)하시면 누락된 노드들이 정상적으로 로드됩니다!
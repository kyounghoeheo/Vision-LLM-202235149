# Project03: Vision-LLM 기반 학교 분실물 자동 분류 및 기록

## 1. 프로젝트 목적

카메라로 분실물을 확인하여 YOLO가 물체를 탐지하고, 종류별 권장 보관함과 접수 시각을 JSON에 기록합니다. 사용자는 마이크로 분실물 접수 내역을 질문하고, Whisper가 음성을 텍스트로 변환하면 Gemma가 기록을 근거로 답변하며 Piper가 음성으로 출력합니다.

시연에서는 실물 분실물을 카메라에 보여주며, 준비물이 부족하면 A4에 인쇄한 물체 사진으로 대체할 수 있습니다. 

## 2. 파일 구조

```text
Vision-LLM-202235149/
├── Project03/
│   ├── vision.py
│   ├── assistant.py
│   ├── README.md
│   └── data/                          # 프로그램 실행 중 생성
│       ├── vision_data.json
│       ├── lost_items.json
│       ├── input.wav
│       └── response.wav
├── src/
│   └── models/
│       ├── YOLO/
│       │   ├── yolo11n_int8.engine   # Jetson 전용 (있을 때 우선 사용)
│       │   └── yolo11n.pt           # Windows 또는 Jetson 대체 모델
│       ├── Gemma4/
│       │   └── google_gemma-4-E2B-it-Q4_K_M.gguf
│       └── Piper/
│           ├── ko_KR-kss-medium.onnx
│           └── ko_KR-kss-medium.onnx.json
├── whisper.cpp/
│   ├── build-cpu/bin/whisper-cli
│   └── models/ggml-base.bin
└── .piper_venv/
    └── bin/python
```

`.engine`, `.pt`, `.gguf`, `.onnx` 모델 파일은 이 압축파일에 포함되어 있지 않습니다. 

## 3. 실행 방식
 저장소 루트 폴더를 VS Code에서 연 다음 터미널에서 실행합니다.

```powershell
py -3.15 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install ultralytics opencv-python
python Project03/vision.py
```

- Windows에서는 `src/models/YOLO/yolo11n.pt`가 있으면 사용하고 없으면 `yolo11n.pt`를 처음 실행할 때 내려받으려고 시도합니다(첫 실행 시 인터넷 필요).
- `yolo11n_int8.engine`는 Jetson의 TensorRT 엔진이므로 Windows에서 사용하지 않습니다.
- 웹캠이 열리지 않으면 `vision.py`의 `CAMERA_ID = 0`을 `1` 등으로 변경합니다.

### 최종 시연
터미널 1 (실시간 카메라 분류):

```bash
cd ~/Vision-LLM-202235149
python3 Project03/vision.py
```

터미널 2 (음성 질문 및 응답):

```bash
cd ~/Vision-LLM-202235149
python3 Project03/assistant.py
```

### Jetson 음성 기능 사전 확인

Whisper CLI와 Piper 별도 가상환경을 사용합니다. 다음 파일이 있는지 확인합니다.

```bash
ls whisper.cpp/build-cpu/bin/whisper-cli
ls whisper.cpp/models/ggml-base.bin
ls src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf
ls src/models/Piper/ko_KR-kss-medium.onnx
ls .piper_venv/bin/python
```

마이크 번호 확인:

```bash
arecord -l
```

## 4. 사용 방법 (발표 시연)

1. 가방 등 실물 분실물을 카메라 중앙의 노란 사각형에 보여줍니다. 실물이 없으면 A4 인쇄 사진을 사용해도 됩니다.
2. YOLO가 동일한 물체를 **연속 6개 처리 프레임** 감지하면 `data/lost_items.json`에 자동으로 등록됩니다.
3. 화면에 `REGISTERED - REMOVE ITEM`이 나타나면 물건(또는 사진)을 카메라에서 제거합니다.
4. `READY - SHOW NEXT ITEM`이 나타나면 휴대전화, 곰인형 등 다음 물건(또는 사진)을 차례대로 보여줍니다.
5. `assistant.py` 터미널에서 Enter를 눌러 "지금 접수된 분실물은 뭐야?", "전자기기는 몇 개야?", "가방은 어디에 보관해야 해?"처럼 질문합니다.
6. Whisper → Gemma → Piper 순서로 음성 답변이 출력됩니다.

키보드 (카메라 창에서):

- `q`: 카메라 프로그램 종료
- `r`: 인식하는 대상이 변경되었는데 `READY`로 돌아오지 않을 경우 다음 분실물 접수 준비

**동일한 물건을 계속 비추는 동안에는 중복 등록하지 않습니다.** 물건 또는 사진을 제거하여 검출이 6개 처리 프레임 동안 사라지면 다음 접수를 허용합니다. 프로그램을 종료했다 다시 실행하면 이전 기록은 유지되지만 같은 물건을 다시 접수할 수 있으므로, 재시연 전에는 중복 내역을 확인하세요.

## 5. 분류 기준

| YOLO 클래스 | 분류 | 권장 보관함 |
|---|---|---|
| backpack / handbag / suitcase | 가방류 | A 보관함 |
| cell phone / laptop | 전자기기 | B 보관함 |
| umbrella | 우산 | C 보관함 |
| teddy bear | 기타 | D 보관함 |

COCO 사전학습 모델이 지원하는 물건만 대상으로 합니다. 지갑, 열쇠, 에어팟, 실제 재질 등은 현재 인식 대상이 아닙니다. 보관함은 **시연용 분류 규칙**이며 실제로 물건을 물리적으로 보관했다는 뜻은 아닙니다.

## 6. 창의성 및 확장성

### 아이디어 설명

학교 분실물을 사람이 일일이 품목별로 구분하는 접수 과정을 단순화하기 위해, 카메라가 인식한 물건을 분류·기록하고 접수 정보를 음성으로 조회할 수 있도록 구성했습니다. 접수 여부는 일정 프레임 이상 같은 물체가 검출되는지로 판단합니다.

### 구현한 내용


- `vision.py`: `cv2.VideoCapture()` → `YOLO.predict()` → `result.boxes`를 Dictionary로 정리 → 자동 접수 이벤트 → `json.dump()` / `os.replace()`로 기록
- `assistant.py`: 기록된 JSON → 자연어 Context → `speech_to_text()` (Whisper CLI) → `Llama.create_chat_completion()` (Gemma) → `text_to_speech()` (Piper)
- Gemma의 답변은 접수 기록을 근거로 생성하며, 실제 품목 인식·분류·접수 여부는 **YOLO와 명시적 규칙**으로 결정합니다.

### 확장성

- 분실물 접수 위치, 습득 장소 등 관리자가 입력하는 항목을 추가할 수 있습니다. 현재 코드에는 구현되어 있지 않습니다.
- 향후 학습 데이터를 확보하면 COCO에서 지원하지 않는 지갑이나 열쇠 등도 분류 대상으로 확장할 수 있습니다.

## 7. 기타 참고 사항

- 실물은 단순한 배경에 하나씩 놓고 촬영합니다. A4 사진을 사용할 때는 대상이 크게 보이고 빛 반사가 없도록 준비합니다.
- **실물과 A4 인쇄 사진 모두 객체 검출이 보장되지는 않습니다.** 발표 전에 사용할 물체를 실제 카메라에서 테스트해야 합니다.
- 화면에 `REGISTERED`가 나오지 않으면 모델 confidence, 사진의 크기·각도, 카메라 조명, 탐지 클래스를 확인합니다.
- 새로운 시연을 빈 기록부터 시작하려면 두 프로그램을 종료하고 `Project03/data/lost_items.json`을 직접 삭제합니다. (기존 데이터는 삭제되므로 필요한 경우 백업하세요.)
- GitHub에는 대형 모델 파일을 올리지 마세요. 프로젝트 코드는 `Project03/`만 커밋하고, `Project03/data/`의 생성 파일도 제출 목적이 없다면 제외하는 것이 좋습니다.
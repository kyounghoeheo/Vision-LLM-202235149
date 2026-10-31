# Object Detection - NMS

## 파일 구조

```text
Vision-LLM-202235149/
└── 261025/
    └── 04/
        └── nms/
            ├── assignment04_nms.py
            └── README.md
```

## 실행 방법

`nms` 폴더에서 실행합니다.

```powershell
python assignment04_nms.py
```

## 필요 라이브러리

```powershell
pip install numpy opencv-python matplotlib scikit-image
```

## 기능

- `prune_low_conf()`: Confidence Score가 임계값보다 낮은 Bounding Box를 제거합니다.
- `apply_nms()`: Confidence Score가 높은 Bounding Box부터 선택하고, IoU 기준을 초과하는 중복 Bounding Box를 제거합니다.
- 임의의 Bounding Box와 Confidence Score를 생성하여 NMS 적용 전후를 시각화합니다.

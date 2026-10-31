# Object Detection - IoU

## 파일 구조

```
Vision-LLM-202235149/
├── 261025/
│   └── 04/
│       └── iou/
│           ├── assignment04.py
│           └── README.md
└── src/
    └── images/
        ├── two_apples1.png
        ├── two_apples2.png
        └── two_apples3.png
```

## 실행 방법

프로젝트 최상위 폴더에서 실행합니다.

```powershell
python 261025/04/assignment04.py
```

## 필요 라이브러리

```powershell
pip install opencv-python numpy matplotlib
```

## 기능

각 이미지에서 빨간 사과와 초록 사과의 Bounding Box를 검출하고 IoU를 계산하여 결과를 출력하고 표시합니다.

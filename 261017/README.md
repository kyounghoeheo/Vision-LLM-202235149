# Assignment 02 - Basketball Contour Detection

## 파일 구조 (필수)

- **assignment02.py**: 농구공 Binary Mask 생성, Morphological Operation 및 외곽선 검출
- **src/**: 컴퓨터 비전 실습에 사용하는 데이터 및 이미지 폴더
- **src/images/basketball.jpg**: 농구공 외곽선 검출에 사용하는 이미지

## 실행 방식 (필수)

1. 필요한 라이브러리 설치
   ```bash
   pip install opencv-python numpy matplotlib
   ```

2. Python 파일 실행
   ```bash
   python assignment02.py
   ```

3. 실행 결과
   - Opening 및 Closing을 이용한 노이즈 제거
   - Binary Mask에서 외곽선 검출
   - 가장 큰 외곽선을 선택하여 농구공 영역 표시
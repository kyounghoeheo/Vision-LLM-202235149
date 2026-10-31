# Convolution CPU/GPU 실습

## 파일 구조 (필수)

- **convolution_cpu_gpu.py**: NumPy(CPU)와 CuPy(GPU)로 2차원 Convolution을 구현하고 결과를 비교합니다.

## 실행 방식 (필수)

`src/images/seagull.jpg`가 있는 프로젝트 루트에서 `python convolution_cpu_gpu.py`를 실행합니다.
CPU/GPU 처리 시간과 세로·가로 Convolution 결과 이미지 6개가 표시됩니다.

## 기타 참고 사항 (과제 & 프로젝트 모두 해당)

```powershell
python -m pip install numpy matplotlib opencv-python "cupy-cuda12x[ctk]"
```

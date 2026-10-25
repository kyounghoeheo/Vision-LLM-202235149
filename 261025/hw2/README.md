# 이미지 잔상 효과 실습

## 파일 구조 (필수)

- **motion_trail_cpu_gpu.py**: NumPy(CPU)와 CuPy(GPU)로 이미지 잔상 효과를 구현하고 처리 시간과 결과 이미지를 비교합니다.

## 실행 방식 (필수)

`src/images/seagull.jpg`가 있는 프로젝트 루트에서 `python motion_trail_cpu_gpu.py` 실행합니다.
원본 이미지, CPU 결과, GPU 결과가 화면에 표시됩니다.

## 기타 참고 사항 (과제 & 프로젝트 모두 해당)

```powershell
python -m pip install numpy matplotlib opencv-python "cupy-cuda12x[ctk]"
```

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# 일반 .py 환경에서 이미지 불러오기 (프로젝트 루트에서 실행하는 파일)
image_path = Path(__file__).resolve().parent.parent / "src" / "images" / "basketball.jpg"
if not image_path.is_file():
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

ball_bgr = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
if ball_bgr is None:
    raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")
ball = cv2.cvtColor(ball_bgr, cv2.COLOR_BGR2RGB)


ball_blur = cv2.GaussianBlur(ball, (7, 7), 0)
ball_blur_hsv = cv2.cvtColor(ball_blur, cv2.COLOR_RGB2HSV)


ball_lower1, ball_upper1 = (115, 15, 165), (145, 120, 240)
ball_lower2, ball_upper2 = (146, 15, 165), (170, 120, 240)
ball_mask1 = cv2.inRange(ball_blur_hsv, ball_lower1, ball_upper1)
ball_mask2 = cv2.inRange(ball_blur_hsv, ball_lower2, ball_upper2)
ball_mask = cv2.bitwise_or(ball_mask1, ball_mask2)


# TODO: Morphological operation을 적용하여 최종 binary mask 생성
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
ball_mask_opened2 = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel, iterations=2)
ball_mask_closed3 = cv2.morphologyEx(ball_mask_opened2, cv2.MORPH_CLOSE, kernel, iterations=3)


# TODO: Binary mask로 외곽선 검출
final_ball_mask = ball_mask_closed3
final_img = ball.copy()
contours, _ = cv2.findContours(
    final_ball_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


# TODO: 최종 농구공 영역 및 외곽선 선택
if len(contours) > 0:
    largest_contour = max(contours, key=cv2.contourArea)

    final_mask = np.zeros_like(final_ball_mask)
    cv2.drawContours(
        final_mask,
        [largest_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    mask_color = np.zeros_like(final_img)
    mask_color[:, :, 1] = final_mask
    alpha = 0.35
    final_img = cv2.addWeighted(final_img, 1.0, mask_color, alpha, 0)

    cv2.drawContours(final_img, [largest_contour], -1, (0, 255, 0), 2)

plt.imshow(final_img)
plt.axis("off")
plt.show()

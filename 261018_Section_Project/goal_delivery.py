# -*- coding: utf-8 -*-
"""Section 2 Project: 손과 초록색 공의 목표 도달 게임.

게임 창 선택 후 q: 종료, r: 처음부터
Jetson Nano에서 USB YUYV 카메라를 사용하는 코드입니다.
"""

# [0] 라이브러리
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# [1] 설정
FLIP_CODE = 1             # 1: 좌우, 0: 상하, -1: 둘 다, None: 반전 없음
SHOW_MASK = True           # 공 검출 확인용 흑백 창도 표시

# 실제 공의 색과 조명에 따라 조정할 수 있습니다.
green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)

MIN_AREA_RATIO = 0.0005    # 화면 면적의 0.05% 미만인 작은 검출은 무시
TARGET_RADIUS_RATIO = 0.12  # 목표 원 반지름: 화면의 짧은 변 * 0.12
HAND_MARGIN_RATIO = 0.10    # 손가락-공 중심 허용 거리: 공 반지름 + 짧은 변 * 0.10
HOLD_FRAMES = 5             # 성공 조건을 연속 5개 처리 프레임에서 만족해야 함
MAX_MISSING_FRAMES = 8      # 미검출 시 예측 원을 보여 줄 최대 처리 프레임 수

#  화면 크기가 달라도 비슷한 위치에 목표가 생깁니다.
# (가로 비율, 세로 비율): 왼쪽 -> 위쪽 -> 오른쪽
target_positions = [(0.25, 0.55), (0.50, 0.38), (0.75, 0.55)]
target_index = 0            # 현재 목표의 인덱스: 0, 1, 2
score = 0
hold_count = 0


# [2] 영상 전처리 설정
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# [3] Kalman Filter 설정
# 상태: [x, y, vx, vy], 측정값: [x, y]
kalman = cv2.KalmanFilter(4, 2)
kalman.transitionMatrix = np.array(
    [[1, 0, 1, 0],
     [0, 1, 0, 1],
     [0, 0, 1, 0],
     [0, 0, 0, 1]], dtype=np.float32,
)
kalman.measurementMatrix = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0]], dtype=np.float32,
)
kalman.processNoiseCov = np.array(
    [[1e-2, 0, 0, 0],
     [0, 1e-2, 0, 0],
     [0, 0, 5e-2, 0],
     [0, 0, 0, 5e-2]], dtype=np.float32,
)
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
kalman.errorCovPost = np.eye(4, dtype=np.float32)
found = False             
missing_frames = 0
last_radius = 0


# [4] MediaPipe 손 인식 모델 로드
project_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(project_dir, "..", "src", "models", "MediaPipe", "hand_landmarker.task")
with open(model_path, "rb") as model_file:
    model_data = model_file.read()

base_option = python.BaseOptions(model_asset_buffer=model_data)
options = vision.HandLandmarkerOptions(base_options=base_option, num_hands=2)
hand_detector = vision.HandLandmarker.create_from_options(options)
connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
finger_tips = (4, 8, 12, 16, 20)

# [5] USB 카메라 YUYV GStreamer 파이프라인
pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    raise RuntimeError("USB 카메라를 열 수 없습니다. 지원 형식을 확인하세요.")

print("게임 시작: 초록색 공을 손 가까이에 두고 1 -> 2 -> 3 목표로 옮기세요.")
print("게임 창을 클릭한 뒤 q: 종료, r: 처음부터. 판정은 화면상의 2D 거리입니다.")

while True:
    # [6] 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("카메라에서 프레임을 받지 못해 종료합니다.")
        break

    # 카메라 화면 반전
    if FLIP_CODE is not None:
        frame = cv2.flip(frame, FLIP_CODE)
    h, w = frame.shape[:2]
    short_side = min(h, w)
    target_radius = int(short_side * TARGET_RADIUS_RATIO)
    hand_margin = short_side * HAND_MARGIN_RATIO

    # [7] Gaussian -> LAB -> CLAHE -> 색 분할
    frame_blur = cv2.GaussianBlur(frame, (7, 7), 0)
    frame_lab = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(frame_lab)
    enhanced_l = clahe.apply(l)
    enhanced_lab = cv2.merge((enhanced_l, a, b))
    mask = cv2.inRange(enhanced_lab, green_lower, green_upper)

    # [8] 모폴로지 노이즈 제거
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    final_mask = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    # [9] 공의 외곽선과 중심 찾기
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    object_detected = False
    center = None
    radius = 0
    largest_contour = None

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) >= h * w * MIN_AREA_RATIO:
            _, radius = cv2.minEnclosingCircle(largest_contour)
            M = cv2.moments(largest_contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            center = (cX, cY)
            object_detected = True

    # [10] 칼만 필터로 공 위치 추적
    tracked_center = None
    if object_detected:
        missing_frames = 0
        last_radius = radius
        if not found:
            # 첫 검출 위치로 필터 초기화
            initial_state = np.array([[center[0]], [center[1]], [0.0], [0.0]], dtype=np.float32)
            kalman.statePre = initial_state.copy()
            kalman.statePost = initial_state.copy()
            kalman.errorCovPost = np.eye(4, dtype=np.float32)
            found = True
        z = np.array([[center[0]], [center[1]]], dtype=np.float32)
        predicted_state = kalman.predict()
        corrected_state = kalman.correct(z)
        tracked_center = (int(corrected_state[0, 0]), int(corrected_state[1, 0]))
    elif found:
        missing_frames += 1
        if missing_frames <= MAX_MISSING_FRAMES:
            predicted_state = kalman.predict()
            tracked_center = (int(predicted_state[0, 0]), int(predicted_state[1, 0]))
        else:
            found = False  # 다음 실제 검출 때 새 위치에서 다시 초기화

    # [11] 손 검출
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = hand_detector.detect(mp_image)

    # [12] 공과 가장 가까운 손가락 끝 찾기
    hand_points = []
    nearest_tip = None
    hand_distance = None
    for hand in result.hand_landmarks:
        points = [(int(p.x * w), int(p.y * h)) for p in hand]
        hand_points.append(points)
        if object_detected:
            for tip_index in finger_tips:
                tip = points[tip_index]
                distance = np.linalg.norm(np.array(tip) - np.array(center))
                if hand_distance is None or distance < hand_distance:
                    hand_distance = distance
                    nearest_tip = tip
    hand_near = hand_distance is not None and hand_distance <= radius + hand_margin

    # [13] 손과 공의 목표 도달 판정
    if target_index < len(target_positions):
        tx, ty = target_positions[target_index]
        target_center = (int(w * tx), int(h * ty))
        if not object_detected:
            status = "SHOW GREEN BALL"
            hold_count = 0
        elif not hand_near:
            status = "KEEP HAND NEAR BALL"
            hold_count = 0
        else:
            measured_distance = np.linalg.norm(np.array(center) - np.array(target_center))
            tracked_distance = np.linalg.norm(np.array(tracked_center) - np.array(target_center))
            if measured_distance <= target_radius and tracked_distance <= target_radius:
                hold_count += 1
                status = "HOLD AT TARGET"
                if hold_count >= HOLD_FRAMES:
                    score += 1
                    target_index += 1
                    hold_count = 0
                    print(f"목표 {score} 성공! 점수: {score}/{len(target_positions)}")
                    status = "SUCCESS - NEXT TARGET"
            else:
                status = "MOVE TO TARGET"
                hold_count = 0
    if target_index == len(target_positions):
        status = "ALL TARGETS COMPLETE! PRESS R"

    # [14] 게임 화면 출력
    display = frame.copy()
    if object_detected:
        largest_mask = np.zeros_like(final_mask)
        cv2.drawContours(largest_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        mask_color = np.zeros_like(display)
        mask_color[:, :, 2] = largest_mask
        display = cv2.addWeighted(display, 1.0, mask_color, 0.15, 0)
        cv2.drawContours(display, [largest_contour], -1, (255, 100, 0), 2)
        cv2.circle(display, center, 5, (255, 100, 0), -1)
    if tracked_center is not None:
        cv2.circle(display, tracked_center, int(last_radius), (0, 255, 255), 2)
        cv2.circle(display, tracked_center, 4, (0, 255, 255), -1)
    for points in hand_points:
        for c in connections:
            cv2.line(display, points[c.start], points[c.end], (0, 255, 0), 2)
        for i, point in enumerate(points):
            color = (0, 0, 255) if i in finger_tips else (255, 0, 0)
            cv2.circle(display, point, 6 if i in finger_tips else 4, color, -1)
    if object_detected and nearest_tip is not None:
        cv2.line(display, center, nearest_tip, (0, 165, 255), 2)

    for i, (tx, ty) in enumerate(target_positions):
        goal = (int(w * tx), int(h * ty))
        if i < target_index:
            color = (0, 200, 0)
        elif i == target_index:
            color = (255, 0, 255)
        else:
            color = (130, 130, 130)
        cv2.circle(display, goal, target_radius, color, 3 if i == target_index else 1)
        cv2.putText(display, str(i + 1), (goal[0] - 10, goal[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    ball_text = "VISIBLE" if object_detected else "MISSING"
    tracking_text = "MEASURED + KALMAN" if object_detected else "SEARCHING"
    if not object_detected and tracked_center is not None:
        tracking_text = "PREDICTED ONLY - NO SCORE"
    cv2.putText(display, f"Score: {score}/{len(target_positions)}  |  Hold: {hold_count}/{HOLD_FRAMES}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display, status, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(display, f"Hands: {len(hand_points)}  |  Ball: {ball_text}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display, tracking_text, (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(display, "Q: quit  |  R: restart  |  Purple: current target", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    cv2.imshow("Green Ball Delivery", display)
    if SHOW_MASK:
        cv2.imshow("Green Ball Mask", final_mask)

    # 키 입력
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("r"):
        target_index = 0
        score = 0
        hold_count = 0
        found = False
        missing_frames = 0
        print("게임을 처음부터 시작합니다.")

# 게임 종료 후 장치 정리
cap.release()
hand_detector.close()
cv2.destroyAllWindows()
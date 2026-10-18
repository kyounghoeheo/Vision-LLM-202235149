import cv2
import numpy as np
from numpy.linalg import inv

def KalmanFilter(mu_prev, sigma_prev, z):
    mu_bar = A_t.dot(mu_prev)
    sigma_bar = A_t.dot(sigma_prev).dot(A_t.transpose()) + R_t
    if z is None:
        return mu_bar, sigma_bar
    else:
        K_t = sigma_bar.dot(C_t.transpose()).dot(inv(C_t.dot(sigma_bar).dot(C_t.transpose()) + Q_t))
        mu = mu_bar + K_t.dot(z - C_t.dot(mu_bar))
        sigma = (np.identity(2) - K_t.dot(C_t)).dot(sigma_bar)
        return mu, sigma


# Kalman filter 변수 정의
A_t = np.array([[1, 1], [0, 1]])
G = np.array([[0.5], [1]])
R_t = G.dot(G.transpose())
C_t = np.array([[1, 0]])
Q_t = np.array([[1]])
mu_t = np.array([[0, 0], [0, 0]])
sigma_t = np.array([[0, 0], [0, 0]])

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
    raise RuntimeError("카메라를 열 수 없습니다. GStreamer 파이프라인과 카메라 연결을 확인하세요.")

green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
contour_lst, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
found = False

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라 프레임을 읽지 못했습니다.")
            break

        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        lab_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

        mask = cv2.inRange(lab_frame, green_lower, green_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (center_x, center_y), radius = cv2.minEnclosingCircle(largest_contour)
            center = (int(round(center_x)), int(round(center_y)))
            radius = int(round(radius))

            if radius > 0:
                object_mask = np.zeros_like(mask)
                cv2.drawContours(
                    object_mask,
                    [largest_contour],
                    -1,
                    255,
                    thickness=cv2.FILLED,
                )
                green_overlay = np.zeros_like(frame)
                green_overlay[object_mask > 0] = (0, 255, 0)
                frame = cv2.addWeighted(frame, 1.0, green_overlay, 0.3, 0)

                cv2.circle(frame, center, radius, (0, 255, 0), 2)
                cv2.circle(frame, center, 4, (0, 0, 255), cv2.FILLED)
        


# TODO: Kalman Filter를 활용하여 객체 추적
# TODO: 탐지된 객체의 중심과 반지름을 구해 프레임 위에 overlay
# TODO: 예측된 객체의 중심과 반지름을 구해 프레임 위에 다른 색으로 overlay

    

# 객체 최초 검출 여부 확인용 boolean
  


        frame = cap.read()[1]

        if len(contour_lst) > 0:
        # 가장 큰 contour 선택
            contour = max(contour_lst, key=cv2.contourArea)
        # 최소 외접원 반지름
            radius = int(cv2.minEnclosingCircle(contour)[1])
        # 무게중심
            center = tuple(map(int, cv2.moments(contour)["m10"] / cv2.moments(contour)["m00"],
                            int(cv2.moments(contour)["m01"] / cv2.moments(contour)["m00"])))
                        

        # 검출된 객체에 파란 원 overlay
            cv2.circle(frame, center, int(radius), (255, 0, 0), 2)
            cv2.circle(frame, center, 4, (0, 0, 255), cv2.FILLED)

        # 객체 최초 검출
            if not found:
                found = True

    # 최초 검출 이후 Kalman Filter 적용
    # 측정값 사용 (visible) -> Prediction & Update
        if found and (len(contour_lst) > 0):
            mu_t, sigma_t = KalmanFilter(mu_t, sigma_t, np.array([list(center)]))
            x_bel, y_bel = mu_t[0][0], mu_t[0][1]

    # 측정값 미사용 (occluded) -> Prediction
        elif found and (len(contour_lst) <= 0):
            mu_t, sigma_t = KalmanFilter(mu_t, sigma_t, None)
            x_bel, y_bel = mu_t[0][0], mu_t[0][1]

    # 예측한 객체 위치에 노란 원 overlay
        cv2.circle(frame, (int(x_bel), int(y_bel)), int(radius), (0, 255, 255), 2)
        cv2.circle(frame, (int(x_bel), int(y_bel)), 4, (0, 255, 255), cv2.FILLED)
        cv2.imshow("Real-time Green Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
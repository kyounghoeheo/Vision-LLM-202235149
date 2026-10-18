import cv2
import numpy as np

pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

# TODO: 배운 기술(이미지 전처리, 특징 추출)을 활용하여 실시간으로 객체를 검출
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다. GStreamer 파이프라인과 카메라 연결을 확인하세요.")


# 초록 LAB lower/upper range
green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)

# TODO: 각 프레임 이미지를 전처리
# TODO: Color Segmentation으로 Binary Mask 생성
# TODO: Morphological Operation을 적용하여 객체 영역 추출
# TODO: 객체의 중심과 반지름을 구해 프레임 위에 overlay
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

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
        cv2.imshow("Real-time Green Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()



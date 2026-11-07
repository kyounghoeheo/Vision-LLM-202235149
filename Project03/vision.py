import cv2
import json
import os
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


# 06 강의: 카메라 -> YOLO -> Dictionary -> JSON
PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
JSON_PATH = DATA_DIR / "vision_data.json"
TEMP_JSON_PATH = DATA_DIR / "vision_data_tmp.json"
RECORD_PATH = DATA_DIR / "lost_items.json"
TEMP_RECORD_PATH = DATA_DIR / "lost_items_tmp.json"

ENGINE_PATH = ROOT_DIR / "src/models/YOLO/yolo11n_int8.engine"
PT_PATH = ROOT_DIR / "src/models/YOLO/yolo11n.pt"
YOLO_MODEL_PATH = (
    str(ENGINE_PATH) if os.name != "nt" and ENGINE_PATH.is_file()
    else str(PT_PATH) if PT_PATH.is_file()
    else "yolo11n.pt"
)

CAMERA_ID = 0
CONFIDENCE = 0.65
STABLE_FRAMES = 6
CLEAR_FRAMES = 6

# COCO 클래스만 사용 (추가 학습 없음)
ITEM_INFO = {
    "backpack": ("가방류", "A 보관함"),
    "handbag": ("가방류", "A 보관함"),
    "suitcase": ("가방류", "A 보관함"),
    "cell phone": ("전자기기", "B 보관함"),
    "laptop": ("전자기기", "B 보관함"),
    "umbrella": ("우산", "C 보관함"),
    "teddy bear": ("기타", "D 보관함"),
}
ITEM_CLASS_IDS = [24, 25, 26, 28, 63, 67, 77]


def save_json(path, temp_path, data):
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    os.replace(temp_path, path)


def load_records():
    if RECORD_PATH.is_file():
        with open(RECORD_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return []


def main():
    DATA_DIR.mkdir(exist_ok=True)
    records = load_records()

    yolo = YOLO(YOLO_MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다. CAMERA_ID를 확인하세요.")
        return

    candidate = None
    stable_count = 0
    clear_count = 0
    ready = True

    print("q : 종료")
    print("r : 다음 분실물 접수 준비 (종이를 교체했으나 자동 초기화되지 않을 때)")
    print("사진 한 장을 카메라 중앙에 보여주면 자동으로 등록됩니다.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("카메라 프레임을 읽지 못했습니다.")
                break

            height, width = frame.shape[:2]
            x_min, x_max = int(width * 0.15), int(width * 0.85)
            y_min, y_max = int(height * 0.10), int(height * 0.90)

            # 1. 현재 프레임 YOLO 탐지
            results = yolo.predict(
                source=frame,
                conf=CONFIDENCE,
                iou=0.5,
                classes=ITEM_CLASS_IDS,
                verbose=False,
            )
            result = results[0]

            # 2. YOLO 결과를 Dictionary 형태로 변환
            objects = []
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                class_name = result.names[class_id]

                # A4 사진의 물체가 중앙 검사 구역에 있는지만 확인
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                if not (x_min <= center_x <= x_max and y_min <= center_y <= y_max):
                    continue
                if class_name not in ITEM_INFO:
                    continue

                objects.append({
                    "class": class_name,
                    "confidence": round(confidence, 3),
                    "bbox": {
                        "x1": int(x1), "y1": int(y1),
                        "x2": int(x2), "y2": int(y2),
                    },
                })

            vision_dict = {
                "image_width": width,
                "image_height": height,
                "objects": objects,
            }

            # 3. 가장 확실한 물체 1개가 연속 검출되면 '접수' 이벤트 발생
            current = max(objects, key=lambda obj: obj["confidence"], default=None)
            if current is not None:
                clear_count = 0
                if candidate == current["class"]:
                    stable_count += 1
                else:
                    candidate = current["class"]
                    stable_count = 1

                if ready and stable_count >= STABLE_FRAMES:
                    class_name = current["class"]
                    category, box_name = ITEM_INFO[class_name]
                    record = {
                        "id": max((item["id"] for item in records), default=0) + 1,
                        "class": class_name,
                        "category": category,
                        "recommended_box": box_name,
                        "registered_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                        "confidence": current["confidence"],
                    }
                    records.append(record)

                    # 4. 강의와 동일하게 임시 JSON 파일 작성 후 교체
                    save_json(JSON_PATH, TEMP_JSON_PATH, vision_dict)
                    save_json(RECORD_PATH, TEMP_RECORD_PATH, records)

                    print(f"[접수 #{record['id']}] {class_name} -> {category} / {box_name}")
                    ready = False
            else:
                candidate = None
                stable_count = 0
                clear_count += 1
                if clear_count >= CLEAR_FRAMES:
                    ready = True

            # 5. YOLO 화면 출력
            output_frame = result.plot()
            cv2.rectangle(output_frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
            status = "READY - SHOW NEXT ITEM" if ready else "REGISTERED - REMOVE ITEM"
            cv2.putText(output_frame, status, (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(output_frame, f"Registered: {len(records)}", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.imshow("Lost & Found - YOLO", output_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                ready = True
                candidate = None
                stable_count = 0
                clear_count = 0
                print("다음 분실물을 접수할 준비가 되었습니다.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

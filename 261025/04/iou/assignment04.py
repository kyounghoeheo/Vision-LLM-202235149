from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


def convert_bbox_coord(bbox):
    x, y, w, h = bbox
    return x, y, x + w, y + h


def calculate_iou(bbox_a, bbox_b):
    xa_1, ya_1, xa_2, ya_2 = bbox_a
    xb_1, yb_1, xb_2, yb_2 = bbox_b

    intersection_x1 = max(xa_1, xb_1)
    intersection_y1 = max(ya_1, yb_1)
    intersection_x2 = min(xa_2, xb_2)
    intersection_y2 = min(ya_2, yb_2)

    w = max(0, intersection_x2 - intersection_x1)
    h = max(0, intersection_y2 - intersection_y1)
    intersection_area = w * h

    area_a = max(0, xa_2 - xa_1) * max(0, ya_2 - ya_1)
    area_b = max(0, xb_2 - xb_1) * max(0, yb_2 - yb_1)
    union_area = area_a + area_b - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def draw_bbox(img, coord1, coord2, color, thickness=3, txt=""):
    _, txt_h = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.rectangle(img, coord1, coord2, color, thickness)
    cv2.putText(img, txt, (coord1[0], coord1[1] - txt_h + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def get_contour(img_bgr):
    red_lower1 = np.array([0, 100, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 100, 50])
    red_upper2 = np.array([180, 255, 255])
    green_lower = np.array([25, 10, 30])
    green_upper = np.array([100, 255, 255])

    img_blr = cv2.GaussianBlur(img_bgr, (7, 7), 0)
    blr_hsv = cv2.cvtColor(img_blr, cv2.COLOR_BGR2HSV)

    red_mask1 = cv2.inRange(blr_hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(blr_hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    grn_mask = cv2.inRange(blr_hsv, green_lower, green_upper)

    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_grn, _ = cv2.findContours(grn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cntr_red, cntr_grn = None, None

    if len(contours_red) > 0:
        cntr_red = max(contours_red, key=cv2.contourArea)
    if len(contours_grn) > 0:
        cntr_grn = max(contours_grn, key=cv2.contourArea)

    return cntr_red, cntr_grn


def main():
    image_dir = Path(__file__).resolve().parents[3] / "src" / "images"
    images_bgr = []

    for i in range(1, 4):
        image_path = image_dir / f"two_apples{i}.png"
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        images_bgr.append(image)

    images_rgb = []
    ious = []

    for i, image in enumerate(images_bgr, start=1):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cntr_red, cntr_grn = get_contour(image)
        if cntr_red is None or cntr_grn is None:
            raise ValueError(f"Could not detect both apples in two_apples{i}.png")

        bbox1 = cv2.boundingRect(cntr_red)
        bbox2 = cv2.boundingRect(cntr_grn)

        x_r, y_r, w_r, h_r = bbox1
        x_g, y_g, w_g, h_g = bbox2

        draw_bbox(image_rgb, (x_r, y_r), (x_r + w_r, y_r + h_r),
                  (255, 0, 0), txt="Red Apple")
        draw_bbox(image_rgb, (x_g, y_g), (x_g + w_g, y_g + h_g),
                  (0, 255, 0), txt="Green Apple")

        iou = calculate_iou(convert_bbox_coord(bbox1), convert_bbox_coord(bbox2))
        print(f"IoU {i}: {iou:.4f}")

        images_rgb.append(image_rgb)
        ious.append(iou)

    plt.figure(figsize=(16, 6))

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(images_rgb[i])
        plt.title(f"IoU: {ious[i]:.4f}")
        plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()

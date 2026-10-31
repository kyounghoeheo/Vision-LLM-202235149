from random import randint, gauss

import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.data


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


def random_bbox(img, min_w=100, min_h=400, max_w=600, max_h=600):
    img_h, img_w = img.shape[:2]
    w = randint(min_w, min(max_w, img_w))
    h = randint(min_h, min(max_h, img_h))
    x1 = randint(0, img_w - w)
    y1 = randint(0, img_h - h)
    return x1, y1, x1 + w, y1 + h


def random_conf():
    return min(1.0, max(0.0, gauss(0.6, 0.1)))


def prune_low_conf(boxes, scores, conf_threshold=0.5):
    pruned_boxes = []
    pruned_scores = []

    for box, score in zip(boxes, scores):
        if score >= conf_threshold:
            pruned_boxes.append(box)
            pruned_scores.append(score)

    return pruned_boxes, pruned_scores


def apply_nms(boxes, scores, iou_threshold=0.5):
    nms_boxes = []
    nms_scores = []

    sorted_indices = np.argsort(scores)[::-1]

    while len(sorted_indices) > 0:
        best_idx = sorted_indices[0]
        nms_boxes.append(boxes[best_idx])
        nms_scores.append(scores[best_idx])

        remaining_indices = []
        for idx in sorted_indices[1:]:
            iou = calculate_iou(boxes[best_idx], boxes[idx])
            if iou <= iou_threshold:
                remaining_indices.append(idx)

        sorted_indices = np.array(remaining_indices, dtype=int)

    return nms_boxes, nms_scores


def main():
    hubble = skimage.data.hubble_deep_field()
    boxes = []
    scores = []

    for _ in range(20):
        boxes.append(random_bbox(hubble))
        scores.append(random_conf())

    original_image = hubble.copy()
    for box, score in zip(boxes, scores):
        draw_bbox(original_image, (box[0], box[1]), (box[2], box[3]),
                  (randint(0, 255), randint(0, 255), randint(0, 255)),
                  txt=f"{score:.2f}")

    new_boxes, new_scores = prune_low_conf(boxes, scores)
    new_boxes, new_scores = apply_nms(new_boxes, new_scores, iou_threshold=0.2)

    print(len(boxes), len(scores))
    print(len(new_boxes), len(new_scores))

    nms_image = hubble.copy()
    for box, score in zip(new_boxes, new_scores):
        draw_bbox(nms_image, (box[0], box[1]), (box[2], box[3]),
                  (randint(0, 255), randint(0, 255), randint(0, 255)),
                  txt=f"{score:.2f}")

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Before NMS")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(nms_image)
    plt.title("After NMS")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()

import cv2
import numpy as np

def preprocess_image(img, size=(224, 224)):
    """Preprocess image for model input."""
    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img_normalized = img_resized / 255.0  # Normalize to [0, 1]
    return np.expand_dims(img_normalized, axis=0).astype(np.float32)

def non_max_suppression(boxes, overlap_thresh=0.3):
    """Suppress overlapping boxes."""
    if len(boxes) == 0:
        return []

    boxes = np.array([[x1, y1, x2, y2, score] for x1, y1, x2, y2, score in boxes])
    x1, y1, x2, y2, scores = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(boxes[i])
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / areas[order[1:]]

        order = order[1:][overlap <= overlap_thresh]
    return keep
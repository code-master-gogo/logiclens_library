import cv2
import numpy as np
from skimage.feature import hog
import pickle
import tensorflow as tf
from .utils import preprocess_image, non_max_suppression

class PersonDetector:
    def __init__(self, hog_model_path="models/hog_svm.pkl", tflite_model_path="models/mobilenet_quant.tflite"):
        # Load HOG + SVM model
        with open(hog_model_path, 'rb') as f:
            self.svm = pickle.load(f)
        self.hog = cv2.HOGDescriptor()

        # Load quantized MobileNet TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def detect(self, image_path, confidence_threshold=0.5):
        """Detect persons in an image."""
        # Load and preprocess image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = preprocess_image(img_rgb, size=(224, 224))  # For MobileNet

        # Step 1: HOG + SVM for initial detection (fast)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hog_features = self.hog.compute(gray)
        boxes, weights = self.hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)

        # Step 2: Refine with MobileNet TFLite (accurate)
        refined_boxes = []
        for (x, y, w, h) in boxes:
            roi = img_resized[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            roi_resized = preprocess_image(roi, size=(224, 224))

            # Run TFLite inference
            self.interpreter.set_tensor(self.input_details[0]['index'], roi_resized)
            self.interpreter.invoke()
            score = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]

            if score > confidence_threshold:
                refined_boxes.append((x, y, x+w, y+h, score))

        # Apply non-maximum suppression to remove overlapping boxes
        final_boxes = non_max_suppression(refined_boxes, overlap_thresh=0.3)
        return final_boxes

    def draw_boxes(self, image_path, boxes):
        """Draw detected boxes on the image."""
        img = cv2.imread(image_path)
        for (x1, y1, x2, y2, score) in boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img
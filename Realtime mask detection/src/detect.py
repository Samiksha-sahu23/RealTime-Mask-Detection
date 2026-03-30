import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1. Setup Absolute Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model paths
MASK_MODEL_PATH = os.path.join(BASE_DIR, "model", "mask_detector.h5")
PROTOTXT_PATH = os.path.join(BASE_DIR, "face_detector", "deploy.prototxt")
WEIGHTS_PATH = os.path.join(BASE_DIR, "face_detector", "res10_300x300_ssd_iter_140000.caffemodel")

# 2. Debugging Check
for path in [MASK_MODEL_PATH, PROTOTXT_PATH, WEIGHTS_PATH]:
    if not os.path.exists(path):
        print(f"Error: Missing file at {path}")
        exit()

print("All model files found. Loading networks...")

# 3. Load Models 
faceNet = cv2.dnn.readNet(PROTOTXT_PATH, WEIGHTS_PATH)
maskNet = load_model(MASK_MODEL_PATH)

def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    # Module 3: Object Detection (Face Localization)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces, locs, preds = [], [], []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Preprocessing for CNN
            face = frame[startY:endY, startX:endX]
            if face.size == 0: continue
            
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        #Pattern Analysis (Inference)
        preds = maskNet.predict(np.array(faces), batch_size=32)

    return (locs, preds)

# 4. Main Video Loop
print("Starting Webcam...")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        
        # Mapping results to classes
        label_idx = np.argmax(pred)
        class_labels = ["Mask On", "No Mask", "Incorrect Mask"]
        
        # Color coding: Green for Mask, Red for No Mask/Incorrect
        color = (0, 255, 0) if label_idx == 0 else (0, 0, 255)
        
        text = f"{class_labels[label_idx]}: {pred[label_idx]*100:.2f}%"
        cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Vision3D-Pro: Mask Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
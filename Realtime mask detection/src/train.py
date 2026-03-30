import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 1. Configuration
BASE_PATH = r"C:\Users\Samiksha sahu\OneDrive\Desktop\FaceMaskDetection\dataset\archive (5)"
IMG_PATH = os.path.join(BASE_PATH, "images")
ANNOT_PATH = os.path.join(BASE_PATH, "annotations")

# All possible classes in the dataset
CLASS_MAP = {"with_mask": 0, "without_mask": 1, "mask_weared_incorrectly": 2}

def load_and_preprocess_data():
    data = []
    labels = []
    
    print("Parsing XML Annotations and Preprocessing Images...")
    
    if not os.path.exists(ANNOT_PATH):
        print(f"Error: Cannot find path {ANNOT_PATH}")
        return None, None

    for xml_file in os.listdir(ANNOT_PATH):
        if not xml_file.endswith(".xml"): continue
        
        tree = ET.parse(os.path.join(ANNOT_PATH, xml_file))
        root = tree.getroot()
        
        img_file = root.find("filename").text
        img_full_path = os.path.join(IMG_PATH, img_file)
        image = cv2.imread(img_full_path)
        
        if image is None: continue

        for obj in root.findall("object"):
            label_name = obj.find("name").text
            if label_name not in CLASS_MAP: continue
            
            # Extract Bounding Box 
            bbox = obj.find("bndbox")
            xmin = max(0, int(bbox.find("xmin").text))
            ymin = max(0, int(bbox.find("ymin").text))
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            
            # Crop & Resize 
            face = image[ymin:ymax, xmin:xmax]
            if face.size == 0: continue # Skip invalid crops
            
            face = cv2.resize(face, (224, 224))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            
            data.append(face)
            labels.append(CLASS_MAP[label_name])
            
    if len(data) == 0:
        print("Error: No valid samples were loaded.")
        return None, None

    # Normalization
    X = np.array(data) / 255.0
    y_raw = np.array(labels)
    
    # Identify unique classes  present in  data
    unique_classes = np.unique(y_raw)
    num_classes = len(unique_classes)
    
    print(f"Loaded {len(data)} samples across {num_classes} active classes.")
    
    # Map labels 
    y_remapped = np.searchsorted(unique_classes, y_raw)
    y = to_categorical(y_remapped, num_classes=num_classes)
    
    return X, y, num_classes

def build_cnn_model(num_classes):
    # Transfer Learning using MobileNetV2 
    baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    
    # Use dynamically detected num_classes to prevent shape mismatch
    headModel = Dense(num_classes, activation="softmax")(headModel)

    model = Model(inputs=baseModel.input, outputs=headModel)
    
    for layer in baseModel.layers:
        layer.trainable = False
        
    return model

if __name__ == "__main__":
    # Load Data and get actual class count
    X, y, detected_classes = load_and_preprocess_data()
    
    if X is not None:
        # Split Data
        (trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

        # Build & Compile
        print(f"Compiling CNN Model for {detected_classes} classes...")
        model = build_cnn_model(detected_classes)
        opt = Adam(learning_rate=1e-4)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

        # Training
        print("Starting Training...")
        H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=10, batch_size=32)

        # Saving the result
        if not os.path.exists("model"): os.makedirs("model")
        model.save("model/mask_detector.h5")
        print("Success: Model saved as model/mask_detector.h5")
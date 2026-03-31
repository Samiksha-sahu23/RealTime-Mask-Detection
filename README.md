# Real-Time Mask Detection System 

## Project
This is a Computer Vision system that is designed to detect and classify face mask usage in real-time. It checks whether the the person in live web camera is wearing the mask properly or not . This gives a solution to a major world problem of people ignoring safety regulations in places where it is required .For example hospital premises where wearing masks is a necessity for preventing further contamination of possible dieases . This will encourage people to take safety measures and in return will help in reducing possible health problems .

<img width="288" height="175" alt="image" src="https://github.com/user-attachments/assets/27eaef19-229c-4e18-b5c4-ac67f1954955" />


---

##  Key Features  
* **Real-Time Inference**: Video feed processes at ~10-12 FPS .
* **SSD Localization**: Model detects face even in varying lighting conditions.
* **Design**: This project has separate training and detection pipelines structure.

---

##  Technical aspects
* **Language**: Python 3.13
* **Models**: MobileNetV2 (Classifier), ResNet-10 SSD (Face Detector)

---

##  Project Structure
```text
FaceMaskDetection/
├── dataset/             # Images and XML Annotations
├── face_detector/       # SSD Model files (deploy.prototxt & caffemodel)
├── model/               # Final trained model (mask_detector.h5)
├── src/
│   ├── train.py         # Data parsing & CNN training script
│   └── detect.py        # Real-time webcam  script
├── requirements.txt     # List of dependencies
---
```

##  Installation & Usage

### 1. Clone the Repository

```bash
git clone [https://github.com/Samiksha-sahu23/RealTime-Mask-Detection.git]
cd RealTime-Mask-Detection

```

### 2. Environment Setup 
```
python -m venv venv
source venv/bin/scripts/activate

# Install required libraries
pip install -r requirements.txt
```
### 3.  Dataset Setup

1. Go to the https://www.kaggle.com/datasets/andrewmvd/face-mask-detection

2.  Download the ZIP file and extract it.

3. Create a dataset folder like given in the project structure and move the annotations and images folders from the extracted ZIP into the dataset folder.


### 4. Points to keep in mind

Ensure that the folder structure matches the project structure given her :

Face detector files should be in face_detector/ folder.

Ensure that the trained model mask_detector.h5 is in the model/ folder.

Update the paths according to your system where it is necessary in all the code . Ensure that the path is correct.

### 5. Training the Model

To train the MobileNetV2 classifier using dataset, run:

```bash
python src/train.py

```

### 6.Real-Time Detection

Ensure your webcam is connected, then execute:

```bash
python src/detect.py
```
o Exit: Click on the video window and press the 'q' key to exit the window and stop the process

### Project Results 

Training Accuracy: 91.90%

Validation Accuracy: 91.65%

Loss: 0.2053

Successful real-time classification of "No Mask" status eg:-
<img width="634" height="508" alt="image" src="https://github.com/user-attachments/assets/fa7b013f-152b-4f6c-aed2-23d94f45ef06" />

### License

MIT License - Developed as a Capstone BYOP Project for VITyarthi.



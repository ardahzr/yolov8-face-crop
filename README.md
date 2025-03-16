# YOLOv8 Face Crop

This Python script uses the `ultralytics` YOLOv8 model to detect faces in images, dynamically enlarges bounding boxes based on face size, and saves cropped face images.

## Features
- Uses YOLOv8 face detection (`yolov8l-face.pt`)
- Filters detections based on confidence score
- Dynamically enlarges bounding boxes based on face size relative to image width
- Saves cropped face images in an output directory

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- Ultralytics (`YOLO`)
- Matplotlib
- NumPy

## Installation
```bash
pip install ultralytics opencv-python matplotlib numpy

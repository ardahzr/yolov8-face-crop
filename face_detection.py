from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt

# Detect faces and enlarge bounding boxes

def detect_faces_and_enlarge_bbox(input_dir, output_dir):
    model = YOLO('/content/yolov8l-face.pt')  # Load YOLOv8 face model

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"{image_file} could not be loaded.")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model(image_rgb)  # Detect faces

        faces = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
        confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores

        if len(faces) == 0:
            print(f"No face detected in {image_file}.")
            continue

        image_height, image_width = image.shape[:2]

        for i, face in enumerate(faces):
            x1, y1, x2, y2 = map(int, face[:4])
            conf = confidences[i]

            if conf > 0.84:  # Process only high-confidence detections
                width = x2 - x1
                height = y2 - y1

                face_ratio = width / image_width
                enlarge_factor = 0.8 - face_ratio

                new_width = int(width * (1 + enlarge_factor))
                new_height = int(height * (1 + enlarge_factor))

                new_x1 = max(0, x1 - (new_width - width) // 2)
                new_y1 = max(0, y1 - (new_height - height) // 2)
                new_x2 = min(image_width, new_x1 + new_width)
                new_y2 = min(image_height, new_y1 + new_height)

                cropped_image = image[new_y1:new_y2, new_x1:new_x2]
                output_image_path = os.path.join(output_dir, f"cropped_{image_file}")
                cv2.imwrite(output_image_path, cropped_image)
                print(f"Saved: {output_image_path}")

# Example usage
input_directory = '/images'  # Input image folder
output_directory = '/cropped_images2'  # Output cropped images folder

detect_faces_and_enlarge_bbox(input_directory, output_directory)

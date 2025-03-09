"""
    ReWoo, for object detection

"""

import torch
import threading
import openai
import cv2
from ultralytics import YOLO

# Load YOLOv8 Model (optimized for speed)
yolo_model = YOLO("yolov8n.pt")  # 'n' (nano) is optimized for speed
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model.to(device)

# Confidence threshold for automatic acceptance
CONFIDENCE_THRESHOLD = 0.7

# Function to detect objects in an image
def detect_objects(image_path):
    results = yolo_model(image_path)
    detections = []

    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            label = result.names[int(cls)]
            detections.append({
                "label": label,
                "confidence": float(conf),
                "bbox": box.cpu().numpy().tolist()
            })
    return detections

# Function to run ReWoo reasoning for uncertain cases
def rewoo_reasoning(label, confidence):
    prompt = f"""
    The object detection model is uncertain about the classification of an object.
    It detected '{label}' with {confidence:.2f} confidence.
    Can you verify if this is correct based on common visual characteristics?
    """
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": "You are an AI visual reasoning expert."},
                  {"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"].strip()

# ReWoo: Parallel Execution
def process_image(image_path):
    detections = detect_objects(image_path)
    final_labels = []

    def process_detection(detection):
        label, confidence = detection["label"], detection["confidence"]
        if confidence < CONFIDENCE_THRESHOLD:
            refined_label = rewoo_reasoning(label, confidence)
        else:
            refined_label = label

        final_labels.append({"label": refined_label, "bbox": detection["bbox"], "confidence": confidence})

    threads = []
    for detection in detections:
        thread = threading.Thread(target=process_detection, args=(detection,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return final_labels

# Example Usage
image_path = "test_image.jpg"  # Replace with actual image path
final_results = process_image(image_path)
print("Final Object Detections:", final_results)

import os
import cv2
import random
import torch
from ultralytics import YOLO

# --- CONFIG ---
video_path = "15sec_input_720p.mp4"
model_path = "your_model.pt"  # <-- Replace with your actual model path
output_dir = "player_dataset"
gallery_dir = os.path.join(output_dir, "gallery")
query_dir = os.path.join(output_dir, "query")
player_class_id = 2  # Only keep "player" class
frame_skip = 3       # Process every Nth frame to reduce redundancy

# --- SETUP ---
os.makedirs(gallery_dir, exist_ok=True)
os.makedirs(query_dir, exist_ok=True)

# Load model
model = YOLO(model_path)

# Open video
cap = cv2.VideoCapture(video_path)
frame_id = 0
player_id_counter = 0
player_crops = []

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    if frame_id % frame_skip != 0:
        frame_id += 1
        continue

    results = model(frame)[0]

    for i, det in enumerate(results.boxes.data):
        x1, y1, x2, y2, conf, cls = det.tolist()
        if int(cls) != player_class_id:
            continue

        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0:
            continue

        filename = f"player_{player_id_counter}_frame{frame_id}.jpg"
        player_crops.append((filename, crop))
        player_id_counter += 1

    frame_id += 1

cap.release()
print(f"[INFO] Extracted {len(player_crops)} player crops")

# --- SPLIT INTO GALLERY & QUERY ---
random.shuffle(player_crops)
split_idx = len(player_crops) // 2

for i, (filename, crop) in enumerate(player_crops):
    target_dir = gallery_dir if i < split_idx else query_dir
    cv2.imwrite(os.path.join(target_dir, filename), crop)

print(f"[DONE] Saved {split_idx} images to gallery and {len(player_crops) - split_idx} to query")

# shoreline_annotation.py

import os
import time
import json
import cv2
import torch
import numpy as np
import sys
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- adjust this if your helper modules live elsewhere ---
sys.path.insert(1, "./functions")

from data_preprocessing   import load_single_image
from pytorch_models       import hed_cnn, pretrained_weights, hed_predict_single
from data_visualisation   import extract_filtered_shoreline_coords

# --- 1. device & model init ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load your HED model
weightsPath = "./models/vgg16_4.pt"
model = hed_cnn().to(device)
model = pretrained_weights(model, weightsPath=weightsPath, applyWeights=True, hedIn=True)
model.eval()

# --- 2. stream + sizes + output folder ---
stream_url = "./videos/twinlakes2.mp4"
cap        = cv2.VideoCapture(stream_url)
imSize     = (480, 640)       # model’s (H, W)
out_dir    = "./annotated_frames/twinlakes2"
os.makedirs(out_dir, exist_ok=True)

# helper: write LabelMe JSON
def write_labelme_json(image_path, coords, image_shape, label="shoreline"):
    h, w = image_shape
    shapes = [{
        "label":      label,
        "points":     [[x,y] for x,y in coords],
        "group_id":   None,
        "shape_type": "polygon",
        "flags":      {}
    }]
    data = {
        "version":     "0.3.3",
        "flags":       {},
        "shapes":      shapes,
        "imagePath":   os.path.basename(image_path),
        "imageData":   None,
        "imageHeight": h,
        "imageWidth":  w,
        "text":        ""
    }
    json_path = os.path.splitext(image_path)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

# main loop
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            # we hit end-of-file (or unrecoverable read error) → exit loop
            print("▶︎ end of video, exiting.")
            break
        # original size
        orig_h, orig_w = frame.shape[:2]

        # preprocess to model size
        img_small = load_single_image(frame, imSize)
        if img_small.max()>1: 
            img_small = img_small / 255.0
        inp = torch.from_numpy(img_small.transpose((2,0,1)))\
                   .unsqueeze(0).float().to(device)

        # forward
        with torch.no_grad():
            pred = hed_predict_single(model, inp)

        # --- 3. extract shoreline coords at model-size ---
        coords_model = extract_filtered_shoreline_coords(
            pred.cpu(),
            thres=0.6,
            blur_ksize=(13,13),
            base_confidence=0.3,
            percentile=60,
            min_contour_length=400
        )

        # --- 4. scale to original ---
        coords_orig = [
            (
                x * (orig_w / imSize[1]),
                y * (orig_h / imSize[0])
            )
            for x,y in coords_model
        ]

        # --- 5. save frame + json ---
        ts   = int(time.time()*1000)
        name = f"frame_{ts}"
        img_path = os.path.join(out_dir, name + ".jpg")
        cv2.imwrite(img_path, frame)
        write_labelme_json(img_path, coords_orig, (orig_h, orig_w))

        # throttle if you like
        time.sleep(0.05)

except KeyboardInterrupt:
    print("Interrupted, cleaning up…")
finally:
    cap.release()
    cv2.destroyAllWindows()

import time
import cv2
import torch
import sys
from datetime import datetime
import os
import numpy as np

# Set environment variables
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add custom functions to the system path
sys.path.insert(1, './functions')
import data_preprocessing
import data_visualisation
import pytorch_models
from data_preprocessing import load_single_image
from data_visualisation import plot_refined_single_prediction
from pytorch_models import hed_cnn, pretrained_weights, hed_predict_single

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("CUDA is available. Using GPU for computations.")
else:
    print("CUDA is not available. Using CPU for computations.")

# Load the model and move it to the selected device
start_time = time.time()
weightsPath = './models/vgg16_4.pt'
hedModel = hed_cnn().to(device)  # Move model to GPU/CPU
hedModel = pretrained_weights(hedModel, weightsPath=weightsPath, applyWeights=True, hedIn=True)
print(f"Load model time: {time.time() - start_time:.4f} seconds")

# Open the video stream
cap = cv2.VideoCapture("./testing/input/videos/walton_lighthouse/walton_lighthouse-2025-02-05-212624Z.mp4")

# Define image size
imSize = (480, 640)

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Create output directories
output_dir = "./testing/output/coords/walton_lighthouse-2025-02-05-212624Z"
os.makedirs(output_dir, exist_ok=True)

frame_counter = 0
retry_counter = 0
max_retries = 20  # Maximum number of retries

while cap.isOpened():

    # Measure time to read one frame from the stream
    start_time = time.time()
    ret, frame = cap.read()
    read_time = time.time() - start_time

    if ret:
        retry_counter = 0  # Reset retry counter on successful frame read
        frame_counter += 1  # Increment frame counter

        # Measure time for image preprocessing
        start_time = time.time()
        imgData = load_single_image(frame, imSize)
        if imgData.max() > 1:
            imgData = imgData / 255
        imgData = torch.from_numpy(imgData.transpose((2, 0, 1))).float().unsqueeze(0).to(device)  # Move input to GPU
        preprocessing_time = time.time() - start_time

        # Measure time for model prediction
        start_time = time.time()
        hedModel.eval()  # Ensure model is in evaluation mode
        with torch.no_grad():  # Disable gradient computation for faster inference
            model_pred = hed_predict_single(hedModel, imgData)  # Prediction on GPU
        prediction_time = time.time() - start_time

        # Measure time for post-processing
        start_time = time.time()
        frame_image, pred_coords = plot_refined_single_prediction(
            imgData.cpu(), model_pred.cpu(), thres=0.5, cvClean=True, imReturn=True
        )  # Move outputs to CPU for OpenCV
        postprocess_time = time.time() - start_time

        # Save the frame and coordinates
        if frame_image is not None and pred_coords is not None:
            # Save the frame as an image
            frame_filename = os.path.join(output_dir, f"frame_{frame_counter:04d}.jpg")
            cv2.imwrite(frame_filename, frame_image)
            print(f"Saved frame: {frame_filename}")

            # Save the coordinates as a text file
            coords_filename = os.path.join(output_dir, f"frame_{frame_counter:04d}_coords.txt")
            np.savetxt(coords_filename, pred_coords, fmt='%d')
            print(f"Saved coordinates: {coords_filename}")

        # Print timings for each step (optional)
        total = read_time + preprocessing_time + prediction_time + postprocess_time
        print(f"Frame {frame_counter}: Total processing time = {total:.4f} seconds")
        if frame_counter == 100:
            break

    else:
        print(f"Error: Could not read frame. Retrying ({retry_counter + 1}/{max_retries})...")
        time.sleep(60)  # Wait for 60 seconds before retrying
        retry_counter += 1

        # Reinitialize cap if retry limit is reached
        if retry_counter >= max_retries:
            print("Max retries reached. Reinitializing video stream...")
            cap.release()
            cap = cv2.VideoCapture("http://stage-ams-nfs.srv.axds.co/stream/adaptive/ucsc/walton_lighthouse/hls.m3u8")
            retry_counter = 0

# Release resources
cap.release()
print("Processing complete. All frames and coordinates saved.")
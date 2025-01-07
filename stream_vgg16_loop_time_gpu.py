import time
import cv2
import torch
import sys
from datetime import datetime 

import os
#os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

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
cap = cv2.VideoCapture("http://stage-ams-nfs.srv.axds.co/stream/adaptive/ucsc/walton_lighthouse/hls.m3u8")
#cap = cv2.VideoCapture("./testing/walton_lighthouse/walton_lighthouse-2024-12-02-221643Z.mp4")

# Define image size
imSize = (960, 1280)
#imSize = (1920, 2560)
#imSize = (320, 480)

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
else:
    print("Video file opened successfully.")

frame_counter = 0
retry_counter = 0
max_retries = 10  # Maximum number of retries

while cap.isOpened():
    # Check current time
    now = datetime.now()
    current_hour = now.hour
    # If the current time is outside 7 AM to 7 PM, handle downtime
    if current_hour < 7 or current_hour >= 19:
        print("STREAM OFF: Current time is outside operational hours (7 AM to 7 PM).")
        time.sleep(300)  # Wait for 5 minutes before checking again
        continue

    # Measure time to read one frame from the stream
    start_time = time.time()
    ret, frame = cap.read()
    read_time = time.time() - start_time

    if ret:
        retry_counter = 0  # Reset retry counter on successful frame read

        # Skip every other frame
        if frame_counter % 7 != 0:
            frame_counter += 1
            continue
        
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
        frame_image = plot_refined_single_prediction(imgData.cpu(), model_pred.cpu(), thres=0.5, cvClean=True, imReturn=True)  # Move outputs to CPU for OpenCV
        postprocess_time = time.time() - start_time

        # Measure time to display the result
        start_time = time.time()
        frame_image = cv2.resize(frame_image, (1280, 960))

        # Create a named window and set it to fullscreen
        # cv2.namedWindow('PreviewWindow', cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty('PreviewWindow', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.namedWindow('PreviewWindow', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('PreviewWindow', 1280, 960)  # Set desired dimensions
        cv2.moveWindow('PreviewWindow', 80, 30) 

        cv2.imshow('PreviewWindow', frame_image)
        display_time = time.time() - start_time

        # Print timings for each step
        print(f"Read time: {read_time:.4f} seconds")
        print(f"Preprocessing time: {preprocessing_time:.4f} seconds")
        print(f"Prediction time: {prediction_time:.4f} seconds")
        print(f"Post-process time: {postprocess_time:.4f} seconds")
        print(f"Display time: {display_time:.4f} seconds")
        total = read_time + preprocessing_time + prediction_time + postprocess_time + display_time
        print(f"Total time: {total:.4f} seconds")
        # Exit loop if 'ESC' key is pressed
        if cv2.waitKey(10) & 0xFF == 27:
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

# Release resources and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
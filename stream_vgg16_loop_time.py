# Load model time: 0.3209 seconds
# Read time: 0.0037 seconds
# Preprocessing time: 0.0058 seconds
# Prediction time: 0.4698 seconds
# Post-process time: 0.2441 seconds
# Display time: 0.0016 seconds
import sys
import time
import cv2
import torch

sys.path.insert(1, './functions')

import data_preprocessing
import data_visualisation
import pytorch_models
from data_preprocessing import load_single_image
from data_visualisation import plot_refined_single_prediction
from pytorch_models import hed_cnn, pretrained_weights, hed_predict_single

# Load the model and measure time taken
start_time = time.time()
weightsPath = './models/vgg16_2.pt'
hedModel = hed_cnn()
hedModel = pretrained_weights(hedModel, weightsPath=weightsPath, applyWeights=True, hedIn=True)
print(f"Load model time: {time.time() - start_time:.4f} seconds")

# Open the local video file for testing
cap = cv2.VideoCapture("./testing/walton_lighthouse-2024-08-05-142219Z.mp4")
# Or open the Walton stream
# cap = cv2.VideoCapture("http://stage-ams-nfs.srv.axds.co/stream/adaptive/ucsc/walton_lighthouse/hls.m3u8")

# Define image size
imSize = (320, 480)

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
else:
    print("Video file opened successfully.")

# Process the video stream
while cap.isOpened():
    # Measure time to read one frame from the stream
    start_time = time.time()
    ret, frame = cap.read()
    read_time = time.time() - start_time

    if ret:
        # Measure time for image preprocessing
        start_time = time.time()
        imgData = load_single_image(frame, imSize)
        if imgData.max() > 1:
            imgData = imgData / 255
        imgData = torch.from_numpy(imgData.transpose((2, 0, 1))).float().unsqueeze(0)
        preprocessing_time = time.time() - start_time

        # Measure time for model prediction
        start_time = time.time()
        model_pred = hed_predict_single(hedModel, imgData)
        prediction_time = time.time() - start_time

        # Measure time for post-processing
        start_time = time.time()
        frame_image = plot_refined_single_prediction(imgData, model_pred, thres=0.8, cvClean=True, imReturn=True)
        postprocess_time = time.time() - start_time

        # Measure time to display the result
        start_time = time.time()
        cv2.imshow('PreviewWindow', frame_image)
        display_time = time.time() - start_time

        # Print timings for each step
        print(f"Read time: {read_time:.4f} seconds")
        print(f"Preprocessing time: {preprocessing_time:.4f} seconds")
        print(f"Prediction time: {prediction_time:.4f} seconds")
        print(f"Post-process time: {postprocess_time:.4f} seconds")
        print(f"Display time: {display_time:.4f} seconds")

        # Exit loop if 'ESC' key is pressed
        if cv2.waitKey(10) & 0xFF == 27:
            break
    else:
        break

# Release resources and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
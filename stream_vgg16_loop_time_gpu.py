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


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"


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
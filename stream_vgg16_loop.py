# Inserting path for custom module
import sys
sys.path.insert(1, './functions')

import data_preprocessing
import data_visualisation
import pytorch_models
from data_preprocessing import load_single_image
from data_visualisation import plot_refined_single_prediction
from pytorch_models import hed_cnn, pretrained_weights, hed_predict_single

import cv2
import torch

weightsPath = 'vgg16.pt'# Import necessary libraries
hedModel = hed_cnn()
hedModel = pretrained_weights(hedModel, weightsPath=weightsPath, applyWeights=True, hedIn=True)

import cv2
import torch



# Open local video file
cap = cv2.VideoCapture("walton_lighthouse-2024-07-09-225235Z.mp4")
imSize = (320,480)

# Check if video file is opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
else:
    print("Video file opened successfully.")

# Process the video stream if it's opened successfully
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        imgData = load_single_image(frame, imSize)
        if imgData.max() > 1:
            imgData = imgData / 255
        imgData = torch.from_numpy(imgData.transpose((2, 0, 1))).float().unsqueeze(0)
        model_pred = hed_predict_single(hedModel, imgData)
        frame_image = plot_refined_single_prediction(imgData, model_pred, thres=0.8, cvClean=True, imReturn=True)
        # Display the processed frame
        cv2.imshow('PreviewWindow', frame_image)

        # Break the loop if 'ESC' key is pressed
        if cv2.waitKey(10) & 0xFF == 27:
            break
    else:
        break
# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()



import sys
import cv2
import torch

sys.path.insert(1, './functions')

import data_preprocessing
import data_visualisation
import pytorch_models
from data_preprocessing import load_single_image
from data_visualisation import plot_refined_single_prediction
from pytorch_models import hed_cnn, pretrained_weights, hed_predict_single

# Load the model 
weightsPath = './models/vgg16_4.pt'
hedModel = hed_cnn()
hedModel = pretrained_weights(hedModel, weightsPath=weightsPath, applyWeights=True, hedIn=True)

# Open the local video file for testing
#cap = cv2.VideoCapture("./testing/walton_lighthouse-2024-08-05-142219Z.mp4")
#cap = cv2.VideoCapture("./testing/check/twinlake.mp4")
cap = cv2.VideoCapture("./testing/poster_clip/untitled2.mp4")

# Or open the Walton stream
#cap = cv2.VideoCapture("http://stage-ams-nfs.srv.axds.co/stream/adaptive/ucsc/walton_lighthouse/hls.m3u8")

# Define image size
imSize = (320, 480)

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
else:
    print("Video file opened successfully.")

# Process the video stream
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        # image preprocessing
        imgData = load_single_image(frame, imSize)
        if imgData.max() > 1:
            imgData = imgData / 255
        imgData = torch.from_numpy(imgData.transpose((2, 0, 1))).float().unsqueeze(0)

        # model prediction
        model_pred = hed_predict_single(hedModel, imgData)

        # post-processing
        frame_image = plot_refined_single_prediction(imgData, model_pred, thres=0.8, cvClean=True, imReturn=True)

        # display the result
        cv2.imshow('PreviewWindow', frame_image)

        # Exit loop if 'ESC' key is pressed
        if cv2.waitKey(10) & 0xFF == 27:
            break
    else:
        break

# Release resources and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
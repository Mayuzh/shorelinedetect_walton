import time
import cv2
import torch
import sys
import threading
from datetime import datetime
import os
from queue import Queue

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(1, './functions')
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

# Video stream URL and parameters
#stream_url = "http://stage-ams-nfs.srv.axds.co/stream/adaptive/ucsc/walton_lighthouse/hls.m3u8"
stream_url = "./testing/input/videos/walton_lighthouse/walton_lighthouse-2024-12-02-221643Z.mp4"
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit(1)
else:
    print("Video stream opened successfully.")

imSize = (480, 640)  # Input image size for processing

# Global variables for frame capture
frame_queue = Queue(maxsize=30)  # Buffer up to 30 frames
capture_running = True

def frame_capture():
    """Thread function to continuously capture frames and put them in a queue."""
    global capture_running, cap
    while capture_running:
        now = datetime.now()
        # Only attempt to capture frames during operational hours (7 AM to 7 PM)
        if now.hour < 7 or now.hour >= 19:
            time.sleep(300)  # Sleep for 5 minutes if outside operational hours
            continue

        ret, frame = cap.read()
        if ret:
            if not frame_queue.full():
                frame_queue.put(frame)
            else:
                print("Frame queue is full, dropping frame.")
        else:
            print("Error: Could not read frame in capture thread. Retrying...")
            time.sleep(1)  # Short wait for transient issues
        time.sleep(0.01)  # Yield control briefly

# Start the frame capture thread
capture_thread = threading.Thread(target=frame_capture)
capture_thread.start()

# Set desired output FPS and calculate frame interval (in seconds)
desired_fps = 30
frame_interval = 1.0 / desired_fps

try:
    while True:
        # Check operational hours (7 AM to 7 PM)
        now = datetime.now()
        if now.hour < 7 or now.hour >= 19:
            print("STREAM OFF: Outside operational hours (7 AM to 7 PM).")
            time.sleep(300)  # Wait 5 minutes before checking again
            continue

        # Safely retrieve the most recent frame from the queue
        if not frame_queue.empty():
            frame = frame_queue.get()
        else:
            print("No frame available, skipping processing.")
            time.sleep(frame_interval)
            continue

        # Preprocess the frame
        start_time = time.time()
        imgData = load_single_image(frame, imSize)
        if imgData.max() > 1:
            imgData = imgData / 255
        imgData = torch.from_numpy(imgData.transpose((2, 0, 1))).float().unsqueeze(0).to(device)
        preprocessing_time = time.time() - start_time

        # Run model prediction
        start_time = time.time()
        hedModel.eval()
        with torch.no_grad():
            model_pred = hed_predict_single(hedModel, imgData)
        prediction_time = time.time() - start_time

        # Post-process prediction to generate output image
        start_time = time.time()
        frame_image, _ = plot_refined_single_prediction(
            imgData.cpu(), model_pred.cpu(), thres=0.6, cvClean=True, imReturn=True
        )
        postprocess_time = time.time() - start_time

        # Prepare and display the output image
        start_time = time.time()
        frame_image = cv2.resize(frame_image, (1280, 960))
        cv2.namedWindow('PreviewWindow', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('PreviewWindow', 1280, 960)
        cv2.moveWindow('PreviewWindow', 80, 30)
        cv2.imshow('PreviewWindow', frame_image)
        display_time = time.time() - start_time

        # Optionally, you can log timing info for debugging
        # print(f"Pre: {preprocessing_time:.4f}, Pred: {prediction_time:.4f}, Post: {postprocess_time:.4f}, Disp: {display_time:.4f}")

        # Exit loop if 'ESC' key is pressed
        if cv2.waitKey(10) & 0xFF == 27:
            break

        # Sleep to maintain the desired frame rate
        time.sleep(frame_interval)

except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    # Clean up resources and stop the capture thread
    capture_running = False
    capture_thread.join()
    cap.release()
    cv2.destroyAllWindows()
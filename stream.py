import time
import cv2
import torch
import sys
import threading
from datetime import datetime 
import os

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
stream_url = "./videos/walton_lighthouse-2024-12-23-233057Z.mp4"
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    print("Video stream opened successfully.")

imSize = (480, 640)  # Input image size for processing

# Global variables for frame capture
latest_frame = None
latest_frame_time = None  # Timestamp in seconds when the frame was captured
capture_running = True
lock = threading.Lock()
retry_counter = 0
max_retries = 5  # Reduced maximum retries for faster recovery

def frame_capture():
    """Thread function to continuously capture frames, along with their timestamps."""
    global latest_frame, latest_frame_time, cap, capture_running, retry_counter
    while capture_running:
        now = datetime.now()
        # Only attempt to capture frames during operational hours (7 AM to 7 PM)
        # if now.hour < 7 or now.hour >= 19:
        #     time.sleep(300)  # Sleep for 5 minutes if outside operational hours
        #     continue

        ret, frame = cap.read()
        if ret:
            with lock:
                latest_frame = frame
                latest_frame_time = time.time()  # Record the capture time
            retry_counter = 0  # Reset retry counter on success
        else:
            retry_counter += 1
            print(f"Error: Could not read frame in capture thread. Retrying ({retry_counter}/{max_retries})...")
            time.sleep(1)  # Short wait for transient issues
            if retry_counter >= max_retries:
                print("Max retries reached in capture thread. Reinitializing video stream...")
                cap.release()
                cap = cv2.VideoCapture(stream_url)
                retry_counter = 0
        time.sleep(0.01)  # Yield control briefly

# Start the frame capture thread
capture_thread = threading.Thread(target=frame_capture)
capture_thread.start()

# Set desired output FPS and calculate frame interval (in seconds)
desired_fps = 7
frame_interval = 1.0 / desired_fps
last_processed_time = 0  # We'll update this with the capture time of the last processed frame

try:
    while True:
        # Check operational hours (7 AM to 7 PM)
        now = datetime.now()
        # if now.hour < 7 or now.hour >= 19:
        #     print("STREAM OFF: Outside operational hours (7 AM to 7 PM).")
        #     time.sleep(300)  # Wait 5 minutes before checking again
        #     continue

        # Safely retrieve the most recent frame and its capture time
        with lock:
            frame = latest_frame.copy() if latest_frame is not None else None
            frame_capture_time = latest_frame_time if latest_frame_time is not None else 0

        if frame is None:
            print("No frame available, skipping processing.")
            continue

        # Check if the new frame's capture time is sufficiently ahead of the last processed frame
        # elapsed = frame_capture_time - last_processed_time
        # if elapsed < frame_interval:
        #     # Sleep the remaining time to match the desired interval
        #     time.sleep(frame_interval - elapsed)
        #     continue

        # Update last processed time using the capture timestamp
        last_processed_time = frame_capture_time

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
        print(f"Pre: {preprocessing_time:.4f}, Pred: {prediction_time:.4f}, Post: {postprocess_time:.4f}, Disp: {display_time:.4f}")

        # Exit loop if 'ESC' key is pressed
        if cv2.waitKey(10) & 0xFF == 27:
            break
except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    # Clean up resources and stop the capture thread
    capture_running = False
    capture_thread.join()
    cap.release()
    cv2.destroyAllWindows()

import time
import cv2
import torch
import sys
import threading
import subprocess
from datetime import datetime 
import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
hedModel = hed_cnn().to(device)
hedModel = pretrained_weights(hedModel, weightsPath=weightsPath, applyWeights=True, hedIn=True)
print(f"Load model time: {time.time() - start_time:.4f} seconds")

# Video stream URL and parameters
#stream_url = "http://stage-ams-nfs.srv.axds.co/stream/adaptive/ucsc/walton_lighthouse/hls.m3u8"
stream_url = "./testing/input/videos/walton_lighthouse/walton_lighthouse-2024-12-02-221643Z.mp4"
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    print("Video stream opened successfully.")

imSize = (480, 640)  # Input image size for processing

# Define output stream settings
WIDTH, HEIGHT = 1280, 720  # Output resolution for streaming
FPS = 30                   # Desired frames per second

# FFmpeg settings for YouTube Live (update with your stream key)
YOUTUBE_RTMP_URL = "rtmps://a.rtmp.youtube.com/live2/8xdz-uzm3-k3wy-p9jr-c3br"

# FFmpeg command to read raw BGR frames from stdin and stream them
ffmpeg_cmd = [
    'ffmpeg',
    '-y',
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', f'{WIDTH}x{HEIGHT}',
    '-r', str(FPS),
    '-i', '-',  # video input from stdin
    '-f', 'lavfi',
    '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',  # silent audio
    '-shortest',  # stop when the shortest input ends
    '-c:v', 'libx264',
    '-preset', 'veryfast',
    '-pix_fmt', 'yuv420p',
    '-g', '50',
    '-b:v', '3000k',
    '-maxrate', '3000k',
    '-bufsize', '6000k',
    '-c:a', 'aac',
    '-b:a', '128k',
    '-ar', '44100',
    '-f', 'flv',
    YOUTUBE_RTMP_URL
]


# Start FFmpeg subprocess
ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# Global variables for frame capture
latest_frame = None
latest_frame_time = None  # Timestamp in seconds when the frame was captured
capture_running = True
lock = threading.Lock()
retry_counter = 0
max_retries = 5  # Reduced maximum retries for faster recovery

def frame_capture():
    """Continuously capture frames (with timestamps) in a separate thread."""
    global latest_frame, latest_frame_time, cap, capture_running, retry_counter
    while capture_running:
        now = datetime.now()
        # Only capture during operational hours (7 AM to 7 PM)
        if now.hour < 7 or now.hour >= 19:
            time.sleep(300)
            continue

        ret, frame = cap.read()
        if ret:
            with lock:
                latest_frame = frame
                latest_frame_time = time.time()
            retry_counter = 0
        else:
            retry_counter += 1
            print(f"Error: Could not read frame in capture thread. Retrying ({retry_counter}/{max_retries})...")
            time.sleep(1)
            if retry_counter >= max_retries:
                print("Max retries reached in capture thread. Reinitializing video stream...")
                cap.release()
                cap = cv2.VideoCapture(stream_url)
                retry_counter = 0
        time.sleep(0.01)

# Start frame capture thread
capture_thread = threading.Thread(target=frame_capture)
capture_thread.start()

# Frame processing timing
frame_interval = 1.0 / FPS
last_processed_time = 0

try:
    while True:
        # Check operational hours
        now = datetime.now()
        if now.hour < 7 or now.hour >= 19:
            print("STREAM OFF: Outside operational hours (7 AM to 7 PM).")
            time.sleep(300)
            continue

        # Safely get the latest captured frame and timestamp
        with lock:
            frame = latest_frame.copy() if latest_frame is not None else None
            frame_capture_time = latest_frame_time if latest_frame_time is not None else 0

        if frame is None:
            print("No frame available, skipping processing.")
            continue

        # Ensure the frame's capture time meets the desired FPS interval
        elapsed = frame_capture_time - last_processed_time
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
            continue

        last_processed_time = frame_capture_time

        # Preprocess the frame
        imgData = load_single_image(frame, imSize)
        if imgData.max() > 1:
            imgData = imgData / 255
        imgData = torch.from_numpy(imgData.transpose((2, 0, 1))).float().unsqueeze(0).to(device)

        # Run ML model prediction
        hedModel.eval()
        with torch.no_grad():
            model_pred = hed_predict_single(hedModel, imgData)

        # Post-process to get final output image
        frame_image, _ = plot_refined_single_prediction(
            imgData.cpu(), model_pred.cpu(), thres=0.6, cvClean=True, imReturn=True
        )

        # Resize to match FFmpeg's expected resolution
        frame_image = cv2.resize(frame_image, (WIDTH, HEIGHT))

        # Write raw frame data to FFmpeg's stdin for streaming
        try:
            ffmpeg_process.stdin.write(frame_image.tobytes())
        except BrokenPipeError:
            print("FFmpeg pipe broken, exiting...")
            break

except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    # Cleanup: stop capture thread, release video capture and close FFmpeg process
    capture_running = False
    capture_thread.join()
    cap.release()
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    print("Streaming stopped and cleaned up.")

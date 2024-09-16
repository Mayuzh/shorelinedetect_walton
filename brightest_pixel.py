import cv2
import numpy as np

def extract_brightest_pixel_shoreline(video_path, output_image_path, max_seconds=None):
    # Initialize the video capture
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    # Get the frame rate of the video to calculate the number of frames to process
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the maximum number of frames to process based on the max_seconds limit
    max_frames = total_frames if not max_seconds else min(total_frames, int(fps * max_seconds))

    # Read the first frame to get the dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error reading the video file")
        return

    # Get the frame dimensions (height, width)
    height, width = frame.shape[:2]

    # Initialize the brightest pixel image (BPI) as all black
    brightest_pixel_img = np.zeros((height, width), dtype=np.uint8)

    # Process each frame in the video
    frame_count = 0
    while ret and frame_count < max_frames:
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Update the brightest pixel image by comparing each pixel value
        brightest_pixel_img = np.maximum(brightest_pixel_img, gray_frame)

        # Read the next frame
        ret, frame = cap.read()
        frame_count += 1

    # Release the video capture object
    cap.release()

    # Save the resulting brightest pixel image
    cv2.imwrite(output_image_path, brightest_pixel_img)

    print(f"Brightest pixel shoreline extraction saved to {output_image_path}")

# Example usage with max 10 seconds of the video
video_path = "./testing/poster_clip/Untitled6.mp4"
output_image_path = "brightest_pixel_shoreline6.png"
extract_brightest_pixel_shoreline(video_path, output_image_path, max_seconds=25)

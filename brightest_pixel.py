import cv2
import numpy as np

def extract_brightest_pixel_shoreline(video_path, output_image_path, max_seconds=None, crop_ratio=0.5):
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

    # Calculate the crop height to exclude the sky region (e.g., crop_ratio = 0.5 will keep the lower half)
    crop_height = int(height * crop_ratio)

    # Initialize the brightest pixel image (BPI) as all black
    brightest_pixel_img = np.zeros((crop_height, width), dtype=np.uint8)

    # Process each frame in the video
    frame_count = 0
    while ret and frame_count < max_frames:
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Crop the grayscale frame to avoid the sky (keep only the lower part)
        cropped_gray_frame = gray_frame[-crop_height:, :]

        # Update the brightest pixel image by comparing each pixel value
        brightest_pixel_img = np.maximum(brightest_pixel_img, cropped_gray_frame)

        # Read the next frame
        ret, frame = cap.read()
        frame_count += 1

    # Release the video capture object
    cap.release()

    # Step 1: Apply Thresholding to isolate the brightest pixels (shoreline)
    threshold_value = 220
    _, thresholded_img = cv2.threshold(brightest_pixel_img, threshold_value, 255, cv2.THRESH_BINARY)

    # Step 2: Post-processing with morphological operations (dilate and erode)
    kernel = np.ones((3, 3), np.uint8)  # Adjust size based on noise size

    # Morphological opening to remove small white dots (noise)
    noise_removed_img = cv2.morphologyEx(thresholded_img, cv2.MORPH_OPEN, kernel)

    # Resize the final image to match the original height (adding black padding to top)
    full_height_img = np.zeros((height, width), dtype=np.uint8)
    full_height_img[-crop_height:, :] = noise_removed_img

    # Save the resulting post-processed brightest pixel image
    cv2.imwrite(output_image_path, full_height_img)

    print(f"Brightest pixel shoreline extraction saved to {output_image_path}")


# Example usage with max 15 seconds of the video
video_path = "./testing/poster_clip/Untitled5.mp4"
output_image_path = "./output/brightest_pixel_shoreline5.png"
extract_brightest_pixel_shoreline(video_path, output_image_path, max_seconds=15, crop_ratio=1.0)


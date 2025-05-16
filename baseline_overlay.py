import os
import json
import cv2
import glob

# Path to your folder
folder_path = './annotated_frames/baseline'  # <-- Replace with your actual folder

# Loop through all JSON files
for json_file in glob.glob(os.path.join(folder_path, '*.json')):
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Get the two points
    points = data['shapes'][0]['points']
    pt1 = tuple(map(int, points[0]))
    pt2 = tuple(map(int, points[1]))

    # Load the corresponding image
    img_path = os.path.join(folder_path, data['imagePath'])
    image = cv2.imread(img_path)

    if image is None:
        print(f"Warning: Image not found for {json_file}")
        continue

    # Draw line
    cv2.line(image, pt1, pt2, (0, 0, 255), thickness=2)

    # Save the overlay image
    output_path = os.path.join(folder_path, f"overlay_{data['imagePath']}")
    cv2.imwrite(output_path, image)

    print(f"Saved overlay to {output_path}")

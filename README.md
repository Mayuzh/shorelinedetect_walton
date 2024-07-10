# Shoreline Detection System

## Overview


## Installation and Setup
Before running the script, ensure you have Python installed on your system. 

```bash
pip install opencv-python
pip install scikit-image
```

## ML Model
The ML models pretrainined with shoreline data is available here: https://drive.google.com/drive/folders/1dFww-SBKHgCnK2Ien5nOHc4rU36jujd_?usp=sharing

## Usage
The script is designed to continuously process video streams for shoreline detection. By default, it connects to an online video stream but can be modified to analyze local video files or other stream URLs.


## Key Components:


## Execution
To run the script, simply execute it in a Python environment:

```bash
py stream_vgg16_loop.py
```

## Customization
You can customize the script to suit different use cases. For example, you can change the video source by modifying the cap = cv2.VideoCapture(...) line with a different stream URL or a local video file path.

## Output
The script displays a window showing the live processed video with red lines marking the movements of shoreline. 

## Disclaimer
This script is for educational and developmental purposes. The accuracy of rip current detection is subject to various factors, including video quality, environmental conditions, and model limitations.

## Contributing
Contributions to enhance the script's functionality and performance are welcome. Please feel free to fork the repository, make improvements, and submit pull requests.

## Acknowledgments
Thanks to Joshua Simmons for the original shoreline detection method: 

https://github.com/simmonsja/cnn-shoreline-detect

Special thanks to OpenCV for their powerful image processing capabilities.

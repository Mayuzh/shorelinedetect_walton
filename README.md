# Shoreline Detection System

## Overview
This Python script utilizes advanced object detection techniques for identifying shoreline movements in video streams. The script employs VGG16, a type of Convolutional Neural Network (CNN), to analyze video footage and detect shorelines. This application is particularly valuable for observing shoreline changes, aiding in coastal management and environmental monitoring efforts.

Note: This script and associated models are prototypes and are currently configured for shoreline detection only from the Walton lighthouse camera. Do not attempt to use it for any other network camera.

## Installation and Setup
Before running the script, ensure you have Python installed on your system. 

```bash
pip install opencv-python
pip install scikit-image
```
For the specific environment requirements, you can refer to the `environment.yml` file. To create a conda environment with these dependencies, use the following command:
```bash
conda env create -f environment.yml
```
## ML Model
The pre-trained machine learning models specifically trained with shoreline data are available [here](https://drive.google.com/drive/folders/1dFww-SBKHgCnK2Ien5nOHc4rU36jujd_?usp=sharing).
+ `vgg16_1.0` is trained to detect shorelines from time-averaged (TIMEX) images, where the shoreline is defined as the wet/dry boundary.
+ `vgg16_2.0` is trained to detect shorelines from individual video frames, where the shoreline is defined as the water/land boundary.

Please download the appropriate model based on your specific use case for shoreline detection.
## Usage
The script is designed to continuously process video streams for shoreline detection. By default, it connects to an online video stream but can be modified to analyze local video files or other stream URLs.

## Execution
To run the script, simply execute it in a Python environment:

```bash
python stream_vgg16_loop.py
python stream_vgg16_loop_time.py
```

## Customization
You can customize the script to suit different use cases. For example, you can change the video source by modifying the cap = cv2.VideoCapture(...) line with a different stream URL or a local video file path.

## Output
The script displays a window showing the live processed video with red curves marking the movements of shoreline. 

## Disclaimer
This script is for educational and developmental purposes. The accuracy of shoreline detection is subject to various factors, including video quality, environmental conditions, and model limitations.

## Contributing
Contributions to enhance the script's functionality and performance are welcome. Please feel free to fork the repository, make improvements, and submit pull requests.

## Acknowledgments
Thanks to Joshua Simmons for the original shoreline detection method: 

https://github.com/simmonsja/cnn-shoreline-detect

Special thanks to OpenCV for their powerful image processing capabilities.

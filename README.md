# Rip Current Detection System

## Overview
This Python script utilizes advanced object detection techniques for identifying rip currents in video streams. The script employs YOLO (You Only Look Once) v8, a state-of-the-art, real-time object detection system, to analyze video footage and detect the presence of rip currents. This application is particularly useful for enhancing safety measures in beach areas by providing real-time rip current alerts. 

**Note:** This script and associated models are prototypes and are currently configured for rip current detection only from the Walton lighthouse camera. Do not attempt to use it for any other network camera.

## Installation and Setup
Before running the script, ensure you have Python installed on your system. The script depends on two primary libraries: OpenCV and Ultralytics YOLO. You can install these dependencies using pip:

```bash
pip install opencv-python
pip install ultralytics
```

## ML Model
The ML models pretrainined with rip current data is available here: https://drive.google.com/drive/folders/1ZM7FOfAbr3yDNliVbZ_LCg7jt3-yucCL?usp=sharing

## Usage
The script is designed to continuously process video streams for rip current detection. By default, it connects to an online video stream but can be modified to analyze local video files or other stream URLs.

Upon detecting a rip current, the script logs the detection with a timestamp and the confidence score into a CSV file named after the current date.

## Key Components:
OpenCV (cv2): For handling video stream capture and image processing.
Ultralytics YOLO: For performing the object detection tasks.
Datetime and Time: For timestamping detections.

## Execution
To run the script, simply execute it in a Python environment:

```bash
stream_yolov8_loop.py
```
or
```bash
stream_yolov8_loop_time.py
```

## Customization
You can customize the script to suit different use cases. For example, you can change the video source by modifying the cap = cv2.VideoCapture(...) line with a different stream URL or a local video file path.

## Output
The script displays a window showing the live processed video with bounding boxes around detected rip currents. Detection details are logged in a CSV file with timestamps and confidence scores.

## Disclaimer
This script is for educational and developmental purposes. The accuracy of rip current detection is subject to various factors, including video quality, environmental conditions, and model limitations.

## Contributing
Contributions to enhance the script's functionality and performance are welcome. Please feel free to fork the repository, make improvements, and submit pull requests.

## License
UC Santa Cruz Noncommercial License

Acceptance
In order to get any license under these terms, you must agree to them as both strict obligations and conditions to all your licenses.

Copyright License
The licensor grants you a copyright license for the software to do everything you might do with the software that would otherwise infringe the licensor's copyright in it for any permitted purpose. However, you may only distribute the software according to Distribution License and make changes or new works based on the software according to Changes and New Works License.

Distribution License
The licensor grants you an additional copyright license to distribute copies of the software. Your license to distribute covers distributing the software with changes and new works permitted by Changes and New Works License.

Notices
You must ensure that anyone who gets a copy of any part of the software from you also gets a copy of these terms, as well as the following copyright notice:

This software is Copyright ©2022. The Regents of the University of California (“Regents”). All Rights Reserved.

Changes and New Works License
The licensor grants you an additional copyright license to make changes and new works based on the software for any permitted purpose.

Noncommercial Purposes
Any noncommercial purpose is a permitted purpose.

Commercial Purposes
Contact Innovation Transfer, UC Santa Cruz, innovation@ucsc.edu , https://officeofresearch.ucsc.edu/iatc/ , for any commercial purpose.

Personal Uses
Personal use for research, experiment, and testing for the benefit of public knowledge, personal study, private entertainment, hobby projects, amateur pursuits, or religious observance, without any anticipated commercial application, is use for a permitted purpose.

Noncommercial Organizations
Use by any charitable organization, educational institution, public research organization, public safety or health organization, environmental protection organization, or government institution is use for a permitted purpose regardless of the source of funding or obligations resulting from the funding.

Fair Use
You may have "fair use" rights for the software under the law. These terms do not limit them.

No Other Rights
These terms do not allow you to sublicense or transfer any of your licenses to anyone else, or prevent the licensor from granting licenses to anyone else.  These terms do not imply any other licenses.

Patent Defense
If you make any written claim that the software infringes or contributes to infringement of any patent, all your licenses for the software granted under these terms end immediately. If your company makes such a claim, all your licenses end immediately for work on behalf of your company.

Violations
The first time you are notified in writing that you have violated any of these terms, or done anything with the software not covered by your licenses, your licenses can nonetheless continue if you come into full compliance with these terms, and take practical steps to correct past violations, within 32 days of receiving notice.  Otherwise, all your licenses end immediately.

No Liability
As far as the law allows, the software comes as is, without any warranty or condition, and the licensor will not be liable to you for any damages arising out of these terms or the use or nature of the software, under any kind of legal claim.

Definitions
The "licensor" is Regents, and the "software" is the software the licensor makes available under these terms.
"You" refers to the individual or entity agreeing to these terms.
"Your company" is any legal entity, sole proprietorship, or other kind of organization that you work for, plus all organizations that have control over, are under the control of, or are under common control with that organization.  
"Control" means ownership of substantially all the assets of an entity, or the power to direct its management and policies by vote, contract, or otherwise.  Control can be direct or indirect.
"Your licenses" are all the licenses granted to you for the software under these terms.
"Use" means anything you do with the software requiring one of your licenses.

## Acknowledgments
Thanks to Ultralytics for the YOLOv8 base models.
Special thanks to OpenCV for their powerful image processing capabilities.

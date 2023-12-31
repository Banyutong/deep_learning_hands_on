# 交大密院Deep learning学习手册
## UM-SJTU JI Deep learning Hands-on Tutorial
# Session 5 - Object Detection with YOLOv5 on Custom Video Data (Reduced Scope)

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Data Preparation](#data-preparation)
- [Object Detection with YOLOv5](#object-detection-with-yolov5)
- [Visualization](#visualization)
- [Conclusion](#conclusion)

---

## Introduction

In this truncated session, we will focus solely on running YOLOv5 for object detection on a custom video and visualizing the results.

---

## Prerequisites

To run the code, you'll need:

- PyTorch
- OpenCV (`cv2`)
- YOLOv5 (from GitHub)

You can clone the YOLOv5 repository from GitHub:

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -U -r requirements.txt
```

---

## Data Preparation

Assume you have a video file named `my_video.mp4`. Place it in a directory of your choosing.

---

## Object Detection with YOLOv5

### Step 1: Run Object Detection on Video

Run the following command to perform object detection:

```bash
python detect.py --source /path/to/my_video.mp4 --weights yolov5s.pt --conf 0.4
```

Replace `/path/to/my_video.mp4` with the path to your video.

This command will generate an output video with bounding boxes around detected objects.

---

## Visualization

### Step 2: View the Output Video

You can use any video player to view the output video generated by YOLOv5, or you can use OpenCV to visualize it programmatically.

Here's a Python code snippet that uses OpenCV to play the output video:

```python
import cv2

# Initialize OpenCV video capture
video = cv2.VideoCapture('runs/detect/exp/your_output_video.mp4')  # Change the path to your output video

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame or video has ended. Exiting.")
        break

    # Display the frame
    cv2.imshow('YOLOv5 Object Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video and destroy OpenCV windows
video.release()
cv2.destroyAllWindows()
```

In this code, OpenCV will read the output video generated by YOLOv5 (`your_output_video.mp4`) and play it in a window. Press 'q' to close the window and stop the video.

---

## Conclusion

In this simplified session, you learned how to run YOLOv5 object detection on a custom video and visualize the results. You can further explore the various configurations and options YOLOv5 offers to fine-tune the detection to your specific needs.

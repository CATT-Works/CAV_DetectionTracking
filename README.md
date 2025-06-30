[//]: # (Image References)
[img_SystemOverview]: ./doc/img/System_overview.jpg "Entire system - big picture"


# Introduction
This repo is a part of the CAV project carried out by [Center of Advanced Transportation Technology](http://www.catt.umd.edu/). The big picture behind this project is to improve the safety at the intersections. Namely: the objects at the intersection (cars, pedestrians etc.) are detected and tracked, and then the information about detected objects is passed to all traffic participants that can receive it. Currently BSM messages are used to broadcast the information.

The entire system is build of three components:
- Object detection and tracking 
- BSM server
- Broadcasting Equipment

![img_SystemOverview]

This repository contains Detection and Tracking component.

# Branch Information
This repository has two main branches:
- **`master`**: Contains the original implementation using FRCNN (TensorFlow) for detection and DeepSort for tracking
- **`YOLO`**: Contains the modernized implementation using YOLOv8/YOLOv11 for both detection and tracking

**Current branch**: `YOLO` - This is the active development branch with the latest improvements.

# Credits
## Current Implementation (YOLO branch)
- For object detection and tracking: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Modern, fast, and accurate object detection and tracking
- For computer vision operations: [OpenCV](https://opencv.org/)
- For deep learning: [PyTorch](https://pytorch.org/)

## Original Implementation (master branch)
- For object detection: [Google TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) (FRCNN)
- For tracking: [Nicolai Wojke Deep Sort Repository](https://github.com/nwojke/deep_sort)

# About the code
This is Alpha version of the code. It works but it was designed as a Proof Of Concept. The code is organized as follows:

## Core Components
- **`./cav/`**: Main folder with the core code
  - `detection.py`: YOLO-based object detection and tracking interface
  - `objects.py`: Object representation and BSM message generation
  - `parameters.py`: Camera calibration and coordinate transformation
  - `visualization.py`: Map visualization and bounding box plotting
  - `lanes.py`: Lane detection and analysis
  - `functions.py`: Utility functions for geometric calculations

## Configuration and Assets
- **`./config/`**: Configuration files for different intersections
- **`./images/`**: Images for visualization purposes (satellite views, icons, lane masks)
- **`./doc/`**: Documentation (currently minimal)

## Demo and Tools
- **`demo.py`**: Command-line interface for running detection and tracking
- **`demo.ipynb`**: Jupyter notebook demonstrating the complete pipeline
- **`./tools/`**: Utility scripts for configuration and analysis

## Dependencies
- **`requirements.txt`**: Python dependencies for the YOLO implementation
- **`./deep_sort/`**: (Legacy) DeepSort implementation (not used in YOLO branch)

# Key Improvements in YOLO Branch

## Performance Enhancements
- **Faster Processing**: YOLO is significantly faster than FRCNN + DeepSort
- **Unified Model**: Single YOLO model handles both detection and tracking
- **Better Accuracy**: Improved detection and tracking performance
- **Reduced Complexity**: Eliminated separate feature extraction pipeline

## Technical Improvements
- **Modern Architecture**: PyTorch-based instead of TensorFlow
- **Built-in Tracking**: YOLO's integrated tracking eliminates need for DeepSort
- **Better Memory Efficiency**: Single model reduces memory footprint
- **Real-time Performance**: Optimized for live video streams

## Usage Examples

### Basic Usage
```bash
# Use default YOLOv8 nano model
python demo.py

# Use YOLOv8 small model
python demo.py --yolo_model yolov8s.pt

# Use YOLOv11 nano model
python demo.py --yolo_model yolov11n.pt
```

### Advanced Usage
```bash
# Custom video stream with YOLOv8 medium model
python demo.py --yolo_model yolov8m.pt --video_stream rtsp://camera.ip/stream

# Enable BSM pushing to server
python demo.py --yolo_model yolov8l.pt --push_bsm

# Save frames and logs
python demo.py --yolo_model yolov8x.pt --frame_folder ./output --logfile tracking.log
```

## Model Options
- **YOLOv8**: `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`
- **YOLOv11**: `yolov11n.pt`, `yolov11s.pt`, `yolov11m.pt`, `yolov11l.pt`, `yolov11x.pt`

# Migration Notes
If you're migrating from the master branch (FRCNN + DeepSort):
1. Install new dependencies: `pip install -r requirements.txt`
2. Update model paths: Use `--yolo_model` instead of `--model`
3. The detection and tracking pipeline is now unified in a single call
4. BSM generation and visualization remain compatible

# Future Development
- Support for newer YOLO versions as they become available
- Enhanced tracking algorithms and performance optimizations
- Integration with additional sensor types
- Improved BSM message formats and standards compliance



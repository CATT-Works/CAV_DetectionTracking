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

# Credits
- For object detection [Google TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) is used.
- For Tracking (deep sort) I am using [Nicolai Wojke Deep Sort Repository](https://github.com/nwojke/deep_sort).

# About the code
This is Alpha version of the code. It works but it was designed as a Proof Of Concept. The code is organized as follows:
- ./cav: main folder with the core code
- ./deep_sort: [Nicolai Wojke](https://github.com/nwojke/deep_sort) code (slightly changed) that is responsible for deep sort based tracking.
- ./config: Folder with configuration files. 
- ./doc: Detailed documentation (in plans, currently empty :<)
- ./images: Images for visualization purposes
- demo.ipynb - Demo that shows how to use this code. 
- demo_noDeepSort.ipynb - an old version of demo that used various cv2 trackets instead of Deep Sort.

# Alternative Implementation
There is also a **`YOLO`** branch available that uses modern YOLO models (YOLOv8/YOLOv10) for both detection and tracking instead of FRCNN + DeepSort. The YOLO implementation offers improved performance and simplified architecture. To use the YOLO version, switch to the YOLO branch:

```bash
git checkout YOLO
```

The YOLO branch includes updated dependencies, simplified detection/tracking pipeline, and enhanced performance while maintaining compatibility with existing BSM generation and visualization components.



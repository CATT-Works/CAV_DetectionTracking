import cv2
import numpy as np
from ultralytics import YOLO

from time import time
from random import randint

from .objects import BoundingBox

class ObjectDetector:
    """
    Class used for object detection with YOLOv8
    Arguments:
        model_name         - YOLOv8 model name or path (e.g., 'yolov8n.pt', 'yolov8s.pt')
        detection_threshold - threshold for detection (default = 0.5)
    Instance variables:
        detection_threshold - threshold for detection
        model              - YOLOv8 model instance
    """
    def __init__(self, model_name='yolov8n.pt', detection_threshold=0.5):
        self.detection_threshold = detection_threshold
        
        # Initialize YOLOv8 model
        try:
            self.model = YOLO(model_name)
            print(f"YOLOv8 model '{model_name}' loaded successfully")
        except Exception as e:
            print(f"Error loading YOLOv8 model '{model_name}': {e}")
            # Fallback to a default model
            self.model = YOLO('yolov8n.pt')
            print("Using fallback model 'yolov8n.pt'")

    def detect(self, image, timestamp=None, returnBBoxes=True, detectThreshold=None):
        """Performs object detection using YOLOv8 and returns bounding boxes
        Arguments:
            image           - image containing objects to detect
            timestamp       - timestamp of the image. If none, current timestamp is taken
            returnBBoxes    - if True list of BoundingBox objects is returned.
            detectThreshold - detection threshold. If none (default) the predefined detection threshold is used
        Returns:
            Tuple of: (boxes, scores, classes) where:
                scores and classes are numpy arrays
                boxes is numpy array if returnBoxes == False otherwise it is a list of BoundingBox objects
        """
        
        if timestamp is None:
            timestamp = time()

        if detectThreshold is None:
            detectThreshold = self.detection_threshold
        
        # Ensure image is in the correct format for YOLOv8
        if isinstance(image, np.ndarray):
            # YOLOv8 expects BGR format (OpenCV default)
            pass
        else:
            image = np.asarray(image)
        
        # Run YOLOv8 detection
        results = self.model.predict(
            source=image,
            conf=detectThreshold,
            verbose=False,
            save=False
        )
        
        # Process results
        if len(results) == 0:
            return [], np.array([]), np.array([])
        
        # Get the first result (single image)
        result = results[0]
        
        if result.boxes is None or len(result.boxes) == 0:
            return [], np.array([]), np.array([])
        
        # Extract detection data
        boxes_data = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2] format
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        
        # Filter by confidence threshold
        mask = scores >= detectThreshold
        boxes_data = boxes_data[mask]
        scores = scores[mask]
        classes = classes[mask]
        
        if returnBBoxes:
            boxes = self.boxes2BoundingBoxes(boxes_data, image.shape, timestamp)
        else:
            boxes = boxes_data
        
        return boxes, scores, classes

    def track(self, image, timestamp=None, returnBBoxes=True, detectThreshold=None):
        """Performs object detection and tracking using YOLOv8's built-in tracking
        Arguments:
            image           - image containing objects to detect and track
            timestamp       - timestamp of the image. If none, current timestamp is taken
            returnBBoxes    - if True list of BoundingBox objects is returned.
            detectThreshold - detection threshold. If none (default) the predefined detection threshold is used
        Returns:
            Tuple of: (boxes, scores, classes, track_ids) where:
                scores, classes, and track_ids are numpy arrays
                boxes is numpy array if returnBoxes == False otherwise it is a list of BoundingBox objects
        """
        
        if timestamp is None:
            timestamp = time()

        if detectThreshold is None:
            detectThreshold = self.detection_threshold
        
        # Ensure image is in the correct format for YOLOv8
        if isinstance(image, np.ndarray):
            pass
        else:
            image = np.asarray(image)
        
        # Run YOLOv8 tracking
        results = self.model.track(
            source=image,
            conf=detectThreshold,
            verbose=False,
            save=False,
            persist=True  # Maintain track IDs across frames
        )
        
        # Process results
        if len(results) == 0:
            return [], np.array([]), np.array([]), np.array([])
        
        # Get the first result (single image)
        result = results[0]
        
        if result.boxes is None or len(result.boxes) == 0:
            return [], np.array([]), np.array([]), np.array([])
        
        # Extract detection and tracking data
        boxes_data = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2] format
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        
        # Get track IDs if available
        if hasattr(result.boxes, 'id') and result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = np.array([])
        
        # Filter by confidence threshold
        mask = scores >= detectThreshold
        boxes_data = boxes_data[mask]
        scores = scores[mask]
        classes = classes[mask]
        if len(track_ids) > 0:
            track_ids = track_ids[mask]
        
        if returnBBoxes:
            boxes = self.boxes2BoundingBoxes(boxes_data, image.shape, timestamp)
        else:
            boxes = boxes_data
        
        return boxes, scores, classes, track_ids
        
    def boxes2BoundingBoxes(self, boxes, imgshape, timestamp=None):
        """
        Converts YOLOv8 detection boxes into list of BoundingBox objects
        Arguments:
            boxes       - np.array of boxes from YOLOv8 in [x1, y1, x2, y2] format
            imgshape    - img shape, could be in format (y, x) or (y, x, ...)
            timestamp   - timestamp of the image.
        Returns:
            list of bounding boxes (objects.BoundingBox objects from this library)
        """
        bboxes = []
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            # Convert to integer coordinates
            xmin = int(x1)
            ymin = int(y1)
            xmax = int(x2)
            ymax = int(y2)
            bboxes.append(BoundingBox(xmin, xmax, ymin, ymax, timestamp))
        return bboxes



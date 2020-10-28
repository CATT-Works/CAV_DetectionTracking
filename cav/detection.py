import cv2
import numpy as np
import tensorflow as tf

from time import time
from random import randint

from .objects import BoundingBox

class ObjectDetector:
    """
    Class used for object detection
    Arguments:
        path_to_graph       - path to protobuf file with network inference. This file is loaded during
                            initialization
        detection_threshold - threshold for detection (default = 0.5)
    Instance variables:
        detection_threshold - threshold for detection

        detection_graph     - tf.Graph()
        sess                - tf.Session()
        image_tensor        - tensor where image is stored, used for detection only
        detection_boxes     - tensor for boxes
        detection_scores    - tensor for scores
        detection_classes   - tensor for classes
        num_detections      - tensor for number of detections
    """
    def __init__(self, path_to_graph, detection_threshold=0.5):
        self.detection_threshold = detection_threshold
        self.tf_version = int(tf.__version__.split('.')[0])
        
        if self.tf_version==1:
            self.detection_graph = tf.Graph()
            with self.detection_graph.as_default():
                od_graph_def = tf.GraphDef()            
                with tf.gfile.GFile(path_to_graph, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')

                    self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                    self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                    self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                    self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                    self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                self.sess = tf.Session(graph=self.detection_graph)
        else:
            self.model = tf.saved_model.load(path_to_graph)
        
            

    def _detect_tf1(self, image, timestamp = None, returnBBoxes = True, detectThreshold = None):
        """Performs object detection and returns bounding boxes
        Arguments:
            image           - image containing the traffic light
            timestamp       - timestamp of the image. If none, current timestamp is taken
            returnBBoxes    - if True list of BoundingBox objects is returned.
            detectThreshold - detection treshold. If none (default) the predefined detection threshold is used
        Returns:
            Tuple of: (boxes, scores, classes) where:
                scores and classes are numpy arrays
                boxes is numpy array if returnBoxes == False otherwise it is a list of BoundingBox objects
        """
        
        with self.detection_graph.as_default():
            image_expanded = np.expand_dims(image, axis=0)

            (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_expanded})

            classes = np.squeeze(classes).astype(np.int32)
            scores = np.squeeze(scores)

            cond = scores > detectThreshold
            boxes = boxes[0][cond]
            classes = classes[cond]
            scores = scores[cond]

            if returnBBoxes:
                boxes = self.boxes2BoundingBoxes(boxes, image.shape, timestamp)

            return boxes, scores, classes


    def detect(self, image, timestamp = None, returnBBoxes = True, detectThreshold = None):
        """Performs object detection and returns bounding boxes
        Arguments:
            image           - image containing the traffic light
            timestamp       - timestamp of the image. If none, current timestamp is taken
            returnBBoxes    - if True list of BoundingBox objects is returned.
            detectThreshold - detection treshold. If none (default) the predefined detection threshold is used
        Returns:
            Tuple of: (boxes, scores, classes) where:
                scores and classes are numpy arrays
                boxes is numpy array if returnBoxes == False otherwise it is a list of BoundingBox objects
        """
        
        if timestamp is None:
            timestamp = time()

        if detectThreshold is None:
            detectThreshold = self.detection_threshold
      
        
        if self.tf_version == 1:
            return _detect_tf1(image, timestamp, returnBBoxes, detectThreshold)
        else: # For Tensorflow 2
            image = np.asarray(image)
            # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
            input_tensor = tf.convert_to_tensor(image)
            # The model expects a batch of images, so add an axis with `tf.newaxis`.
            input_tensor = input_tensor[tf.newaxis,...]
            model_fn = self.model.signatures['serving_default']
            output_dict = model_fn(input_tensor)        
  
            # All outputs are batches tensors.
            # Convert to numpy arrays, and take index [0] to remove the batch dimension.
            # We're only interested in the first num_detections.
            num_detections = int(output_dict.pop('num_detections'))
            output_dict = {key:value[0, :num_detections].numpy() 
                           for key,value in output_dict.items()}
            output_dict['num_detections'] = num_detections

            # detection_classes should be ints.
            output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)                
                
            scores = output_dict['detection_scores']
            cond = scores > detectThreshold
            boxes = output_dict['detection_boxes'][cond]
            classes = output_dict['detection_classes'][cond]
            scores = scores[cond]                

            if returnBBoxes:
                boxes = self.boxes2BoundingBoxes(boxes, image.shape, timestamp)            
            
            return boxes, scores, classes            
                
        
    def boxes2BoundingBoxes(self, boxes, imgshape, timestamp = None):
        """
        Changes np.array of boxes (output from tensorflow object detection API) into list od bounding boxes
        Arguments:
            boxes       - np.array of boxes (first element of detection_boxes tensor from tf object detection API)
            imgshape    - img shape, could be in format (y, x) or (y, x, ...)
            timestamp       - timestamp of the image.
        Returns:
            list of bounding boxes (objects.BoundingBox objects from this library)
        """
        y = imgshape[0]
        x = imgshape[1]
        bboxes = []
        for box in boxes:
            ymin, xmin, ymax, xmax = box.tolist()
            ymin = int(y * ymin)
            xmin = int(x * xmin)
            ymax = int(y * ymax)
            xmax = int(x * xmax)
            bboxes.append(BoundingBox(xmin, xmax, ymin, ymax, timestamp))
        return bboxes



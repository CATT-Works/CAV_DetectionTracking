"""
This module runs ODT with a specified parameters. It is
the more robust version of the demo.ipynb file
Author: Przemek Sekula
Date: August 2023
Updated: YOLOv11 migration - removed DeepSort, added YOLOv11 tracking
"""

import socket
import json
import os
import sys
import argparse
from time import time

import numpy as np
import cv2

from cav.detection import ObjectDetector
from cav.parameters import Parameters
from cav.visualization import Map, plotBoxes
from cav.objects import Object, ObjectType

parser = argparse.ArgumentParser(description='Runs object detection and tracking with YOLOv11')

parser.add_argument(
    '-p', '--push_bsm',
    action='store_true',
    help='Pushes bsm message to the server')

parser.add_argument(
    '-t', '--time',
    type=int,
    default=60,
    help='Time [min] to process the stream. -1 means non-stop processing')

parser.add_argument(
    '-s', '--video_stream',
    type=str,
    default='rtmp://cctv.ritis.org/vod/CHART_CCTV_000109f6004b00a6004af03676235daa.vod',
    # default='rtsp://10.228.17.253/1',
    # default = None,
    help='Video stream. If not provided, the local camera is used.'
)

parser.add_argument(
    '--yolo_model',
    type=str,
    default='yolov11n.pt',
    help='YOLOv11 model name (e.g., yolov11n.pt, yolov11s.pt, yolov11m.pt, yolov11l.pt, yolov11x.pt)'
)

parser.add_argument(
    '-l', '--logfile',
    type=str,
    default=None,
    help='log file'
)

parser.add_argument(
    '--frame_folder',
    type=str,
    default=None,
    help='folder to save frames'
)

parser.add_argument(
    '--bsm_server',
    type=str,
    default='10.228.16.251',
    help='BSM server address'
)

parser.add_argument(
    '--bsm_port',
    type=int,
    default=65432,
    help='BSM server port'
)

parser.add_argument(
    '--bsm_buff',
    type=int,
    default=4096,
    help='BSM server buffer'
)

args = parser.parse_args()

VIDEO_X = 640
VIDEO_Y = 480
FRAMES_SEC = 15

MAX_BOXES_TO_DRAW = 20
MIN_SCORE_THRESH = 0.5

def send_bsm(my_socket, bsm):
    """
    Send message to the BSM server
    """
    data = {
        'mode': 'push',
        'msg': bsm
    }

    msg = json.dumps(data)
    msg = str.encode(msg)
    my_socket.sendall(msg)
    data = my_socket.recv(1024)
    return data

def create_object_from_detection(box, score, class_id, track_id, timestamp):
    """
    Create an Object instance from YOLOv10 detection results
    """
    # Map YOLOv10 class IDs to ObjectType (assuming COCO format)
    class_mapping = {
        0: ObjectType.Person,
        1: ObjectType.Bicycle,
        2: ObjectType.Car,
        3: ObjectType.Motorcycle,
        4: ObjectType.Bus,
        5: ObjectType.Truck,
        6: ObjectType.Car,  # Default to car for unknown classes
    }
    
    # Convert numpy values to Python native types
    class_id = int(class_id) if hasattr(class_id, 'item') else int(class_id)
    track_id = int(track_id) if track_id is not None and hasattr(track_id, 'item') else track_id
    score = float(score) if hasattr(score, 'item') else float(score)
    
    object_type = class_mapping.get(class_id, ObjectType.Car)
    obj = Object(object_type)
    
    # Add the bounding box to the object
    obj.addBoundingBox(box)
    
    # Set track ID
    obj.track_id = track_id
    
    # Assign a color based on track ID
    if track_id is not None:
        np.random.seed(track_id)
        obj.color = tuple(np.random.randint(0, 255, 3).tolist())
    else:
        obj.color = (0, 255, 0)  # Default green
    
    return obj

print(f'Loading YOLOv10 model: {args.yolo_model}')
od = ObjectDetector(args.yolo_model)
print('Model loaded successfully')

params = Parameters()
params.generateParameters('./config/params.json')
mymap = Map('./images/SkyView.jpg', './config/icons.json', params)

if args.video_stream is None:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture()
    cap.open(args.video_stream)

objects = []
frame_nr = 0
start_time = time()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as bsm_socket:
    if args.push_bsm:
        bsm_socket.connect((args.bsm_server, args.bsm_port))
    
    while cap.isOpened():
        curr_time = max(time() - start_time, 0.1)
        sys.stdout.write(f'{frame_nr} frames done in {curr_time:.1f} seconds '
                         f'({frame_nr/curr_time:.2f} frames/sec)    \r')

        ret, image = cap.read()
        if not ret or image is None:
            print(f'Failed to read frame {frame_nr}')
            frame_nr += 1
            continue

        # Single YOLOv10 tracking call - replaces detection + encoding + tracking
        boxes, scores, classes, track_ids = od.track(image, timestamp=time())
        
        # Process tracking results
        objects = []
        plotboxes = []
        plotcolors = []
        
        if len(boxes) > 0:
            print(f'{len(boxes)} objects tracked on frame {frame_nr}')
            
            for i, (box, score, class_id, track_id) in enumerate(zip(boxes, scores, classes, track_ids)):
                # Create Object instance from detection
                obj = create_object_from_detection(box, score, class_id, track_id, time())
                objects.append(obj)
                
                # Prepare for visualization
                plotboxes.append(box)
                plotcolors.append(obj.color)
        
        # Visualization
        if len(plotboxes) > 0:
            vid = plotBoxes(image, plotboxes, colors=plotcolors)
        else:
            vid = image.copy()
            mapimg = mymap.getMap()

        # BSM generation and transmission
        bsm_list = []
        if len(objects) > 0:
            for obj in objects:
                bsm_list.append(
                    obj.getBsm(
                        retDic=False,
                        params=params,
                        roundValues=True,
                        includeNone=False))

            if args.push_bsm:
                data = send_bsm(bsm_socket, json.dumps(bsm_list))

            if args.logfile is not None:
                logfile_path = os.path.join('./logs', args.logfile)
                with open(logfile_path, 'a', encoding='utf-8') as logfile:
                    for obj in objects:
                        line = f'{frame_nr},{time()},{obj.getParams(asCsv=True)}'
                        print(line, file=logfile)

        # Frame saving
        if args.frame_folder is not None:
            img_nr = str(frame_nr).zfill(7)
            with open(os.path.join(
                args.frame_folder,
                '_img_list.csv'),
                'a',
                    encoding='utf-8') as logfile:
                line = f'{img_nr},{time()}'
                print(line, file=logfile)
            cv2.imwrite(
                os.path.join(
                    args.frame_folder,
                    f'im_{img_nr}.jpg'),
                image)

        frame_nr += 1

        if (args.time >= 0) and (curr_time / 60 > args.time):
            break

curr_time = max(time() - start_time, 0.1)
print(f'\n\n{frame_nr} frames done in {curr_time:.1f} seconds '
      f'({frame_nr/curr_time:.2f} frames/sec)')
cap.release()

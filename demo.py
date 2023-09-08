"""
This module runs ODT with a specified parameters. It is
the more robust version of the demo.ipynb file
Author: Przemek Sekula
Date: August 2023
"""

import socket
import json
import os
import sys
import argparse
from time import time

import numpy as np
import cv2
import tensorflow as tf

from cav.detection import ObjectDetector
from cav.parameters import Parameters
from cav.visualization import Map, plotBoxes

# Deep sort imports
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection


parser = argparse.ArgumentParser(description='Runs object detection')


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
    '-m', '--model',
    type=str,
    default='./models/frcnninference/saved_model/',
    help='Path to the model'
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
    help='log file'
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

ENCODER_PATH = "./models/mars/mars-small128.pb"
ENCODER_BATCH_SIZE = 32
ENCODER_INPUT_NAME = "images"
ENCODER_OUTPUT_NAME = "features"

VIDEO_X = 640
VIDEO_Y = 480
FRAMES_SEC = 15

MAX_BOXES_TO_DRAW = 20
MIN_SCORE_THRESH = 0.5
IOU_COMMON_THRESHOLD = 0.50
NOT_DETECTED_TRHESHOLD = 1


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


class ImageEncoder():
    """
    Encode images for the purposes of DeepSort algorithm.
    """

    def __init__(self, checkpoint_filename, input_name="images",
                 output_name="features"):

        self.tf_version = int(tf.__version__.split('.', maxsplit=1)[0])

        if self.tf_version == 1:
            self.session = tf.Session()
            with tf.io.gfile.GFile(checkpoint_filename, "rb") as file_handle:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(file_handle.read())

        else:
            self.session = tf.compat.v1.Session()

            with tf.io.gfile.GFile(checkpoint_filename, "rb") as file_handle:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(file_handle.read())

        tf.import_graph_def(graph_def, name="net")

        if self.tf_version == 1:
            self.input_var = tf.get_default_graph().get_tensor_by_name(
                f"net/{input_name}:0")
            self.output_var = tf.get_default_graph().get_tensor_by_name(
                f"net/{output_name}:0")
        else:  # TF 2
            self.input_var = tf.compat.v1.get_default_graph(
            ).get_tensor_by_name(f"{input_name}:0")
            self.output_var = tf.compat.v1.get_default_graph(
            ).get_tensor_by_name(f"{output_name}:0")

        assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4
        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def __call__(self, data_x, batch_size=32):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        _run_in_batches(
            lambda x: self.session.run(self.output_var, feed_dict=x),
            {self.input_var: data_x}, out, batch_size)
        return out


def create_box_encoder(model_filename, input_name="images",
                       output_name="features", batch_size=32):
    """
    Create a box encoder for DeepSort algorithm.
    This function looks ugly, it should be refactored in future.
    """

    image_encoder = ImageEncoder(model_filename, input_name, output_name)
    image_shape = image_encoder.image_shape

    def encoder(my_image, my_boxes):
        image_patches = []
        for my_box in my_boxes:
            patch = extract_image_patch(my_image, my_box, image_shape[:2])
            if patch is None:
                print(
                    "WARNING: Failed to extract image patch: %s." %
                    str(my_box))
                patch = np.random.uniform(
                    0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches, batch_size)

    return encoder


def _run_in_batches(func, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = func(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = func(batch_data_dict)


def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    start_x, start_y, end_x, end_y = bbox
    image = image[start_y:end_y, start_x:end_x]
    image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image


image_encoder = ImageEncoder(
    ENCODER_PATH,
    ENCODER_INPUT_NAME,
    ENCODER_OUTPUT_NAME)
encoder = create_box_encoder(ENCODER_PATH, batch_size=32)

MAX_COSINE_DISTANCE = 0.2
NN_BUDGET = 100

metric = nn_matching.NearestNeighborDistanceMetric(
    "cosine", MAX_COSINE_DISTANCE, NN_BUDGET)


print('Loading od model...')
od = ObjectDetector(args.model)
print('Model loaded')


params = Parameters()
params.generateParameters('./config/params.json')
mymap = Map('./images/SkyView.jpg', './config/icons.json', params)


if args.video_stream is None:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture()
    cap.open(args.video_stream)

objects = []

results = []
colors = {}


tracker = Tracker(metric)

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

        boxes, scores, classes = od.detect(image)
        if len(boxes) >= 1:

            boxes_array = [[box.xLeft, box.yTop, box.xRight -
                            box.xLeft, box.yBottom - box.yTop] for box in boxes]
            boxes_array = np.array(boxes_array)
            bgr_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            features = encoder(bgr_image, boxes_array)
            detections = []

            for box, score, objClass, f_vector in zip(
                    boxes, scores, classes, features):
                detection = Detection(
                    [box.xLeft, box.yTop, box.xRight - box.xLeft,
                        box.yBottom - box.yTop],  # BBox
                    score, f_vector,
                    objClass
                )
                detection.bbox = box
                detections.append(detection)

            tracker.predict()
            tracker.update(detections)

        else:
            tracker.predict()

        plotboxes = []
        plotcolors = []
        objects = []

        if len(tracker.tracks) >= 1:
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                obj = track.trackedObject

                if obj is not None:
                    plotbox = obj.bboxes[-1]
                    plotbox.trackId = track.track_id
                    plotboxes.append(plotbox)
                    plotcolors.append(obj.color)
                    objects.append(obj)

            if len(plotboxes) >= 1:
                vid = plotBoxes(image, plotboxes, colors=plotcolors)
            else:
                vid = image.copy()
                mapimg = mymap.getMap()

        bsm_list = []

        if len(objects) > 0:
            for obj in objects:
                bsm_list.append(
                    obj.getBsm(
                        retDic=False,
                        params=params,
                        roundValues=True,
                        includeNone=False))

            # print (bsm_list)
            if args.push_bsm:
                data = send_bsm(bsm_socket, json.dumps(bsm_list))
                # print ("Response: {}\n".format(data))

            if args.logfile is not None:
                logfile_path = os.path.join('./logs', args.logfile)
                with open(logfile_path, 'a', encoding='utf-8') as logfile:
                    for obj in objects:
                        line = f'{frame_nr},{time()},{obj.getParams(asCsv=True)}'
                        print(line, file=logfile)

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

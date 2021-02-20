import os
import cv2
import json
import numpy as np
from random import randint



class Icons:
    """
    Class for loading and keeping icons and masks. These icons can be displayed on the map. Class may be configured
    with the json file (../config/icons.json by default)
    Parameters:
        folder: folder with icons
        filenames: dictionary { class_nr : filename }
        icons = None # Dictionary { class_nr : original icon image }
        iconsMasked = None # Dictionary { class_nr : masked icon imaged }
    """
    def __init__(self, config_file):
        self.folder = None # folder with icons
        self.filenames = None # Dictionary { class_nr : filename }
        self.icons = None # Dictionary { class_nr : original icon image }
        self.iconsMasked = None # Dictionary { class_nr : masked icon imaged }
        self.initializeIcons(config_file)

    def initializeIcons(self, config_file):
        with open(config_file) as f:
            data = json.load(f)
        self.folder = data['folder']
        self.filenames = {int(k):v for k,v in data['filenames'].items()}

        self.icons = {}
        self.iconsMasked = {}
        for k, v in self.filenames.items():
            icon = cv2.imread(os.path.join(self.folder, v))
            tmp = np.sum(icon, axis=2)
            tmpmask = 1 * (tmp < 3 * 255)
            self.iconsMasked[k] = 1 * np.logical_not(tmpmask)
            fg = icon.copy()
            for channel in range (3):
                fg[:, :, channel] = np.multiply(fg[:, :, channel], tmpmask)
            self.icons[k] = fg



class Map:
    """
    Class that keeps the image with the map and returns this image with (or without) the corresponding objects
    """
    def __init__(self, mapfile, config_file, params):
        self.icons = Icons(config_file=config_file)
        self.mapimg = cv2.imread(mapfile)
        self.params = params


    def getMap(self):
        """
        Returns the copy of the map image
        """
        return self.mapimg.copy()

    
    def addObjects(self, objects, use_obj_color = False):
        """
        Returns the copy of the image with objects added. Uses last bouding boxes. Updates parameters if necessary.
        Arguments:
            objects - list ob objects (instances ob Object class)
            icons - icons related to objects. If none, icons are collected automatically
        Returns:
            OpenCV image
        """

        newmap = self.mapimg.copy()
        mapx = newmap.shape[1]
        mapy = newmap.shape[0]

        for obj in objects:
            box = obj.updateBBoxParams(self.params)
            x = int(round(box.birdEyeX))
            y = int(round(box.birdEyeY))
            det_class = int(obj.type)

            
            if (x >= 16) and (x <= mapx-16) and (y >= 32) and (y <= mapy):
                bg = newmap[y-32:y, x-16:x+16, :]
                for channel in range (3):
                    bg[:, :, channel] = np.multiply(bg[:, :, channel], self.icons.iconsMasked[det_class])
                if use_obj_color:
                    dst = np.multiply((self.icons.icons[det_class] > 0).astype(np.uint8), obj.color).astype(np.uint8)
                    
                    dst = cv2.add(bg, dst)
                else:
                    dst = cv2.add(bg, self.icons.icons[det_class])
                newmap[y-32:y, x-16:x+16, :] = dst

        return newmap






def plotBoxes(image, boxes, colors=None, copy_img=True):
    """
    Plots bounding boxes to the image
    Arguments:
        image       - image we want to add the boxes to
        boxes       - boxes (list of instances of BoundingBox objects from this library)
        colors      - tuple or list with colors. If none, the colors will be set randomly
        copy_img    - if True this function is doing image.copy() before plotting bounding boxes
    Returns:
        Image with bounding boxes
    """

    if copy_img:
        image = image.copy()

    for i, box in enumerate(boxes):
        if colors is None:
            color = (randint(0, 255), randint(0, 255), randint(0, 255))
        else:
            color = colors[i]
        p1 = (int(box.xLeft), int(box.yTop))
        p2 = (int(box.xRight), int(box.yBottom))
        cv2.rectangle(image, p1, p2, color, 2, 1)
    return image

def bsmImg(bsm, imgsize = (450, 450), framecolor = (0, 0, 0)):
    """
    Returns an image with bsm message
    Arguments:
        bsm     - bsm message (as a dictionary)
        imgsize - size of image (y, x). Default = (450, 450)
        framecolor    - tuple with a BGR color of the frame around image (default = black, no color)
    Returns:
        Image (numpy array)
    """
    maxx = imgsize[1]
    maxy = imgsize[0]

    font = cv2.FONT_HERSHEY_SIMPLEX

    img = np.zeros((maxy, maxx, 3), np.uint8)
    cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), framecolor, 15)
    y0, dy = 40, 25
    text = json.dumps(bsm, indent=4)
    for i, line in enumerate(text.split('\n')):
        y = y0 + i * dy
        kv = line.split(':')
        cv2.putText(img, kv[0], (15, y), font, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
        if len(kv) > 1:
            cv2.putText(img, ':', (185, y), font, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, kv[1], (195, y), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img



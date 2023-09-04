import numpy as np
import cv2
import matplotlib.pyplot as plt

def displayImages(im1, im2, brg2rgb1=True, brg2rgb2=True, im1_title='Original Image', im2_title = 'Processed image', fontsize=20):
    """
    Displays two images.
    Arguments:
        im2, im2           - images to be displayed
        brg2rgb1, brg2rgb2 - if true, the BGR2RGB should be applied on images 1 and 2
        im1title, im2title - titles of images
        fontsize           - fontsize
    Returns: 
        Handles to the images
    """
    
    
    iscolor1 = len(im1.shape) > 2
    iscolor2 = len(im2.shape) > 2 
    if brg2rgb1 and iscolor1:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    if brg2rgb1 and iscolor2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    if iscolor1:
        ax1.imshow(im1)
    else:
        ax1.imshow(im1, cmap='gray')
    ax1.set_title(im1_title, fontsize=fontsize)
    if iscolor2:
        ax2.imshow(im2)
    else:
        ax2.imshow(im2, cmap='gray')
    ax2.set_title(im2_title, fontsize=fontsize)
    return ax1, ax2

def extractFrame(video_path = None, cap = None, frameno = 1, dest_file = None, display = True):
    """
    Extracts (and optionally saves) one frame from the video
    Arguments:
        video_path - path to the video
        cap        - captured video (results of cv2.VideoCapture). Works only if video_path is None
        frameno    - number of frame to be extracted
        dest_file  - where should the frame be saved
        display    - it True, the frame is displayed        
    """
    if video_path is None:
        if cap is None:
             raise Exception('Either video_path or cap must be set.')
    else:
        cap = cv2.VideoCapture(video_path)
        
    cap.set(1,frameno);
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    ret, image = cap.read()
    if dest_file is not None:
        cv2.imwrite(dest_file,image)
    
    if display:
        im2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        f = plt.figure(figsize=(20, 10))
        plt.imshow(im2)
    return image

def displayPoints(points, M, img1_path, img2_path):
    """
    Displays points (source and transformed) in two images.
    First four points are display red, all others blue.
    Arguments:
        points    - points to be displayed
        M         - perspective transform matrix
        img1_path - path to image with source points
        img2_path - path to image with transformed points
    Returns:
        transfomed points
    """
    
    src = np.array([points], dtype='float32')

    trans = cv2.perspectiveTransform(src, M)
    
    for a, b in zip(src[0], trans[0]):
        print ('{} -> {}'.format(a, b))
    
    img = cv2.imread(img1_path)
    mapimg = cv2.imread(img2_path)
    
    ax1, ax2 = displayImages(img, mapimg)

    if True: #show points
        for i, coor in enumerate(src[0]):
            if i < 4:
                ax1.plot(coor[0], coor[1], '.', color='red')
            else:
                ax1.plot(coor[0], coor[1], '.', color='blue')
        for i, coor in enumerate(trans[0]):
            if i < 4:
                ax2.plot(coor[0], coor[1], '.', color='red')
            else:
                ax2.plot(coor[0], coor[1], '.', color='blue')
    return trans
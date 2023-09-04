import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas import DataFrame

class Lanes:
    def __init__(self, path, params = None, queue_size=50):
        self.loadMask(path, params)
        self.lanesQueue = {}
        self.lanesCount = {}
        self.laneChanges = np.zeros((self.nrLanes, self.nrLanes)).astype(int)        
        
        for i in range(self.nrLanes):
            self.lanesQueue[i+1] = [None] * queue_size
            self.lanesCount[i+1] = 0


    def updateMask(self, params = None):
        if (params is not None) and params.lanes_mask is not None:
            for i, color in enumerate(params.lanes_mask):
                self.mask[self.mask == color] = i+1
 
        else:
            tmpline = self.mask[-1, :]
            for i in range(self.nrLanes):
                color = tmpline[(tmpline > i).argmax(axis=0)]
                self.mask[self.mask == color] = i+1
            
    def loadMask(self, path, params = None):
        self.mask = (255*plt.imread(path)).astype(int)
        if (len(self.mask.shape) == 3) and (self.mask.shape[2] > 1):
            self.mask = self.mask[:, :, 0]
        self.nrLanes = len(np.unique(self.mask)) - 1
        self.updateMask(params)

        
        
    def getMeanSpeed(self, lane, timeStamp = None, speedScope = None, lookback = 10):
        speeds = np.array([])
        for obj in self.lanesQueue[lane]:
            if obj is None:
                continue
            if len(obj.bboxes) <= lookback:
                continue
            if (timeStamp is not None) and (speedScope is not None):
                if obj.bboxes[-1].timeStamp < timeStamp - speedScope:
                    continue
            speed = obj.getSpeed(lookback = lookback)[0]
            if speed is not None:
                speeds = np.append(speeds, speed) 
        if len(speeds) > 0:
            #return np.mean(speeds)
            return np.median(speeds)
        return None
        
        
    def addObject(self, obj, minBoxes = 10, classes = None):
        """
        Adds an object to the current lane
        Arguments:
            obj       - object to be added
            minBoxes - only objects with minBoxes or more are added 
            classes   - list of acceptable classes. If None, all objects are added
        Returns:
            -2 - object does not belong to the desired class
            -1 - object does not have enough bounding boxes
             0 - object is not on the lane
             X - number of lane
            
        """
        if len(obj.bboxes) < minBoxes:
            return -1

        if classes is not None:
            raise('Classes Not Implemented!')
        
        
        x = int (obj.bboxes[-1].x) 
        y = int (obj.bboxes[-1].y) 
        
        if y == self.mask.shape[0]:
            y -=1
            
        if (y < self.mask.shape[0]) & (x < self.mask.shape[1]):
            lane = self.mask[y, x]
        else:
            lane = 0
        
        if lane > 0:
            if obj not in self.lanesQueue[lane]:
                self.lanesQueue[lane].append(obj)
                self.lanesQueue[lane].pop(0)
                self.lanesCount[lane] += 1
                
                if hasattr(obj, 'laneNr'):
                    self.laneChanges[obj.laneNr-1, lane-1] += 1                
                obj.laneNr = lane

        return lane
    
    def addObjects(self, objects, minBoxes = 10, classes = None):
        for obj in objects:
            self.add_object(obj, minBoxes = 10, classes = None)
        
    def _color_matrix(self, df, color):
        return [[color] * df.values.shape[1]] * df.values.shape[0]

    def getDataTables(self, imgsize = (270, 300), timeStamp = None, speedScope = 60,
                     cell_color = '#FFFF00', header_color = '#00FFFF'):
        """
        Arguments:
            imgsize   - size of the image with table
            timeStamp - timeStamp of the frame.
            speedScope - check speed for the objects not older than last X seconds. Works only
                         if timeStamp is not none. Otherwise all the objects are considered.
        """
        
        
        
        backend_ =  mpl.get_backend() 
        mpl.use("Agg")  # Prevent showing stuff
        df = DataFrame(columns = ['Cars', 'Speed [mph]'])
        for i in range(self.nrLanes):
            speed = self.getMeanSpeed(i+1)
            if speed is None:
                speed = '-'
            else:
                speed = int(np.round(speed * 2.23694)) # m/s to miles/h
            df.loc[i] = [self.lanesCount[i+1], speed]        

            
        lane_labels = ['Lane {}'.format(x+1) for x in range(self.nrLanes)]            
            
        fig = plt.figure()
        ax = plt.subplot(111, frame_on=False) # no visible frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        dpi = plt.gcf().get_dpi()
        fig.set_size_inches(imgsize[1]/float(dpi),imgsize[0]/2/float(dpi))

        nr_values = df.values.shape[0] * df.values.shape[1]
        tmp = ax.table(cellText=df.values, colLabels=df.keys(), rowLabels = lane_labels, 
                       loc='center', cellLoc='center', 
                       cellColours=self._color_matrix(df, cell_color), 
                       colColours=self._color_matrix(df, header_color)[0], rowColours=[header_color]*self.nrLanes
                      )
        fig.canvas.draw()
        
        
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))  
        
        plt.close()
        
        fig = plt.figure()
        ax = plt.subplot(111, frame_on=False) # no visible frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        dpi = plt.gcf().get_dpi()
        fig.set_size_inches(imgsize[1]/float(dpi),imgsize[0]/2/float(dpi))
        
        tmp = ax.table(cellText=self.laneChanges, 
                       colLabels=lane_labels, rowLabels = lane_labels, 
                       loc='center', cellLoc='center', 
                       cellColours=[[cell_color] * self.nrLanes] * self.nrLanes, 
                       colColours=[header_color]*self.nrLanes, rowColours=[header_color]*self.nrLanes
                      )
        fig.canvas.draw()
        
        data2 = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data2 = data2.reshape(fig.canvas.get_width_height()[::-1] + (3,))          
        
        data = np.concatenate((data, data2), axis = 0)

        white_filter = np.sum(data, axis=2) == 3*255
        data[white_filter, :] = 0
        plt.close()
        mpl.use(backend_) # Reset backend      
        return data

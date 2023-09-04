import cv2
import json
import numpy as np

class Parameters:
    """
    Parameters used for transferring between camera view and Bird Eye view,
    and between Bird Eye view and lat / lon
    List of Parameters:

        unwarp_M:       The matrix used for unwarping (from Camera view to Bird Eye View)
        unwarp_Minv:    Inverted unwarp_M (from Bird Eye to Camera View)
        lonA, lonB:     longitude = lonA * x + lonB, where x - pixel coordinate
        latA, latB:     latitude = latA * y + latB, where y - pixel coordinate

    This class contains also methods that allow to load data from json file and generate the necessary
    parameters

    """

    def __init__(self):
        self.unwarp_M = None # The matrix used for unwarping (from Camera view to Bird Eye View)
        self.unwarp_Minv = None # Inverted unwarp_M (from Bird Eye to Camera View)
        self.lonA, self.lonB = None, None # longitude = lonA * x + lonB, where x - pixel coordinate
        self.latA, self.latB = None, None # latitude = latA * y + latB, where y - pixel coordinate

        self.elevation = None # Elevation of an intersection
        
        self.lanes_mask = None

    def __generate_unwarp_matrices(self, cameraPoints, birdEyePoints):
        """
        Generates the matrices responsible for switching between Camera and Bird Eye views.
        Arguments:
            cameraPoints  - list of lists of pixel coordinates from the Camera Image
                            [ [x1, y1], [x2, y2], [x3, y3], [x4, y4] ]
            birdEyePoints - list of lists of pixel coordinates from the Bird Eye Image
                            [ [x1, y1], [x2, y2], [x3, y3], [x4, y4] ]
        """

        # Check data

        assert isinstance(cameraPoints, (list, tuple)), (
            'cameraPoints should be a list or a tuple. It is a {}'.format(type(cameraPoints))
        )
        assert isinstance(birdEyePoints, (list, tuple)), (
            'birdEyePoints should be a list or a tuple. It is a {}'.format(type(birdEyePoints))
        )
        assert len(cameraPoints) == 4, (
            'Len of cameraPoints should be 4. It is {}'.format(len(cameraPoints))
        )
        assert len(birdEyePoints) == 4, (
            'Len of birdEyePoints should be 4. It is {}'.format(len(birdEyePoints))
        )
        for i in range(4):
            assert isinstance(cameraPoints[i], (list, tuple)), (
                'cameraPoints[{}] should be a list or a tuple. It is a {}'
                    .format(i, type(cameraPoints[i]))
            )
            assert isinstance(birdEyePoints[i], (list, tuple)), (
                'birdEyePoints[{}] should be a list or a tuple. It is a {}'
                    .format(i, type(birdEyePoints[i]))
            )
            assert len(cameraPoints[i]) == 2, (
                'Len of cameraPoints[{}] should be 2. It is {}'.format(i, len(cameraPoints))
            )
            assert len(birdEyePoints[i]) == 2, (
                'Len of birdEyePoints[{}] should be 2. It is {}'.format(i, len(birdEyePoints))
            )

        cameraPoints = np.float32(cameraPoints)
        birdEyePoints = np.float32(birdEyePoints)

        self.unwarp_M = cv2.getPerspectiveTransform(cameraPoints, birdEyePoints)
        self.unwarp_Minv = cv2.getPerspectiveTransform(birdEyePoints, cameraPoints)

    def __generate_latlon_coefs(self, birdEyeCoordinates, latLonCoordinates):
        """
        Generates the coeffitiens responsible for going from pixels to latitudes, longitudes.
        Arguments:
            birdEyeCoordinates -    list of lists of coordinates from the Bird Eye Image
                                    [ [x1, y1], [x2, y2] ]
            latLonCoordinates -     list of lists of longitudes and latitudes corresponding with
                                    points stored in birdEyeCoordinates
                                    [ [lon1, lat1], [lon2, lat2] ]
        """

        # Check data

        assert isinstance(birdEyeCoordinates, (list, tuple)), (
            'birdEyeCoordinates should be a list or tuple. It is a {}'.format(type(birdEyeCoordinates))
        )
        assert isinstance(latLonCoordinates, (list, tuple)), (
            'latLonCoordinates should be a list or tuple. It is a {}'.format(type(latLonCoordinates))
        )
        assert len(birdEyeCoordinates) == 2, (
            'Len of birdEyeCoordinates should be 2. It is {}'.format(len(birdEyeCoordinates))
        )
        assert len(latLonCoordinates) == 2, (
            'Len of latLonCoordinates should be 2. It is {}'.format(len(latLonCoordinates))
        )
        for i in range(2):
            assert isinstance(birdEyeCoordinates[i], (list, tuple)), (
                'birdEyeCoordinates[{}] should be a list or tuple. It is a {}'
                    .format(i, type(birdEyeCoordinates[i]))
            )
            assert isinstance(latLonCoordinates[i], (list, tuple)), (
                'latLonCoordinates[{}] should be a list or tuple. It is a {}'
                    .format(i, type(latLonCoordinates[i]))
            )
            assert len(birdEyeCoordinates[i]) == 2, (
                'Len of birdEyeCoordinates[{}] should be 2. It is {}'.format(i, len(birdEyeCoordinates))
            )
            assert len(latLonCoordinates) == 2, (
                'Len of latLonCoordinates[{}] should be 2. It is {}'.format(i, len(latLonCoordinates))
            )

        x1, y1 = birdEyeCoordinates[0]
        x2, y2 = birdEyeCoordinates[1]
        lon1, lat1 = latLonCoordinates[0]
        lon2, lat2 = latLonCoordinates[1]

        self.lonA = (lon1 - lon2) / (x1 - x2)
        self.lonB = lon1 - x1 * self.lonA

        self.latA = (lat1 - lat2) / (y1 - y2)
        self.latB = lat1 - y1 * self.latA

    def generateParameters(self, jsonfile):
        """
        Computes and sets parameters based on the given json file
        Arguments:
            jsonfile - path to the file
        """

        with open(jsonfile) as f:
            data = json.load(f)
        self.__generate_unwarp_matrices(data['cameraPoints'], data['birdEyePoints'])
        self.__generate_latlon_coefs(data['birdEyeCoordinates'], data['latLonCoordinates'])

        if 'elevation' in data:
            self.elevation = data['elevation']
            
        if 'lanes_mask' in data:
            self.lanes_mask = data['lanes_mask']

            
    def birdEye2Geocoordinates(self, x, y):
        """
        Computes longitude and latitude given the image point coordinates
        Arguments:
            x, y - point coordinates (bird eye view)
        Returns:
            longitude, latitude (float, float) - geocoordinates
        """
        longitude = self.lonA * x + self.lonB
        latitude = self.latA * y + self.latB
        return longitude, latitude
    
    def geocoordinates2BirdEye(self, longitude, latitude):
        """
        Computes birdeye view x and y given the geocoordinates
        Arguments:
            longitude, latitude - geocoordinates
        Returns:
            x, y (int, int) - point coordinates
        """
        x = (longitude - self.lonB) / self.lonA
        y = (latitude - self.latB) / self.latA
        
        return np.round(x).astype(int), np.round(y).astype(int)
    
    def camera2BirdEye(self, x, y):
        src = np.array([[[x, y]]], dtype='float32')
        trans = cv2.perspectiveTransform(src, self.unwarp_M)
        return trans[0][0]
    
    def camera2Geocoordinates(self, x, y):
        birdEyeX, birdEyeY = self.camera2BirdEye(x, y)
        return self.birdEye2Geocoordinates(birdEyeX, birdEyeY)
    
params.json

# Parameters for translating among camera View, bird Eye View and Latitude / Longitude
{
  "videoShape": [x, y],  # Shape of the video Frames in pixels 
  "birdEyeViewShape": [x, y] # Shape of the related Bird Eye image in pixels 

  # Points used for unwarping
  "cameraPoints": [ [x1, y1], [x2, y2], [x3, y3], [x4, y4] ], 
  "birdEyePoints": [ [x1, y1], [x2, y2], [x3, y3], [x4, y4] ], 

  # Points used for going from bird Eye View to lon / lat.
  # Usually x1, y1, lon1, lat1 refer to the upper left corner of the image and 
  # x2, y2, lon2, lat2 refer to the lower right corner
  "birdEyeCoordinates": [ [x1, y1], [x2, y2] ]
  "latLonCoordinates" : [ [lon1, lat1], [lon2, lat2] ] 
}

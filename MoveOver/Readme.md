# Move Over
This folder is entirely devoted to MoveOver project.

## How To
The first working example is in `D8_581` folder. If you want to apply the code for your own video, you should:
- create a folder
- copy all the notebooks from `D8_581` to your folder
- copy .json files from `D8_581` to your folder
- prepare configs. `PrepareCongis.ipynb` will help you with this. Your config should be defined in `params.json` file
- generate detections. This is the most time-consuming task, so I extracted it to separate notebook `CreateDetections`. You need an inference of Tensorflow Object Detection zoo model to do this (not provided with this repo). The outputs of this notebook are as follows:
    - `./data/detections.p` - file with detections (bounding boxes)
    - `../frames/folder_name/frames_raw/*.jpg` raw frames for visualizations
- Generate lane detections with `detect_lanes.ipynb` notebook. The output of this notebook is:
    - `./data/lanes_detections.csv` - file where the objects (vehicles) are assigned to lanes, frame by frame
- Perform lane analysis with `Lane_Analysis.ipynb` notebook. This notebook provides the outputs as follows:
    - `./data/actions.csv` - file with detected actions
    - `../frames/folder_name/frames/*.jpg` frames for video with visualization
    - `../videos/folder_name.mp4` - video file with visualizations
    - `./data/car_lanes.csv` - traffic density for each lane (NOT DONE YET)
    - `./data/speeds.csv` - table with speeds for each lane (NOT DONE YET)
- Run `OutputProcessed.ipynb` to get a nice, shareable output. The output of this notebook is:
    - `./data/output_processed.csv` - shareable output file (columns describe below)
    
## TO DO
- generate `car_lanes.csv` file
- generate `speeds.csv` file
- add visualizations for `car_lanes.csv` and `speeds.csv`
- add compression for .mp4 file
- refactor the code in `Lane_Analysis.ipynb` to enhance clarity. Will be done later, when I am sure that it works well.


## Files description
- `./data/output_processed.csv`
    - `objectId` - id of an object (vehicle) build of UniquePythonId_CreationTimeStamp
    - `frame_start` - first frame where the vehicle was detected
    - `frame_end` - last frame where the vehicle was detected
    - `objectType` - 1 for passenger cars, 2 for trucks
    - `speed` - average speed \[MpH\]
    - `action` - action undertaken by the vehicle. One of:
        - `noNeed` - no need to move over, it wasn't on the adjacent lane
        - `moveOver` - the vehicle moved over
        - `noMoveOver` - the vehicle didn't move over
    - `can_change` - the object could/couldn't change the lane \[True/False\]. Set only for `noMoveOver` vehicles
    - `slowed` - the object slowed down \[True/False\]. Set only for `noMoveOver` vehicles
    
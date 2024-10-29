## Package information
This package contains all the detectors/launch/config files for using rgbd camera detecting human position in 3D space.

- **Package Name:** `human_rgbd_det_uol`
- **Inputs:** 
  - RGB raw image
  - Depth map
- **Outputs:**
  - 2D bounding box
  - 2D bounding box with mask (optional)
  - 2D bounding box with skeleton (optional)
  - Human position and orientation in 3D space
**Note:** if need to visualise the results using MarkerArray in Rviz, need to launch trackers/human_tracker_uol/marker_publisher.py.

## Running the Code

Below are the instructions to run different detection modes. Ensure you have the appropriate YAML configuration settings before running each command.

### 1. 2D Bounding Box + Human Position and Orientation

To run the detection with a 2D bounding box and human position and orientation, use the following command:

```bash
ros2 launch human_rgbd_det_uol launch_multi_zed_detection.py
```

Make sure the YAML file is set with the following argument:

```xml
<arg name="yolo_model" default="yolov8m.pt"/>
```

### 2. 2D Bounding Box with Mask + Human Position and Orientation

For detection with a 2D bounding box including masks, run:

```bash
ros2 launch human_rgbd_det_uol launch_multi_zed_detection.py
```

Ensure the YAML configuration includes:

```xml
<arg name="yolo_model" default="yolov8m-seg.pt"/>
```

### 3. 2D Bounding Box with Skeleton + Human Position and Orientation

To detect 2D bounding boxes with skeleton and human position and orientation, use:

```bash
ros2 launch human_rgbd_det_uol tracker_depth_skeleton.launch.xml
```

Ensure the YAML file has the following argument:

```xml
<arg name="yolo_model" default="yolov8n-pose.pt"/>
```



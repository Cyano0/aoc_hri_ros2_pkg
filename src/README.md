# aoc_hri
Repository with ROS2 packages that cover the human-robot interaction component of the Agri-Opencore project


## SETUP 

### PREREQUISITES:
- Ubuntu 22.04
- Ros2 Humble

### Packages List
- Detectors:
    - RGBD_camera :
      - package name: human_rgbd_det_uol
      - contains all the detectors/launch/config for rgbd camera
  - Fisheye_camera:
    - package name: human_fisheye_det_uol
    - contains all the detectors/launch/config for fisheye camera
  - Lidar:
    - package name: human_lidar_det_uol
    - contains all the detectors/launch/config for lidar
  - Camera-Lidar:
    - package name: human_camlidar_det_uol
    - contains all the detectors/launch/config for using lidar + camera (fisheye _ rgbd)
- Trackers:
    - package name: human_tracker_uol
    - contains all the detectors/launch/config for trackers
- Visualiser: (part of Tracker) 
    - package name: human_visualiser_uol
    - contains all the detectors/launch/config for sending Markers (e.g. bounding box) to visualise on Rviz
- Rosbag:
    - rosbags here
- Docker:
    - dockefile here

  Please refer to the README file of each package for detailed information.

### Git clone repositories
1. Create a ROS2 workspace and a source directory (`src`):

```bash
mkdir -p ~/{ROS2_WORKSPACE}/src
```
2. In the `src` directory, clone this [aoc_hri] repository (https://github.com/LCAS/aoc_hri.git) repository:

```bash
cd ~/{ROS2_WORKSPACE}/src
git clone git@github.com:LCAS/human_detection_rgbd_camera.git
```
3. Additional repository:
   
    3.1 In the `src` directory, clone the [ultralytics_ros](https://github.com/Alpaca-zip/ultralytics_ros.git) repository. Further, install dependencies and build the workspace: 

    ```
    cd ~/{ROS2_WORKSPACE}/src
    GIT_LFS_SKIP_SMUDGE=1 git clone -b humble-devel https://github.com/Alpaca-zip/ultralytics_ros.git 
    rosdep install -r -y -i --from-paths .
    python3 -m pip install -r ultralytics_ros/requirements.txt 
    cd ~/{ROS2_WORKSPACE} && $ colcon build
    source ~/.bashrc
    ```

      **Note**: make sure **Git Large File Storage (LFS)** is installed already as it is required by **ultralytics_ros **. 

    3.2 In the `src` directory, install the [mmdetection3D](https://github.com/open-mmlab/mmdetection3d?tab=readme-ov-file) repository. Please follow the instruction: https://mmdetection3d.readthedocs.io/en/latest/get_started.html.

5. Additional information:

    It is common to have this error notice when building the workspace during Step 3:
    
    ```bash
    CMake Error at CmakeLists.txt:16 (find packages)
    ```
    
    In this case, please try the following methods in order:
       
   (1) Make sure that you have this package installed:
          
           ```bash     
           sudo apt install ros-${ROS_DISTRO}-ament-cmake-clang-format
           ```
           
           ```bash
           sudo apt install ros-humble-ament-cmake-clang-format
           ```
          
   (2) Authorise permissions to edit the workspace directory. Try giving read/write permission of the workspace to all users using:
          
           ```bash
           cd ~/{ROS2_WORKSPACE}
           sudo chmod 777 -R .
           ```
          
   (3) Then build using colcon build in the current directory.
          
           ```bash
           cd ~/{ROS2_WORKSPACE} && $ colcon build
           source ~/.bashrc
           ```

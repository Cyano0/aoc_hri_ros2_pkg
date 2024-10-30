import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

def generate_launch_description():
    robot_model = os.environ['ROBOT_MODEL']
    # Configuration file paths
    if robot_model == 'hunter':
        config_file_1 = PathJoinSubstitution(
            [get_package_share_directory('human_rgbd_det_uol'), 'config', 'front_camera_hunter.yaml']
        )
        config_file_2 = PathJoinSubstitution(
            [get_package_share_directory('human_rgbd_det_uol'), 'config', 'back_camera_hunter.yaml']
        )
    elif robot_model == 'dogtooth':
        config_file_1 = PathJoinSubstitution(
            [get_package_share_directory('human_rgbd_det_uol'), 'config', 'front_camera_dogtooth.yaml']
        )
        config_file_2 = PathJoinSubstitution(
            [get_package_share_directory('human_rgbd_det_uol'), 'config', 'back_camera_dogtooth.yaml']
        )
    else:
        print("Please double chack the config files and robot model.")
        config_file_1 = PathJoinSubstitution(
            [get_package_share_directory('human_rgbd_det_uol'), 'config', 'front_camera_hunter.yaml']
        )
        config_file_2 = PathJoinSubstitution(
            [get_package_share_directory('human_rgbd_det_uol'), 'config', 'back_camera_hunter.yaml']
        )

    # Declare launch arguments
    debug = LaunchConfiguration('debug')

    return LaunchDescription([
        DeclareLaunchArgument(
            'debug',
            default_value='false',
            description='Flag to enable debugging nodes'
        ),

        # Launch the first instance of the human_detection node
        Node(
            package='human_rgbd_det_uol',
            executable='tracker_depth.py',
            name='tracker_depth_node_front',
            output='screen',
            parameters=[config_file_1]
        ),

        # Launch the second instance of the human_detection node
        Node(
            package='human_rgbd_det_uol',
            executable='tracker_depth.py',
            name='tracker_depth_node_back',
            output='screen',
            parameters=[config_file_2]
        ),

        # Conditionally launch the image_view node based on the debug parameter
        # Node(
        #     condition=IfCondition(debug),
        #     package='image_view',
        #     executable='image_view',
        #     name='image_view_node',
        #     output='screen',
        #     remappings=[('image', '/back_yolo_image')]
        # ),
    ])
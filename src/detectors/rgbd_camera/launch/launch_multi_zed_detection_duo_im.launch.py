#!/usr/bin/env python3
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
        config_file = PathJoinSubstitution(
            [get_package_share_directory('human_detection'), 'config', 'two_cameras_hunter.yaml']
        )
    elif robot_model == 'dogtooth':
        config_file = PathJoinSubstitution(
            [get_package_share_directory('human_detection'), 'config', 'two_cameras_dogtooth.yaml']
        )
    else:
        print("Please double chack the config files and robot model.")
        config_file= PathJoinSubstitution(
            [get_package_share_directory('human_detection'), 'config', 'two_cameras_hunter.yaml']
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
            package='human_detection',
            executable='tracker_depth_duo_im.py',
            name='tracker_depth_node',
            output='screen',
            parameters=[config_file]
        ),
    ])
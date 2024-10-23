#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():
    config_file = PathJoinSubstitution(
            [get_package_share_directory('human_tracker_uol'), 'config', 'marker_pub_params.yaml']
        )

    return LaunchDescription([
        # DeclareLaunchArgument(
        #     'config_file',
        #     default_value=config,
        #     description='Path to the config file'
        # ),
        Node(
            package='human_tracker_uol',
            executable='marker_publisher.py',
            name='bounding_box_publisher',
            output='screen',
            parameters=[config_file]
        )
    ])

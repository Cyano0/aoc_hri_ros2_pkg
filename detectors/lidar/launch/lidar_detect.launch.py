#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('human_lidar_det_uol'),
        'config',
        'mmdet3d_pointpillars_config.yaml'
    )

    return LaunchDescription([
        Node(
            package='human_lidar_det_uol',
            executable='pointcloud_detect.py',
            name='mmdet3d_node',
            output='screen',
            parameters=[config]
        )
    ])
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
    config_file_1 = PathJoinSubstitution(
        [get_package_share_directory('human_detection'), 'config', 'marker_pub_params_back.yaml']
    )
    config_file_2 = PathJoinSubstitution(
        [get_package_share_directory('human_detection'), 'config', 'marker_pub_params_front.yaml']
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
            executable='marker_publisher.py',
            name='bounding_box_publisher_front',
            output='screen',
            parameters=[config_file_2]
        ),

        # Launch the second instance of the human_detection node
        Node(
            package='human_detection',
            executable='marker_publisher.py',
            name='bounding_box_publisher_back',
            output='screen',
            parameters=[config_file_1]
        ),

    ])
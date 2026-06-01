import os
import math
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir       = get_package_share_directory('iolair')
    map_yaml_file = os.path.join(pkg_dir, 'maps', 'SLAM_map.yaml')

    return LaunchDescription([

        # 1. Nav2 Map Server
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{'yaml_filename': map_yaml_file}]
        ),

        # 2. Lifecycle Manager — activates map_server
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            parameters=[
                {'node_names': ['map_server']},
                {'autostart': True},
            ]
        ),

        # 3. Odometry — same initial_yaw as SLAM so frames agree
        Node(
            package='iolair',
            executable='odometry',
            name='puzzlebot_odometry',
            output='screen',
            parameters=[{
                'initial_yaw': math.pi,
            }]
        ),

        # 4. MCL — initial_yaw + lidar_yaw_offset must match SLAM launch
        Node(
            package='iolair',
            executable='mcl',
            name='puzzlebot_mcl',
            output='screen',
            parameters=[{
                'num_particles':    300,
                'beam_skip':        6,
                'lidar_yaw_offset': math.pi,   # same as slam.launch.py
                'initial_yaw':      math.pi,   # matches odometry start
                'map_frame':       'map',
                'odom_frame':      'odom',
                'base_frame':      'base_link',
            }]
        ),

        # 5. Controller
        Node(
            package='iolair',
            executable='controller',
            name='puzzlebot_controller',
            output='screen',
        ),
    ])
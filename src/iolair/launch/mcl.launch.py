import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Dynamically find where colcon installs the iolair package assets
    pkg_dir = get_package_share_directory('iolair')
    
    # Resolves the path to your map file inside the installed share directory
    map_yaml_file = os.path.join(pkg_dir, 'maps', 'slam_map.yaml') 

    return LaunchDescription([
        # 1. Nav2 Map Server Node
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{'yaml_filename': map_yaml_file}]
        ),

        # 2. Lifecycle Manager (Crucial! Activates the map_server lifecycle states)
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            parameters=[{'node_names': ['map_server']},
                        {'autostart': True}]
        ),

        # 3. Puzzlebot Odometry Node
        Node(
            package='iolair',
            executable='odometry',
            name='puzzlebot_odometry',
            output='screen'
        ),

        # 4. Puzzlebot MCL Node (Monte Carlo Localization Filter)
        Node(
            package='iolair',
            executable='mcl',
            name='puzzlebot_mcl',
            output='screen',
            parameters=[{
                'num_particles': 300,
                'beam_skip': 6,
                'map_frame': 'map',
                'odom_frame': 'odom',
                'base_frame': 'base_link'
            }]
        ),

        # 5. Puzzlebot Controller Node
        Node(
            package='iolair',
            executable='controller',
            name='puzzlebot_controller',
            output='screen'
        )
    ])
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    pkg_dir = get_package_share_directory('iolair')
    
    # Resolves the path to your map file inside the installed share directory
    map_yaml_file = os.path.join(pkg_dir, 'maps', 'slam_map.yaml') 

    return LaunchDescription([
        # 1. Nodo de Odometría del Puzzlebot
        Node(
            package='iolair',
            executable='odometry',
            name='puzzlebot_odometry',
            output='screen'
        ),

        # 2. Nodo ArUco Localizer (Landmark Anchoring)
        Node(
            package='iolair',
            executable='aruco_localizer',
            name='aruco_localizer',
            output='screen',
            # Aquí puedes sobreescribir los parámetros declarados en tu script si lo requieres
            parameters=[{
                'camera_to_base_x': 0.11,
                'camera_to_base_y': 0.00,
                'camera_to_base_z': 0.10,
                'anchor_min_dist': 0.20,
                'anchor_max_dist': 3.50,
                'anchor_reobserve': 0.30,
                'r_base_pos': 0.03,
                'r_base_yaw': 0.04,
                'distance_noise_k': 0.025,
                'publish_rate': 10.0
            }]
        ),

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

        # 3. Nodo del Controlador del Puzzlebot
        Node(
            package='iolair',
            executable='controller',
            name='puzzlebot_controller',
            output='screen'
        )
    ])
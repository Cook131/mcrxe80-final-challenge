"""
astar_test.launch.py
====================
Launch para probar el A* planner con goals desde RViz.

Nodos:
  1. map_server        — publica el mapa en /map
  2. lifecycle_manager — activa map_server
  3. astar_planner     — planificador A*
  4. rviz_goal_bridge  — convierte clicks de RViz → /astar/goal

Flujo:
  RViz "Publish Point" click → /clicked_point
        → rviz_goal_bridge   → /astar/goal
        → astar_planner      → /astar/path  + /goal
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    iolair_dir = get_package_share_directory('iolair')

    map_yaml_arg = DeclareLaunchArgument(
        'map_yaml',
        default_value=os.path.join(iolair_dir, 'maps', 'slam_map.yaml'),
        description='Ruta al archivo YAML del mapa'
    )
    map_yaml = LaunchConfiguration('map_yaml')

    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'yaml_filename': map_yaml}]
    )

    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager',
        output='screen',
        parameters=[
            {'node_names': ['map_server']},
            {'autostart': True},
        ]
    )

    astar_planner_node = Node(
        package='iolair',
        executable='astar_planner',
        name='astar_planner',
        output='screen',
        parameters=[{
            'slam_map_topic':     '/slam_map',
            'static_map_topic':   '/map',
            'odom_topic':         '/odom',
            'goal_in_topic':      '/astar/goal',
            'goal_out_topic':     '/goal',
            'inflation_radius':   0.15,
            'waypoint_threshold': 0.10,
            'occupied_threshold': 65,
            'allow_diagonal':     False,
        }]
    )

    rviz_goal_bridge_node = Node(
        package='iolair',
        executable='rviz_goal_bridge',
        name='rviz_goal_bridge',
        output='screen',
    )

    return LaunchDescription([
        map_yaml_arg,
        map_server_node,
        lifecycle_manager_node,
        astar_planner_node,
        rviz_goal_bridge_node,
    ])
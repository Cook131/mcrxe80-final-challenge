"""
astar_test.launch.py
====================
Launch minimalista para probar SOLO el A* planner.

Nodos que levanta:
  1. map_server          — publica el mapa pre-construido en /map
  2. lifecycle_manager   — activa map_server automáticamente
  3. astar_planner       — escucha /map, /odom y /astar/goal

Lo que NO incluye (para simplificar la prueba):
  - odometry   → reemplazado con un publisher manual de /odom desde terminal
  - go_to_goal → el waypoint output (/goal) lo lees con ros2 topic echo
  - controller, bug_IBA, aruco → no necesarios para validar el planner

Flujo de prueba:
  1. Lanzar este archivo
  2. Publicar una pose falsa en /odom para que el nodo sepa dónde está el robot
  3. Publicar un goal en /astar/goal
  4. Observar /astar/path en RViz y /astar/status en terminal

IMPORTANTE — entry point requerido en setup.py de iolair:
  'astar_planner = iolair.astar_planner:main',
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

    # ── 1. Map Server ──────────────────────────────────────────────────────
    # Publica el mapa en /map (OccupancyGrid) con QoS TRANSIENT_LOCAL
    # para que el A* lo reciba aunque se suscriba después de que se publicó
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'yaml_filename': map_yaml}]
    )

    # ── 2. Lifecycle Manager ───────────────────────────────────────────────
    # Necesario para que map_server pase al estado "active" y empiece a publicar
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

    # ── 3. A* Planner ──────────────────────────────────────────────────────
    astar_planner_node = Node(
        package='iolair',
        executable='astar_planner',
        name='astar_planner',
        output='screen',
        parameters=[{
            'slam_map_topic':     '/slam_map',   # prioridad 1 (SLAM en vivo)
            'static_map_topic':   '/map',         # prioridad 2 (mapa estático)
            'odom_topic':         '/odom',
            'goal_in_topic':      '/astar/goal',
            'goal_out_topic':     '/goal',
            'inflation_radius':   0.15,
            'waypoint_threshold': 0.10,
            'occupied_threshold': 65,
            'allow_diagonal':     True,
        }]
    )

    return LaunchDescription([
        map_yaml_arg,
        map_server_node,
        lifecycle_manager_node,
        astar_planner_node,
    ])
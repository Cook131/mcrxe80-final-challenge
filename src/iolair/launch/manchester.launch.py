"""
aruco_bug_iba.launch.py
=======================
Launches the full navigation stack for the Puzzlebot using:
  - ArUco Detector        (puzzlebot pkg) — detects ArUco markers from camera
  - ArUco Localizer       (iolair pkg)    — landmark-anchoring EKF correction
  - ArUco Map Publisher   (iolair pkg)    — visualiza landmarks en RViz
  - A* Planner            (iolair pkg)    — path planning on occupancy grid
  - Bug IBA / BugReflex   (iolair pkg)    — safety reflex layer (LiDAR-based)
  - Odometry / EKF        (iolair pkg)    — wheel encoder + EKF dead-reckoning
  - Controller            (iolair pkg)    — PID wheel velocity controller
  - Map Server            (nav2)          — serves the pre-built slam_map
  - Nav2 Lifecycle Mgr    (nav2)          — activates map_server automatically

Uso
---
  # Modo por defecto (full — todas las fuentes)
  ros2 launch iolair aruco_bug_iba.launch.py

  # Solo ArUco
  ros2 launch iolair aruco_bug_iba.launch.py ekf_mode:=aruco

  # Solo encoders
  ros2 launch iolair aruco_bug_iba.launch.py ekf_mode:=odometry_only

  # Mapa personalizado
  ros2 launch iolair aruco_bug_iba.launch.py map_yaml:=/ruta/a/mi_mapa.yaml
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ── Package share directories ──────────────────────────────────────────
    iolair_dir = get_package_share_directory('iolair')

    landmarks_yaml = os.path.join(iolair_dir, 'maps', 'aruco_landmarks.yaml')

    # ── Launch arguments ───────────────────────────────────────────────────
    ekf_mode_arg = DeclareLaunchArgument(
        'ekf_mode',
        default_value='aruco',
        description=(
            'Modo del EKF de odometría. '
            'Opciones: odometry_only | aruco | mcl | icp | full'
        ),
    )

    map_yaml_arg = DeclareLaunchArgument(
        'map_yaml',
        default_value=os.path.join(iolair_dir, 'maps', 'SLAM_map.yaml'),
        description='Ruta absoluta al archivo YAML del mapa'
    )

    # ── 1. ArUco Detector ──────────────────────────────────────────────────
    aruco_detector_node = Node(
        package='puzzlebot',
        executable='aruco_detector',
        name='aruco_detector',
        output='screen',
    )

    # ── 2. Odometry (EKF) ──────────────────────────────────────────────────
    # FIX: eliminada la coma al final que lo convertía en tuple
    odometry_node = Node(
        package='iolair',
        executable='odometry',
        name='puzzlebot_odometry',
        output='screen',
        parameters=[{
            'ekf_mode':     LaunchConfiguration('ekf_mode'),
            'wheel_radius': 0.05,
            'wheel_base':   0.19,
            'rate':         50.0,
        }]
    )

    # ── 3. ArUco Localizer ─────────────────────────────────────────────────
    # FIX: agregado landmarks_file para cargar posiciones predefinidas del YAML
    aruco_localizer_node = Node(
        package='iolair',
        executable='aruco_localizer',
        name='aruco_localizer',
        output='screen',
        parameters=[{
            'landmarks_file':    landmarks_yaml,
            'camera_to_base_x':  0.03,
            'camera_to_base_y':  0.07,
            'camera_to_base_z':  0.13,
            'anchor_min_dist':   0.20,
            'anchor_max_dist':   3.50,
            'anchor_reobserve':  0.30,
            'r_base_pos':        0.03,
            'r_base_yaw':        0.04,
            'distance_noise_k':  0.025,
            'publish_rate':      10.0,
        }]
    )

    # ── 4. ArUco Map Publisher — visualiza landmarks en RViz ───────────────
    aruco_map_publisher_node = Node(
        package='iolair',
        executable='aruco_map_publisher',
        name='aruco_map_publisher',
        output='screen',
        parameters=[{
            'landmarks_file': landmarks_yaml,
            'publish_rate':   1.0,
            'sphere_scale':   0.08,
            'text_scale':     0.12,
        }]
    )

    # ── 5. A* Planner ──────────────────────────────────────────────────────
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
            'allow_diagonal':     True,
        }]
    )

    # ── 6. Go-to-Goal ──────────────────────────────────────────────────────
    go_to_goal_node = Node(
        package='iolair',
        executable='go_to_goal',
        name='puzzlebot_go_to_goal',
        output='screen',
    )

    # ── 7. Bug IBA ─────────────────────────────────────────────────────────
    bug_iba_node = Node(
        package='iolair',
        executable='bug_IBA',
        name='bug_reflex',
        output='screen',
        parameters=[{
            'warn_dist':      0.55,
            'emergency_dist': 0.22,
            'stop_dist':      0.10,
            'reflex_v':       0.04,
            'reflex_w':       0.65,
            'reflex_hold_ms': 350,
            'front_half_deg': 30.0,
            'side_half_deg':  35.0,
            'hysteresis':     0.06,
        }]
    )

    # ── 8. Controller ──────────────────────────────────────────────────────
    controller_node = Node(
        package='iolair',
        executable='controller',
        name='puzzlebot_controller',
        output='screen',
    )

    # ── 9. Map Server ──────────────────────────────────────────────────────
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'yaml_filename': LaunchConfiguration('map_yaml')}]
    )

    # ── 10. Nav2 Lifecycle Manager ─────────────────────────────────────────
    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[
            {'node_names': ['map_server']},
            {'autostart': True},
        ]
    )

    rviz_goal_bridge_node = Node(
        package='iolair',
        executable='rviz_goal_bridge',
        name='rviz_goal_bridge',
        output='screen',
    )

    # ── LaunchDescription ─────────────────────────────────────────────────
    # FIX: ekf_mode_arg agregado aquí (antes faltaba)
    return LaunchDescription([
        ekf_mode_arg,
        map_yaml_arg,

        # Percepción
        aruco_detector_node,

        # Estimación de estado
        odometry_node,
        aruco_localizer_node,
        aruco_map_publisher_node,

        # Planeación
        astar_planner_node,

        # Navegación
        go_to_goal_node,

        # Capa de seguridad
        bug_iba_node,

        # Actuación
        controller_node,

        # Infraestructura del mapa
        map_server_node,
        lifecycle_manager_node,
        rviz_goal_bridge_node,
    ])
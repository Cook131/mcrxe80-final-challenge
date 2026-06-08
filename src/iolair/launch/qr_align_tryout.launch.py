"""
qr_align_tryout.launch.py
=========================
Tryout para QRCollectNode en pasillo.
Asume que la cámara ya está corriendo en /camera_raw/compressed.

Nodos levantados:
  aruco_detector_node  — detección ArUco + QR  (suscribe /camera_raw/compressed)
  qr_align_node        — FSM alignment QR (v7)
  qr_zone_checker      — chequeo de zona QR
  qr_detector          — detección QR

Uso:
  ros2 launch iolair qr_align_tryout.launch.py zone:=rack
  ros2 launch iolair qr_align_tryout.launch.py zone:=conveyor

Trigger manual (otra terminal):
  ros2 topic pub --once /collect/trigger std_msgs/String "data: rack"
  ros2 topic pub --once /collect/trigger std_msgs/String "data: conveyor"
"""

import os
import math

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ── Package share directories ──────────────────────────────────────────
    iolair_dir = get_package_share_directory('iolair')

    # ── Launch arguments ──────────────────────────────────────────────────
    map_yaml_arg = DeclareLaunchArgument(
        'map_yaml',
        default_value=os.path.join(iolair_dir, 'maps', 'SLAM_map.yaml'),
        description='Absolute path to the map YAML file'
    )
    map_yaml = LaunchConfiguration('map_yaml')
    landmarks_yaml_file = os.path.join(iolair_dir, 'configs', 'aruco_landmarks.yaml')

    # ── 1. QR Align — FSM de alineación (v7, parámetros internos) ─────────
    qr_align = Node(
        package    = 'Navigation',
        executable = 'qr_align_node',
        name       = 'qr_align_node',
        output     = 'screen',
    )

    # ── 2. QR Zone Checker ────────────────────────────────────────────────
    qr_zone_checker = Node(
        package    = 'Vision',
        executable = 'qr_zone_checker',
        name       = 'qr_zone_checker',
        output     = 'screen',
    )

    # ── 3. ArUco Detector ─────────────────────────────────────────────────
    aruco_detector_node = Node(
        package='Vision',
        executable='aruco_detector',
        name='aruco_detector',
        output='screen',
    )

    # ── 4. Odometry (EKF) — wheel encoders + ArUco EKF fusion ─────────────
    odometry_node = Node(
        package='iolair',
        executable='odometry',
        name='puzzlebot_odometry',
        output='screen',
        parameters=[{
            'wheel_radius': 0.045,
            'wheel_base':   0.19,
            'rate':         50.0,
            'initial_yaw':  math.pi,
        }]
    )

    # ── 5. ArUco Localizer — landmark-anchoring correction for the EKF ────
    aruco_localizer_node = Node(
        package='LocalizationMapping',
        executable='aruco_localizer',
        name='aruco_localizer',
        output='screen',
        parameters=[{
            'landmarks_file':   landmarks_yaml_file,
            'camera_to_base_x': 0.07,
            'camera_to_base_y': 0.08,
            'camera_to_base_z': 0.15,
            'anchor_min_dist':  0.2,
            'anchor_max_dist':  3.50,
            'anchor_reobserve': 0.1,
            'r_base_pos':       0.03,
            'r_base_yaw':       0.04,
            'distance_noise_k': 0.025,
            'publish_rate':     10.0,
        }]
    )

    # ── 6. ArUco Map Publisher — landmark markers for RViz ────────────────
    aruco_map_node = Node(
        package='LocalizationMapping',
        executable='aruco_map_publisher',
        name='aruco_map_publisher',
        output='screen',
        parameters=[{
            'landmarks_file': landmarks_yaml_file,
            'publish_rate':   1.0,
            'sphere_scale':   0.2,
            'text_scale':     0.15,
        }]
    )

    # ── 7. A* Planner — global path planner on occupancy grid ─────────────
    astar_planner_node = Node(
        package='Navigation',
        executable='astar_planner',
        name='astar_planner',
        output='screen',
        parameters=[{
            'slam_map_topic':     '/slam_map',
            'static_map_topic':   '/map',
            'odom_topic':         '/odom',
            'goal_in_topic':      '/astar/goal',
            'goal_out_topic':     '/goal',
            'inflation_radius':   0.2,
            'waypoint_threshold': 0.10,
            'occupied_threshold': 50,
            'allow_diagonal':     False,
        }]
    )

    # ── 8. Go-to-Goal — publica en /cmd_raw por defecto ───────────────────
    go_to_goal_node = Node(
        package='iolair',
        executable='go_to_goal',
        name='puzzlebot_go_to_goal',
        output='screen',
    )

    # ── 9. Controller — PID wheel velocity controller ──────────────────────
    controller_node = Node(
        package='iolair',
        executable='controller',
        name='puzzlebot_controller',
        output='screen',
    )

    # ── 10. Map Server — serves the pre-built SLAM map as /map ────────────
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'yaml_filename': map_yaml}]
    )

    # ── 11. Nav2 Lifecycle Manager — auto-activates map_server ────────────
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

    # ── 12. RViz Goal Bridge ───────────────────────────────────────────────
    rviz_goal_bridge_node = Node(
        package='Navigation',
        executable='rviz_goal_bridge',
        name='rviz_goal_bridge',
        output='screen',
    )

    # ── 13. MCL — Monte Carlo Localization ────────────────────────────────
    mcl_node = Node(
        package='LocalizationMapping',
        executable='mcl',
        name='puzzlebot_mcl',
        output='screen',
        parameters=[{
            'num_particles':    500,
            'lidar_yaw_offset': math.pi,
            'initial_yaw':      math.pi,
            'map_frame':        'map',
            'odom_frame':       'odom',
            'base_frame':       'base_link',
            'pose_ema_alpha':     0.25,
            'beam_skip':          10,
            'sigma_hit':          0.38,
            'resample_interval':  3,
        }]
    )

    # ── 14. SLAM ───────────────────────────────────────────────────────────
    slam_node = Node(
        package='LocalizationMapping',
        executable='slam',
        name='slam_node',
        output='screen',
        parameters=[{
            'scan_topic':       '/scan',
            'lidar_yaw_offset':  math.pi,
            'resolution':        0.05,
            'map_init_size':     400,
            'map_origin_x':     -10.0,
            'map_origin_y':     -10.0,
            'lidar_max_range':   8.0,
            'beam_skip':         1,
            'target_beams':      360,
            'log_odds_occ':      0.85,
            'log_odds_free':     0.40,
            'log_odds_max':      3.5,
            'log_odds_min':     -3.5,
            'use_icp':           True,
            'icp_max_iter':      50,
            'icp_tolerance':     1e-4,
            'publish_rate':      1.0,
            'tf_rate':           20.0,
            'occ_thresh':        0.65,
            'free_thresh':       0.35,
        }]
    )

    # ── 15. VFH+ — capa de evasión ────────────────────────────────────────
    # go_to_goal publica en /cmd_raw → vfh_plus filtra → /cmd_vel
    # Durante alineación qr_align publica /align/active=True
    # y vfh_plus cede el control directo.
    vfh_plus_node = Node(
        package='Navigation',
        executable='vfh_plus',
        name='vfh_plus',
        output='screen',
        parameters=[{
            'robot_radius_m'    : 0.18,
            'safety_margin_m'   : 0.10,
            'warn_dist'         : 0.65,
            'emergency_dist'    : 0.35,
            'stop_dist'         : 0.14,
            'num_sectors'       : 180,
            'hist_threshold'    : 8.0,
            'smoothing_window'  : 5,
            'influence_radius_m': 1.20,
            'max_v'             : 0.22,
            'max_w'             : 1.20,
            'kp_heading'        : 2.00,
            'lidar_yaw_offset'  : 3.14159,
        }],
    )

    # ── 16. QR Detector ───────────────────────────────────────────────────
    qr_detector = Node(
        package='Vision',
        executable='qr_detector',
        name='qr_detector',
        output='screen',
    )

    return LaunchDescription([
        map_yaml_arg,

        # Map infrastructure (primero — map_server debe estar activo
        # antes de que A* intente suscribirse a /map)
        map_server_node,
        lifecycle_manager_node,

        # Perception
        aruco_detector_node,
        qr_detector,
        qr_zone_checker,

        # State estimation
        odometry_node,
        aruco_localizer_node,
        mcl_node,
        slam_node,

        # Visualisation
        aruco_map_node,

        # Path planning
        astar_planner_node,

        # Navigation
        go_to_goal_node,
        vfh_plus_node,

        # QR alignment FSM
        qr_align,

        # Actuation
        controller_node,
        rviz_goal_bridge_node,
    ])
"""
manchester.launch.py  (actualizado con mission_manager)
========================================================
Agrega mission_manager al stack existente.
Todo lo demás permanece igual al original.
"""

import os
import math

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    iolair_dir = get_package_share_directory('iolair')

    map_yaml_arg = DeclareLaunchArgument(
        'map_yaml',
        default_value=os.path.join(iolair_dir, 'maps', 'SLAM_map.yaml'),
        description='Absolute path to the map YAML file'
    )
    map_yaml = LaunchConfiguration('map_yaml')

    landmarks_yaml_file = os.path.join(iolair_dir, 'configs', 'aruco_landmarks.yaml')
    exploration_yaml    = os.path.join(iolair_dir, 'configs', 'exploration_waypoints.yaml')

    # ── 1. ArUco Detector ─────────────────────────────────────────────────
    aruco_detector_node = Node(
        package='puzzlebot', executable='aruco_detector',
        name='aruco_detector', output='screen',
    )

    # ── 2. Odometry (EKF) ─────────────────────────────────────────────────
    odometry_node = Node(
        package='iolair', executable='odometry',
        name='puzzlebot_odometry', output='screen',
        parameters=[{
            'wheel_radius': 0.05,
            'wheel_base':   0.19,
            'rate':         50.0,
            'initial_yaw':  math.pi,
        }]
    )

    # ── 3. ArUco Localizer ────────────────────────────────────────────────
    aruco_localizer_node = Node(
        package='iolair', executable='aruco_localizer',
        name='aruco_localizer', output='screen',
        parameters=[{
            'landmarks_file':   landmarks_yaml_file,
            'camera_to_base_x': 0.05,
            'camera_to_base_y': 0.07,
            'camera_to_base_z': 0.13,
            'anchor_min_dist':  0.2,
            'anchor_max_dist':  3.50,
            'anchor_reobserve': 0.1,
            'r_base_pos':       0.03,
            'r_base_yaw':       0.04,
            'distance_noise_k': 0.025,
            'publish_rate':     10.0,
        }]
    )

    # ── 4. ArUco Map Publisher ────────────────────────────────────────────
    aruco_map_node = Node(
        package='iolair', executable='aruco_map_publisher',
        name='aruco_map_publisher', output='screen',
        parameters=[{
            'landmarks_file': landmarks_yaml_file,
            'publish_rate':   1.0,
            'sphere_scale':   0.095,
            'text_scale':     0.15,
        }]
    )

    # ── 5. A* Planner ─────────────────────────────────────────────────────
    astar_planner_node = Node(
        package='iolair', executable='astar_planner',
        name='astar_planner', output='screen',
        parameters=[{
            'slam_map_topic':     '/slam_map',
            'static_map_topic':   '/map',
            'odom_topic':         '/odom',
            'goal_in_topic':      '/astar/goal',
            'goal_out_topic':     '/goal',
            'inflation_radius':   0.21,
            'waypoint_threshold': 0.10,
            'occupied_threshold': 65,
            'allow_diagonal':     True,
        }]
    )

    # ── 6. Go-to-Goal ─────────────────────────────────────────────────────
    go_to_goal_node = Node(
        package='iolair', executable='go_to_goal',
        name='puzzlebot_go_to_goal', output='screen',
    )

    # ── 7. Bug IBA ────────────────────────────────────────────────────────
    bug_iba_node = Node(
        package='iolair', executable='bug_IBA',
        name='bug_reflex', output='screen',
        parameters=[{
            'warn_dist':         0.45,
            'emergency_dist':    0.22,
            'stop_dist':         0.13,
            'reflex_v':          0.1,
            'reflex_w':          0.50,
            'reflex_hold_ms':    350,
            'front_half_deg':    30.0,
            'side_half_deg':     35.0,
            'hysteresis':        0.03,
            'replan_cooldown_s': 2.0,
            'm_line_tol':        0.12,
            'bug2_min_follow_m': 0.20,
        }]
    )

    # ── 8. Controller ─────────────────────────────────────────────────────
    controller_node = Node(
        package='iolair', executable='controller',
        name='puzzlebot_controller', output='screen',
    )

    # ── 9. Map Server ─────────────────────────────────────────────────────
    map_server_node = Node(
        package='nav2_map_server', executable='map_server',
        name='map_server', output='screen',
        parameters=[{'yaml_filename': map_yaml}]
    )

    # ── 10. Lifecycle Manager ─────────────────────────────────────────────
    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_localization', output='screen',
        parameters=[
            {'node_names': ['map_server']},
            {'autostart': True},
        ]
    )

    # ── 11. RViz Goal Bridge ──────────────────────────────────────────────
    rviz_goal_bridge_node = Node(
        package='iolair', executable='rviz_goal_bridge',
        name='rviz_goal_bridge', output='screen',
    )

    # ── 12. MCL ───────────────────────────────────────────────────────────
    mcl_node = Node(
        package='iolair', executable='mcl',
        name='puzzlebot_mcl', output='screen',
        parameters=[{
            'num_particles':    300,
            'beam_skip':        6,
            'lidar_yaw_offset': math.pi,
            'initial_yaw':      math.pi,
            'map_frame':        'map',
            'odom_frame':       'odom',
            'base_frame':       'base_link',
        }]
    )

    # ── 13. SLAM ──────────────────────────────────────────────────────────
    slam_node = Node(
        package='iolair', executable='slam',
        name='slam_node', output='screen',
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

    # ── 14. YOLO Vision ───────────────────────────────────────────────────
    yolo_node = Node(
        package='puzzlebot', executable='yolo_vision',
        name='yolo_detector', output='screen',
    )

    # ── 15. Mission Manager (FSM central) ─────────────────────────────────
    mission_manager_node = Node(
        package='iolair', executable='mission_manager',
        name='mission_manager', output='screen',
        parameters=[{
            'fsm_rate_hz':      10.0,
            'goal_timeout_s':   90.0,
            'qr_detect_dist':    1.5,
            'truck_zone_id':    20,
            'exploration_file': exploration_yaml,
        }]
    )

    return LaunchDescription([
        map_yaml_arg,

        # Infraestructura de mapa
        map_server_node,
        lifecycle_manager_node,

        # Percepción
        aruco_detector_node,
        yolo_node,

        # Estimación de estado
        odometry_node,
        aruco_localizer_node,
        mcl_node,
        slam_node,

        # Visualización
        aruco_map_node,

        # Planificación
        astar_planner_node,

        # Navegación
        go_to_goal_node,
        rviz_goal_bridge_node,

        # Safety
        bug_iba_node,

        # Actuación
        controller_node,

        # FSM Central (último — depende de todos los anteriores)
        mission_manager_node,
    ])
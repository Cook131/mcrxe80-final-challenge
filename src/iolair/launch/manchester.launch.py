"""
manchester.launch.py
====================
Launches the full navigation stack for the Puzzlebot using:
  - ArUco Detector        (puzzlebot pkg) — detects ArUco markers from camera
  - ArUco Localizer       (iolair pkg)    — landmark-anchoring EKF correction
  - Odometry / EKF        (iolair pkg)    — wheel encoder + EKF dead-reckoning
                                            publishes TF: map → odom → base_link
  - A* Planner            (iolair pkg)    — path planning on occupancy grid
  - Go-to-Goal            (iolair pkg)    — drives toward each A* waypoint
  - Bug IBA / BugReflex   (iolair pkg)    — safety reflex layer BUG2+LateralCtrl v4.1
  - Controller            (iolair pkg)    — PID wheel velocity controller
  - Map Server            (nav2)          — serves the pre-built SLAM map
  - Nav2 Lifecycle Mgr    (nav2)          — activates map_server automatically
  - ArUco Map Publisher   (iolair pkg)    — visualises landmark positions in RViz

TF tree
-------
  map ──(EKF ArUco correction)──> odom ──(raw wheel odometry)──> base_link

  The map→odom offset is computed and broadcast by puzzlebotOdometry every
  50 Hz cycle.  It stays at identity until the first ArUco measurement update
  fires, then tracks the full 2-D correction offset.

Robot geometry (Puzzlebot)
--------------------------
  Width:    0.22 m  →  half-width  = 0.11 m
  Length:   0.30 m  →  half-length = 0.15 m
  Diagonal: sqrt(0.11²+0.15²) ≈ 0.186 m  (bounding-circle radius)

Parameter derivation — bug_IBA v4.1
------------------------------------
  stop_dist        = diagonal + 5 cm safety margin  = 0.186 + 0.05  ≈ 0.14 m
                     (v4.1: aumentado de 0.13 para coincidir con geometría real)
  emergency_dist   = stop + braking distance at reflex_v over hold_ms
                   = 0.14 + 0.10*0.35                              ≈ 0.35 m
                     (v4.1: aumentado de 0.30 — reacciona antes de estar pegado)
  warn_dist        = emergency + soft-braking zone                  = 0.65 m (unchanged)
  hysteresis       = ~25 % of (emergency - stop) gap
                   = 0.25 * (0.35 - 0.14)                          ≈ 0.05 m
                     (v4.1: recalculado por nuevos stop/emergency)
  wall_follow_dist = distancia lateral deseada al muro durante BUG2 = 0.40 m
                     (v4.1 NEW — P-controller lateral, elimina "pasar rozando")
  wall_follow_kp   = ganancia proporcional del P-controller lateral  = 1.20
  wall_follow_w_max= límite angular del P-controller                 = 0.80 rad/s
  reflex_v         = 0.06 m/s (v4.1: reducido — velocidad base wall-follow;
                     la velocidad real es adaptativa según distancia al frente)
  m_line_tol       = 0.15 m  (v4.1: ligeramente aumentado de 0.12 para robustez)
  bug2_min_follow  = 0.30 m  (v4.1: aumentado de 0.20 — evita salidas falsas)
  bug2_min_time_s  = 1.0 s   (v4.1 NEW — tiempo mínimo en wall-follow anti-ruido)
  inflation_r      = half-width + 10 cm clearance = 0.11 + 0.10 = 0.21 m
                     (was 0.35 — overly conservative, caused narrow-passage failures)
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

    # ── Launch arguments ───────────────────────────────────────────────────
    map_yaml_arg = DeclareLaunchArgument(
        'map_yaml',
        default_value=os.path.join(iolair_dir, 'maps', 'SLAM_map.yaml'),
        description='Absolute path to the map YAML file'
    )
    map_yaml = LaunchConfiguration('map_yaml')

    landmarks_yaml_file = os.path.join(iolair_dir, 'configs', 'aruco_landmarks.yaml')

    # ── 1. ArUco Detector — detects markers, publishes /aruco/waypoint ─────
    aruco_detector_node = Node(
        package='Vision',
        executable='aruco_detector',
        name='aruco_detector',
        output='screen',
    )

    # ── 2. Odometry (EKF) — wheel encoders + ArUco EKF fusion ─────────────
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

    # ── 3. ArUco Localizer — landmark-anchoring correction for the EKF ─────
    aruco_localizer_node = Node(
        package='LocalizationMapping',
        executable='aruco_localizer',
        name='aruco_localizer',
        output='screen',
        parameters=[{
            'landmarks_file':   landmarks_yaml_file,
            'camera_to_base_x': 0.07,
            'camera_to_base_y': 0.05,
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

    # ── 4. ArUco Map Publisher — landmark markers for RViz ─────────────────
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

    # ── 5. A* Planner — global path planner on occupancy grid ──────────────
    # inflation_radius: half-width (0.11) + 10cm clearance = 0.21m
    # Reduced from 0.35 — that value blocked navigable passages narrower than
    # 70cm, which is overkill for a 22cm-wide robot.
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
            'inflation_radius':   0.25,
            'waypoint_threshold': 0.10,
            'occupied_threshold': 50,
            'allow_diagonal':     False,
        }]
    )

    # ── 6. Go-to-Goal — drives the robot toward each A* waypoint ───────────
    go_to_goal_node = Node(
        package='iolair',
        executable='go_to_goal',
        name='puzzlebot_go_to_goal',
        output='screen',
    )

    vfh_node = Node(
        package    = 'Navigation',
        executable = 'vfh_plus',
        name       = 'vfh_plus',
        output     = 'screen',
        parameters =[{
            # Geometria del Robot
            'robot_radius': 0.18,      # m (half-width of the robot)
            'safety_margin': 0.7,      # m (extra clearance to avoid collisions)

            # Parámetros de comportamiento
            'warn_dist': 0.9,          # m (start soft braking zone)
            'emergency_dist': 0.30,    # m (start hard braking zone)
            'stop_dist': 0.14,         # m (distance to consider a collision)
            'hysteresis': 0.04,        # m (hysteresis for collision detection)

            'num_sectors': 200,        # number of sectors in the polar histogram
            'hist_threshold': 8.0,     # threshold for obstacle detection in the histogram
            'smoothing_window': 5,     # window size for smoothing the histogram
            'influence_radius': 1.0,   # radius of influence for obstacles in the histogram
            'a_weight': 1.0,           # weight for the angle difference in the cost function
            'b_weight': 1.0,           # weight for the distance to the goal

            'max_v': 0.22,             # m/s (maximum linear velocity)
            'min_v': 0.05,             # m/s (minimum linear velocity)
            'max_w': 1.00,              # rad/s (maximum angular velocity)
            'kp_heading': 2.0,         # proportional gain for heading control
            'speed_reduction': 0.5,    # factor to reduce speed when obstacles are detected

            'lidar_yaw_offset': math.pi,  # radians (rotation from LiDAR frame to robot forward)
            'replan_cooldown_s': 2.0,     # seconds (minimum time between replanning)
        }]
    )

    # ── 8. Controller — PID wheel velocity controller ──────────────────────
    controller_node = Node(
        package='iolair',
        executable='controller',
        name='puzzlebot_controller',
        output='screen',
    )

    # ── 9. Map Server — serves the pre-built SLAM map as /map ──────────────
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'yaml_filename': map_yaml}]
    )

    # ── 10. Nav2 Lifecycle Manager — auto-activates map_server ─────────────
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
        package='Navigation',
        executable='rviz_goal_bridge',
        name='rviz_goal_bridge',
        output='screen',
    )

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
            'pose_ema_alpha':     0.25,   # 0.25 si aún hay jitter, 0.5 si responde lento
            'beam_skip':          10,
            'sigma_hit':          0.38,
            'resample_interval':  3,
        }]
    )

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

    # ── Assemble LaunchDescription ─────────────────────────────────────────
    return LaunchDescription([
        map_yaml_arg,

        # Map infrastructure (start first — map_server must be active before
        # A* planner tries to subscribe to /map)
        map_server_node,
        lifecycle_manager_node,

        # Perception
        aruco_detector_node,

        # State estimation — odometry must come before aruco_localizer so the
        # /odom subscriber inside aruco_localizer finds the topic immediately
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

        # Safety layer
        vfh_node,

        # Actuation
        controller_node,
        rviz_goal_bridge_node,
    ])
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
  - Bug IBA / BugReflex   (iolair pkg)    — safety reflex layer BUG2 (LiDAR-based)
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

Parameter derivation
--------------------
  stop_dist        = diagonal + 5 cm safety margin  = 0.186 + 0.05  ≈ 0.24 m
  emergency_dist   = stop + braking distance at max_v over hold_ms
                   = 0.24 + 0.22*0.35                              ≈ 0.32 m
  warn_dist        = emergency + soft-braking zone                  = 0.65 m  (unchanged)
  hysteresis       = ~25% of (emergency - stop) gap
                   = 0.25 * (0.32-0.24)                             ≈ 0.03 m
  inflation_r      = half-width + 10 cm clearance  = 0.11 + 0.10   = 0.21 m
                     (was 0.35 — overly conservative, caused narrow-passage failures)
  reflex_v         = reduced to 0.08 m/s (was 0.14) — safer arc speed in tight spaces
  m_line_tol       = 0.12 m — tolerancia perpendicular a línea M de BUG2
  bug2_min_follow  = 0.20 m — distancia mínima wall-follow antes de chequear salida
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
        package='puzzlebot',
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
            'wheel_radius': 0.05,
            'wheel_base':   0.19,
            'rate':         50.0,
            'initial_yaw':  math.pi,
        }]
    )

    # ── 3. ArUco Localizer — landmark-anchoring correction for the EKF ─────
    aruco_localizer_node = Node(
        package='iolair',
        executable='aruco_localizer',
        name='aruco_localizer',
        output='screen',
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

    # ── 4. ArUco Map Publisher — landmark markers for RViz ─────────────────
    aruco_map_node = Node(
        package='iolair',
        executable='aruco_map_publisher',
        name='aruco_map_publisher',
        output='screen',
        parameters=[{
            'landmarks_file': landmarks_yaml_file,
            'publish_rate':   1.0,
            'sphere_scale':   0.095,
            'text_scale':     0.15,
        }]
    )

    # ── 5. A* Planner — global path planner on occupancy grid ──────────────
    # inflation_radius: half-width (0.11) + 10cm clearance = 0.21m
    # Reduced from 0.35 — that value blocked navigable passages narrower than
    # 70cm, which is overkill for a 22cm-wide robot.
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
            'inflation_radius':   0.2,   # was 0.35 — see header derivation
            'waypoint_threshold': 0.10,
            'occupied_threshold': 65,
            'allow_diagonal':     True,
        }]
    )

    # ── 6. Go-to-Goal — drives the robot toward each A* waypoint ───────────
    go_to_goal_node = Node(
        package='iolair',
        executable='go_to_goal',
        name='puzzlebot_go_to_goal',
        output='screen',
    )

    # ── 7. Bug IBA — safety reflex layer BUG2, intercepts /cmd_raw → /cmd_vel
    #
    # stop_dist        0.24m  robot diagonal (0.186) + 5cm margin
    # emergency_dist   0.32m  unchanged — correct for max_v + hold_ms
    # warn_dist        0.65m  unchanged — good soft-braking zone
    # hysteresis       0.03m  was 0.08 — reduced to 25% of (emg-stop) gap;
    #                          0.08 consumed 67% of that gap causing sticky TURN
    # reflex_v         0.08   was 0.14 — safer arc speed in tight spaces
    # reflex_w         0.50   unchanged
    # m_line_tol       0.12m  BUG2 v4.0 — tolerancia perpendicular a línea M
    # bug2_min_follow  0.20m  BUG2 v4.0 — distancia mínima wall-follow antes
    #                          de chequear salida (evita salir en el hit point)
    bug_iba_node = Node(
        package='iolair',
        executable='bug_IBA',
        name='bug_reflex',
        output='screen',
        parameters=[{
            'warn_dist':           0.5,   # unchanged
            'emergency_dist':      0.3,   # unchanged
            'stop_dist':           0.2,   # was 0.20 — below robot diagonal!
            'reflex_v':            0.1,   # was 0.14 — safer in tight spaces
            'reflex_w':            0.50,   # unchanged
            'reflex_hold_ms':      350,    # unchanged
            'front_half_deg':      30.0,   # unchanged
            'side_half_deg':       35.0,   # unchanged
            'hysteresis':          0.03,   # was 0.08 — was too sticky
            'replan_cooldown_s':   2.0,    # v3.0
            'm_line_tol':          0.1,   # v4.0 BUG2 — tolerancia línea M
            'bug2_min_follow_m':   0.40,   # v4.0 BUG2 — distancia mínima wall-follow
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
        package='iolair',
        executable='rviz_goal_bridge',
        name='rviz_goal_bridge',
        output='screen',
    )

    mcl_node = Node(
        package='iolair',
        executable='mcl',
        name='puzzlebot_mcl',
        output='screen',
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

    slam_node = Node(
        package='iolair',
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
        bug_iba_node,

        # Actuation
        controller_node,
        rviz_goal_bridge_node,
    ])
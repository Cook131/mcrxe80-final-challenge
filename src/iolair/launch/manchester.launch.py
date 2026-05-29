"""
aruco_bug_iba.launch.py
=======================
Launches the full navigation stack for the Puzzlebot using:
  - ArUco Detector        (puzzlebot pkg) — detects ArUco markers from camera
  - ArUco Localizer       (iolair pkg)    — landmark-anchoring EKF correction
  - A* Planner            (iolair pkg)    — path planning on occupancy grid
  - Bug IBA / BugReflex   (iolair pkg)    — safety reflex layer (LiDAR-based)
  - Odometry / EKF        (iolair pkg)    — wheel encoder + EKF dead-reckoning
  - Controller            (iolair pkg)    — PID wheel velocity controller
  - Map Server            (nav2)          — serves the pre-built slam_map
  - Nav2 Lifecycle Mgr    (nav2)          — activates map_server automatically

Topic pipeline recap
--------------------
  camera/image_raw
      └─► aruco_detector   →  /aruco/id, /aruco/distance, /aruco/angle,
                               /aruco/waypoint, /aruco/label, /aruco/imagen

  /VelocityEncR, /VelocityEncL
      └─► odometry (EKF)   →  /odom

  /aruco/waypoint + /odom
      └─► aruco_localizer  →  /aruco/pose  →  (fused back into EKF)

  /map (or /slam_map if SLAM is active)
  /odom + /astar/goal (Pose2D)
      └─► astar_planner    →  /goal (Pose2D waypoints, one at a time)
                               /astar/path (nav_msgs/Path, for RViz)
                               /astar/status (String)

  /goal (Pose2D from A*)
      └─► go_to_goal       →  /cmd_raw  (Twist)

  /cmd_raw + /scan (LaserScan)
      └─► bug_IBA          →  /cmd_vel  (with safety reflexes)

  /cmd_vel
      └─► controller       →  /VelocitySetR, /VelocitySetL  →  firmware

NOTE — setup.py entry point required
-------------------------------------
  'astar_planner' must be registered in src/iolair/setup.py:
      'astar_planner = iolair.astar_planner:main',
  Then rebuild with:  colcon build --packages-select iolair
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

    # ── Launch arguments (easy to override from CLI) ───────────────────────
    map_yaml_arg = DeclareLaunchArgument(
        'map_yaml',
        default_value=os.path.join(iolair_dir, 'maps', 'slam_map.yaml'),
        description='Absolute path to the map YAML file'
    )
    map_yaml = LaunchConfiguration('map_yaml')

    # ── 1. ArUco Detector — detects ArUco markers, publishes pose data ─────
    aruco_detector_node = Node(
        package='puzzlebot',
        executable='aruco_detector',
        name='aruco_detector',
        output='screen',
    )

    # ── 2. Odometry (EKF) — wheel encoders + multi-source EKF fusion ───────
    odometry_node = Node(
        package='iolair',
        executable='odometry',
        name='puzzlebot_odometry',
        output='screen',
    )

    # ── 3. ArUco Localizer — landmark-anchoring correction for the EKF ─────
    #       Subscribes:  /aruco/waypoint (PoseStamped), /odom (Odometry)
    #       Publishes:   /aruco/pose (PoseWithCovarianceStamped) → EKF
    aruco_localizer_node = Node(
        package='iolair',
        executable='aruco_localizer',
        name='aruco_localizer',
        output='screen',
        parameters=[{
            # Camera-to-base_link offsets [m]
            'camera_to_base_x': 0.10,
            'camera_to_base_y': 0.00,
            'camera_to_base_z': 0.13,
            # Landmark anchoring distances [m]
            'anchor_min_dist':  0.20,
            'anchor_max_dist':  3.50,
            'anchor_reobserve': 0.30,
            # EKF noise terms
            'r_base_pos':        0.03,
            'r_base_yaw':        0.04,
            'distance_noise_k':  0.025,
            # Publishing rate [Hz]
            'publish_rate':      10.0,
        }]
    )

    # ── 4. A* Planner — global path planner on occupancy grid ──────────────
    #       Subscribes:  /map or /slam_map (OccupancyGrid, auto-detected)
    #                    /odom             (Odometry)
    #                    /astar/goal       (Pose2D)  ← send goal here
    #       Publishes:   /goal             (Pose2D)  → go_to_goal waypoints
    #                    /astar/path       (Path)    → RViz visualisation
    #                    /astar/status     (String)
    astar_planner_node = Node(
        package='iolair',
        executable='astar_planner',
        name='astar_planner',
        output='screen',
        parameters=[{
            # Map topic priority: A* listens to /slam_map first, falls back to /map
            'slam_map_topic':     '/slam_map',
            'static_map_topic':   '/map',
            'odom_topic':         '/odom',
            # Goal input (publish a Pose2D here to trigger planning)
            'goal_in_topic':      '/astar/goal',
            # Waypoint output consumed by go_to_goal
            'goal_out_topic':     '/goal',
            # Planner tuning
            'inflation_radius':   0.15,   # obstacle inflation [m]
            'waypoint_threshold': 0.10,   # distance to consider waypoint reached [m]
            'occupied_threshold': 65,     # occupancy value treated as obstacle (0-100)
            'allow_diagonal':     True,   # 8-connected A* (vs 4-connected)
        }]
    )

    # ── 5. Go-to-Goal — drives the robot toward each A* waypoint ───────────
    #       Subscribes:  /goal (Pose2D from A*), /odom (Odometry)
    #       Publishes:   /cmd_raw (Twist) → bug_IBA
    go_to_goal_node = Node(
        package='iolair',
        executable='go_to_goal',
        name='puzzlebot_go_to_goal',
        output='screen',
    )

    # ── 6. Bug IBA (BugReflex safety layer) — intercepts /cmd_raw → /cmd_vel
    #       Subscribes:  /cmd_raw (Twist), /scan (LaserScan)
    #       Publishes:   /cmd_vel (Twist), /reflex_status (String)
    bug_iba_node = Node(
        package='iolair',
        executable='bug_IBA',
        name='bug_reflex',
        output='screen',
        parameters=[{
            # Safety distances [m]
            'warn_dist':       0.55,   # start of predictive braking zone
            'emergency_dist':  0.22,   # triggers escape arc
            'stop_dist':       0.10,   # triggers full stop
            # Escape arc velocities
            'reflex_v':        0.04,   # linear  [m/s]
            'reflex_w':        0.65,   # angular [rad/s]
            # Timing / hysteresis
            'reflex_hold_ms':  350,    # minimum reflex hold time [ms]
            'front_half_deg':  30.0,   # frontal sector half-angle [deg]
            'side_half_deg':   35.0,   # lateral sector half-angle [deg]
            'hysteresis':      0.06,   # deactivation margin [m]\
        }]
    )

    # ── 7. Controller — PID wheel velocity controller ──────────────────────
    #       Subscribes:  /cmd_vel, /VelocityEncR, /VelocityEncL
    #       Publishes:   /VelocitySetR, /VelocitySetL  → firmware
    controller_node = Node(
        package='iolair',
        executable='controller',
        name='puzzlebot_controller',
        output='screen',
    )

    # ── 8. Map Server — serves the pre-built SLAM map as /map ──────────────
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'yaml_filename': map_yaml}]
    )

    # ── 9. Nav2 Lifecycle Manager — auto-activates map_server ──────────────
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

    # ── Assemble LaunchDescription ─────────────────────────────────────────
    return LaunchDescription([
        map_yaml_arg,

        # Perception
        aruco_detector_node,

        # State estimation
        odometry_node,
        aruco_localizer_node,

        # Path planning
        astar_planner_node,

        # Navigation
        go_to_goal_node,

        # Safety layer
        bug_iba_node,

        # Actuation
        controller_node,

        # Map infrastructure
        map_server_node,
        lifecycle_manager_node,
    ])
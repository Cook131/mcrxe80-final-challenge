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
  - Bug IBA / BugReflex   (iolair pkg)    — safety reflex layer (LiDAR-based)
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
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import math


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
    # Broadcasts BOTH:
    #   TF map  → odom        (ArUco correction offset, updated at ~10 Hz)
    #   TF odom → base_link   (raw dead-reckoning, updated at 50 Hz)
    # /odom topic carries the corrected pose with frame_id='map'.
    # FIX: removed undeclared 'ekf_mode' parameter.
    odometry_node = Node(
        package='iolair',
        executable='odometry',
        name='puzzlebot_odometry',
        output='screen',
        parameters=[{
            'wheel_radius': 0.05,
            'wheel_base':   0.19,
            'rate':         50.0,
            'initial_yaw': math.pi,
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
            'anchor_reobserve': 0.05,
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
            'sphere_scale':   0.08,
            'text_scale':     0.12,
        }]
    )

    # ── 5. A* Planner — global path planner on occupancy grid ──────────────
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
            'inflation_radius':   0.35,
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

    # ── 7. Bug IBA — safety reflex layer, intercepts /cmd_raw → /cmd_vel ───
    bug_iba_node = Node(
        package='iolair',
        executable='bug_IBA',
        name='bug_reflex',
        output='screen',
        parameters=[{
            'warn_dist':       0.65,
            'emergency_dist':  0.32,
            'stop_dist':       0.20,
            'reflex_v':        0.14,
            'reflex_w':        0.5,
            'reflex_hold_ms':  350,
            'front_half_deg':  30.0,
            'side_half_deg':   35.0,
            'hysteresis':      0.08,
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
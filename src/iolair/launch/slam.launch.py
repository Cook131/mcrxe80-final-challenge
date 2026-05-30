#!/usr/bin/env python3
"""
slam_launch.py — Puzzlebot SLAM on the REAL robot (no Gazebo, no nav2)
=======================================================================
Starts only the ROS 2 nodes needed to build a map with the physical
Puzzlebot.  No Gazebo, no nav2, no MCL required.

Node graph
----------
  firmware ──/VelocityEncR, /VelocityEncL──► odometry   → /odom + TF odom→base_link
  /scan    ──────────────────────────────► slam_node   → /slam_map + TF map→odom
  teleop   ──/cmd_vel──► controller ──/VelocitySetR,L──► firmware

How to use
----------
  1. Build and source:
       colcon build --packages-select iolair
       source install/setup.bash

  2. Launch:
       ros2 launch iolair slam_launch.py

  3. Drive with W/S/A/D in the terminal where teleop is running.

  4. Save the map when finished:
       ros2 service call /slam/save_map std_srvs/srv/Trigger
     → writes /tmp/slam_map.pgm and /tmp/slam_map.yaml

  5. Copy the map files to iolair/maps/ and rebuild.

  6. Monitor diagnostics:
       ros2 topic echo /slam/diagnostics
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction


def generate_launch_description():

    # ── 1. Static TF: base_link → lidar_link ──────────────────────────────
    # Tells RViz where the LiDAR sits on the robot frame.
    # Adjust the Z value (0.14) if your physical LiDAR height is different.
    static_tf_lidar = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_lidar',
        arguments=['0', '0', '0.17', '3.1419', '0', '0', 'base_link', 'lidar_link'],
    )

    # ── 2. Odometry node ──────────────────────────────────────────────────
    # Reads /VelocityEncR and /VelocityEncL from the Puzzlebot firmware.
    # Publishes /odom and broadcasts the odom → base_link TF.
    odometry_node = Node(
        package='iolair',
        executable='odometry',
        name='puzzlebot_odom_node',
        output='screen',
    )

    # ── 3. Controller node ────────────────────────────────────────────────
    # Converts /cmd_vel (Twist) → /VelocitySetR and /VelocitySetL (Float32).
    controller_node = Node(
        package='iolair',
        executable='controller',
        name='puzzlebot_main_controller',
        output='screen',
    )

    # ── 4. SLAM node (delayed 2 s so /odom is already flowing) ────────────
    # Subscribes to /scan and /odom.
    # Publishes /slam_map, /slam_pose, /slam/diagnostics, and map→odom TF.
    # Call /slam/save_map (std_srvs/Trigger) to dump the map to disk.
    slam_node = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='iolair',
                executable='slam',
                name='slam_node',
                output='screen',
                parameters=[{
                    # ── Topic ─────────────────────────────────────────────
                    # Must match the topic your LiDAR driver publishes.
                    'scan_topic':      '/scan',

                    # ── Grid ──────────────────────────────────────────────
                    'resolution':      0.05,   # 5 cm per cell
                    'map_init_size':   400,    # 400×400 cells = 20 m × 20 m
                    'map_origin_x':   -10.0,   # world X of grid bottom-left [m]
                    'map_origin_y':   -10.0,   # world Y of grid bottom-left [m]

                    # ── LiDAR ─────────────────────────────────────────────
                    # Match this to your sensor's usable range:
                    #   RPLiDAR A1 → 8 m | A2 → 10 m | A3 → 25 m
                    'lidar_max_range': 8.0,
                    # beam_skip=0  → adaptive (auto sub-samples to target_beams).
                    # beam_skip=1  → all beams (best quality, most CPU).
                    # beam_skip=3  → recommended for slow Raspberry Pi.
                    'beam_skip':       1,      # adaptive mode
                    'target_beams':    360,    # beams used per scan in adaptive mode

                    # ── Log-odds update model ─────────────────────────────
                    'log_odds_occ':    0.85,
                    'log_odds_free':   0.40,
                    'log_odds_max':    3.5,
                    'log_odds_min':   -3.5,

                    # ── ICP scan correction ───────────────────────────────
                    # Compensates for odometry drift (scan-to-scan +
                    # scan-to-map).  Set use_icp: False if the Pi is overloaded
                    # (monitor /slam/diagnostics → avg_scan_ms).
                    'use_icp':            True,
                    'icp_max_iter':       50,
                    'icp_tolerance':      1e-4,
                    'icp_max_correction': 0.4,  # m — discard wilder jumps

                    # ── Motion gate ───────────────────────────────────────
                    # Map is only updated when the robot has moved this much.
                    # Prevents noise accumulation when standing still.
                    'min_travel_m':    0.02,   # metres
                    'min_travel_rad':  0.02,   # radians (~1.1 °)

                    # ── Publishing ────────────────────────────────────────
                    'publish_rate':    1.0,    # Hz — map topic (lower = less network load)
                    'tf_rate':         20.0,   # Hz — map→odom TF

                    # ── Map save destination ──────────────────────────────
                    'save_map_path':   '/home/serch/mcrxe80-final-challenge/src/iolair/maps/SLAM_map',
                }],
            )
        ],
    )

    return LaunchDescription([
        static_tf_lidar,
        odometry_node,
        controller_node,
        slam_node,
    ])
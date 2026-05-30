#!/usr/bin/env python3
"""
slam_launch.py — Puzzlebot SLAM on the REAL robot (no Gazebo, no nav2)
=======================================================================
Starts only the ROS 2 nodes needed to build a map with the physical
Puzzlebot.  No Gazebo, no nav2, no MCL required.

Node graph
----------
  firmware ──/VelocityEncR, /VelocityEncL──► odometry      → /odom + TF odom→base_link
  /scan    ──────────────────────────────► slam_node      → /slam_map + TF map→odom
  /slam_map ─────────────────────────────► map_merger_node → /merged_map
  teleop   ──/cmd_vel──► controller ──/VelocitySetR,L──► firmware

How to use
----------
  1. Build and source:
       colcon build --packages-select iolair
       source install/setup.bash

  2. Launch:
       ros2 launch iolair slam_launch.py

  3. Drive with W/S/A/D in the terminal where teleop is running.

  4. Save the merged map when finished:
       ros2 service call /map_merger/save_map std_srvs/srv/Trigger
     → writes /home/serch/mcrxe80-final-challenge/src/iolair/maps/SLAM_map.pgm
              /home/serch/mcrxe80-final-challenge/src/iolair/maps/SLAM_map.yaml

  5. Copy the map files to iolair/maps/ and rebuild.

  6. Monitor diagnostics:
       ros2 topic echo /slam/diagnostics
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction
import math


def generate_launch_description():

    # ── 1. Static TF: base_link → lidar_link ──────────────────────────────
    # Tells RViz where the LiDAR sits on the robot frame.
    # Adjust the Z value (0.14) if your physical LiDAR height is different.
    static_tf_lidar = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_lidar',
        arguments=['0', '0', '0.14', '0', '0', '0', 'base_link', 'lidar_link'],
    )

    # ── 2. Odometry node ──────────────────────────────────────────────────
    # Reads /VelocityEncR and /VelocityEncL from the Puzzlebot firmware.
    # Publishes /odom and broadcasts the odom → base_link TF.
    odometry_node = Node(
        package='iolair',
        executable='odometry',
        name='puzzlebot_odom_node',
        output='screen',
        parameters=[{'initial_yaw': math.pi}],
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
    # Publishes /slam_map, /slam_pose, and map→odom TF.
    slam_node = TimerAction(
        period=2.0,
        actions=[
            Node(
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
        ],
    )

    # ── 5. Map merger node (delayed 3 s — starts after slam_node is up) ───
    # Subscribes to /slam_map and accumulates every snapshot into a single
    # persistent merged map published on /merged_map.
    # Call /map_merger/save_map (std_srvs/Trigger) to dump the map to disk.
    map_merger_node = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='iolair',
                executable='map_merger',
                name='map_merger_node',
                output='screen',
                parameters=[{
                    'source_map_topic':  '/slam_map',
                    'publish_rate':       1.0,
                    'map_frame':         'map',
                    'resolution':         0.05,
                    'lo_occ':             0.85,
                    'lo_free':            0.40,
                    'lo_max':             5.0,
                    'lo_min':            -5.0,
                    'occ_thresh':         0.65,
                    'free_thresh':        0.35,
                    'save_map_path':     '/home/serch/mcrxe80-final-challenge/src/iolair/maps/SLAM_map',
                }]
            )
        ],
    )

    return LaunchDescription([
        static_tf_lidar,
        odometry_node,
        controller_node,
        slam_node,
        map_merger_node,
    ])
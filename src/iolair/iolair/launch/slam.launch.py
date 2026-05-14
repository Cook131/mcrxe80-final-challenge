#!/usr/bin/env python3
"""
slamReal.launch.py — Puzzlebot SLAM on the REAL robot (no Gazebo, no nav2)
===========================================================================
Starts only the ROS 2 software nodes needed to build a map with the
physical Puzzlebot.  Gazebo, gazebo_ros and every simulation dependency
are intentionally absent.

Hardware assumptions
--------------------
* The Puzzlebot's on-board firmware already publishes:
    /VelocityEncR  (std_msgs/Float32) — right encoder angular velocity [rad/s]
    /VelocityEncL  (std_msgs/Float32) — left  encoder angular velocity [rad/s]
    /lidar         (sensor_msgs/LaserScan) — 2-D LiDAR scan
  and subscribes to:
    /VelocitySetR  (std_msgs/Float32) — right wheel set-point [rad/s]
    /VelocitySetL  (std_msgs/Float32) — left  wheel set-point [rad/s]

* You run this launch on the robot's Raspberry Pi (or your laptop
  connected to the same ROS 2 domain / same network).

Node graph (no nav2, no MCL)
-----------------------------
  firmware  ──/VelocityEncR, /VelocityEncL──►  odometry_node
                                                   │ /odom + TF odom→base_link
                                                   ▼
  /lidar  ──────────────────────────────────►  slam_node
                                                   │ /slam_map  (OccupancyGrid)
                                                   │ /slam_pose (PoseStamped)
                                                   │ TF map→odom
                                                   ▼
  teleop ──/cmd_vel──► controller_node ──/VelocitySetR,L──► firmware

How to use
----------
1.  ssh into the robot (or run on-board):
        ros2 launch iolair slamReal.launch.py

2.  Drive with the teleop terminal that opens (W/S/A/D, Q to quit).

3.  In a second terminal, monitor the map:
        ros2 topic echo /slam_map --no-arr   # metadata only
        rviz2 -d <path>/rviz/puzzlebot.rviz  # if you have a desktop nearby

4.  When the map looks complete, save it:
        ros2 service call /slam/save_map std_srvs/srv/Trigger

    This writes /tmp/slam_map.pgm and /tmp/slam_map.yaml.
    Copy them to iolair/maps/ and rebuild to use with map_publisher.
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction


def generate_launch_description():

    # ── 1. Static TF frames ────────────────────────────────────────────────
    # map → odom  : slam_node broadcasts this dynamically (20 Hz).
    #               We publish a one-shot identity bootstrap so RViz does not
    #               error out in the first ~2 seconds before slam_node is up.
    static_tf_map_odom_boot = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_map_odom_boot',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
    )

    # odom → base_link : odometry_node broadcasts this dynamically.
    # LiDAR sensor frame — distance from base_link to the LiDAR centre.
    # Adjust the Z offset (third argument) if your LiDAR sits at a
    # different height on the physical robot.
    static_tf_lidar = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_lidar',
        arguments=['0', '0', '0.14', '0', '0', '0', 'base_link', 'lidar_link'],
    )

    # ── 2. Odometry node ──────────────────────────────────────────────────
    # Reads /VelocityEncR and /VelocityEncL from the firmware and
    # publishes /odom + the odom→base_link TF transform.
    odometry_node = Node(
        package='iolair',
        executable='odometry',
        name='odometry_node',
        output='screen',
        parameters=[{
            'wheel_radius': 0.05,    # metres — measure your physical wheel
            'wheel_base':   0.18,    # metres — distance between wheel centres
            'publish_rate': 100.0,   # Hz
        }],
    )

    # ── 3. Robot controller ───────────────────────────────────────────────
    # Converts /cmd_vel (Twist) → /VelocitySetR and /VelocitySetL (Float32).
    controller_node = Node(
        package='iolair',
        executable='controller',
        name='puzzlebot_controller',
        output='screen',
        parameters=[{
            'wheel_radius': 0.05,
            'wheel_base':   0.19,   # use the same value as in robot_controller.py
        }],
    )

    # ── 4. SLAM node (starts 2 s after odom to ensure /odom is flowing) ──
    # Subscribes to /lidar and /odom.
    # Publishes /slam_map, /slam_pose, and the map→odom TF.
    # Exposes /slam/save_map service to persist the map to disk.
    slam_node = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='iolair',
                executable='slam',
                name='slam_node',
                output='screen',
                parameters=[{
                    # ── Grid ──────────────────────────────────────────────
                    'resolution':       0.05,    # 5 cm / cell
                    'map_init_size':    400,      # 400×400 = 20 m × 20 m
                    'map_origin_x':    -10.0,     # world X of grid bottom-left [m]
                    'map_origin_y':    -10.0,     # world Y of grid bottom-left [m]

                    # ── LiDAR ─────────────────────────────────────────────
                    # Set lidar_max_range to the usable range of YOUR sensor.
                    # Common values: RPLiDAR A1 → 8 m, A2 → 10 m, A3 → 25 m.
                    'lidar_max_range':  8.0,
                    # beam_skip=1 uses all beams (best quality, more CPU).
                    # Increase to 3-4 on a slow Raspberry Pi.
                    'beam_skip':        2,

                    # ── Log-odds model ────────────────────────────────────
                    'log_odds_occ':     0.85,
                    'log_odds_free':    0.40,
                    'log_odds_max':     3.5,
                    'log_odds_min':    -3.5,

                    # ── ICP scan matching ─────────────────────────────────
                    # On a Raspberry Pi 4 with ~180 beams this adds ~5 ms
                    # per scan.  Set use_icp: False if the Pi is struggling.
                    'use_icp':          True,
                    'icp_max_iter':     15,
                    'icp_tolerance':    1e-4,

                    # ── Publishing ────────────────────────────────────────
                    'publish_rate':     1.0,      # Hz — reduce on slow Pi

                    # ── Map save path ─────────────────────────────────────
                    'save_map_path':    '/tmp/slam_map',
                }],
            )
        ],
    )

    # ── 5. Keyboard teleop ────────────────────────────────────────────────
    # Opens in the same terminal (no xterm needed on the robot).
    # W = forward, S = backward, A = turn left, D = turn right, Q = quit.
    teleop_node = Node(
        package='iolair',
        executable='teleop',
        name='puzzlebot_teleop',
        output='screen',
    )

    return LaunchDescription([
        static_tf_map_odom_boot,
        static_tf_lidar,
        odometry_node,
        controller_node,
        slam_node,
        teleop_node,
    ])
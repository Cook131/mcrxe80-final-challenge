#!/usr/bin/env python3
"""
sim.launch.py — Puzzlebot SLAM simulation
==========================================
Stack: ROS 2 Humble + classic Gazebo 11 + gazebo_ros bridge.
No nav2. No ros_gz_bridge. No Ignition/Fortress/Harmonic.

Launch order:
  1. Gazebo classic (gzserver + gzclient)  via gazebo_ros launch
  2. Static TF frames
  3. map_publisher_node  (reads .pgm/.yaml → /map, latched)
  4. Odometry node
  5. Robot controller
  6. MCL node  (delayed 3 s so /map is already latched)
  7. RViz2

The custom DiffDynamicPlugin publishes encoder velocities and subscribes
to cmd_vel directly as Gazebo topics — no bridge needed for those.
The camera and LiDAR publish via libgazebo_ros_camera / libgazebo_ros_ray_sensor
plugins embedded in the SDF, so topics appear directly on the ROS graph.
"""

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    pkg_share     = get_package_share_directory('iolair')
    gazebo_ros    = get_package_share_directory('gazebo_ros')

    world_file    = os.path.join(pkg_share, 'worlds', 'puzzlebot_world.world')
    gazebo_models = os.path.join(pkg_share, 'gazebo')   # contains puzzlebot/
    plugin_dir    = os.path.join(pkg_share, 'gazebo', 'plugins')
    map_yaml      = os.path.join(pkg_share, 'maps', 'puzzlebot_map.yaml')
    rviz_file     = os.path.join(pkg_share, 'rviz', 'puzzlebot.rviz')

    # ── 1. Gazebo classic ──────────────────────────────────────────────────
    # gazebo_ros provides gzserver.launch.py + gzclient.launch.py
    # (or gazebo.launch.py which starts both).
    # GAZEBO_MODEL_PATH tells Gazebo where to find the puzzlebot model.
    # GAZEBO_PLUGIN_PATH tells Gazebo where libDiffDynamicPlugin.so lives.
    import os as _os
    _os.environ['GAZEBO_MODEL_PATH'] = \
        gazebo_models + ':' + _os.environ.get('GAZEBO_MODEL_PATH', '')
    _os.environ['GAZEBO_PLUGIN_PATH'] = \
        plugin_dir + ':' + _os.environ.get('GAZEBO_PLUGIN_PATH', '')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': world_file, 'verbose': 'false'}.items(),
    )

    # ── 2. Static TF frames (sensors only) ────────────────────────────────
    # map → odom  : broadcast dynamically by mcl_node   (localization correction)
    # odom → base_link : broadcast dynamically by odometry_node (dead reckoning)
    # Only sensor offsets are static — they never change.

    # Bootstrap map→odom as identity so RViz doesn't error before MCL starts.
    # The MCL node (dynamic TransformBroadcaster) overrides this automatically
    # once its first pose estimate is published (~3 s after launch).
    static_tf_map_odom_boot = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_map_odom_boot',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
    )

    # sensor frames — must match <frameName> in the SDF plugins
    static_tf_lidar = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_lidar',
        arguments=['0', '0', '0.14', '0', '0', '0', 'base_link', 'lidar_link'],
    )

    static_tf_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_camera',
        arguments=['0.05', '0', '0.11', '0', '0', '0', 'base_link', 'camera_link'],
    )

    # ── 3. Map publisher ───────────────────────────────────────────────────
    map_publisher = Node(
        package='iolair',
        executable='map_publisher',
        name='map_publisher_node',
        output='screen',
        parameters=[{'map_yaml_path': map_yaml}],
    )

    # ── 4. Odometry and controller nodes are no longer needed ─────────────
    # libgazebo_ros_diff_drive publishes /odom and odom→base_link TF natively.
    # It also reads /cmd_vel directly — teleop works without a controller node.

    # ── 6. MCL node (delayed so /map is already latched) ──────────────────
    mcl_node = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='iolair',
                executable='mcl',
                name='mcl_node',
                output='screen',
                parameters=[{
                    'init_x':   0.0,
                    'init_y':   0.0,
                    'init_yaw': 0.0,
                }],
            )
        ],
    )

    # ── 6b. SLAM node (starts after MCL so it can use /mcl_pose) ──────────
    # Publishes the live-built map on /slam_map (TRANSIENT_LOCAL).
    # Coexists with map_publisher_node (/map) — they use different topics.
    slam_node = TimerAction(
        period=5.0,   # wait for MCL particles to settle a bit first
        actions=[
            Node(
                package='iolair',
                executable='slam',
                name='slam_node',
                output='screen',
                parameters=[{
                    'resolution':       0.05,
                    'map_origin_x':    -10.0,
                    'map_origin_y':    -10.0,
                    'lidar_max_range':  10.0,
                    'beam_skip':        2,
                    'publish_rate':     2.0,
                }],
            )
        ],
    )

    # ── 7. RViz ────────────────────────────────────────────────────────────
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_file],
        output='screen',
    )

    return LaunchDescription([
        gazebo,
        static_tf_map_odom_boot,
        static_tf_lidar,
        static_tf_camera,
        map_publisher,
        mcl_node,
        slam_node,
        rviz_node,
    ])
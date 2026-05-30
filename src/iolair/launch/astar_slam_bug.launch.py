

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction
import math


def generate_launch_description():

    # ── 1. Static TF: base_link → lidar_link ──────────────────────────────
    static_tf_lidar = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_lidar',
        arguments=['0', '0', '0.17', '3.1419', '0', '0', 'base_link', 'lidar_link'],
    )

    # ── 2. Odometry node ──────────────────────────────────────────────────
    odometry_node = Node(
        package='iolair',
        executable='odometry',
        name='puzzlebot_odom_node',
        output='screen',
        parameters=[{'initial_yaw': math.pi}],
    )

    # ── 3. Controller node ────────────────────────────────────────────────
    controller_node = Node(
        package='iolair',
        executable='controller',
        name='puzzlebot_main_controller',
        output='screen',
    )

    # ── 4. SLAM node (delayed 2 s so /odom is already flowing) ────────────
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
                    'save_map_path':    '/home/serch/mcrxe80-final-challenge/src/iolair/maps/SLAM_map',
                }]
            )
        ],
    )

    # ── 5. A* planner node ────────────────────────────────────────────────
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
            'waypoint_threshold': 0.12,
            'occupied_threshold': 50,
            'allow_diagonal':     False,
        }]
    )

    # ── 6. RViz goal bridge ───────────────────────────────────────────────
    rviz_goal_bridge_node = Node(
        package='iolair',
        executable='rviz_goal_bridge',
        name='rviz_goal_bridge',
        output='screen',
    )

    # ── 7. Go-to-goal node ────────────────────────────────────────────────
    go_to_goal_node = Node(
        package='iolair',
        executable='go_to_goal',
        name='puzzlebot_go_to_goal',
        output='screen',
    )

    # ── 8. Bug reflex / IBA explorer node ────────────────────────────────
    bug_iba_node = Node(
        package='iolair',
        executable='bug_IBA',      # ← was 'bug_reflex', must match setup.py
        name='bug_reflex',
        output='screen',
        parameters=[{
            'lidar_yaw_offset': math.pi,
        }]
    )

    return LaunchDescription([
        static_tf_lidar,
        odometry_node,
        controller_node,
        slam_node,
        astar_planner_node,
        rviz_goal_bridge_node,
        go_to_goal_node,
        bug_iba_node,
    ])
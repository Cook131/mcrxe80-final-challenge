"""
qr_align_tryout.launch.py
=========================
Tryout para QRCollectNode en pasillo.
Asume que la cámara ya está corriendo en /camera_raw/compressed.

Nodos levantados:
  aruco_detector_node  — detección ArUco + QR  (suscribe /camera_raw/compressed)
  qr_collect_node      — FSM alignment QR

Uso:
  ros2 launch puzzlebot qr_align_tryout.launch.py zone:=rack
  ros2 launch puzzlebot qr_align_tryout.launch.py zone:=conveyor

Trigger manual (otra terminal):
  ros2 topic pub --once /collect/trigger std_msgs/String "data: rack"
  ros2 topic pub --once /collect/trigger std_msgs/String "data: conveyor"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ── ArUco Detector ────────────────────────────────────────────────────────
    # Suscribe /camera_raw/compressed directamente — sin remap necesario.

    # ── QRCollectNode ─────────────────────────────────────────────────────────
    qr_collect = Node(
        package    = 'Vision',
        executable = 'qr_collect_node',
        name       = 'qr_collect_node',
        output     = 'screen',
        parameters = [{
            'kp_angle'             : 0.018,
            'kd_angle'             : 0.004,
            'kp_dist'              : 0.40,
            'kd_dist'              : 0.08,
            'angle_tol_deg'        : 4.0,
            'approach_dist'        : 0.28,
            'approach_handoff_dist': 0.80,
            'dist_tol'             : 0.03,
            'max_angular'          : 0.45,
            'max_linear'           : 0.18,
            'extract_speed'        : 0.08,
            'extract_time'         : 0.6,
            'reverse_speed'        : 0.10,
            'reverse_time'         : 1.8,
            'qr_timeout'           : 2.5,
            'lift_timeout'         : 8.0,
            'nav_approach_timeout' : 30.0,
            'fsm_rate_hz'          : 20.0,
        }],
    )

    qr_zone_checker = Node(
        package    = 'Vision',
        executable = 'qr_zone_checker',
        name       = 'qr_zone_checker',
        output     = 'screen',
    )

    log_ready = LogInfo(
        msg = [
            '\n',
            '╔══════════════════════════════════════════════════════╗\n',
            '║        QR ALIGNMENT TRYOUT — Iolair Puzzlebot        ║\n',
            '╠══════════════════════════════════════════════════════╣\n',
            '║  Zona       : ', LaunchConfiguration('zone'), '\n',
            '║                                                      ║\n',
            '║  Dispara (otra terminal):                            ║\n',
            '║    ros2 topic pub --once /collect/trigger \\          ║\n',
            '║      std_msgs/String "data: rack"                    ║\n',
            '║                                                      ║\n',
            '║  Monitorea:                                          ║\n',
            '║    ros2 topic echo /collect/done                     ║\n',
            '║    ros2 topic echo /aruco/qr                         ║\n',
            '║    ros2 topic echo /aruco/qr/angle                   ║\n',
            '╚══════════════════════════════════════════════════════╝\n',
        ]
    )

    return LaunchDescription([
        log_ready,
        qr_collect,
        qr_zone_checker,
    ])

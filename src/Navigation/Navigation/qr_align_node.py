"""
qr_align_tryout.launch.py (Modificado)
=====================================
Tryout para QRCollectNode v2.
Asume que manchester.launch ya está activo (Cámara, ArUco Detector y A* funcionando).

Nodos levantados:
  qr_collect_node      — FSM alignment QR + Integración A*
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    zone_arg = DeclareLaunchArgument(
        'zone',
        default_value='rack',
        description="Zona de recolección: 'rack' (N1) o 'conveyor' (N2)",
    )

    # ── QRCollectNode (v2 con integración A*/GoToGoal) ──────────────────────
    # Este nodo ahora se comunica con /astar/goal y /odom
    qr_collect = Node(
        package    = 'puzzlebot',
        executable = 'qr_collect_node', # Asegúrate que el script v2 sea este ejecutable
        name       = 'qr_collect_node',
        output     = 'screen',
        parameters = [{
            'kp_angle'             : 0.018,
            'kd_angle'             : 0.004,
            'kp_dist'              : 0.40,
            'kd_dist'              : 0.08,
            'angle_tol_deg'        : 4.0,
            'approach_dist'        : 0.28,
            'approach_handoff_dist': 0.80, # Distancia donde A* entrega al PD
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

    log_ready = LogInfo(
        msg = [
            '\n',
            '╔══════════════════════════════════════════════════════╗\n',
            '║   QR ALIGN v2 (A* Ready) — Iolair Puzzlebot          ║\n',
            '╠══════════════════════════════════════════════════════╣\n',
            '║  Asumiendo que manchester.launch está activo...      ║\n',
            '║  Zona       : ', LaunchConfiguration('zone'), '\n',
            '║                                                      ║\n',
            '║  Dispara (otra terminal):                            ║\n',
            '║    ros2 topic pub --once /collect/trigger \\          ║\n',
            '║      std_msgs/String "data: ', LaunchConfiguration('zone'), '"\n',
            '╚══════════════════════════════════════════════════════╝\n',
        ]
    )

    return LaunchDescription([
        zone_arg,
        log_ready,
        qr_collect,
    ])

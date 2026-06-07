"""
full_collect_delivery.launch.py
================================
Pipeline completo de recolección y entrega de pallet.

Secuencia autónoma
------------------
  1. El robot navega con el stack completo (A* + MCL + EKF + VFH+).
  2. Al llegar a un pallet, qr_align_node detecta el QR, se alinea,
     recoge el pallet (lift n1/n2 → HOLD) y retrocede.
  3. qr_align_node publica /collect/done "SUCCESS" y /collect/qr_payload
     con el nombre de la empresa del QR.
  4. truck_align_node recibe el trigger en /truck_align/cmd, va al
     waypoint truck_detection, encuentra el logo YOLO de la empresa,
     se alinea 24 cm a la derecha, avanza 22 cm para meter las horquillas
     y baja el pallet ("down").
  5. truck_align_node publica /truck_align/done "SUCCESS".

Nodos levantados
-----------------
  Percepción:
    aruco_detector        — Vision        (QR + ArUco → qr_detector executable)
    yolo_detector_node    — Vision        (logos YOLO, /yolo/detecciones)
    spi_servo_node        — puzzlebot     (control lift FPGA via SPI)

  Localización y mapa:
    map_server            — nav2_map_server
    lifecycle_manager     — nav2_lifecycle_manager
    odometry              — iolair
    aruco_localizer       — LocalizationMapping
    aruco_map_publisher   — LocalizationMapping
    slam_node             — LocalizationMapping
    mcl                   — LocalizationMapping

  Navegación:
    astar_planner         — Navigation
    go_to_goal            — iolair        (publica /cmd_raw)
    vfh_plus              — iolair        (filtra /cmd_raw → /cmd_vel)
    controller            — iolair
    rviz_goal_bridge      — Navigation

  Misión:
    qr_align_node         — Navigation
    truck_align_node      — Navigation

Uso
---
  ros2 launch iolair full_collect_delivery.launch.py

  Trigger manual de recolección (zona rack):
    ros2 topic pub --once /collect/trigger std_msgs/String "data: rack"

  Trigger manual de entrega (camión wolmar):
    ros2 topic pub --once /truck_align/cmd std_msgs/String "data: wolmar"

  Trigger manual de entrega (camión emezon):
    ros2 topic pub --once /truck_align/cmd std_msgs/String "data: emezon"

Waypoints
---------
  El archivo waypoints_file debe tener una entrada "truck_detection":

    waypoints:
      - name: truck_detection
        x: 2.150    # ← coordenadas reales del mapa
        y: 0.450

  Medir en el robot físico con:
    ros2 topic echo /odom --once | grep -A2 "position:"

Parámetros ajustables (sin recompilar)
---------------------------------------
  qr_align_node:
    align_stop_dist      — distancia centro robot → QR al alinearse (0.35 m)
    forklift_reach_m     — largo horquilla, define stop de APPROACH_FINAL (0.20 m)
    cam_fwd_m / cam_left_m — offset cámara respecto a base_link

  truck_align_node:
    lateral_offset_m     — offset lateral derecho al logo del camión (0.24 m)
    stop_dist_m          — distancia alineación al camión (0.30 m)
    forklift_len_m       — largo físico de la horquilla (0.15 m)
    insert_depth_m       — cuánto entra la punta al camión (0.07 m)
    advance_speed        — velocidad de inserción (0.08 m/s)
    waypoints_file       — ruta al YAML de waypoints
"""

import math
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ── Directorios de paquetes ────────────────────────────────────────────
    iolair_dir = get_package_share_directory('iolair')

    # ── Rutas de archivos de configuración ────────────────────────────────
    map_yaml_default      = os.path.join(iolair_dir, 'maps',    'SLAM_map.yaml')
    landmarks_yaml        = os.path.join(iolair_dir, 'configs', 'aruco_landmarks.yaml')
    waypoints_yaml        = os.path.join(iolair_dir, 'configs', 'waypoints.yaml')

    # ── Launch arguments ──────────────────────────────────────────────────
    map_yaml_arg = DeclareLaunchArgument(
        'map_yaml',
        default_value=map_yaml_default,
        description='Ruta al YAML del mapa pre-construido',
    )
    map_yaml = LaunchConfiguration('map_yaml')

    # ══════════════════════════════════════════════════════════════════════
    # PERCEPCIÓN
    # ══════════════════════════════════════════════════════════════════════

    # QR detector — detección QR + ArUco landmarks
    # Suscribe /camera_raw/compressed directamente (sin remap).
    # Publica /qr/data, /qr/distance, /qr/angle  → qr_align_node
    # Publica /aruco/distance, /aruco/angle, /aruco/id → aruco_localizer
    aruco_detector_node = Node(
        package='Vision',
        executable='qr_detector',
        name='qr_detector',
        output='screen',
    )

    # YOLO detector — logos de camiones
    # Publica /yolo/detecciones (JSON) e /yolo/imagen.
    yolo_detector_node = Node(
        package='Vision',
        executable='yolo_vision',
        name='yolo_detector_node',
        output='screen',
    )

    # SPI servo — control del lift FPGA
    # Suscribe /lift_auto (String), publica /lift_done (String).
    # ══════════════════════════════════════════════════════════════════════
    # MAPA E INFRAESTRUCTURA
    # ══════════════════════════════════════════════════════════════════════

    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'yaml_filename': map_yaml}],
    )

    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[
            {'node_names': ['map_server']},
            {'autostart': True},
        ],
    )

    # ══════════════════════════════════════════════════════════════════════
    # LOCALIZACIÓN
    # ══════════════════════════════════════════════════════════════════════

    odometry_node = Node(
        package='iolair',
        executable='odometry',
        name='puzzlebot_odometry',
        output='screen',
        parameters=[{
            'wheel_radius': 0.045,
            'wheel_base':   0.19,
            'rate':         50.0,
            'initial_yaw':  math.pi,
        }],
    )

    aruco_localizer_node = Node(
        package='LocalizationMapping',
        executable='aruco_localizer',
        name='aruco_localizer',
        output='screen',
        parameters=[{
            'landmarks_file':   landmarks_yaml,
            'camera_to_base_x': 0.07,
            'camera_to_base_y': 0.08,
            'camera_to_base_z': 0.15,
            'anchor_min_dist':  0.2,
            'anchor_max_dist':  3.50,
            'anchor_reobserve': 0.1,
            'r_base_pos':       0.03,
            'r_base_yaw':       0.04,
            'distance_noise_k': 0.025,
            'publish_rate':     10.0,
        }],
    )

    aruco_map_node = Node(
        package='LocalizationMapping',
        executable='aruco_map_publisher',
        name='aruco_map_publisher',
        output='screen',
        parameters=[{
            'landmarks_file': landmarks_yaml,
            'publish_rate':   1.0,
            'sphere_scale':   0.2,
            'text_scale':     0.15,
        }],
    )

    mcl_node = Node(
        package='LocalizationMapping',
        executable='mcl',
        name='puzzlebot_mcl',
        output='screen',
        parameters=[{
            'num_particles':    500,
            'lidar_yaw_offset': math.pi,
            'initial_yaw':      math.pi,
            'map_frame':        'map',
            'odom_frame':       'odom',
            'base_frame':       'base_link',
            'pose_ema_alpha':   0.25,
            'beam_skip':        10,
            'sigma_hit':        0.38,
            'resample_interval': 3,
        }],
    )

    slam_node = Node(
        package='LocalizationMapping',
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
        }],
    )

    # ══════════════════════════════════════════════════════════════════════
    # NAVEGACIÓN
    # ══════════════════════════════════════════════════════════════════════

    astar_planner_node = Node(
        package='Navigation',
        executable='astar_planner',
        name='astar_planner',
        output='screen',
        parameters=[{
            'slam_map_topic':     '/slam_map',
            'static_map_topic':   '/map',
            'odom_topic':         '/odom',
            'goal_in_topic':      '/astar/goal',
            'goal_out_topic':     '/goal',
            'inflation_radius':   0.2,
            'waypoint_threshold': 0.10,
            'occupied_threshold': 50,
            'allow_diagonal':     False,
        }],
    )

    # go_to_goal publica en /cmd_raw; vfh_plus filtra → /cmd_vel.
    # Durante ALIGNING/ADVANCING ambos nodos de misión publican
    # /align/active=True y vfh_plus cede el control directo.
    go_to_goal_node = Node(
        package='iolair',
        executable='go_to_goal',
        name='puzzlebot_go_to_goal',
        output='screen',
        remappings=[('/cmd_vel', '/cmd_raw')],
    )

    vfh_plus_node = Node(
        package='iolair',
        executable='vfh_plus',
        name='vfh_plus',
        output='screen',
        parameters=[{
            'robot_radius_m':     0.18,
            'safety_margin_m':    0.10,
            'warn_dist':          0.65,
            'emergency_dist':     0.35,
            'stop_dist':          0.14,
            'num_sectors':        180,
            'hist_threshold':     8.0,
            'smoothing_window':   5,
            'influence_radius_m': 1.20,
            'max_v':              0.22,
            'max_w':              1.20,
            'kp_heading':         2.00,
            'lidar_yaw_offset':   math.pi,
        }],
        remappings=[
            ('/cmd_raw', '/cmd_raw'),
            ('/cmd_vel', '/cmd_vel'),
        ],
    )

    controller_node = Node(
        package='iolair',
        executable='controller',
        name='puzzlebot_controller',
        output='screen',
    )

    rviz_goal_bridge_node = Node(
        package='Navigation',
        executable='rviz_goal_bridge',
        name='rviz_goal_bridge',
        output='screen',
    )

    # ══════════════════════════════════════════════════════════════════════
    # MISIÓN — QR ALIGN + TRUCK ALIGN
    # ══════════════════════════════════════════════════════════════════════

    # qr_align_node — recolección de pallet vía ArUco/QR
    # Trigger:  ros2 topic pub --once /collect/trigger std_msgs/String "data: rack"
    # Done:     /collect/done "SUCCESS" | "ABORT"
    # Payload:  /collect/qr_payload  (nombre empresa del QR)
    qr_align_node = Node(
        package='Navigation',
        executable='qr_align_node',
        name='qr_align_node',
        output='screen',
        parameters=[{
            # Geometría de alineación
            'align_stop_dist':     0.35,   # m — centro robot → QR al alinearse
            'forklift_reach_m':    0.20,   # m — stop de APPROACH_FINAL
            'align_lateral_tol':   0.03,   # m — tolerancia lateral
            'angle_tol_deg':       4.0,    # ° — tolerancia angular
            'goal_replan_dist':    0.06,   # m — mín. desplazamiento para replanificar
            # Offset de cámara respecto a base_link
            'cam_fwd_m':           0.15,
            'cam_left_m':          0.07,
            'cam_offset_deg':      0.0,
            # Lift
            'lift_timeout':        8.0,
            # Timeouts
            'align_timeout':       20.0,
            'approach_timeout':    15.0,
            'search_timeout':      10.0,
            'qr_timeout':          2.5,
            # Retroceso post-recolección
            'back_away_speed':     0.10,
            'back_away_time':      1.8,
            # Barrido RECOVER_SCAN
            'scan_range_deg':      30.0,
            'scan_speed_dps':      20.0,
            'scan_max_attempts':   3,
            # General
            'fsm_rate_hz':         20.0,
        }],
    )

    # truck_align_node — entrega de pallet vía YOLO
    # Trigger:  ros2 topic pub --once /truck_align/cmd std_msgs/String "data: wolmar"
    #           (acepta: wolmar | emezon | popsi | nalmart | nemezon | nepsi)
    # Done:     /truck_align/done "SUCCESS" | "ABORT"
    #
    # Secuencia interna:
    #   GOTO_DETECTION_WP → SEARCH_TRUCK → ALIGNING → APPROACH_FINAL
    #   → ADVANCING (inserta horquillas) → LOWERING → BACK_AWAY → DONE
    truck_align_node = Node(
        package='Navigation',
        executable='truck_align_node',
        name='truck_align_node',
        output='screen',
        parameters=[{
            # Waypoint de detección (YAML con entrada "truck_detection")
            'waypoints_file':     waypoints_yaml,
            'detection_wp_name':  'truck_detection',
            # Geometría de entrega
            'lateral_offset_m':   0.24,   # m — offset derecho al logo
            'stop_dist_m':        0.30,   # m — distancia alineación al camión
            'final_dist_m':       0.20,   # m — umbral llegada APPROACH_FINAL
            # Geometría de inserción de horquillas
            # advance_dist = stop_dist - forklift_len + insert_depth
            #              = 0.30 - 0.15 + 0.07 = 0.22 m
            'forklift_len_m':     0.15,   # m — largo físico de la horquilla
            'insert_depth_m':     0.07,   # m — cuánto entra la punta al camión
            'advance_speed':      0.08,   # m/s — velocidad lenta de inserción
            'advance_timeout_s':  8.0,    # s — timeout de seguridad ADVANCING
            # Lift
            'lift_timeout_s':     8.0,
            # Timeouts
            'goto_timeout_s':     30.0,
            'align_timeout_s':    20.0,
            'approach_timeout_s': 15.0,
            'search_timeout_s':   12.0,
            # Barrido de búsqueda YOLO
            'scan_range_deg':     40.0,
            'scan_speed_dps':     20.0,
            'scan_max_attempts':  3,
            # Retroceso post-entrega
            'back_away_speed':    0.10,
            'back_away_time':     2.0,
            # Cámara
            'logo_timeout_s':     1.5,
            'cam_fov_h_deg':      62.0,
            # General
            'fsm_rate_hz':        20.0,
            'goal_replan_dist':   0.06,
        }],
    )

    # ══════════════════════════════════════════════════════════════════════
    # LAUNCH DESCRIPTION — orden de arranque
    # ══════════════════════════════════════════════════════════════════════
    #
    # Orden recomendado:
    #   1. Argumentos y mapa  — map_server debe estar activo antes de A*
    #   2. Percepción         — cámara y sensores
    #   3. Localización       — odometría antes de aruco_localizer
    #   4. Planificación      — A* puede suscribirse a /map ya activo
    #   5. Actuación          — controller y VFH+
    #   6. Misión             — qr_align y truck_align al final

    return LaunchDescription([
        # ── Argumentos ────────────────────────────────────────────────────
        map_yaml_arg,

        # ── Mapa ──────────────────────────────────────────────────────────
        map_server_node,
        lifecycle_manager_node,

        # ── Percepción ────────────────────────────────────────────────────
        aruco_detector_node,
        yolo_detector_node,

        # ── Localización ──────────────────────────────────────────────────
        odometry_node,
        aruco_localizer_node,
        aruco_map_node,
        mcl_node,
        slam_node,

        # ── Planificación y navegación ────────────────────────────────────
        astar_planner_node,
        go_to_goal_node,
        vfh_plus_node,
        controller_node,
        rviz_goal_bridge_node,

        # ── Misión ────────────────────────────────────────────────────────
        qr_align_node,
        truck_align_node,
    ])
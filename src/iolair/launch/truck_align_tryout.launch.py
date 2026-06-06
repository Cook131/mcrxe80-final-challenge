"""
truck_align_tryout.launch.py
============================
Tryout para TruckAlignNode:
  - El robot está enfrente de los tres camiones
  - Se le dice a cuál alinearse vía arg 'target'
  - El nodo hace SCAN → ALIGN → NAV_APPROACH → ALIGNED

Nodos levantados:
  yolo_detector_node   — detecta logos nalmart / nemezon / nepsi
  truck_align_node     — FSM de alineación visual

Uso:
  ros2 launch puzzlebot truck_align_tryout.launch.py target:=wolmar
  ros2 launch puzzlebot truck_align_tryout.launch.py target:=emezon
  ros2 launch puzzlebot truck_align_tryout.launch.py target:=popsi

Targets válidos (mapean a clases YOLO del modelo):
  wolmar  →  nalmart
  emezon  →  nemezon
  popsi   →  nepsi

Monitorea resultado:
  ros2 topic echo /truck_align/result
  ros2 topic echo /truck_align/status
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# Mapeo arg CLI → clase YOLO interna
_TARGET_MAP = {
    'wolmar': 'nalmart',
    'emezon': 'nemezon',
    'popsi' : 'nepsi',
}


def generate_launch_description():

    target_arg = DeclareLaunchArgument(
        'target',
        default_value='wolmar',
        description="Camión objetivo: 'wolmar', 'emezon' o 'popsi'",
    )

    # ── YoloDetectorNode ──────────────────────────────────────────────────────
    yolo_detector = Node(
        package    = 'puzzlebot',
        executable = 'yolo_detector_node',
        name       = 'yolo_detector_node',
        output     = 'screen',
        # Los parámetros WEIGHTS / CAMERA_TOPIC / CONF / DEVICE / IMGSZ están
        # hardcodeados como constantes al tope de yolo_vision.py.
        # Si los moviste a declare_parameter puedes sobreescribir aquí.
        parameters = [{
            'confidence': 0.65,
        }],
        remappings = [
            ('/camera_raw/compressed', '/camera/image_raw/compressed'),
        ],
    )

    # ── TruckAlignNode ────────────────────────────────────────────────────────
    # El cmd de alineación se publica automáticamente 2 s después del arranque
    # por el one-shot node de abajo. El target_class viene del arg CLI mapeado.
    truck_align = Node(
        package    = 'puzzlebot',
        executable = 'truck_align_node',
        name       = 'truck_align_node',
        output     = 'screen',
        parameters = [{
            # Geometría de cámara — ajusta fx según tu calibración
            'frame_width_px'         : 320,
            'logo_real_width_m'      : 0.35,
            'focal_length_px'        : 186.0,
            'approach_stop_dist'     : 0.40,
            # Control P angular
            'yolo_kp'                : 0.005,
            'max_w'                  : 0.35,
            # Umbrales de alineación
            'err_ok_px'              : 22,
            'confirm_ticks'          : 3,
            'min_conf'               : 0.60,
            # Búsqueda
            'scan_speed'             : 0.22,
            'scan_timeout_s'         : 18.0,
            # Timeouts de las fases siguientes
            'align_timeout_s'        : 10.0,
            'nav_approach_timeout_s' : 30.0,
            'fsm_rate_hz'            : 20.0,
        }],
        remappings = [
            ('/odom',    '/odom'),
            ('/cmd_vel', '/cmd_vel'),
        ],
    )

    # ── One-shot publisher — dispara el cmd de alineación ─────────────────────
    # Publica  "align:<yolo_class>"  en /truck_align/cmd
    # El mapeo wolmar→nalmart / emezon→nemezon / popsi→nepsi se hace aquí.
    #
    # Como LaunchConfiguration no puede hacer dict-lookup en tiempo de launch,
    # lanzamos un nodo Python inline que lee el arg y publica el string correcto.
    trigger_node = Node(
        package    = 'puzzlebot',
        executable = 'truck_align_trigger',   # script mínimo — ver nota al pie
        name       = 'truck_align_trigger',
        output     = 'screen',
        parameters = [{
            'target'    : LaunchConfiguration('target'),
            'delay_s'   : 2.0,   # espera antes de publicar (nodos listos)
            # Mapeo amigable → clase YOLO
            'wolmar_class': 'nalmart',
            'emezon_class': 'nemezon',
            'popsi_class' : 'nepsi',
        }],
    )

    log_ready = LogInfo(
        msg = [
            '\n',
            '╔══════════════════════════════════════════════════════╗\n',
            '║      TRUCK ALIGNMENT TRYOUT — Iolair Puzzlebot       ║\n',
            '╠══════════════════════════════════════════════════════╣\n',
            '║  Target     : ', LaunchConfiguration('target'), '\n',
            '║  (YOLO map) :  wolmar→nalmart  emezon→nemezon        ║\n',
            '║               popsi→nepsi                            ║\n',
            '║                                                      ║\n',
            '║  O dispara manualmente:                              ║\n',
            '║    ros2 topic pub --once /truck_align/cmd \\          ║\n',
            '║      std_msgs/String "data: align:nalmart"           ║\n',
            '║                                                      ║\n',
            '║  Monitorea resultado:                                 ║\n',
            '║    ros2 topic echo /truck_align/result               ║\n',
            '║    ros2 topic echo /truck_align/status               ║\n',
            '╚══════════════════════════════════════════════════════╝\n',
        ]
    )

    return LaunchDescription([
        target_arg,
        log_ready,
        yolo_detector,
        truck_align,
        # trigger_node,   # <-- descomenta si tienes truck_align_trigger.py
        # Si no, usa el one-liner de arriba en la terminal
    ])


# ══════════════════════════════════════════════════════════════════════════════
# NOTA: truck_align_trigger.py  (mini-nodo helper, pégalo en puzzlebot/scripts)
# ══════════════════════════════════════════════════════════════════════════════
#
# #!/usr/bin/env python3
# import time, rclpy
# from rclpy.node import Node
# from std_msgs.msg import String
#
# _MAP = {'wolmar':'nalmart', 'emezon':'nemezon', 'popsi':'nepsi'}
#
# class TriggerNode(Node):
#     def __init__(self):
#         super().__init__('truck_align_trigger')
#         self.declare_parameter('target', 'wolmar')
#         self.declare_parameter('delay_s', 2.0)
#         for k in ('wolmar_class','emezon_class','popsi_class'):
#             self.declare_parameter(k, '')
#         self._pub = self.create_publisher(String, '/truck_align/cmd', 10)
#         self.create_timer(0.1, self._tick)
#         self._sent = False
#         self._t0   = time.monotonic()
#
#     def _tick(self):
#         if self._sent:
#             return
#         delay = self.get_parameter('delay_s').value
#         if time.monotonic() - self._t0 < delay:
#             return
#         target = self.get_parameter('target').value.lower()
#         yolo_class = _MAP.get(target, target)
#         msg = String(); msg.data = f'align:{yolo_class}'
#         self._pub.publish(msg)
#         self.get_logger().info(f'Trigger enviado: {msg.data}')
#         self._sent = True
#
# def main(args=None):
#     rclpy.init(args=args)
#     n = TriggerNode()
#     rclpy.spin_once(n, timeout_sec=5.0)
#     n.destroy_node(); rclpy.shutdown()
#
# if __name__ == '__main__': main()

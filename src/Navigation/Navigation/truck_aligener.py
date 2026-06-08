#!/usr/bin/env python3
"""
truck_align_node.py — Iolair Truck Align + Pallet Delivery  [v1]
=================================================================
Nodo de entrega de pallet. Se activa después de que qr_align_node
completa la recolección y el robot lleva el pallet levantado.

Secuencia completa
------------------
  1. IDLE
       Espera trigger en /truck_align/cmd (String con nombre del camión).
       El payload del QR recibido en /collect/qr_payload determina qué
       clase de logo YOLO buscar (ver _PAYLOAD_TO_CLASS).

  2. GOTO_DETECTION_WP
       Publica el waypoint con ID configurado (por defecto 15) del YAML de
       waypoints hacia /astar/goal y espera confirmación de llegada
       vía /astar/status == "GOAL_REACHED".

  3. SEARCH_TRUCK
       Rota in-place (barrido ±scan_range_deg) mientras YOLO no detecta
       la clase objetivo. Si no encuentra el logo tras scan_max_attempts
       completos → ABORT.

  4. ALIGNING
       Con el logo visible, calcula el bearing al centro del logo
       desde la posición del robot en el frame mundo y publica un goal
       lateral a /astar/goal. El goal se posiciona a:
         - stop_dist metros en profundidad frente al logo
         - lateral_offset_m (24 cm) a la DERECHA del logo
       "Derecha" = vector perpendicular derecho al bearing robot→logo.

  5. APPROACH_FINAL
       Avanza con /goal (GoToGoal directo, bypassa A*) hasta que la
       cámara reporta que el logo está dentro de final_dist_m.
       _stop() explícito al llegar.

  6. ADVANCING
       El robot avanza en línea recta a velocidad lenta (advance_speed)
       durante advance_dist_m metros (cronometrado), con el pallet
       todavía levantado, para meter las horquillas dentro del camión.
       Cálculo geométrico:
         advance_dist = stop_dist_m - forklift_len_m + insert_depth_m
                      = 0.30 - 0.15 + 0.07 = 0.22 m  (default)
       Al terminar el avance el robot se detiene.

  7. LOWERING
       Publica "down" en /lift_auto y espera /lift_done == "DOWN"
       con timeout lift_timeout_s.

  8. BACK_AWAY
       Retrocede back_away_time segundos a back_away_speed m/s.

  9. DONE
       Publica /truck_align/done "SUCCESS", reactiva VFH+,
       vuelve a IDLE.

  ABORT en cualquier estado:
       Publica /truck_align/done "ABORT", reactiva VFH+,
       vuelve a IDLE. Si el lift estaba en HOLD, intenta bajar antes.

Parámetros configurables (ros2 param)
--------------------------------------
  waypoints_file     str    ruta al YAML con waypoints
                            (ej. ~/iolair_ws/src/iolair/config/waypoints.yaml)
  detection_wp_id    int    ID del waypoint en el YAML (por defecto 15)
  lateral_offset_m   float  offset lateral a la derecha del logo (0.24 m)
  stop_dist_m        float  distancia al logo al detenerse (0.30 m)
  final_dist_m       float  umbral de llegada en APPROACH_FINAL (0.20 m)
  forklift_len_m     float  largo físico de la horquilla, frente robot → punta (0.15 m)
  insert_depth_m     float  cuánto debe penetrar la punta en el camión (0.07 m)
  advance_speed      float  velocidad de avance de inserción (0.08 m/s)
  advance_timeout_s  float  timeout de seguridad para ADVANCING (8.0 s)
  align_timeout_s    float  timeout de ALIGNING (20.0 s)
  approach_timeout_s float  timeout de APPROACH_FINAL (15.0 s)
  lift_timeout_s     float  timeout de LOWERING (8.0 s)
  goto_timeout_s     float  timeout de GOTO_DETECTION_WP (30.0 s)
  search_timeout_s   float  timeout de un barrido completo sin logo (12.0 s)
  scan_range_deg     float  semiángulo del barrido de búsqueda (40.0 °)
  scan_speed_dps     float  velocidad angular del barrido (20.0 °/s)
  scan_max_attempts  int    intentos antes de ABORT (3)
  back_away_speed    float  velocidad de retroceso (0.10 m/s)
  back_away_time     float  duración del retroceso (2.0 s)
  fsm_rate_hz        float  frecuencia del tick de la FSM (20.0 Hz)
  goal_replan_dist   float  mínimo desplazamiento del goal para replanificar (0.06 m)
  logo_timeout_s     float  tiempo sin detección antes de considerar logo perdido (1.5 s)
  cam_fov_h_deg      float  FOV horizontal de la cámara (62.0 °) — para calcular ángulo
                            del logo desde píxeles del bbox

Topics
------
  SUB  /truck_align/cmd       std_msgs/String  — nombre del camión objetivo
                                                 (mapeo a clase YOLO vía _PAYLOAD_TO_CLASS)
  SUB  /collect/qr_payload    std_msgs/String  — payload del QR (ya procesado por qr_align)
  SUB  /yolo/detecciones      std_msgs/String  — JSON de detecciones YOLO
  SUB  /odom                  nav_msgs/Odometry
  SUB  /astar/status          std_msgs/String
  SUB  /lift_done             std_msgs/String
  PUB  /astar/goal            geometry_msgs/Pose2D
  PUB  /goal                  geometry_msgs/Pose2D
  PUB  /cmd_vel               geometry_msgs/Twist
  PUB  /lift_auto             std_msgs/String
  PUB  /align/active          std_msgs/Bool
  PUB  /truck_align/done      std_msgs/String  — "SUCCESS" | "ABORT"

Mapeo payload QR → clase YOLO
-------------------------------
  "wolmar"  / "nalmart"  → "nalmart"
  "emezon"  / "nemezon"  → "nemezon"
  "popsi"   / "nepsi"    → "nepsi"

  El mapeo acepta tanto el nombre amigable del camión (del QR) como
  la clase interna de YOLO directamente, para flexibilidad.
"""

import json
import math
import os
import time

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos  import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Pose2D, Twist
from nav_msgs.msg      import Odometry
from std_msgs.msg      import Bool, String


# ══════════════════════════════════════════════════════════════════════════════
# Mapeo nombre-camión → clase YOLO
# Acepta nombre amigable (del QR) O nombre interno de YOLO
# ══════════════════════════════════════════════════════════════════════════════
_PAYLOAD_TO_CLASS = {
    # nombre amigable → clase YOLO
    'wolmar':  'nalmart',
    'emezon':  'nemezon',
    'popsi':   'nepsi',
    # clase YOLO directa (pass-through, por si el QR ya tiene el nombre interno)
    'nalmart': 'nalmart',
    'nemezon': 'nemezon',
    'nepsi':   'nepsi',
}


# ══════════════════════════════════════════════════════════════════════════════
# Estados FSM
# ══════════════════════════════════════════════════════════════════════════════
class _S:
    IDLE               = 'IDLE'
    GOTO_DETECTION_WP  = 'GOTO_DETECTION_WP'
    SEARCH_TRUCK       = 'SEARCH_TRUCK'
    ALIGNING           = 'ALIGNING'
    APPROACH_FINAL     = 'APPROACH_FINAL'
    ADVANCING          = 'ADVANCING'
    LOWERING           = 'LOWERING'
    BACK_AWAY          = 'BACK_AWAY'
    DONE               = 'DONE'
    ABORT              = 'ABORT'


# ══════════════════════════════════════════════════════════════════════════════
class TruckAlignNode(Node):

    def __init__(self):
        super().__init__('truck_align_node')

        # ── Parámetros ────────────────────────────────────────────────────
        self.declare_parameter('waypoints_file',     '')
        self.declare_parameter('detection_wp_id',    15)     # <--- ID modificado
        self.declare_parameter('lateral_offset_m',   0.24)
        self.declare_parameter('stop_dist_m',        0.30)
        self.declare_parameter('final_dist_m',       0.20)
        self.declare_parameter('forklift_len_m',     0.15)   # largo físico de la horquilla
        self.declare_parameter('insert_depth_m',     0.07)   # cuánto entra la punta al camión
        self.declare_parameter('advance_speed',      0.08)   # m/s — lento para no perder el pallet
        self.declare_parameter('advance_timeout_s',  8.0)    # timeout de seguridad
        self.declare_parameter('align_timeout_s',    20.0)
        self.declare_parameter('approach_timeout_s', 15.0)
        self.declare_parameter('lift_timeout_s',     8.0)
        self.declare_parameter('goto_timeout_s',     30.0)
        self.declare_parameter('search_timeout_s',   12.0)
        self.declare_parameter('scan_range_deg',     40.0)
        self.declare_parameter('scan_speed_dps',     20.0)
        self.declare_parameter('scan_max_attempts',  3)
        self.declare_parameter('back_away_speed',    0.10)
        self.declare_parameter('back_away_time',     2.0)
        self.declare_parameter('fsm_rate_hz',        20.0)
        self.declare_parameter('goal_replan_dist',   0.06)
        self.declare_parameter('logo_timeout_s',     1.5)
        self.declare_parameter('cam_fov_h_deg',      62.0)

        self._p = lambda n: self.get_parameter(n).value

        # ── Estado FSM ────────────────────────────────────────────────────
        self._state       = _S.IDLE
        self._state_entry = time.monotonic()

        # ── Target ────────────────────────────────────────────────────────
        self._target_class   = ''   # clase YOLO a buscar
        self._lift_was_held  = False  # saber si hay pallet arriba al hacer ABORT

        # ── Detección YOLO ────────────────────────────────────────────────
        # Ángulo horizontal del logo (grados, relativo al heading del robot)
        # positivo = logo a la derecha
        self._logo_angle  = 0.0
        self._logo_stamp  = 0.0     # time.monotonic() del último frame con logo
        self._logo_seen   = False   # hay detección válida en la ventana de tiempo

        # ── Pose odométrica ───────────────────────────────────────────────
        self._rx  = 0.0
        self._ry  = 0.0
        self._rth = 0.0

        # ── A* / GoToGoal ─────────────────────────────────────────────────
        self._astar_status = ''
        self._last_goal_x  = None
        self._last_goal_y  = None

        # ── Lift ──────────────────────────────────────────────────────────
        self._lift_done_label = ''

        # ── ADVANCING ─────────────────────────────────────────────────────
        self._advance_dist_m     = 0.0    # metros a recorrer calculados en APPROACH_FINAL
        self._advance_start_odom = (0.0, 0.0)  # pose de inicio del avance

        # ── SEARCH_TRUCK ──────────────────────────────────────────────────
        self._scan_phase           = 'LEFT'
        self._scan_phase_start_yaw = 0.0
        self._scan_attempts        = 0

        # ── QoS ───────────────────────────────────────────────────────────
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Suscriptores ──────────────────────────────────────────────────
        self.create_subscription(String,   '/truck_align/cmd',    self._cb_cmd,          10)
        self.create_subscription(String,   '/collect/qr_payload', self._cb_qr_payload,   10)
        self.create_subscription(String,   '/yolo/detecciones',   self._cb_yolo,         qos_be)
        self.create_subscription(Odometry, '/odom',               self._cb_odom,         10)
        self.create_subscription(String,   '/astar/status',       self._cb_astar_status, 10)
        self.create_subscription(String,   '/lift_done',          self._cb_lift_done,    10)

        # ── Publicadores ──────────────────────────────────────────────────
        self._pub_astar  = self.create_publisher(Pose2D, '/astar/goal',        10)
        self._pub_wp     = self.create_publisher(Pose2D, '/goal',              10)
        self._pub_cmd    = self.create_publisher(Twist,  '/cmd_vel',           10)
        self._pub_lift   = self.create_publisher(String, '/lift_auto',         10)
        self._pub_active = self.create_publisher(Bool,   '/align/active',      10)
        self._pub_done   = self.create_publisher(String, '/truck_align/done',  10)

        # ── Timer FSM ─────────────────────────────────────────────────────
        self.create_timer(1.0 / float(self._p('fsm_rate_hz')), self._tick)

        # ── Cargar waypoints ──────────────────────────────────────────────
        self._detection_wp = self._load_detection_wp()

        self.get_logger().info(
            'truck_align_node v2 listo\n'
            f'  detection_wp (ID {self._p("detection_wp_id")})=({self._detection_wp[0]:.3f}, {self._detection_wp[1]:.3f})\n'
            f'  lateral_offset={self._p("lateral_offset_m")}m  '
            f'stop_dist={self._p("stop_dist_m")}m  '
            f'final_dist={self._p("final_dist_m")}m\n'
            f'  forklift_len={self._p("forklift_len_m")}m  '
            f'insert_depth={self._p("insert_depth_m")}m  '
            f'→ advance={self._p("stop_dist_m")-self._p("forklift_len_m")+self._p("insert_depth_m"):.3f}m '
            f'@ {self._p("advance_speed")}m/s\n'
            f'  Mapeo QR→YOLO: wolmar→nalmart | emezon→nemezon | popsi→nepsi'
        )

    # ══════════════════════════════════════════════════════════════════════
    # CARGA DE WAYPOINTS
    # ══════════════════════════════════════════════════════════════════════

    def _load_detection_wp(self):
        """
        Carga la posición (x, y) del waypoint utilizando el ID desde
        el YAML de waypoints. Si no se encuentra, devuelve (0.0, 0.0)
        con una advertencia y el operador puede corregir con ros2 param.

        Formato YAML soportado (dos variantes):
          # Variante A — lista de dicts con campos 'id', 'x', 'y'
          waypoints:
            - id: 15
              x: 1.234
              y: 5.678

          # Variante B — dict nombrado (id como key)
          waypoints:
            15:
              x: 1.234
              y: 5.678
        """
        wp_file = self._p('waypoints_file')
        wp_id   = self._p('detection_wp_id')

        if not wp_file:
            self.get_logger().warn(
                f'waypoints_file no configurado. '
                f'Usando (0.0, 0.0) para ID {wp_id}. '
                f'Pasa el parámetro: '
                f'--ros-args -p waypoints_file:=/ruta/waypoints.yaml')
            return (0.0, 0.0)

        expanded = os.path.expanduser(wp_file)
        if not os.path.exists(expanded):
            self.get_logger().error(
                f'Archivo de waypoints no encontrado: {expanded}')
            return (0.0, 0.0)

        try:
            with open(expanded, 'r') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            self.get_logger().error(f'Error leyendo YAML: {e}')
            return (0.0, 0.0)

        wps = data.get('waypoints', {})

        # Variante A: lista
        if isinstance(wps, list):
            for wp in wps:
                if isinstance(wp, dict) and wp.get('id') == wp_id:
                    x = float(wp.get('x', 0.0))
                    y = float(wp.get('y', 0.0))
                    self.get_logger().info(
                        f'[WP] ID {wp_id} cargado: ({x:.3f}, {y:.3f})')
                    return (x, y)
            self.get_logger().error(
                f'Waypoint ID {wp_id} no encontrado en lista. '
                f'IDs disponibles: '
                f'{[w.get("id","?") for w in wps if isinstance(w, dict)]}')
            return (0.0, 0.0)

        # Variante B: dict
        if isinstance(wps, dict):
            # Probar el key tanto como entero (15) o como string ("15")
            entry = wps.get(wp_id) or wps.get(str(wp_id))
            if entry:
                x = float(entry.get('x', 0.0))
                y = float(entry.get('y', 0.0))
                self.get_logger().info(
                    f'[WP] ID {wp_id} cargado: ({x:.3f}, {y:.3f})')
                return (x, y)
            self.get_logger().error(
                f'Waypoint ID {wp_id} no encontrado en dict. '
                f'Claves: {list(wps.keys())}')
            return (0.0, 0.0)

        self.get_logger().error(f'Formato YAML de waypoints no reconocido.')
        return (0.0, 0.0)

    # ══════════════════════════════════════════════════════════════════════
    # CALLBACKS
    # ══════════════════════════════════════════════════════════════════════

    def _cb_cmd(self, msg: String):
        """
        Trigger principal. Acepta el nombre del camión objetivo.
        El nombre puede ser el nombre amigable (wolmar, emezon, popsi)
        o directamente la clase YOLO (nalmart, nemezon, nepsi).
        También acepta 'abort' para cancelar en cualquier momento.
        """
        cmd = msg.data.strip().lower()

        if cmd == 'abort':
            if self._state != _S.IDLE:
                self.get_logger().warn('[TruckAlign] ABORT recibido')
                self._transition(_S.ABORT)
            return

        if self._state != _S.IDLE:
            self.get_logger().warn(
                f'[TruckAlign] Trigger ignorado — estado: {self._state}')
            return

        yolo_class = _PAYLOAD_TO_CLASS.get(cmd)
        if yolo_class is None:
            self.get_logger().error(
                f'[TruckAlign] Nombre desconocido: "{cmd}". '
                f'Válidos: {list(_PAYLOAD_TO_CLASS.keys())}')
            return

        self._target_class  = yolo_class
        self._lift_was_held = True   # asumimos que llegamos con pallet arriba
        self.get_logger().info(
            f'[TruckAlign] Trigger: "{cmd}" → clase YOLO: "{yolo_class}"')
        self._set_vfh_bypass(True)
        self._transition(_S.GOTO_DETECTION_WP)

    def _cb_qr_payload(self, msg: String):
        """
        Escucha el payload del QR publicado por qr_align_node.
        Si llega durante IDLE y es un nombre conocido, pre-carga
        la clase objetivo para que cuando llegue el trigger ya esté listo.
        Este callback NO activa la FSM por sí solo.
        """
        if self._state != _S.IDLE:
            return
        payload = msg.data.strip().lower()
        yolo_class = _PAYLOAD_TO_CLASS.get(payload)
        if yolo_class and yolo_class != self._target_class:
            self.get_logger().info(
                f'[TruckAlign] QR payload pre-cargado: '
                f'"{payload}" → clase: "{yolo_class}"')
            self._target_class = yolo_class

    def _cb_yolo(self, msg: String):
        """
        Procesa detecciones YOLO (JSON). Extrae el ángulo horizontal
        del logo objetivo (si está presente) respecto al centro de imagen.
        El ángulo se usa para estimar la dirección mundo al logo.

        Ángulo positivo = logo a la DERECHA del centro de imagen.
        """
        if self._state not in (_S.SEARCH_TRUCK, _S.ALIGNING, _S.APPROACH_FINAL, _S.ADVANCING):
            return
        if not self._target_class:
            return

        try:
            detections = json.loads(msg.data)
        except Exception:
            return

        # Filtrar detecciones de la clase objetivo; quedarse con la de mayor confianza
        candidates = [
            d for d in detections
            if d.get('class') == self._target_class
        ]
        if not candidates:
            return

        best = max(candidates, key=lambda d: d.get('conf', 0.0))
        x1, y1, x2, y2 = best['bbox']

        # Ancho de la imagen: estimado desde el bbox más externo.
        # Para el ángulo horizontal necesitamos el ancho total de imagen.
        # Lo inferimos asumiendo que las detecciones anteriores cubrieron
        # el rango completo, o usamos cam_fov_h_deg con anchura fija.
        # La forma más robusta: calcular ángulo desde el offset normalizado
        # del centro del bbox respecto al centro de imagen.
        #
        # Nota: yolo_detector_node no publica el tamaño de imagen junto con
        # las detecciones. Asumimos IMGSZ=320 como referencia (configurable
        # vía cam_fov_h_deg sin necesitar el tamaño exacto).
        # El ángulo en rad: tan(α) = (px_offset / px_half_width) * tan(fov/2)
        # Con IMGSZ=320: px_half_width = 160

        # Centro del bbox en píxeles
        box_cx = (x1 + x2) / 2.0

        # Offset normalizado respecto al centro de imagen (asumimos IMGSZ=320)
        # Si IMGSZ cambia, ajustar img_half_width o usar parámetro
        img_half_width = 160.0   # IMGSZ=320 / 2

        fov_half_rad = math.radians(self._p('cam_fov_h_deg') / 2.0)
        angle_rad = math.atan(
            (box_cx - img_half_width) / img_half_width * math.tan(fov_half_rad)
        )
        self._logo_angle = math.degrees(angle_rad)
        self._logo_stamp = time.monotonic()
        self._logo_seen  = True

    def _cb_odom(self, msg: Odometry):
        self._rx  = msg.pose.pose.position.x
        self._ry  = msg.pose.pose.position.y
        q         = msg.pose.pose.orientation
        self._rth = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    def _cb_astar_status(self, msg: String):
        prev = self._astar_status
        self._astar_status = msg.data.strip()
        if self._astar_status != prev:
            self.get_logger().debug(f'[A*] {self._astar_status}')

    def _cb_lift_done(self, msg: String):
        # Solo procesar durante LOWERING para evitar contaminación
        if self._state != _S.LOWERING:
            return
        label = msg.data.strip()
        if label:
            self.get_logger().info(f'[Lift] /lift_done: {label}')
            self._lift_done_label = label

    # ══════════════════════════════════════════════════════════════════════
    # FSM TICK
    # ══════════════════════════════════════════════════════════════════════

    def _tick(self):
        s = self._state

        if s == _S.IDLE:
            return

        # ── GOTO_DETECTION_WP ─────────────────────────────────────────────
        elif s == _S.GOTO_DETECTION_WP:
            # Publicar goal de A* al waypoint truck_detection en la primera entrada
            if self._time_in_state() < 0.1:
                self._publish_detection_wp_goal()
                return

            if self._astar_status == 'GOAL_REACHED':
                self.get_logger().info(
                    f'[GOTO_DETECTION_WP] Llegada a waypoint ID {self._p("detection_wp_id")} → SEARCH_TRUCK')
                self._scan_phase           = 'LEFT'
                self._scan_phase_start_yaw = self._rth
                self._scan_attempts        = 0
                self._transition(_S.SEARCH_TRUCK)
                return

            if self._time_in_state() > self._p('goto_timeout_s'):
                self.get_logger().warn('[GOTO_DETECTION_WP] Timeout → ABORT')
                self._transition(_S.ABORT)

        # ── SEARCH_TRUCK ──────────────────────────────────────────────────
        elif s == _S.SEARCH_TRUCK:
            if self._logo_fresh():
                self._stop()
                self.get_logger().info(
                    f'[SEARCH_TRUCK] Logo "{self._target_class}" encontrado '
                    f'angle={self._logo_angle:+.1f}° → ALIGNING')
                self._transition(_S.ALIGNING)
                return

            self._tick_scan()

        # ── ALIGNING ──────────────────────────────────────────────────────
        elif s == _S.ALIGNING:
            if not self._logo_fresh():
                self.get_logger().warn('[ALIGNING] Logo perdido → SEARCH_TRUCK')
                self._scan_phase           = 'LEFT'
                self._scan_phase_start_yaw = self._rth
                self._scan_attempts        = 0
                self._transition(_S.SEARCH_TRUCK)
                return

            if self._time_in_state() > self._p('align_timeout_s'):
                self.get_logger().warn('[ALIGNING] Timeout → ABORT')
                self._transition(_S.ABORT)
                return

            goal = self._compute_delivery_goal(self._p('stop_dist_m'))

            # Verificar si ya estamos cerca: distancia del robot al goal
            dist_to_goal = math.hypot(goal.x - self._rx, goal.y - self._ry)

            if dist_to_goal <= self._p('final_dist_m'):
                self.get_logger().info(
                    f'[ALIGNING] En posición dist_to_goal={dist_to_goal:.3f}m '
                    f'→ APPROACH_FINAL')
                self._publish_final_goal()
                self._transition(_S.APPROACH_FINAL)
                return

            self._publish_align_goal_if_needed(goal)

        # ── APPROACH_FINAL ────────────────────────────────────────────────
        elif s == _S.APPROACH_FINAL:
            if self._logo_fresh():
                goal   = self._compute_delivery_goal(self._p('stop_dist_m'))
                dist   = math.hypot(goal.x - self._rx, goal.y - self._ry)

                if dist <= self._p('final_dist_m'):
                    # Calcular distancia de avance antes de detener el GoToGoal
                    adv = (self._p('stop_dist_m')
                           - self._p('forklift_len_m')
                           + self._p('insert_depth_m'))
                    self.get_logger().info(
                        f'[APPROACH_FINAL] Alineado dist={dist:.3f}m → '
                        f'ADVANCING {adv:.3f}m a {self._p("advance_speed")}m/s')
                    self._stop()
                    self._advance_dist_m    = adv
                    self._advance_start_odom = (self._rx, self._ry)
                    self._transition(_S.ADVANCING)
                    return

            if self._time_in_state() > self._p('approach_timeout_s'):
                self.get_logger().warn('[APPROACH_FINAL] Timeout → ABORT')
                self._transition(_S.ABORT)

        # ── ADVANCING ─────────────────────────────────────────────────────
        elif s == _S.ADVANCING:
            # Avance en línea recta con el pallet levantado para insertar
            # las horquillas dentro del camión.
            # Usamos odometría integrada (distancia euclidiana desde el
            # punto de inicio) para medir el recorrido, no solo tiempo,
            # lo que es más robusto a variaciones de velocidad.
            traveled = math.hypot(
                self._rx - self._advance_start_odom[0],
                self._ry - self._advance_start_odom[1],
            )

            if traveled < self._advance_dist_m:
                cmd = Twist()
                cmd.linear.x = float(self._p('advance_speed'))
                self._pub_cmd.publish(cmd)
            else:
                self._stop()
                self.get_logger().info(
                    f'[ADVANCING] {traveled:.3f}m recorridos ✔ → LOWERING')
                self._transition(_S.LOWERING)
                return

            if self._time_in_state() > self._p('advance_timeout_s'):
                self.get_logger().error('[ADVANCING] Timeout → ABORT')
                self._transition(_S.ABORT)

        # ── LOWERING ──────────────────────────────────────────────────────
        elif s == _S.LOWERING:
            # Primera entrada: inicializar y enviar comando
            if self._time_in_state() < 0.05:
                self._lift_done_label = ''
                self.get_logger().info('[LOWERING] Enviando "down" al lift')
                self._pub_lift.publish(String(data='down'))
                return

            if self._lift_done_label == 'DOWN':
                self._lift_was_held = False
                self.get_logger().info('[LOWERING] Pallet bajado ✔ → BACK_AWAY')
                self._transition(_S.BACK_AWAY)
                return

            if self._time_in_state() > self._p('lift_timeout_s'):
                self.get_logger().error('[LOWERING] Timeout → ABORT')
                self._transition(_S.ABORT)

        # ── BACK_AWAY ─────────────────────────────────────────────────────
        elif s == _S.BACK_AWAY:
            elapsed = self._time_in_state()
            if elapsed < self._p('back_away_time'):
                cmd = Twist()
                cmd.linear.x = -abs(self._p('back_away_speed'))
                self._pub_cmd.publish(cmd)
            else:
                self._stop()
                self.get_logger().info('[BACK_AWAY] Completado → DONE')
                self._transition(_S.DONE)

        # ── DONE ──────────────────────────────────────────────────────────
        elif s == _S.DONE:
            self.get_logger().info(
                f'[TruckAlign] ✔ Entrega completada para "{self._target_class}"')
            self._set_vfh_bypass(False)
            self._pub_done.publish(String(data='SUCCESS'))
            self._reset()
            self._transition(_S.IDLE)

        # ── ABORT ─────────────────────────────────────────────────────────
        elif s == _S.ABORT:
            self._stop()
            # Si el lift tiene el pallet arriba, intentar bajar antes de abortar
            if self._lift_was_held:
                self.get_logger().warn(
                    '[ABORT] Pallet posiblemente arriba — enviando "down"')
                self._pub_lift.publish(String(data='down'))
            self.get_logger().warn('[TruckAlign] ❌ ABORT')
            self._set_vfh_bypass(False)
            self._pub_done.publish(String(data='ABORT'))
            self._reset()
            self._transition(_S.IDLE)

    # ══════════════════════════════════════════════════════════════════════
    # SEARCH_TRUCK — barrido in-place
    # ══════════════════════════════════════════════════════════════════════

    def _tick_scan(self):
        """Barrido izquierda-derecha-centro buscando el logo del camión."""
        if self._time_in_state() > self._p('search_timeout_s'):
            self._stop()
            self._scan_attempts += 1
            max_att = int(self._p('scan_max_attempts'))
            self.get_logger().warn(
                f'[SEARCH_TRUCK] Timeout de barrido '
                f'(intento {self._scan_attempts}/{max_att})')
            if self._scan_attempts >= max_att:
                self.get_logger().error(
                    '[SEARCH_TRUCK] Sin logo tras todos los intentos → ABORT')
                self._transition(_S.ABORT)
            else:
                # Reiniciar barrido desde posición actual
                self._scan_phase           = 'LEFT'
                self._scan_phase_start_yaw = self._rth
                self._state_entry          = time.monotonic()
            return

        scan_range = self._p('scan_range_deg')
        scan_speed = math.radians(self._p('scan_speed_dps'))
        delta_deg  = math.degrees(
            self._angle_diff(self._rth, self._scan_phase_start_yaw))

        if self._scan_phase == 'LEFT':
            if delta_deg < scan_range:
                self._pub_cmd.publish(self._spin_cmd(+scan_speed))
            else:
                self._stop()
                self._scan_phase           = 'RIGHT'
                self._scan_phase_start_yaw = self._rth
                self.get_logger().info(
                    f'[SEARCH_TRUCK] LEFT ({delta_deg:+.1f}°) → RIGHT')

        elif self._scan_phase == 'RIGHT':
            if delta_deg > -2.0 * scan_range:
                self._pub_cmd.publish(self._spin_cmd(-scan_speed))
            else:
                self._stop()
                self._scan_phase           = 'CENTER'
                self._scan_phase_start_yaw = self._rth
                self.get_logger().info(
                    f'[SEARCH_TRUCK] RIGHT ({delta_deg:+.1f}°) → CENTER')

        elif self._scan_phase == 'CENTER':
            if delta_deg < scan_range:
                self._pub_cmd.publish(self._spin_cmd(+scan_speed))
            else:
                self._stop()
                # Barrido completo sin resultado — se manejará en el próximo tick
                # vía timeout de search_timeout_s

    # ══════════════════════════════════════════════════════════════════════
    # GOALS
    # ══════════════════════════════════════════════════════════════════════

    def _publish_detection_wp_goal(self):
        """Publica el waypoint truck_detection hacia A*."""
        goal       = Pose2D()
        goal.x     = self._detection_wp[0]
        goal.y     = self._detection_wp[1]
        # El robot debe llegar mirando hacia donde están los camiones.
        # Se usa 0.0 como theta (heading indiferente); A* lo ajustará.
        goal.theta = 0.0
        self._pub_astar.publish(goal)
        self.get_logger().info(
            f'[GOTO_DETECTION_WP] A* goal → '
            f'({goal.x:.3f}, {goal.y:.3f})')

    def _publish_align_goal_if_needed(self, goal: Pose2D):
        """Publica goal a A* si se movió suficiente del anterior."""
        if self._last_goal_x is not None:
            dx = abs(goal.x - self._last_goal_x)
            dy = abs(goal.y - self._last_goal_y)
            if dx < self._p('goal_replan_dist') and dy < self._p('goal_replan_dist'):
                return
        self._pub_astar.publish(goal)
        self._last_goal_x = goal.x
        self._last_goal_y = goal.y
        self.get_logger().info(
            f'[ALIGNING] A* goal → ({goal.x:.3f}, {goal.y:.3f}) '
            f'θ={math.degrees(goal.theta):.1f}°  '
            f'[logo_angle={self._logo_angle:+.1f}°]')

    def _publish_final_goal(self):
        """Publica goal final a GoToGoal directo (bypassa A*)."""
        goal = self._compute_delivery_goal(self._p('stop_dist_m'))
        self._pub_wp.publish(goal)
        self.get_logger().info(
            f'[APPROACH_FINAL] GoToGoal directo → '
            f'({goal.x:.3f}, {goal.y:.3f}) '
            f'θ={math.degrees(goal.theta):.1f}°')

    def _compute_delivery_goal(self, stop_dist: float) -> Pose2D:
        """
        Calcula el goal de entrega con offset lateral.

        Geometría:
          - El logo del camión se estima en el frame mundo a partir del
            ángulo horizontal reportado por YOLO y la pose del robot.
          - bearing_to_logo = heading del robot + logo_angle_rad
            (ángulo positivo = logo a la derecha)
          - El robot debe quedar a stop_dist del logo en profundidad
            Y a lateral_offset_m a la DERECHA del logo.
          - "Derecha" del robot = perpendicular derecha al bearing robot→logo.

        Como YOLO no da distancia métrica al logo, estimamos la posición
        del logo proyectando a una distancia heurística desde el robot.
        La distancia heurística se toma como stop_dist + un margen
        (_LOGO_PROJ_DIST) para tener un vector de dirección estable.
        Lo que importa es la DIRECCIÓN, no la posición absoluta del logo.
        """
        _LOGO_PROJ_DIST = 2.0   # distancia de proyección heurística [m]

        # Bearing mundo al logo
        bearing = self._rth + math.radians(self._logo_angle)

        # Posición estimada del logo en el frame mundo
        # (proyección a distancia heurística desde el robot)
        logo_x = self._rx + _LOGO_PROJ_DIST * math.cos(bearing)
        logo_y = self._ry + _LOGO_PROJ_DIST * math.sin(bearing)

        # Vector forward (robot→logo)
        fwd_x = math.cos(bearing)
        fwd_y = math.sin(bearing)

        # Vector right: perpendicular derecha al forward
        # right = rotate(forward, -90°) = (sin(bearing), -cos(bearing))
        right_x =  math.sin(bearing)
        right_y = -math.cos(bearing)

        # Goal: retroceder stop_dist desde el logo + desplazar lateral a la derecha
        lateral = float(self._p('lateral_offset_m'))
        gx = logo_x - stop_dist * fwd_x + lateral * right_x
        gy = logo_y - stop_dist * fwd_y + lateral * right_y

        goal       = Pose2D()
        goal.x     = gx
        goal.y     = gy
        goal.theta = bearing   # el robot debe mirar hacia el camión al llegar
        return goal

    # ══════════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _logo_fresh(self) -> bool:
        """True si hay detección del logo objetivo reciente."""
        return (
            self._logo_seen
            and self._target_class != ''
            and (time.monotonic() - self._logo_stamp) < self._p('logo_timeout_s')
        )

    def _stop(self):
        self._pub_cmd.publish(Twist())

    def _time_in_state(self) -> float:
        return time.monotonic() - self._state_entry

    def _set_vfh_bypass(self, active: bool):
        self._pub_active.publish(Bool(data=active))
        label = 'ON  (evasión inhibida)' if active else 'OFF (evasión normal)'
        self.get_logger().info(f'[VFH+] /align/active → {label}')

    def _transition(self, new_state: str):
        if new_state == self._state:
            return
        self.get_logger().info(f'[FSM] {self._state} → {new_state}')
        self._state       = new_state
        self._state_entry = time.monotonic()
        self._last_goal_x = None
        self._last_goal_y = None

    def _reset(self):
        self._target_class         = ''
        self._lift_was_held        = False
        self._logo_angle           = 0.0
        self._logo_stamp           = 0.0
        self._logo_seen            = False
        self._lift_done_label      = ''
        self._astar_status         = ''
        self._last_goal_x          = None
        self._last_goal_y          = None
        self._advance_dist_m       = 0.0
        self._advance_start_odom   = (0.0, 0.0)
        self._scan_phase           = 'LEFT'
        self._scan_phase_start_yaw = 0.0
        self._scan_attempts        = 0

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Diferencia angular con signo en [-π, π]: a − b."""
        d = a - b
        while d >  math.pi: d -= 2.0 * math.pi
        while d < -math.pi: d += 2.0 * math.pi
        return d

    @staticmethod
    def _spin_cmd(angular_z: float) -> Twist:
        cmd = Twist()
        cmd.angular.z = angular_z
        return cmd


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = TruckAlignNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
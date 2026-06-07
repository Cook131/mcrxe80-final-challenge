#!/usr/bin/env python3
"""
qr_align_node.py — Iolair QR Align + Collect  [v4]
====================================================
Correcciones respecto a v3
--------------------------
  Bug 1: _tick_recover_scan usaba getattr y hardcodeaba max_attempts=3
         en vez de leer el parámetro scan_max_attempts.
  Bug 2: _cb_qr_angle no actualizaba _qr_stamp; la visibilidad del QR
         podía expirar aunque el ángulo llegara fresco.
  Bug 3: APPROACH_FINAL esperaba GOAL_REACHED de A*, pero el goal se
         publicó en /goal (GoToGoal directo, sin A*). A* nunca emite
         GOAL_REACHED para ese waypoint → timeout garantizado. Solución:
         criterio de llegada por distancia base_link→QR < approach_final_dist.
  Bug 4: close_enough usaba self._qr_dist (dist cámara→QR). Con offsets de
         cámara corregidos, la dist base_link→QR difiere en 7-15 cm a rangos
         cortos. Ahora se usa dist_bl calculada desde base_link.
  Bug 5: lateral_err mezclaba self._qr_dist (hipotenusa cámara) con
         angle_err_robot (ángulo desde base_link). Ahora usa dist_bl.
  Mejora: _qr_visible() verificaba solo que hubiera algún payload; ahora
         comprueba que el payload coincida con _target_payload para no
         seguir QRs ajenos al pallet objetivo.

Máquina de estados
------------------
  IDLE → SEARCH_QR → ALIGNING → APPROACH_FINAL → HOLD → BACK_AWAY → DELIVERY
                ↘ RECOVER_SCAN ↗  (safeguard pérdida QR en SEARCH_QR / ALIGNING)

Topics
------
  SUB:  /collect/trigger     (std_msgs/String)   "rack" | "conveyor" | "abort"
  PUB:  /collect/done        (std_msgs/String)   "SUCCESS" | "ABORT"
  SUB:  /aruco/qr            (std_msgs/String)
  SUB:  /aruco/qr/distance   (std_msgs/Float32)  metros en plano XZ (cámara)
  SUB:  /aruco/qr/angle      (std_msgs/Float32)  grados, + = derecha (cámara)
  SUB:  /odom                (nav_msgs/Odometry)
  SUB:  /astar/status        (std_msgs/String)
  PUB:  /astar/goal          (geometry_msgs/Pose2D)
  PUB:  /goal                (geometry_msgs/Pose2D)  GoToGoal directo (tramo final)
  PUB:  /cmd_vel             (geometry_msgs/Twist)   SOLO BACK_AWAY y RECOVER_SCAN
  PUB:  /lift_auto           (std_msgs/String)
  SUB:  /lift_done           (std_msgs/String)
  PUB:  /align/active        (std_msgs/Bool)
  PUB:  /collect/qr_payload  (std_msgs/String)

Parámetros ROS2
---------------
  align_stop_dist       float  0.35   Distancia base_link→QR para considerar alineado [m]
  approach_final_dist   float  0.05   Distancia base_link→QR para considerar llegada [m]
  align_lateral_tol     float  0.03   Tolerancia lateral [m]
  angle_tol_deg         float  4.0    Tolerancia angular desde base_link [°]
  goal_replan_dist      float  0.06   Umbral de cambio en mundo para re-publicar goal [m]
  back_away_speed       float  0.10   Velocidad retroceso [m/s]
  back_away_time        float  1.8    Duración retroceso [s]
  lift_timeout          float  8.0    Timeout /lift_done [s]
  align_timeout         float  20.0   Timeout ALIGNING [s]
  approach_timeout      float  15.0   Timeout APPROACH_FINAL [s]
  search_timeout        float  10.0   Timeout SEARCH_QR [s]
  qr_timeout            float  2.5    Segundos sin QR para considerar pérdida [s]
  cam_offset_deg        float  0.0    Offset angular residual (normalmente 0) [°]
  cam_fwd_m             float  0.15   Offset cámara adelante de base_link [m]
  cam_left_m            float  0.07   Offset cámara a la izquierda de base_link [m]
  fsm_rate_hz           float  20.0   Frecuencia del tick [Hz]
  scan_range_deg        float  30.0   Semi-amplitud barrido recuperación [°]
  scan_speed_dps        float  20.0   Velocidad barrido [°/s]
  scan_max_attempts     int    3      Intentos de barrido antes de ABORT
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos  import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist, Pose2D
from nav_msgs.msg      import Odometry
from std_msgs.msg      import Bool, Float32, String


# ══════════════════════════════════════════════════════════════════════════════
# Estados internos
# ══════════════════════════════════════════════════════════════════════════════

class _S:
    IDLE           = 'IDLE'
    SEARCH_QR      = 'SEARCH_QR'
    ALIGNING       = 'ALIGNING'
    RECOVER_SCAN   = 'RECOVER_SCAN'
    APPROACH_FINAL = 'APPROACH_FINAL'
    HOLD           = 'HOLD'
    BACK_AWAY      = 'BACK_AWAY'
    DELIVERY       = 'DELIVERY'
    ABORT          = 'ABORT'


_ZONE_LIFT = {
    'rack':     ('n1', 'AT_N1'),
    'conveyor': ('n2', 'AT_N2'),
}


# ══════════════════════════════════════════════════════════════════════════════
class QRAlignNode(Node):

    def __init__(self):
        super().__init__('qr_align_node')

        # ── Parámetros ────────────────────────────────────────────────────
        self.declare_parameter('align_stop_dist',     0.35)
        self.declare_parameter('approach_final_dist', 0.05)
        self.declare_parameter('align_lateral_tol',   0.03)
        self.declare_parameter('angle_tol_deg',       4.0)
        self.declare_parameter('goal_replan_dist',    0.06)
        self.declare_parameter('back_away_speed',     0.10)
        self.declare_parameter('back_away_time',      1.8)
        self.declare_parameter('lift_timeout',        8.0)
        self.declare_parameter('align_timeout',       20.0)
        self.declare_parameter('approach_timeout',    15.0)
        self.declare_parameter('search_timeout',      10.0)
        self.declare_parameter('qr_timeout',          2.5)
        self.declare_parameter('cam_offset_deg',      0.0)
        self.declare_parameter('cam_fwd_m',           0.15)
        self.declare_parameter('cam_left_m',          0.07)
        self.declare_parameter('fsm_rate_hz',         20.0)
        self.declare_parameter('scan_range_deg',      30.0)
        self.declare_parameter('scan_speed_dps',      20.0)
        self.declare_parameter('scan_max_attempts',   3)

        self._p = lambda n: self.get_parameter(n).value

        # ── Estado FSM ────────────────────────────────────────────────────
        self._state       = _S.IDLE
        self._state_entry = time.monotonic()

        self._zone        = ''
        self._lift_cmd    = ''
        self._lift_expect = ''

        # ── Datos QR ──────────────────────────────────────────────────────
        self._qr_payload     = ''       # último payload recibido
        self._target_payload = ''       # payload del pallet objetivo (del trigger)
        self._qr_angle       = 0.0     # grados, + = derecha, frame cámara
        self._qr_dist        = 999.0   # metros, dist cámara→QR en plano XZ
        self._qr_stamp       = 0.0

        # ── Pose odométrica ───────────────────────────────────────────────
        self._rx  = 0.0
        self._ry  = 0.0
        self._rth = 0.0

        # ── A*/GoToGoal ───────────────────────────────────────────────────
        self._astar_status = ''
        self._last_goal_x  = None
        self._last_goal_y  = None

        # ── RECOVER_SCAN ──────────────────────────────────────────────────
        self._scan_return_state    = _S.SEARCH_QR
        self._scan_phase           = 'LEFT'
        self._scan_phase_start_yaw = 0.0
        self._scan_attempts        = 0

        # ── Lift ──────────────────────────────────────────────────────────
        self._lift_done_label = ''

        # ── QOS ───────────────────────────────────────────────────────────
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Suscriptores ──────────────────────────────────────────────────
        self.create_subscription(String,   '/collect/trigger',   self._cb_trigger,      10)
        self.create_subscription(String,   '/aruco/qr',          self._cb_qr,           qos_be)
        self.create_subscription(Float32,  '/aruco/qr/distance', self._cb_qr_dist,      qos_be)
        self.create_subscription(Float32,  '/aruco/qr/angle',    self._cb_qr_angle,     qos_be)
        self.create_subscription(String,   '/lift_done',         self._cb_lift_done,    10)
        self.create_subscription(Odometry, '/odom',              self._cb_odom,         10)
        self.create_subscription(String,   '/astar/status',      self._cb_astar_status, 10)

        # ── Publicadores ──────────────────────────────────────────────────
        self._pub_cmd     = self.create_publisher(Twist,  '/cmd_vel',            10)
        self._pub_lift    = self.create_publisher(String, '/lift_auto',          10)
        self._pub_done    = self.create_publisher(String, '/collect/done',       10)
        self._pub_payload = self.create_publisher(String, '/collect/qr_payload', 10)
        self._pub_astar   = self.create_publisher(Pose2D, '/astar/goal',         10)
        self._pub_wp      = self.create_publisher(Pose2D, '/goal',               10)
        self._pub_active  = self.create_publisher(Bool,   '/align/active',       10)

        # ── Timer FSM ─────────────────────────────────────────────────────
        self.create_timer(1.0 / float(self._p('fsm_rate_hz')), self._tick)

        self.get_logger().info(
            'qr_align_node v4 listo\n'
            f'  align_stop_dist={self._p("align_stop_dist")}m  '
            f'approach_final_dist={self._p("approach_final_dist")}m  '
            f'cam=[fwd={self._p("cam_fwd_m")}m, left={self._p("cam_left_m")}m]'
        )

    # ══════════════════════════════════════════════════════════════════════
    # CALLBACKS
    # ══════════════════════════════════════════════════════════════════════

    def _cb_trigger(self, msg: String):
        cmd = msg.data.strip().lower()

        if cmd == 'abort':
            if self._state != _S.IDLE:
                self.get_logger().warn('[Collect] ABORT recibido por FSM')
                self._transition(_S.ABORT)
            return

        if cmd not in _ZONE_LIFT:
            self.get_logger().warn(f'[Collect] Zona desconocida: "{cmd}" — ignorado')
            return

        if self._state != _S.IDLE:
            self.get_logger().warn(
                f'[Collect] Trigger ignorado — estado: {self._state}')
            return

        self._zone       = cmd
        self._lift_cmd, self._lift_expect = _ZONE_LIFT[cmd]
        # El payload objetivo se aprende del primer QR que se vea; se fija en
        # _cb_qr la primera vez. Se limpia en _reset para el siguiente ciclo.
        self._target_payload = ''
        self.get_logger().info(
            f'[Collect] Trigger zona="{cmd}" → lift_cmd={self._lift_cmd}')

        self._set_vfh_bypass(True)
        self._transition(_S.SEARCH_QR)

    def _cb_qr(self, msg: String):
        payload = msg.data.strip()
        if not payload:
            return
        # Fijar el payload objetivo con el primer QR visible tras el trigger
        if self._target_payload == '' and self._state not in (_S.IDLE, _S.ABORT):
            self._target_payload = payload
            self.get_logger().info(f'[QR] Payload objetivo fijado: "{payload}"')
        if payload == self._target_payload or self._target_payload == '':
            if payload != self._qr_payload:
                self.get_logger().info(f'[QR] Payload: {payload}')
                self._qr_payload = payload
                self._pub_payload.publish(String(data=payload))
            self._qr_stamp = time.monotonic()

    def _cb_qr_dist(self, msg: Float32):
        self._qr_dist  = float(msg.data)
        self._qr_stamp = time.monotonic()

    def _cb_qr_angle(self, msg: Float32):
        # Fix Bug 2: actualizar _qr_stamp aquí también.
        # El offset lateral ya se corrige en aruco_detector._angle_distance;
        # cam_offset_deg queda en 0.0, preservado solo por compatibilidad.
        self._qr_angle = float(msg.data) + float(self._p('cam_offset_deg'))
        self._qr_stamp = time.monotonic()

    def _cb_lift_done(self, msg: String):
        label = msg.data.strip()
        if label:
            self.get_logger().info(f'[Lift] /lift_done: {label}')
            self._lift_done_label = label

    def _cb_odom(self, msg: Odometry):
        self._rx  = msg.pose.pose.position.x
        self._ry  = msg.pose.pose.position.y
        q         = msg.pose.pose.orientation
        self._rth = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def _cb_astar_status(self, msg: String):
        prev = self._astar_status
        self._astar_status = msg.data.strip()
        if self._astar_status != prev:
            self.get_logger().debug(f'[A*] status: {self._astar_status}')

    # ══════════════════════════════════════════════════════════════════════
    # FSM TICK
    # ══════════════════════════════════════════════════════════════════════

    def _tick(self):
        s = self._state

        if s == _S.IDLE:
            return

        # ── SEARCH_QR ─────────────────────────────────────────────────────
        elif s == _S.SEARCH_QR:
            if self._qr_visible():
                self.get_logger().info(
                    f'[SEARCH_QR] QR detectado dist={self._qr_dist:.2f}m '
                    f'angle={self._qr_angle:+.1f}° → ALIGNING')
                self._transition(_S.ALIGNING)
                return
            if self._time_in_state() > self._p('search_timeout'):
                self.get_logger().warn('[SEARCH_QR] Timeout → RECOVER_SCAN')
                self._start_recover_scan(return_to=_S.SEARCH_QR)

        # ── ALIGNING ──────────────────────────────────────────────────────
        elif s == _S.ALIGNING:
            if not self._qr_visible():
                if (time.monotonic() - self._qr_stamp) > self._p('qr_timeout'):
                    self.get_logger().warn('[ALIGNING] QR perdido → RECOVER_SCAN')
                    self._start_recover_scan(return_to=_S.ALIGNING)
                return

            if self._time_in_state() > self._p('align_timeout'):
                self.get_logger().warn('[ALIGNING] Timeout → ABORT')
                self._transition(_S.ABORT)
                return

            # Geometría desde base_link (corregida de offsets de cámara)
            qr_x, qr_y = self._qr_world_pos()
            bearing_robot = math.atan2(qr_y - self._ry, qr_x - self._rx)
            angle_err_robot = math.degrees(
                self._angle_diff(bearing_robot, self._rth))

            # Fix Bug 4+5: usar dist base_link→QR, no dist cámara→QR
            dist_bl = math.hypot(qr_x - self._rx, qr_y - self._ry)
            lateral_err = dist_bl * math.sin(math.radians(angle_err_robot))

            aligned = (
                abs(angle_err_robot) < self._p('angle_tol_deg')
                and abs(lateral_err)  < self._p('align_lateral_tol')
            )
            # Fix Bug 4: close_enough usa dist_bl
            close_enough = dist_bl <= self._p('align_stop_dist')

            if aligned and close_enough:
                self.get_logger().info(
                    f'[ALIGNING] ✔ dist_bl={dist_bl:.3f}m  '
                    f'lateral={lateral_err*100:.1f}cm  '
                    f'angle_bl={angle_err_robot:+.1f}° → APPROACH_FINAL')
                self._publish_approach_final_goal()
                self._transition(_S.APPROACH_FINAL)
                return

            self._publish_align_goal_if_needed()

        # ── RECOVER_SCAN ──────────────────────────────────────────────────
        elif s == _S.RECOVER_SCAN:
            self._tick_recover_scan()

        # ── APPROACH_FINAL ────────────────────────────────────────────────
        elif s == _S.APPROACH_FINAL:
            # Fix Bug 3: GoToGoal directo no genera GOAL_REACHED en /astar/status.
            # Criterio de llegada: dist base_link→QR < approach_final_dist.
            # Si el QR no es visible usamos el timeout como fallback.
            if self._qr_visible():
                qr_x, qr_y = self._qr_world_pos()
                dist_bl = math.hypot(qr_x - self._rx, qr_y - self._ry)
                if dist_bl <= self._p('approach_final_dist'):
                    self.get_logger().info(
                        f'[APPROACH_FINAL] Llegada confirmada dist_bl={dist_bl:.3f}m → HOLD')
                    self._transition(_S.HOLD)
                    return

            if self._time_in_state() > self._p('approach_timeout'):
                self.get_logger().warn('[APPROACH_FINAL] Timeout → ABORT')
                self._transition(_S.ABORT)

        # ── HOLD ──────────────────────────────────────────────────────────
        elif s == _S.HOLD:
            if self._time_in_state() < 0.1:
                self.get_logger().info(f'[HOLD] Subiendo lift: {self._lift_cmd}')
                self._lift_done_label = ''
                self._pub_lift.publish(String(data=self._lift_cmd))
                return

            if self._lift_done_label == self._lift_expect:
                self.get_logger().info(
                    f'[HOLD] Lift en {self._lift_done_label} → elevando a hold')
                self._pub_lift.publish(String(data='hold'))
                self._lift_done_label = ''
                self.get_logger().info('[HOLD] Pallet recogido → /collect/done SUCCESS')
                self._pub_done.publish(String(data='SUCCESS'))
                self._transition(_S.BACK_AWAY)
                return

            if self._time_in_state() > self._p('lift_timeout'):
                self.get_logger().error(
                    f'[HOLD] Timeout esperando {self._lift_expect} → ABORT')
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
                self.get_logger().info('[BACK_AWAY] Completado → DELIVERY')
                self._set_vfh_bypass(False)
                self._transition(_S.DELIVERY)

        # ── DELIVERY ──────────────────────────────────────────────────────
        elif s == _S.DELIVERY:
            self.get_logger().info(
                f'[DELIVERY] Control cedido a FSM — '
                f'payload="{self._qr_payload}" zona="{self._zone}"')
            self._reset()
            self._transition(_S.IDLE)

        # ── ABORT ─────────────────────────────────────────────────────────
        elif s == _S.ABORT:
            self._stop()
            if self._lift_cmd:
                self._pub_lift.publish(String(data='down'))
            self.get_logger().warn('[Collect] ❌ ABORT')
            self._set_vfh_bypass(False)
            self._pub_done.publish(String(data='ABORT'))
            self._reset()
            self._transition(_S.IDLE)

    # ══════════════════════════════════════════════════════════════════════
    # RECOVER_SCAN
    # ══════════════════════════════════════════════════════════════════════

    def _start_recover_scan(self, return_to: str):
        self._scan_return_state    = return_to
        self._scan_phase           = 'LEFT'
        self._scan_phase_start_yaw = self._rth
        self.get_logger().warn(
            f'[RECOVER_SCAN] Iniciando barrido ±{self._p("scan_range_deg")}°  '
            f'→ volveré a {return_to} si encuentro el QR')
        self._transition(_S.RECOVER_SCAN)

    def _tick_recover_scan(self):
        if self._qr_visible():
            self._stop()
            self.get_logger().info(
                f'[RECOVER_SCAN] QR recuperado en fase {self._scan_phase} '
                f'dist={self._qr_dist:.2f}m angle={self._qr_angle:+.1f}° '
                f'→ {self._scan_return_state}')
            self._transition(self._scan_return_state)
            return

        scan_range = self._p('scan_range_deg')
        scan_speed = math.radians(self._p('scan_speed_dps'))
        delta_yaw_deg = math.degrees(
            self._angle_diff(self._rth, self._scan_phase_start_yaw))

        if self._scan_phase == 'LEFT':
            if delta_yaw_deg < scan_range:
                self._pub_cmd.publish(self._spin_cmd(+scan_speed))
            else:
                self._stop()
                self.get_logger().info(
                    f'[RECOVER_SCAN] LEFT completado ({delta_yaw_deg:+.1f}°) → RIGHT')
                self._scan_phase           = 'RIGHT'
                self._scan_phase_start_yaw = self._rth

        elif self._scan_phase == 'RIGHT':
            if delta_yaw_deg > -2.0 * scan_range:
                self._pub_cmd.publish(self._spin_cmd(-scan_speed))
            else:
                self._stop()
                self.get_logger().info(
                    f'[RECOVER_SCAN] RIGHT completado ({delta_yaw_deg:+.1f}°) → CENTER')
                self._scan_phase           = 'CENTER'
                self._scan_phase_start_yaw = self._rth

        elif self._scan_phase == 'CENTER':
            if delta_yaw_deg < scan_range:
                self._pub_cmd.publish(self._spin_cmd(+scan_speed))
            else:
                self._stop()
                # Fix Bug 1: usar self._scan_attempts y leer scan_max_attempts
                self._scan_attempts += 1
                max_att = int(self._p('scan_max_attempts'))
                self.get_logger().warn(
                    f'[RECOVER_SCAN] Barrido completo sin QR '
                    f'(intento {self._scan_attempts}/{max_att})')
                if self._scan_attempts >= max_att:
                    self.get_logger().error(
                        '[RECOVER_SCAN] Sin QR tras todos los intentos → ABORT')
                    self._transition(_S.ABORT)
                else:
                    self._scan_phase           = 'LEFT'
                    self._scan_phase_start_yaw = self._rth

    # ══════════════════════════════════════════════════════════════════════
    # GOALS
    # ══════════════════════════════════════════════════════════════════════

    def _publish_align_goal_if_needed(self):
        goal = self._compute_align_goal(self._p('align_stop_dist'))

        if self._last_goal_x is not None:
            dx = abs(goal.x - self._last_goal_x)
            dy = abs(goal.y - self._last_goal_y)
            if dx < self._p('goal_replan_dist') and dy < self._p('goal_replan_dist'):
                return

        self._pub_astar.publish(goal)
        self._last_goal_x = goal.x
        self._last_goal_y = goal.y

        self.get_logger().info(
            f'[ALIGNING] Goal → ({goal.x:.3f}, {goal.y:.3f}) '
            f'θ={math.degrees(goal.theta):.1f}°  '
            f'[QR dist={self._qr_dist:.2f}m ang={self._qr_angle:+.1f}°]'
        )

    def _publish_approach_final_goal(self):
        goal = self._compute_align_goal(self._p('approach_final_dist'))
        self._pub_wp.publish(goal)
        self.get_logger().info(
            f'[APPROACH_FINAL] Goal directo → ({goal.x:.3f}, {goal.y:.3f}) '
            f'θ={math.degrees(goal.theta):.1f}°'
        )

    def _compute_align_goal(self, stop_dist: float) -> Pose2D:
        """
        Calcula un Pose2D en coordenadas mundo tal que base_link quede a
        stop_dist metros del QR mirando directamente hacia él.

        Pasos:
          1. Posición de la cámara en mundo (base_link + offset rotado).
          2. Posición del QR en mundo (desde la cámara + bearing_cam).
          3. Bearing base_link→QR (orientación correcta del robot al llegar).
          4. Goal = QR retrocedido stop_dist en la dirección base_link→QR.
        """
        CAM_FWD  = float(self._p('cam_fwd_m'))
        CAM_LEFT = float(self._p('cam_left_m'))

        qr_x, qr_y = self._qr_world_pos_with_offsets(CAM_FWD, CAM_LEFT)
        bearing_robot = math.atan2(qr_y - self._ry, qr_x - self._rx)

        gx = qr_x - stop_dist * math.cos(bearing_robot)
        gy = qr_y - stop_dist * math.sin(bearing_robot)

        goal = Pose2D()
        goal.x     = gx
        goal.y     = gy
        goal.theta = bearing_robot
        return goal

    # ══════════════════════════════════════════════════════════════════════
    # GEOMETRÍA
    # ══════════════════════════════════════════════════════════════════════

    def _qr_world_pos(self):
        """Posición del QR en mundo usando los offsets de cámara declarados."""
        return self._qr_world_pos_with_offsets(
            float(self._p('cam_fwd_m')),
            float(self._p('cam_left_m')),
        )

    def _qr_world_pos_with_offsets(self, cam_fwd: float, cam_left: float):
        """
        Calcula (qr_x, qr_y) en coordenadas mundo a partir de la pose del
        robot, los offsets físicos de la cámara y la medición ángulo+distancia.
        """
        # Posición de la cámara en mundo
        cam_x = self._rx + cam_fwd  * math.cos(self._rth) - cam_left * math.sin(self._rth)
        cam_y = self._ry + cam_fwd  * math.sin(self._rth) + cam_left * math.cos(self._rth)

        # Bearing cámara→QR en mundo
        bearing_cam = self._rth + math.radians(self._qr_angle)

        # Posición del QR
        qr_x = cam_x + self._qr_dist * math.cos(bearing_cam)
        qr_y = cam_y + self._qr_dist * math.sin(bearing_cam)
        return qr_x, qr_y

    # ══════════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _qr_visible(self) -> bool:
        # Mejora: verifica también que el payload coincida con el objetivo.
        payload_ok = (
            self._target_payload == ''               # aún no fijado → aceptar cualquiera
            or self._qr_payload == self._target_payload
        )
        return (
            payload_ok
            and self._qr_payload != ''
            and (time.monotonic() - self._qr_stamp) < self._p('qr_timeout')
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
        self._zone             = ''
        self._lift_cmd         = ''
        self._lift_expect      = ''
        self._lift_done_label  = ''
        self._qr_payload       = ''
        self._target_payload   = ''
        self._qr_angle         = 0.0
        self._qr_dist          = 999.0
        self._astar_status     = ''
        self._last_goal_x      = None
        self._last_goal_y      = None
        self._scan_return_state    = _S.SEARCH_QR
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
    node = QRAlignNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

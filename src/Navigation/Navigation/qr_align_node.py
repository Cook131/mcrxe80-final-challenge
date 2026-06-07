#!/usr/bin/env python3
"""
qr_align_node.py — Iolair QR Align + Collect  [v5]
====================================================
Correcciones respecto a v4 — compatibilidad spi_servo_node v2
--------------------------------------------------------------
  Fix S1 — _cb_lift_done filtra mensajes fuera del estado HOLD.
            El servo publica /lift_done:"DOWN" al arrancar (FPGA en IDLE=0);
            ese mensaje no debe contaminar _lift_done_label cuando el nodo
            está en ALIGNING, APPROACH_FINAL, etc.

  Fix S2 — Estado HOLD dividido en dos fases explícitas:
            Fase A: envía n1/n2 → espera AT_N1/AT_N2
            Fase B: envía hold  → espera HOLD confirmado → BACK_AWAY
            El servo v2 requiere que el FPGA esté en AT_N1(4) o AT_N2(6)
            para aceptar "hold" (VALID_FROM['hold'] = {4,6}).  Publicar
            "hold" y salir en el mismo tick garantizaba rechazo silencioso.

  Fix S3 — ABORT envía "down" solo si el lift llegó a HOLD.
            El servo requiere HOLD(8) para aceptar "down"
            (VALID_FROM['down'] = {8}). Enviarlo desde AT_N1/AT_N2 era
            ignorado silenciosamente y dejaba el lift arriba.
            Ahora se rastrea _lift_reached_hold para saber si es seguro
            enviar "down" en caso de abort.

Secuencia de lift tras los fixes
---------------------------------
  qr_align publica "n1"
    → FPGA: IDLE(0) → TO_N1(3) → AT_N1(4)
    → servo publica /lift_done:"AT_N1"
  qr_align recibe AT_N1 → publica "hold"
    → FPGA: AT_N1(4) → LIFTING(7) → HOLD(8)
    → servo publica /lift_done:"HOLD"
  qr_align recibe HOLD → publica /collect/done SUCCESS → BACK_AWAY

Máquina de estados
------------------
  IDLE → SEARCH_QR → ALIGNING → APPROACH_FINAL → HOLD → BACK_AWAY → DELIVERY
                ↘ RECOVER_SCAN ↗

Topics / Parámetros: sin cambios respecto a v4.
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
        self._qr_payload     = ''
        self._target_payload = ''
        self._qr_angle       = 0.0
        self._qr_dist        = 999.0
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
        self._lift_done_label  = ''
        # Fix S2: fase del estado HOLD ('WAIT_LEVEL' o 'WAIT_HOLD')
        self._lift_phase       = 'WAIT_LEVEL'
        # Fix S3: saber si el FPGA llegó a HOLD para poder enviar "down" en abort
        self._lift_reached_hold = False

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
            'qr_align_node v5 listo\n'
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
        self._target_payload = ''
        self.get_logger().info(
            f'[Collect] Trigger zona="{cmd}" → lift_cmd={self._lift_cmd}')

        self._set_vfh_bypass(True)
        self._transition(_S.SEARCH_QR)

    def _cb_qr(self, msg: String):
        payload = msg.data.strip()
        if not payload:
            return
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
        self._qr_angle = float(msg.data) + float(self._p('cam_offset_deg'))
        self._qr_stamp = time.monotonic()

    def _cb_lift_done(self, msg: String):
        # Fix S1: solo procesar /lift_done cuando estamos en HOLD.
        # El servo publica "DOWN" al arrancar (FPGA IDLE=0) y en cualquier
        # momento que el FPGA llegue a un estado estable. Ignorar fuera de HOLD
        # evita que esos mensajes corrompan _lift_done_label.
        if self._state != _S.HOLD:
            return
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

            qr_x, qr_y     = self._qr_world_pos()
            bearing_robot   = math.atan2(qr_y - self._ry, qr_x - self._rx)
            angle_err_robot = math.degrees(self._angle_diff(bearing_robot, self._rth))
            dist_bl         = math.hypot(qr_x - self._rx, qr_y - self._ry)
            lateral_err     = dist_bl * math.sin(math.radians(angle_err_robot))

            aligned     = (abs(angle_err_robot) < self._p('angle_tol_deg')
                           and abs(lateral_err)  < self._p('align_lateral_tol'))
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
            if self._qr_visible():
                qr_x, qr_y = self._qr_world_pos()
                dist_bl = math.hypot(qr_x - self._rx, qr_y - self._ry)
                if dist_bl <= self._p('approach_final_dist'):
                    self.get_logger().info(
                        f'[APPROACH_FINAL] Llegada confirmada '
                        f'dist_bl={dist_bl:.3f}m → HOLD')
                    self._transition(_S.HOLD)
                    return

            if self._time_in_state() > self._p('approach_timeout'):
                self.get_logger().warn('[APPROACH_FINAL] Timeout → ABORT')
                self._transition(_S.ABORT)

        # ── HOLD ──────────────────────────────────────────────────────────
        #
        # Fix S2: dos fases secuenciales con confirmación de cada una.
        #
        # WAIT_LEVEL: envía "n1"/"n2" en la entrada, espera AT_N1/AT_N2.
        #             spi_servo acepta n1/n2 solo desde IDLE(0).
        #
        # WAIT_HOLD:  envía "hold" al llegar a AT_N1/AT_N2, espera HOLD.
        #             spi_servo acepta "hold" solo desde AT_N1(4)/AT_N2(6).
        #             Solo tras confirmar HOLD se avanza a BACK_AWAY.
        #
        elif s == _S.HOLD:

            if self._time_in_state() < 0.05:
                # Entrada al estado: inicializar fase y enviar primer comando
                self._lift_done_label  = ''
                self._lift_phase       = 'WAIT_LEVEL'
                self._lift_reached_hold = False
                self.get_logger().info(
                    f'[HOLD] Fase WAIT_LEVEL → enviando: {self._lift_cmd}')
                self._pub_lift.publish(String(data=self._lift_cmd))
                return

            if self._time_in_state() > self._p('lift_timeout'):
                self.get_logger().error(
                    f'[HOLD] Timeout en fase {self._lift_phase} → ABORT')
                self._transition(_S.ABORT)
                return

            if self._lift_phase == 'WAIT_LEVEL':
                # Esperando que el FPGA llegue a AT_N1 o AT_N2
                if self._lift_done_label == self._lift_expect:
                    self.get_logger().info(
                        f'[HOLD] {self._lift_done_label} confirmado → '
                        f'fase WAIT_HOLD, enviando: hold')
                    self._lift_done_label = ''
                    self._lift_phase      = 'WAIT_HOLD'
                    # Fix S2: "hold" solo llega al FPGA cuando está en AT_N1/AT_N2.
                    # El servo v2 lo acepta aquí porque acabamos de confirmar ese estado.
                    self._pub_lift.publish(String(data='hold'))

            elif self._lift_phase == 'WAIT_HOLD':
                # Esperando confirmación de HOLD del FPGA
                if self._lift_done_label == 'HOLD':
                    self._lift_reached_hold = True   # Fix S3: marcar para abort seguro
                    self.get_logger().info(
                        '[HOLD] HOLD confirmado — pallet asegurado → '
                        '/collect/done SUCCESS → BACK_AWAY')
                    self._pub_done.publish(String(data='SUCCESS'))
                    self._transition(_S.BACK_AWAY)

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
            # Fix S3: "down" solo es válido desde HOLD(8).
            # Si el lift llegó a HOLD antes del abort, enviarlo es seguro.
            # En cualquier otro caso (AT_N1, AT_N2, en tránsito) el servo
            # lo rechazaría silenciosamente y el lift quedaría arriba.
            if self._lift_reached_hold:
                self.get_logger().info('[ABORT] Lift en HOLD — enviando down')
                self._pub_lift.publish(String(data='down'))
            elif self._lift_cmd:
                self.get_logger().warn(
                    '[ABORT] Lift no llegó a HOLD — "down" omitido; '
                    'el operador debe bajar el lift manualmente')
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
        CAM_FWD  = float(self._p('cam_fwd_m'))
        CAM_LEFT = float(self._p('cam_left_m'))

        qr_x, qr_y   = self._qr_world_pos_with_offsets(CAM_FWD, CAM_LEFT)
        bearing_robot = math.atan2(qr_y - self._ry, qr_x - self._rx)

        gx = qr_x - stop_dist * math.cos(bearing_robot)
        gy = qr_y - stop_dist * math.sin(bearing_robot)

        goal       = Pose2D()
        goal.x     = gx
        goal.y     = gy
        goal.theta = bearing_robot
        return goal

    # ══════════════════════════════════════════════════════════════════════
    # GEOMETRÍA
    # ══════════════════════════════════════════════════════════════════════

    def _qr_world_pos(self):
        return self._qr_world_pos_with_offsets(
            float(self._p('cam_fwd_m')),
            float(self._p('cam_left_m')),
        )

    def _qr_world_pos_with_offsets(self, cam_fwd: float, cam_left: float):
        cam_x = (self._rx
                 + cam_fwd  * math.cos(self._rth)
                 - cam_left * math.sin(self._rth))
        cam_y = (self._ry
                 + cam_fwd  * math.sin(self._rth)
                 + cam_left * math.cos(self._rth))

        bearing_cam = self._rth + math.radians(self._qr_angle)

        qr_x = cam_x + self._qr_dist * math.cos(bearing_cam)
        qr_y = cam_y + self._qr_dist * math.sin(bearing_cam)
        return qr_x, qr_y

    # ══════════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _qr_visible(self) -> bool:
        payload_ok = (
            self._target_payload == ''
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
        self._zone              = ''
        self._lift_cmd          = ''
        self._lift_expect       = ''
        self._lift_done_label   = ''
        self._lift_phase        = 'WAIT_LEVEL'
        self._lift_reached_hold = False
        self._qr_payload        = ''
        self._target_payload    = ''
        self._qr_angle          = 0.0
        self._qr_dist           = 999.0
        self._astar_status      = ''
        self._last_goal_x       = None
        self._last_goal_y       = None
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

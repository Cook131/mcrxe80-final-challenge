#!/usr/bin/env python3
"""
qr_align_node.py — Iolair QR Align + Collect  [v7]
====================================================
Cambios respecto a v6
---------------------

  Fix H1 — Control directo en ALIGNING y APPROACH_FINAL.
            El nodo ya NO usa A* / go_to_goal / VFH+ para moverse.
            Desde el momento en que se detecta el QR (SEARCH_QR → ALIGNING),
            qr_align_node toma control total de /cmd_vel con un controlador
            PD propio. El bypass de VFH+ se activa al arrancar el trigger
            y se mantiene hasta BACK_AWAY.

  Fix H2 — Se eliminan _pub_astar, _pub_wp y toda la lógica de
            replanificación A* (_publish_align_goal_if_needed,
            _publish_approach_final_goal, _last_goal_x/y,
            goal_replan_dist). Ya no son necesarios.

  Fix H3 — Nuevos parámetros de control directo:
              kp_angle    — ganancia proporcional angular
              kd_angle    — ganancia derivativa angular
              kp_dist     — ganancia proporcional de distancia
              kd_dist     — ganancia derivativa de distancia
              max_angular — límite de velocidad angular (rad/s)
              max_linear  — límite de velocidad lineal (m/s)
            El controlador angular usa PD sobre el error de bearing.
            El controlador lineal avanza solo si |angle_err| < 2× tol,
            usando PD sobre (dist - align_stop_dist).

  Fix H4 — APPROACH_FINAL también usa control directo PD sobre
            (dist - forklift_reach_m). Antes publicaba a /goal
            (go_to_goal) lo que seguía dependiendo del pipeline A*.

  (Fixes G1-G4 y S1-S3 de v5/v6 conservados sin cambios)

Secuencia completa
------------------
  1. SEARCH_QR      → detecta QR; activa bypass VFH+
  2. ALIGNING       → PD directo /cmd_vel hasta aligned + close_enough
  3. APPROACH_FINAL → PD directo /cmd_vel hasta dist ≤ forklift_reach_m
  4. HOLD           → lift n1/n2 → AT_N1/AT_N2 → hold → HOLD
  5. BACK_AWAY      → retroceso lineal directo
  6. DELIVERY       → cede control; desactiva bypass

Topics QR (estandarizados):
  Suscribe: /qr/data      (std_msgs/String)   — contenido del QR
            /qr/distance  (std_msgs/Float32)   — distancia en metros
            /qr/angle     (std_msgs/Float32)   — ángulo horizontal en grados
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos  import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
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

        # Geometría de alineación
        self.declare_parameter('align_stop_dist',   0.2)   # m — centro robot → QR en ALIGNING
        self.declare_parameter('forklift_reach_m',  0.20)   # m — stop en APPROACH_FINAL
        self.declare_parameter('align_lateral_tol', 0.03)   # m — tolerancia lateral
        self.declare_parameter('angle_tol_deg',     4.0)    # ° — tolerancia angular

        # Control directo PD (Fix H3)
        self.declare_parameter('kp_angle',    0.018)   # rad/s por grado de error
        self.declare_parameter('kd_angle',    0.004)   # amortiguación angular
        self.declare_parameter('kp_dist',     0.40)    # m/s por metro de error
        self.declare_parameter('kd_dist',     0.08)    # amortiguación lineal
        self.declare_parameter('max_angular', 0.45)    # rad/s
        self.declare_parameter('max_linear',  0.18)    # m/s

        # Offset de cámara respecto a base_link
        self.declare_parameter('cam_fwd_m',     0.15)
        self.declare_parameter('cam_left_m',    0.07)
        self.declare_parameter('cam_offset_deg', 0.0)

        # Lift
        self.declare_parameter('lift_timeout', 8.0)

        # Timeouts
        self.declare_parameter('align_timeout',    20.0)
        self.declare_parameter('approach_timeout', 15.0)
        self.declare_parameter('search_timeout',   10.0)
        self.declare_parameter('qr_timeout',        2.5)

        # Retroceso post-recolección
        self.declare_parameter('back_away_speed', 0.10)
        self.declare_parameter('back_away_time',  1.8)

        # Barrido RECOVER_SCAN
        self.declare_parameter('scan_range_deg',    30.0)
        self.declare_parameter('scan_speed_dps',    20.0)
        self.declare_parameter('scan_max_attempts',  3)

        # General
        self.declare_parameter('fsm_rate_hz', 20.0)

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
        self._qr_angle       = 0.0    # grados, relativo al heading del robot
        self._qr_dist        = 999.0  # metros, desde la cámara al QR
        self._qr_stamp       = 0.0

        # ── Pose odométrica ───────────────────────────────────────────────
        self._rx  = 0.0
        self._ry  = 0.0
        self._rth = 0.0

        # ── Derivadas para control PD ──────────────────────────────────────
        self._prev_angle_err = 0.0
        self._prev_dist_err  = 0.0
        self._prev_tick_t    = time.monotonic()

        # ── RECOVER_SCAN ──────────────────────────────────────────────────
        self._scan_return_state    = _S.SEARCH_QR
        self._scan_phase           = 'LEFT'
        self._scan_phase_start_yaw = 0.0
        self._scan_attempts        = 0

        # ── Lift ──────────────────────────────────────────────────────────
        self._lift_done_label   = ''
        self._lift_phase        = 'WAIT_LEVEL'
        self._lift_reached_hold = False

        # ── QOS ───────────────────────────────────────────────────────────
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Suscriptores ──────────────────────────────────────────────────
        self.create_subscription(String,   '/collect/trigger', self._cb_trigger,      10)
        self.create_subscription(String,   '/qr/data',         self._cb_qr,           qos_be)
        self.create_subscription(Float32,  '/qr/distance',     self._cb_qr_dist,      qos_be)
        self.create_subscription(Float32,  '/qr/angle',        self._cb_qr_angle,     qos_be)
        self.create_subscription(String,   '/lift_done',       self._cb_lift_done,    10)
        self.create_subscription(Odometry, '/odom',            self._cb_odom,         10)

        # ── Publicadores ──────────────────────────────────────────────────
        self._pub_cmd       = self.create_publisher(Twist,  '/cmd_vel',            10)
        self._pub_lift      = self.create_publisher(String, '/lift_auto',          10)
        self._pub_done      = self.create_publisher(String, '/collect/done',       10)
        self._pub_payload   = self.create_publisher(String, '/collect/qr_payload', 10)
        self._pub_active    = self.create_publisher(Bool,   '/align/active',       10)
        self._pub_nav_pause = self.create_publisher(Bool,   '/nav_pause',          10)

        # ── Timer FSM ─────────────────────────────────────────────────────
        self.create_timer(1.0 / float(self._p('fsm_rate_hz')), self._tick)

        self.get_logger().info(
            'qr_align_node v7 listo (control directo — sin A*)\n'
            f'  align_stop_dist={self._p("align_stop_dist")}m  '
            f'forklift_reach_m={self._p("forklift_reach_m")}m\n'
            f'  kp_angle={self._p("kp_angle")}  kd_angle={self._p("kd_angle")}  '
            f'max_angular={self._p("max_angular")} rad/s\n'
            f'  kp_dist={self._p("kp_dist")}  kd_dist={self._p("kd_dist")}  '
            f'max_linear={self._p("max_linear")} m/s\n'
            f'  cam=[fwd={self._p("cam_fwd_m")}m, left={self._p("cam_left_m")}m]\n'
            f'  QR topics: /qr/data | /qr/distance | /qr/angle'
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

        self._zone        = cmd
        self._lift_cmd, self._lift_expect = _ZONE_LIFT[cmd]
        self._target_payload = ''

        # Fix H1: bypass activo desde el trigger — qr_align es dueño de
        # /cmd_vel durante toda la operación de recolección.
        self._set_vfh_bypass(True)

        self.get_logger().info(
            f'[Collect] Trigger zona="{cmd}" → lift_cmd={self._lift_cmd}')
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
        # Solo procesar /lift_done cuando estamos en HOLD (Fix S1 v5)
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
                self._reset_pd()
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
                    self._stop()
                    self._start_recover_scan(return_to=_S.ALIGNING)
                return

            if self._time_in_state() > self._p('align_timeout'):
                self.get_logger().warn('[ALIGNING] Timeout → ABORT')
                self._transition(_S.ABORT)
                return

            qr_x, qr_y = self._estimate_qr_world_pos()

            # Bearing del centro del robot hacia el QR (Fix G1)
            bearing    = math.atan2(qr_y - self._ry, qr_x - self._rx)
            angle_err  = math.degrees(self._angle_diff(bearing, self._rth))
            dist_robot = math.hypot(qr_x - self._rx, qr_y - self._ry)
            lateral_err = dist_robot * math.sin(math.radians(angle_err))

            aligned     = (abs(angle_err) < self._p('angle_tol_deg')
                           and abs(lateral_err) < self._p('align_lateral_tol'))
            close_enough = dist_robot <= self._p('align_stop_dist')

            if aligned and close_enough:
                self.get_logger().info(
                    f'[ALIGNING] ✔ dist={dist_robot:.3f}m  '
                    f'lateral={lateral_err*100:.1f}cm  '
                    f'angle={angle_err:+.1f}° → APPROACH_FINAL')
                self._stop()
                self._reset_pd()
                self._transition(_S.APPROACH_FINAL)
                return

            # Fix H1/H3: control PD directo — sin A*, sin go_to_goal
            self._pub_cmd.publish(
                self._pd_cmd(angle_err, dist_robot, self._p('align_stop_dist')))

        # ── RECOVER_SCAN ──────────────────────────────────────────────────
        elif s == _S.RECOVER_SCAN:
            self._tick_recover_scan()

        # ── APPROACH_FINAL ────────────────────────────────────────────────
        elif s == _S.APPROACH_FINAL:
            if self._qr_visible():
                qr_x, qr_y = self._estimate_qr_world_pos()
                dist_robot  = math.hypot(qr_x - self._rx, qr_y - self._ry)

                if dist_robot <= self._p('forklift_reach_m'):
                    self.get_logger().info(
                        f'[APPROACH_FINAL] Llegada confirmada '
                        f'dist={dist_robot:.3f}m → HOLD')
                    self._stop()   # Fix G2
                    self._transition(_S.HOLD)
                    return

                # Fix H4: control PD directo hacia el QR
                bearing   = math.atan2(qr_y - self._ry, qr_x - self._rx)
                angle_err = math.degrees(self._angle_diff(bearing, self._rth))
                self._pub_cmd.publish(
                    self._pd_cmd(angle_err, dist_robot, self._p('forklift_reach_m')))
            else:
                # QR no visible en approach: frenar y esperar
                self._stop()

            if self._time_in_state() > self._p('approach_timeout'):
                self.get_logger().warn('[APPROACH_FINAL] Timeout → ABORT')
                self._transition(_S.ABORT)

        # ── HOLD ──────────────────────────────────────────────────────────
        elif s == _S.HOLD:
            if self._time_in_state() < 0.05:
                self._lift_done_label   = ''
                self._lift_phase        = 'WAIT_LEVEL'
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
                if self._lift_done_label == self._lift_expect:
                    self.get_logger().info(
                        f'[HOLD] {self._lift_done_label} confirmado → '
                        f'fase WAIT_HOLD, enviando: hold')
                    self._lift_done_label = ''
                    self._lift_phase      = 'WAIT_HOLD'
                    self._pub_lift.publish(String(data='hold'))

            elif self._lift_phase == 'WAIT_HOLD':
                if self._lift_done_label == 'HOLD':
                    self._lift_reached_hold = True
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
            self._reset_pd()
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
    # CONTROL PD DIRECTO
    # ══════════════════════════════════════════════════════════════════════

    def _pd_cmd(self, angle_err_deg: float, dist: float, target_dist: float) -> Twist:
        """
        Controlador PD directo sobre error angular y error de distancia.

        angle_err_deg — error angular en grados (bearing − heading del robot)
        dist          — distancia actual centro robot → QR (m)
        target_dist   — distancia objetivo (m)

        El término D usa la diferencia respecto al tick anterior;
        si el dt es cero o muy pequeño se ignora la derivada.
        """
        now = time.monotonic()
        dt  = now - self._prev_tick_t
        self._prev_tick_t = now

        # ── Angular PD ────────────────────────────────────────────────────
        d_angle = ((angle_err_deg - self._prev_angle_err) / dt
                   if dt > 1e-4 else 0.0)
        self._prev_angle_err = angle_err_deg

        w = (float(self._p('kp_angle')) * angle_err_deg
             + float(self._p('kd_angle')) * d_angle)
        w = max(-float(self._p('max_angular')),
                min( float(self._p('max_angular')), w))

        # ── Lineal PD — solo si el ángulo ya está razonablemente alineado ─
        dist_err = dist - target_dist
        d_dist   = ((dist_err - self._prev_dist_err) / dt
                    if dt > 1e-4 else 0.0)
        self._prev_dist_err = dist_err

        v = 0.0
        if abs(angle_err_deg) < self._p('angle_tol_deg') * 2.0 and dist_err > 0:
            v = (float(self._p('kp_dist')) * dist_err
                 + float(self._p('kd_dist')) * d_dist)
            v = max(0.0, min(float(self._p('max_linear')), v))

        cmd = Twist()
        cmd.linear.x  = v
        cmd.angular.z = w
        return cmd

    def _reset_pd(self):
        """Reinicia las derivadas del controlador PD."""
        self._prev_angle_err = 0.0
        self._prev_dist_err  = 0.0
        self._prev_tick_t    = time.monotonic()

    # ══════════════════════════════════════════════════════════════════════
    # GEOMETRÍA
    # ══════════════════════════════════════════════════════════════════════

    def _estimate_qr_world_pos(self):
        """
        Estima la posición del QR en coordenadas mundo proyectando desde
        la posición de la cámara (Fix G4).

        Offset cámara respecto a base_link:
          cam_x = rx + cam_fwd·cos(rth) − cam_left·sin(rth)
          cam_y = ry + cam_fwd·sin(rth) + cam_left·cos(rth)

        Bearing de la cámara al QR:
          bearing_cam = rth + qr_angle_rad   (qr_angle ya incluye cam_offset_deg)
        """
        CAM_FWD  = float(self._p('cam_fwd_m'))
        CAM_LEFT = float(self._p('cam_left_m'))

        cam_x = (self._rx
                 + CAM_FWD  * math.cos(self._rth)
                 - CAM_LEFT * math.sin(self._rth))
        cam_y = (self._ry
                 + CAM_FWD  * math.sin(self._rth)
                 + CAM_LEFT * math.cos(self._rth))

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
        self._pub_nav_pause.publish(Bool(data=active))  # pausa go_to_goal + A*
        label = 'ON  (nav pausada, evasión inhibida)' if active else 'OFF (nav reanudada)'
        self.get_logger().info(f'[VFH+] /align/active + /nav_pause → {label}')

    def _transition(self, new_state: str):
        if new_state == self._state:
            return
        self.get_logger().info(f'[FSM] {self._state} → {new_state}')
        self._state       = new_state
        self._state_entry = time.monotonic()

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
        self._qr_stamp          = 0.0
        self._scan_return_state    = _S.SEARCH_QR
        self._scan_phase           = 'LEFT'
        self._scan_phase_start_yaw = 0.0
        self._scan_attempts        = 0
        self._reset_pd()

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Diferencia angular con signo en [−π, π]: a − b."""
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
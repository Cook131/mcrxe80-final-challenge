#!/usr/bin/env python3
"""
qr_collect_node.py — Iolair QR Align + Collect  [v2 — integración A*/GoToGoal]
================================================================================
Cambios respecto a v1
---------------------
  * Nuevo estado NAV_APPROACH: cuando el QR está visible pero lejos, publica
    un goal relativo en /astar/goal (coordenadas mundo) en lugar de cmd_vel
    directo. GoToGoal + A* manejan la ruta completa hasta ~approach_handoff_dist.

  * El PD directo (/cmd_vel) sólo se activa para el tramo fino:
      dist ≤ approach_handoff_dist  →  estados ALIGN / ADVANCE como antes.

  * Suscripción a /odom para conocer pose del robot y convertir offset QR
    (ángulo+distancia) a coordenadas mundo absolutas para el goal.

  * Suscripción a /astar/status para saber cuándo A* terminó (GOAL_REACHED).

  * Nueva lógica de "approach ciego": si el QR se pierde DURANTE ADVANCE
    (muy cerca, robot pasa por encima), se asume que ya llegamos y se pasa
    directamente a LIFT_CMD en lugar de abortar.

Integración con FSM (mission_manager.py)
-----------------------------------------
  SUB:  /collect/trigger     (std_msgs/String)
          "rack"     → nivel de recolección N1  (racks bajos)
          "conveyor" → nivel de recolección N2  (conveyors altos)
          "abort"    → cancela operación activa

  PUB:  /collect/done        (std_msgs/String)
          "SUCCESS"  → pallet recogido, listo para GO2GOAL
          "ABORT"    → operación cancelada o timeout

  SUB:  /aruco/qr            (std_msgs/String)   — payload QR
  SUB:  /aruco/qr/distance   (std_msgs/Float32)  — distancia plano XZ [m]
  SUB:  /aruco/qr/angle      (std_msgs/Float32)  — ángulo horizontal [°], + = derecha

  SUB:  /odom                (nav_msgs/Odometry)  — pose robot en mundo
  SUB:  /astar/status        (std_msgs/String)    — GOAL_REACHED / EXECUTING / ...
  PUB:  /astar/goal          (geometry_msgs/Pose2D) — goal absoluto para A*

  PUB:  /cmd_vel             (geometry_msgs/Twist)
  PUB:  /lift_auto           (std_msgs/String)   — n1 | n2 | hold | down
  SUB:  /lift_done           (std_msgs/String)   — AT_N1 | AT_N2 | HOLD | DOWN

  PUB:  /collect/qr_payload  (std_msgs/String)   — reenvía contenido QR a FSM

Máquina de estados interna
---------------------------
  IDLE
    → NAV_APPROACH  al recibir /collect/trigger si QR visible y dist > handoff
    → ALIGN         al recibir /collect/trigger si QR visible y dist ≤ handoff

  NAV_APPROACH
    → ALIGN         cuando A* publica GOAL_REACHED  (o dist ≤ handoff)
    → IDLE          si se pierde el QR antes del handoff y timeout
    → ABORT         si A* tarda demasiado (nav_approach_timeout)

  ALIGN
    → ADVANCE       cuando |angle| < angle_tol  AND  dist > approach_dist
    → LIFT_CMD      cuando |angle| < angle_tol  AND  dist ≤ approach_dist
    → ABORT         si se pierde el QR (timeout)

  ADVANCE
    → LIFT_CMD      al llegar a dist ≤ approach_dist  AND  |angle| < angle_tol
    → LIFT_CMD      si se pierde el QR (approach ciego — asumimos llegada)
    → ALIGN         si ángulo se desvía > angle_tol * 2

  LIFT_CMD → WAIT_LIFT → EXTRACT → REVERSE → DONE / ABORT

Parámetros ROS2 configurables
------------------------------
  kp_angle              float  0.018    P angular [rad/s / °]
  kd_angle              float  0.004    D angular [rad/s / (°/s)]
  kp_dist               float  0.40     P lineal  [m/s / m]
  kd_dist               float  0.08     D lineal  [m/s / (m/s)]
  angle_tol_deg         float  4.0      Tolerancia angular alineación [°]
  approach_dist         float  0.28     Distancia objetivo de parada [m]
  approach_handoff_dist float  0.80     Distancia donde A* entrega al PD [m]
  dist_tol              float  0.03     Tolerancia de distancia [m]
  cam_offset_deg        float  0.0      Offset angular cámara→base_link [°]
  max_angular           float  0.45     Velocidad angular máx [rad/s]
  max_linear            float  0.18     Velocidad lineal máx [m/s]
  extract_speed         float  0.08     Velocidad de encaje [m/s]
  extract_time          float  0.6      Duración avance de encaje [s]
  reverse_speed         float  0.10     Velocidad de retroceso [m/s]
  reverse_time          float  1.8      Duración retroceso [s]
  qr_timeout            float  2.5      Segundos sin QR antes de pausar [s]
  lift_timeout          float  8.0      Timeout esperando /lift_done [s]
  nav_approach_timeout  float  30.0     Timeout máximo en NAV_APPROACH [s]
  fsm_rate_hz           float  20.0     Frecuencia del tick interno [Hz]
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
    IDLE         = 'IDLE'
    NAV_APPROACH = 'NAV_APPROACH'   # A* lleva al robot hasta handoff_dist
    ALIGN        = 'ALIGN'
    ADVANCE      = 'ADVANCE'
    LIFT_CMD     = 'LIFT_CMD'
    WAIT_LIFT    = 'WAIT_LIFT'
    EXTRACT      = 'EXTRACT'
    REVERSE      = 'REVERSE'
    DONE         = 'DONE'
    ABORT        = 'ABORT'


# Mapa zona → comando lift y estado esperado en /lift_done
_ZONE_LIFT = {
    'rack':     ('n1', 'AT_N1'),
    'conveyor': ('n2', 'AT_N2'),
}


# ══════════════════════════════════════════════════════════════════════════════
class QRCollectNode(Node):

    def __init__(self):
        super().__init__('qr_collect_node')

        # ── Parámetros ────────────────────────────────────────────────────
        self.declare_parameter('kp_angle',              0.018)
        self.declare_parameter('kd_angle',              0.004)
        self.declare_parameter('kp_dist',               0.40)
        self.declare_parameter('kd_dist',               0.08)
        self.declare_parameter('angle_tol_deg',         4.0)
        self.declare_parameter('approach_dist',         0.28)
        self.declare_parameter('approach_handoff_dist', 0.80)   # A* → PD
        self.declare_parameter('dist_tol',              0.03)
        self.declare_parameter('cam_offset_deg',        0.0)
        self.declare_parameter('max_angular',           0.45)
        self.declare_parameter('max_linear',            0.18)
        self.declare_parameter('extract_speed',         0.08)
        self.declare_parameter('extract_time',          0.6)
        self.declare_parameter('reverse_speed',         0.10)
        self.declare_parameter('reverse_time',          1.8)
        self.declare_parameter('qr_timeout',            2.5)
        self.declare_parameter('lift_timeout',          8.0)
        self.declare_parameter('nav_approach_timeout',  30.0)
        self.declare_parameter('fsm_rate_hz',           20.0)

        self._p = lambda n: self.get_parameter(n).value

        # ── Estado FSM interno ────────────────────────────────────────────
        self._state        = _S.IDLE
        self._prev_state   = None
        self._state_entry  = time.monotonic()

        self._zone         = ''
        self._lift_cmd     = ''
        self._lift_expect  = ''

        # ── Datos de percepción ───────────────────────────────────────────
        self._qr_payload   = ''
        self._qr_angle     = 0.0    # grados, + = derecha
        self._qr_dist      = 999.0  # metros
        self._qr_stamp     = 0.0

        # ── Pose del robot (de /odom) ─────────────────────────────────────
        self._robot_x  = 0.0
        self._robot_y  = 0.0
        self._robot_th = 0.0        # yaw en radianes

        # ── Estado A* ─────────────────────────────────────────────────────
        self._astar_status = ''     # último string de /astar/status

        # ── Control PD — derivada ─────────────────────────────────────────
        self._prev_angle_err = 0.0
        self._prev_dist_err  = 0.0
        self._prev_ctrl_t    = time.monotonic()

        # ── Estado del lift ───────────────────────────────────────────────
        self._lift_done_label = ''

        # ── QOS sensor (BEST_EFFORT para /aruco/*) ────────────────────────
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Suscriptores ──────────────────────────────────────────────────
        self.create_subscription(String,  '/collect/trigger',   self._cb_trigger,    10)
        self.create_subscription(String,  '/aruco/qr',          self._cb_qr,         qos_be)
        self.create_subscription(Float32, '/aruco/qr/distance', self._cb_qr_dist,    qos_be)
        self.create_subscription(Float32, '/aruco/qr/angle',    self._cb_qr_angle,   qos_be)
        self.create_subscription(String,  '/lift_done',         self._cb_lift_done,  10)
        self.create_subscription(Odometry,'/odom',              self._cb_odom,       10)
        self.create_subscription(String,  '/astar/status',      self._cb_astar_status, 10)

        # ── Publicadores ──────────────────────────────────────────────────
        self._pub_cmd     = self.create_publisher(Twist,  '/cmd_vel',            10)
        self._pub_lift    = self.create_publisher(String, '/lift_auto',          10)
        self._pub_done    = self.create_publisher(String, '/collect/done',       10)
        self._pub_payload = self.create_publisher(String, '/collect/qr_payload', 10)
        self._pub_goal    = self.create_publisher(Pose2D, '/astar/goal',         10)
        self._pub_active  = self.create_publisher(Bool,   '/align/active',     10)

        # ── Timer FSM ─────────────────────────────────────────────────────
        rate = float(self._p('fsm_rate_hz'))
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            'qr_collect_node v2 listo (A*/GoToGoal integrado)\n'
            f'  handoff_dist={self._p("approach_handoff_dist")}m  '
            f'approach_dist={self._p("approach_dist")}m  '
            f'angle_tol={self._p("angle_tol_deg")}°'
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
                f'[Collect] Trigger ignorado — estado actual: {self._state}')
            return

        self._zone       = cmd
        self._lift_cmd, self._lift_expect = _ZONE_LIFT[cmd]
        self.get_logger().info(
            f'[Collect] Trigger zona="{cmd}" → lift_cmd={self._lift_cmd}')

        # Decidir si el approach es largo (A*) o ya estamos cerca (PD directo)
        if self._qr_visible() and self._qr_dist > self._p('approach_handoff_dist'):
            self._publish_approach_goal()
            self._transition(_S.NAV_APPROACH)
        else:
            self._transition(_S.ALIGN)
        # Notificar al VFH que el align tiene el control fino
        self._pub_active.publish(Bool(data=True))
        self.get_logger().info('[Collect] /align/active → True (VFH bypass ON)')

    def _cb_qr(self, msg: String):
        payload = msg.data.strip()
        if payload:
            if payload != self._qr_payload:
                self.get_logger().info(f'[Percepción] QR: {payload}')
                self._qr_payload = payload
                self._pub_payload.publish(String(data=payload))
            self._qr_stamp = time.monotonic()

    def _cb_qr_dist(self, msg: Float32):
        self._qr_dist  = float(msg.data)
        self._qr_stamp = time.monotonic()

    def _cb_qr_angle(self, msg: Float32):
        raw = float(msg.data)
        self._qr_angle = raw + float(self._p('cam_offset_deg'))
        self._qr_stamp = time.monotonic()

    def _cb_lift_done(self, msg: String):
        label = msg.data.strip()
        if label:
            self.get_logger().info(f'[Lift] /lift_done: {label}')
            self._lift_done_label = label

    def _cb_odom(self, msg: Odometry):
        self._robot_x  = msg.pose.pose.position.x
        self._robot_y  = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self._robot_th = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def _cb_astar_status(self, msg: String):
        self._astar_status = msg.data.strip()

    # ══════════════════════════════════════════════════════════════════════
    # HELPERS A* / POSE
    # ══════════════════════════════════════════════════════════════════════

    def _publish_approach_goal(self):
        """
        Convierte la detección actual (ángulo + distancia en frame cámara)
        a coordenadas mundo absolutas y las publica en /astar/goal.

        El goal se coloca a (approach_handoff_dist + dist_tol) metros del QR
        en la dirección del robot, para que A* lleve al robot hasta el punto
        de handoff donde el PD toma el control fino.
        """
        angle_rad = math.radians(self._qr_angle)   # ángulo lateral en cámara
        # Dirección absoluta al QR en frame mundo
        bearing_world = self._robot_th + angle_rad

        # Posición estimada del QR en mundo
        qr_wx = self._robot_x + self._qr_dist * math.cos(bearing_world)
        qr_wy = self._robot_y + self._qr_dist * math.sin(bearing_world)

        # Goal: detenerse a handoff_dist delante del QR (mirando desde el robot)
        handoff = self._p('approach_handoff_dist')
        goal_x = qr_wx - handoff * math.cos(bearing_world)
        goal_y = qr_wy - handoff * math.sin(bearing_world)

        goal_msg = Pose2D()
        goal_msg.x = goal_x
        goal_msg.y = goal_y
        self._pub_goal.publish(goal_msg)

        self.get_logger().info(
            f'[NAV_APPROACH] Goal publicado en A*: '
            f'({goal_x:.3f}, {goal_y:.3f})  '
            f'[QR estimado: ({qr_wx:.3f}, {qr_wy:.3f}), '
            f'dist={self._qr_dist:.2f}m, ang={self._qr_angle:+.1f}°]'
        )

    # ══════════════════════════════════════════════════════════════════════
    # FSM TICK
    # ══════════════════════════════════════════════════════════════════════

    def _tick(self):
        s = self._state

        # ── IDLE ──────────────────────────────────────────────────────────
        if s == _S.IDLE:
            return

        # ── NAV_APPROACH ──────────────────────────────────────────────────
        elif s == _S.NAV_APPROACH:
            # Timeout global por si A* se cuelga
            if self._time_in_state() > self._p('nav_approach_timeout'):
                self.get_logger().error(
                    f'[NAV_APPROACH] Timeout {self._p("nav_approach_timeout")}s → ABORT')
                self._transition(_S.ABORT)
                return

            # QR visible: actualizar el goal si la distancia cambió mucho
            # (el robot se movió, el QR se vio desde otro ángulo)
            if self._qr_visible():
                if self._qr_dist <= self._p('approach_handoff_dist'):
                    # Ya estamos en zona de handoff — A* ya no es necesario
                    self.get_logger().info(
                        f'[NAV_APPROACH] Handoff alcanzado '
                        f'dist={self._qr_dist:.3f}m → ALIGN')
                    self._transition(_S.ALIGN)
                    return
                # Else: todavía lejos, A* sigue navegando. No republiquemos
                # el goal en cada tick para no saturar el planner.
            else:
                # QR no visible durante NAV_APPROACH es tolerable un momento
                if (time.monotonic() - self._qr_stamp) > self._p('qr_timeout') * 2:
                    self.get_logger().warn('[NAV_APPROACH] QR perdido demasiado tiempo → ABORT')
                    self._transition(_S.ABORT)
                    return

            # Esperar confirmación de A*
            if self._astar_status == 'GOAL_REACHED':
                self.get_logger().info('[NAV_APPROACH] A* GOAL_REACHED → ALIGN')
                self._transition(_S.ALIGN)

        # ── ALIGN ─────────────────────────────────────────────────────────
        elif s == _S.ALIGN:
            if not self._qr_visible():
                self._stop()
                if self._time_in_state() > self._p('qr_timeout') * 3:
                    self.get_logger().warn('[ALIGN] QR perdido — ABORT')
                    self._transition(_S.ABORT)
                return

            angle_err = self._qr_angle
            dist_err  = self._qr_dist - self._p('approach_dist')

            cmd = self._pd_control(angle_err, dist_err)
            self._pub_cmd.publish(cmd)

            aligned = (
                abs(angle_err) < self._p('angle_tol_deg')
                and abs(dist_err)  > self._p('dist_tol')
            )
            arrived = (
                abs(angle_err) < self._p('angle_tol_deg')
                and abs(dist_err)  <= self._p('dist_tol')
            )

            if arrived:
                self._stop()
                self.get_logger().info(
                    f'[ALIGN] Alineado y en distancia '
                    f'angle={angle_err:+.1f}° dist={self._qr_dist:.3f}m → LIFT_CMD')
                self._transition(_S.LIFT_CMD)
            elif aligned:
                self._stop()
                self.get_logger().info(
                    f'[ALIGN] Alineado, avanzando '
                    f'angle={angle_err:+.1f}° dist={self._qr_dist:.3f}m → ADVANCE')
                self._transition(_S.ADVANCE)

        # ── ADVANCE ───────────────────────────────────────────────────────
        elif s == _S.ADVANCE:
            if not self._qr_visible():
                self._stop()
                if self._time_in_state() > self._p('qr_timeout'):
                    # QR perdido por approach ciego: asumimos que llegamos
                    self.get_logger().info(
                        '[ADVANCE] QR perdido — approach ciego asumido como llegada → LIFT_CMD')
                    self._transition(_S.LIFT_CMD)
                return

            angle_err = self._qr_angle
            dist_err  = self._qr_dist - self._p('approach_dist')

            if abs(angle_err) > self._p('angle_tol_deg') * 2.0:
                self._stop()
                self.get_logger().info(
                    f'[ADVANCE] Desvío angular {angle_err:+.1f}° → ALIGN')
                self._transition(_S.ALIGN)
                return

            cmd = self._pd_control(angle_err, dist_err, angular_scale=0.5)
            self._pub_cmd.publish(cmd)

            if abs(dist_err) <= self._p('dist_tol'):
                self._stop()
                self.get_logger().info(
                    f'[ADVANCE] Distancia alcanzada {self._qr_dist:.3f}m → LIFT_CMD')
                self._transition(_S.LIFT_CMD)

        # ── LIFT_CMD ──────────────────────────────────────────────────────
        elif s == _S.LIFT_CMD:
            if self._time_in_state() < 0.1:
                self.get_logger().info(
                    f'[Lift] Enviando comando: {self._lift_cmd}')
                self._pub_lift.publish(String(data=self._lift_cmd))
                self._lift_done_label = ''
            self._transition(_S.WAIT_LIFT)

        # ── WAIT_LIFT ─────────────────────────────────────────────────────
        elif s == _S.WAIT_LIFT:
            if self._lift_done_label == self._lift_expect:
                self.get_logger().info(
                    f'[Lift] Confirmado: {self._lift_done_label} → EXTRACT')
                self._transition(_S.EXTRACT)
            elif self._time_in_state() > self._p('lift_timeout'):
                self.get_logger().error(
                    f'[Lift] Timeout esperando {self._lift_expect} → ABORT')
                self._transition(_S.ABORT)

        # ── EXTRACT ───────────────────────────────────────────────────────
        elif s == _S.EXTRACT:
            elapsed = self._time_in_state()
            if elapsed < self._p('extract_time'):
                cmd = Twist()
                cmd.linear.x = self._p('extract_speed')
                self._pub_cmd.publish(cmd)
            else:
                self._stop()
                self.get_logger().info('[EXTRACT] Encaje completado → REVERSE')
                self._pub_lift.publish(String(data='hold'))
                self._transition(_S.REVERSE)

        # ── REVERSE ───────────────────────────────────────────────────────
        elif s == _S.REVERSE:
            elapsed = self._time_in_state()
            if elapsed < self._p('reverse_time'):
                cmd = Twist()
                cmd.linear.x = -self._p('reverse_speed')
                self._pub_cmd.publish(cmd)
            else:
                self._stop()
                self.get_logger().info('[REVERSE] Pallet asegurado → DONE')
                self._transition(_S.DONE)

        # ── DONE ──────────────────────────────────────────────────────────
        elif s == _S.DONE:
            self._stop()
            self.get_logger().info(
                f'[Collect] ✅ SUCCESS — payload="{self._qr_payload}" '
                f'zona="{self._zone}"')
            self._pub_active.publish(Bool(data=False))
            self.get_logger().info('[Collect] /align/active → False (VFH bypass OFF)')
            self._pub_done.publish(String(data='SUCCESS'))
            self._reset()
            self._transition(_S.IDLE)

        # ── ABORT ─────────────────────────────────────────────────────────
        elif s == _S.ABORT:
            self._stop()
            if self._lift_cmd:
                self._pub_lift.publish(String(data='down'))
            self.get_logger().warn('[Collect] ❌ ABORT')
            self._pub_active.publish(Bool(data=False))
            self.get_logger().info('[Collect] /align/active → False (VFH bypass OFF)')
            self._pub_done.publish(String(data='ABORT'))
            self._reset()
            self._transition(_S.IDLE)

    # ══════════════════════════════════════════════════════════════════════
    # CONTROL PD
    # ══════════════════════════════════════════════════════════════════════

    def _pd_control(
        self,
        angle_err: float,
        dist_err: float,
        angular_scale: float = 1.0,
    ) -> Twist:
        """
        Control PD desacoplado.
          angle_err : error angular en grados (+ = derecha)
          dist_err  : error de distancia en metros (+ = lejos)
          angular_scale: atenuación angular durante avance (evitar zigzag)
        """
        now = time.monotonic()
        dt  = max(now - self._prev_ctrl_t, 1e-3)

        d_angle = (angle_err - self._prev_angle_err) / dt
        d_dist  = (dist_err  - self._prev_dist_err)  / dt

        kp_a = self._p('kp_angle');  kd_a = self._p('kd_angle')
        kp_d = self._p('kp_dist');   kd_d = self._p('kd_dist')

        # ángulo + = derecha → angular.z negativo (giro antihorario)
        angular_z = -(kp_a * angle_err + kd_a * d_angle) * angular_scale
        linear_x  =   kp_d * dist_err  + kd_d * d_dist

        max_w = self._p('max_angular');  max_v = self._p('max_linear')
        angular_z = max(-max_w, min(max_w, angular_z))
        linear_x  = max(0.0, min(max_v, linear_x))   # no retroceder en PD

        self._prev_angle_err = angle_err
        self._prev_dist_err  = dist_err
        self._prev_ctrl_t    = now

        cmd = Twist()
        cmd.angular.z = angular_z
        cmd.linear.x  = linear_x
        return cmd

    # ══════════════════════════════════════════════════════════════════════
    # UTILIDADES
    # ══════════════════════════════════════════════════════════════════════

    def _qr_visible(self) -> bool:
        return (
            self._qr_payload != ''
            and (time.monotonic() - self._qr_stamp) < self._p('qr_timeout')
        )

    def _stop(self):
        self._pub_cmd.publish(Twist())

    def _time_in_state(self) -> float:
        return time.monotonic() - self._state_entry

    def _transition(self, new_state: str):
        if new_state != self._state:
            self.get_logger().info(f'[FSM] {self._state} → {new_state}')
            self._prev_state  = self._state
            self._state       = new_state
            self._state_entry = time.monotonic()
            self._prev_angle_err = 0.0
            self._prev_dist_err  = 0.0
            self._prev_ctrl_t    = time.monotonic()

    def _reset(self):
        self._zone            = ''
        self._lift_cmd        = ''
        self._lift_expect     = ''
        self._lift_done_label = ''
        self._qr_payload      = ''
        self._qr_angle        = 0.0
        self._qr_dist         = 999.0
        self._astar_status    = ''


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = QRCollectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
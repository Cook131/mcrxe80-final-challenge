#!/usr/bin/env python3
"""
qr_aligner_node.py — Puzzlebot / Iolair
════════════════════════════════════════════════════════════════
Alinea el robot frente al QR y coordina el lifter.

Geometría confirmada desde map_pista.pgm:
───────────────────────────────────────────────────────────────
  CONVEYOR  x∈[-2.81,-1.60]  cualquier y
            → cara +X  → alinear Y,  heading =  0.0 rad
            → goal_x = qr_x + APPROACH_DIST

  RACK_AZUL_SUR  x∈[-1.70,-0.35],  y∈[-1.0,-0.4]
            → cara +Y  → alinear X,  heading = +π/2
            → goal_y = qr_y + APPROACH_DIST

  RACK_AZUL_NOR  x∈[-1.70,-0.35],  y∈[0.2, 0.8]
            → cara -Y  → alinear X,  heading = -π/2
            → goal_y = qr_y - APPROACH_DIST

  RACK_AMARILLO  x∈[0.20, 0.70]   cualquier y
            → cara -X  → alinear Y,  heading = ±π
            → goal_x = qr_x - APPROACH_DIST

FSM:
  IDLE → ALIGN_PERP → FACE_QR → SEND_LIFT → APPROACH → DONE → IDLE

  ALIGN_PERP : mueve el robot en el eje paralelo a la pared del QR
               hasta que su posición perpendicular coincide con qr_perp
  FACE_QR    : gira en su lugar al heading_target
  SEND_LIFT  : publica /lift_auto (n1=rack, n2=conveyor), espera /lift_done
  APPROACH   : go-to-goal PD hacia el punto a APPROACH_DIST de la pared

Tópicos:
  SUB  /qr/world_pos      geometry_msgs/PointStamped
  SUB  /collect/trigger   std_msgs/String   'conveyor' | 'rack'
  SUB  /odom              nav_msgs/Odometry
  SUB  /lift_done         std_msgs/Bool
  PUB  /cmd_vel           geometry_msgs/Twist
  PUB  /lift_auto         std_msgs/String   'n1' | 'n2'
  PUB  /align/active      std_msgs/Bool     True mientras activo (pausa VFH+)
  PUB  /align/done        std_msgs/Bool     True al terminar
"""

import math

import rclpy
import rclpy.duration
from rclpy.node import Node

from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg      import Odometry
from std_msgs.msg      import Bool, String


# ════════════════════════════════════════════════════════════════
# Parámetros de tuning
# ════════════════════════════════════════════════════════════════
APPROACH_DIST = 0.20    # m — distancia a la pared donde se detiene

ALIGN_TOL    = 0.03     # m   — tolerancia alineación perpendicular
HEADING_TOL  = 0.05     # rad — tolerancia orientación
GOAL_TOL     = 0.03     # m   — tolerancia llegada al goal

KP_ALIGN     = 0.6      # P alineación perpendicular
KD_ALIGN     = 0.08     # D alineación perpendicular
KP_TH        = 1.2      # P rotación
KP_GTG       = 0.7      # P go-to-goal
KD_GTG       = 0.10     # D go-to-goal
KP_HEAD      = 1.0      # P corrección heading durante approach

MAX_LIN      = 0.18     # m/s
MAX_ANG      = 0.8      # rad/s
MIN_LIN      = 0.04     # m/s

LIFT_TIMEOUT_S = 10.0   # s — timeout esperando /lift_done

# Lift command por trigger
LIFT_CMD = {'rack': 'n1', 'conveyor': 'n2'}

# ── Regiones del mapa (confirmadas de map_pista.pgm) ────────────
# Cada región define: eje de alineación, heading, lado del goal
REGIONS = [
    # nombre          x_range           y_range         align  heading   goal_side
    ('CONVEYOR',    (-2.81, -1.60),  (-1.84,  1.84),   'Y',   0.0,      +1),  # goal en +X
    ('RACK_AZ_SUR', (-1.70, -0.35), (-1.00, -0.40),   'X',  +math.pi/2, +1),  # goal en +Y
    ('RACK_AZ_NOR', (-1.70, -0.35), ( 0.20,  0.80),   'X',  -math.pi/2, -1),  # goal en -Y
    ('RACK_AMAR',   ( 0.20,  0.70), (-1.84,  1.84),   'Y',   math.pi,   -1),  # goal en -X
]
# align='Y' → robot se mueve en Y hasta robot_y≈qr_y,  goal en X
# align='X' → robot se mueve en X hasta robot_x≈qr_x,  goal en Y
# goal_side: +1 significa goal = qr_perp + APPROACH_DIST
#            -1 significa goal = qr_perp - APPROACH_DIST

# ════════════════════════════════════════════════════════════════

def _norm(a: float) -> float:
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _classify_qr(qx: float, qy: float):
    """Devuelve la región que contiene (qx,qy) o None."""
    for name, (xmn,xmx), (ymn,ymx), align, heading, goal_side in REGIONS:
        if xmn <= qx <= xmx and ymn <= qy <= ymx:
            return name, align, heading, goal_side
    return None, None, None, None


# ════════════════════════════════════════════════════════════════
class QRAligner(Node):

    IDLE      = 'IDLE'
    ALIGN     = 'ALIGN'       # mover en eje paralelo a la pared
    FACE_QR   = 'FACE_QR'
    SEND_LIFT = 'SEND_LIFT'
    APPROACH  = 'APPROACH'
    DONE      = 'DONE'

    def __init__(self):
        super().__init__('qr_aligner')

        # ── Estado robot ──────────────────────────────────────
        self.robot_x     = 0.0
        self.robot_y     = 0.0
        self.robot_theta = 0.0

        # ── Datos QR ──────────────────────────────────────────
        self.qr_x    = None
        self.qr_y    = None
        self.trigger = None       # 'rack' | 'conveyor'

        # ── Geometría calculada al inicio ─────────────────────
        self.align_axis     = None  # 'X' | 'Y'
        self.heading_target = None  # rad
        self.goal_x         = None  # punto approach
        self.goal_y         = None
        self.region_name    = None

        # ── FSM ───────────────────────────────────────────────
        self.state       = self.IDLE
        self._lift_done  = False
        self._lift_timer = None
        self._lift_poll  = None

        # ── PD state ──────────────────────────────────────────
        self._prev_err_align = 0.0
        self._prev_dist_gtg  = 0.0

        # ── Subs / Pubs ───────────────────────────────────────
        self.create_subscription(
            PointStamped, '/qr/world_pos',    self._cb_world_pos, 10)
        self.create_subscription(
            String,       '/collect/trigger', self._cb_trigger,   10)
        self.create_subscription(
            Odometry,     '/odom',            self._cb_odom,      10)
        self.create_subscription(
            Bool,         '/lift_done',       self._cb_lift_done, 10)

        self.pub_cmd    = self.create_publisher(Twist,  '/cmd_vel',      10)
        self.pub_lift   = self.create_publisher(String, '/lift_auto',    10)
        self.pub_active = self.create_publisher(Bool,   '/align/active', 10)
        self.pub_done   = self.create_publisher(Bool,   '/align/done',   10)

        self.create_timer(0.05, self._control_loop)   # 20 Hz

        self.get_logger().info(
            'QR Aligner listo\n'
            '  CONVEYOR  → align Y, heading  0°,   goal +X\n'
            '  RACK_SUR  → align X, heading +90°,  goal +Y\n'
            '  RACK_NOR  → align X, heading -90°,  goal -Y\n'
            '  RACK_AMAR → align Y, heading ±180°, goal -X'
        )

    # ════════════════════════════════════════════════════════════
    # Callbacks
    # ════════════════════════════════════════════════════════════

    def _cb_odom(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_theta = math.atan2(
            2*(q.w*q.z + q.x*q.y),
            1 - 2*(q.y*q.y + q.z*q.z))

    def _cb_world_pos(self, msg: PointStamped):
        self.qr_x = msg.point.x
        self.qr_y = msg.point.y
        if self.state == self.IDLE and self.trigger is not None:
            self._start()

    def _cb_trigger(self, msg: String):
        self.trigger = msg.data
        if self.state == self.IDLE and self.qr_x is not None:
            self._start()

    def _cb_lift_done(self, msg: Bool):
        if msg.data:
            self._lift_done = True

    # ════════════════════════════════════════════════════════════
    # Inicio — clasificar QR y calcular goal
    # ════════════════════════════════════════════════════════════

    def _start(self):
        region, align, heading, goal_side = _classify_qr(self.qr_x, self.qr_y)

        if region is None:
            self.get_logger().warn(
                f'QR ({self.qr_x:.2f},{self.qr_y:.2f}) fuera de toda región conocida, ignorando')
            return

        self.region_name    = region
        self.align_axis     = align
        self.heading_target = heading

        # Para RACK_AMARILLO el heading depende de qué lado está el robot
        if region == 'RACK_AMAR':
            # El robot debe mirar hacia -X; ajustar a +π o -π según posición actual
            self.heading_target = math.pi if self.robot_x > self.qr_x else -math.pi

        if align == 'Y':
            # Robot se alinea en Y → goal en X a APPROACH_DIST de qr_x
            self.goal_x = self.qr_x + goal_side * APPROACH_DIST
            self.goal_y = self.qr_y
        else:
            # Robot se alinea en X → goal en Y a APPROACH_DIST de qr_y
            self.goal_x = self.qr_x
            self.goal_y = self.qr_y + goal_side * APPROACH_DIST

        self._prev_err_align = 0.0
        self._prev_dist_gtg  = 0.0
        self._lift_done      = False

        self.pub_active.publish(Bool(data=True))
        self._transition(self.ALIGN)

        self.get_logger().info(
            f'[QR Aligner] START  region={region}  trigger={self.trigger}\n'
            f'  QR: ({self.qr_x:.2f}, {self.qr_y:.2f})\n'
            f'  align_axis={align}  heading={math.degrees(self.heading_target):.1f}°\n'
            f'  goal: ({self.goal_x:.2f}, {self.goal_y:.2f})'
        )

    # ════════════════════════════════════════════════════════════
    # Control loop 20 Hz
    # ════════════════════════════════════════════════════════════

    def _control_loop(self):
        if   self.state == self.IDLE:      return
        elif self.state == self.ALIGN:     self._do_align()
        elif self.state == self.FACE_QR:   self._do_face_qr()
        elif self.state == self.APPROACH:  self._do_approach()
        # SEND_LIFT y DONE se manejan por eventos, no por loop

    # ── ALIGN ─────────────────────────────────────────────────
    def _do_align(self):
        """
        Mueve el robot en el eje paralelo a la cara del QR:
          align='Y' → mueve el robot hasta robot_y ≈ qr_y
          align='X' → mueve el robot hasta robot_x ≈ qr_x

        Usa go-to-goal 2D hacia un punto intermedio que solo
        difiere del robot en el eje de alineación.
        """
        if self.align_axis == 'Y':
            err   = self.qr_y - self.robot_y
            # Punto intermedio: mismo X del robot, Y del QR
            int_x = self.robot_x
            int_y = self.qr_y
        else:
            err   = self.qr_x - self.robot_x
            int_x = self.qr_x
            int_y = self.robot_y

        derr = (err - self._prev_err_align) / 0.05
        self._prev_err_align = err

        if abs(err) < ALIGN_TOL:
            self._stop()
            self.get_logger().info(
                f'[ALIGN] ✓  err_{self.align_axis}={err:.3f} m')
            self._transition(self.FACE_QR)
            return

        # Ángulo al punto intermedio
        dx = int_x - self.robot_x
        dy = int_y - self.robot_y
        angle_to_int = math.atan2(dy, dx)
        err_th = _norm(angle_to_int - self.robot_theta)

        # Si el error angular es grande, girar primero
        if abs(err_th) > 0.20:
            twist = Twist()
            twist.angular.z = _clamp(KP_TH * err_th, -MAX_ANG, MAX_ANG)
            self.pub_cmd.publish(twist)
            return

        dist  = abs(err)
        v_lin = _clamp(KP_ALIGN * dist + KD_ALIGN * abs(derr), MIN_LIN, MAX_LIN)
        v_ang = _clamp(KP_TH * err_th, -MAX_ANG, MAX_ANG)

        twist = Twist()
        twist.linear.x  = v_lin
        twist.angular.z = v_ang
        self.pub_cmd.publish(twist)

    # ── FACE_QR ───────────────────────────────────────────────
    def _do_face_qr(self):
        err_th = _norm(self.heading_target - self.robot_theta)

        if abs(err_th) < HEADING_TOL:
            self._stop()
            self.get_logger().info(
                f'[FACE_QR] ✓  θ={math.degrees(self.robot_theta):.1f}°'
                f'  target={math.degrees(self.heading_target):.1f}°')
            self._transition(self.SEND_LIFT)
            return

        twist = Twist()
        twist.angular.z = _clamp(KP_TH * err_th, -MAX_ANG, MAX_ANG)
        self.pub_cmd.publish(twist)

    # ── APPROACH ─────────────────────────────────────────────
    def _do_approach(self):
        dx   = self.goal_x - self.robot_x
        dy   = self.goal_y - self.robot_y
        dist = math.sqrt(dx*dx + dy*dy)

        derr = (dist - self._prev_dist_gtg) / 0.05
        self._prev_dist_gtg = dist

        if dist < GOAL_TOL:
            self._stop()
            self.get_logger().info(
                f'[APPROACH] ✓  goal ({self.goal_x:.2f},{self.goal_y:.2f}) alcanzado')
            self._transition(self.DONE)
            return

        angle_to_goal = math.atan2(dy, dx)
        err_th = _norm(angle_to_goal - self.robot_theta)

        v_lin  = _clamp(KP_GTG * dist + KD_GTG * abs(derr), MIN_LIN, MAX_LIN)
        v_lin *= max(0.0, 1.0 - abs(err_th) / math.pi)
        v_ang  = _clamp(KP_HEAD * err_th, -MAX_ANG, MAX_ANG)

        twist = Twist()
        twist.linear.x  = v_lin
        twist.angular.z = v_ang
        self.pub_cmd.publish(twist)

    # ════════════════════════════════════════════════════════════
    # Transiciones FSM
    # ════════════════════════════════════════════════════════════

    def _transition(self, new_state: str):
        self.get_logger().info(f'[FSM] {self.state} → {new_state}')
        self.state = new_state

        if new_state == self.SEND_LIFT:
            self._lift_done = False
            cmd = LIFT_CMD.get(self.trigger, 'n1')
            self.pub_lift.publish(String(data=cmd))
            self.get_logger().info(
                f'[SEND_LIFT] /lift_auto = {cmd}  (trigger={self.trigger})')
            # Timeout por si lift_done nunca llega
            self._lift_timer = self.create_timer(
                LIFT_TIMEOUT_S, self._lift_timeout)
            # Polling cada 100 ms
            self._lift_poll = self.create_timer(0.1, self._poll_lift)

        elif new_state == self.DONE:
            self.pub_active.publish(Bool(data=False))
            self.pub_done.publish(Bool(data=True))
            self.get_logger().info(
                f'[DONE] ✓ ciclo completo  region={self.region_name}')
            # Reset
            self.qr_x        = None
            self.qr_y        = None
            self.trigger     = None
            self.region_name = None
            self.state       = self.IDLE

    def _poll_lift(self):
        """Avanza a APPROACH cuando lift_done llega."""
        if self.state != self.SEND_LIFT:
            self._cancel_lift_timers()
            return
        if self._lift_done:
            self._cancel_lift_timers()
            self._transition(self.APPROACH)

    def _lift_timeout(self):
        if self.state == self.SEND_LIFT:
            self.get_logger().warn(
                f'[SEND_LIFT] timeout {LIFT_TIMEOUT_S}s → avanzando sin confirmación')
            self._cancel_lift_timers()
            self._transition(self.APPROACH)

    def _cancel_lift_timers(self):
        for t in [self._lift_timer, self._lift_poll]:
            if t is not None:
                t.cancel()
        self._lift_timer = None
        self._lift_poll  = None

    # ════════════════════════════════════════════════════════════
    # Helpers
    # ════════════════════════════════════════════════════════════

    def _stop(self):
        self.pub_cmd.publish(Twist())


# ════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = QRAligner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
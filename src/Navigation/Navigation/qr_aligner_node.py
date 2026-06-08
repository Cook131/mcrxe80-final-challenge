#!/usr/bin/env python3
"""
qr_aligner_node.py — Puzzlebot / Iolair  v11
════════════════════════════════════════════════════════════════
FLUJO:
  IDLE
    → espera GOAL_REACHED (A*) + /collect/trigger

  SWEEP
    → gira en eje buscando QR
    → al detectar: congela (qr_x, qr_y), calcula yaw_tgt y stop_point
    → publica /align/active = True  (toma control total)

  SEND_LIFT_CMD
    → rack → n1 / conveyor → n2
    → espera AT_N1/AT_N2 (o timeout)

  FACE_WALL
    → giro en sitio al yaw_tgt:
        rack     → π  (mirar -X, siempre)
        conveyor → ±π/2 o 0/π según lado relativo del QR

  MOVE_CLOSE
    → avanza recto (yaw fijo) hasta stop_point (a STOP_DIST del QR)

  SEND_HOLD
    → publica /lift_auto = 'hold'
    → espera HOLD (o timeout)

  BACK_AWAY
    → retrocede BACK_DIST metros

  DONE → IDLE  (/align/active = False)

Arranque manual:
  ros2 topic pub --once /align/force_start std_msgs/msg/String "data: rack"
"""

import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg      import Odometry
from std_msgs.msg      import Bool, String


# ════════════════════════════════════════════════════════════════
# Parámetros
# ════════════════════════════════════════════════════════════════

STOP_DIST  = 0.10   # m — distancia final al QR
BACK_DIST  = 0.15   # m — retroceso

GOAL_TOL    = 0.02  # m
HEADING_TOL = 0.05  # rad

KP_LIN = 0.8;  MIN_LIN = 0.04;  MAX_LIN = 0.18
KP_ANG = 1.5;  MAX_ANG = 0.8
BACK_SPD = 0.08  # m/s

SWEEP_SPD     = 0.35
SWEEP_MAX_RAD = math.pi

GOAL_REACHED_WINDOW_S = 30.0
MAX_RETRIES = 4

SWEEP_TIMEOUT_S = 25.0
FACE_TIMEOUT_S  = 8.0
MOVE_TIMEOUT_S  = 20.0
LIFT_TIMEOUT_S  = 12.0
HOLD_TIMEOUT_S  = 8.0
BACK_TIMEOUT_S  = BACK_DIST / BACK_SPD + 4.0

LIFT_LV_LABELS = {'AT_N1', 'AT_N2'}
LIFT_HO_LABELS = {'HOLD'}
LIFT_CMD = {'rack': 'n1', 'conveyor': 'n2'}


# ════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════

def _norm(a):
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ════════════════════════════════════════════════════════════════
class QRAligner(Node):

    IDLE          = 'IDLE'
    SWEEP         = 'SWEEP'
    SEND_LIFT_CMD = 'SEND_LIFT_CMD'
    FACE_WALL     = 'FACE_WALL'
    MOVE_CLOSE    = 'MOVE_CLOSE'
    SEND_HOLD     = 'SEND_HOLD'
    BACK_AWAY     = 'BACK_AWAY'
    DONE          = 'DONE'

    def __init__(self):
        super().__init__('qr_aligner')

        # ── Pose ──────────────────────────────────────────────
        self.rx = self.ry = self.rth = 0.0

        # ── Activación ────────────────────────────────────────
        self._trigger_pending = None
        self._trigger         = None
        self._goal_reached_ts = None

        # ── QR congelado ──────────────────────────────────────
        self._qr_x = self._qr_y = None
        self._stop_x = self._stop_y = None   # punto a STOP_DIST del QR
        self._yaw_tgt = None

        # ── Sweep ─────────────────────────────────────────────
        self._sw_dir   = +1
        self._sw_start = None
        self._sw_swept = 0.0

        # ── Back ──────────────────────────────────────────────
        self._back_ox = self._back_oy = None

        # ── Lift ──────────────────────────────────────────────
        self._lift_label = None
        self._lift_timer = self._lift_poll = None

        # ── FSM ───────────────────────────────────────────────
        self.state        = self.IDLE
        self._state_timer = None
        self._retry_count = 0

        # ── Subs ──────────────────────────────────────────────
        self.create_subscription(Odometry,     '/odom',              self._cb_odom,      10)
        self.create_subscription(PointStamped, '/qr/world_pos',      self._cb_world_pos, 10)
        self.create_subscription(String,       '/collect/trigger',   self._cb_trigger,   10)
        self.create_subscription(String,       '/astar/status',      self._cb_astar,     10)
        self.create_subscription(String,       '/lift_done',         self._cb_lift_done, 10)
        self.create_subscription(String,       '/align/force_start', self._cb_force,     10)

        # ── Pubs ──────────────────────────────────────────────
        self.pub_cmd    = self.create_publisher(Twist,  '/cmd_vel',      10)
        self.pub_lift   = self.create_publisher(String, '/lift_auto',    10)
        self.pub_active = self.create_publisher(Bool,   '/align/active', 10)
        self.pub_done   = self.create_publisher(Bool,   '/align/done',   10)

        self.create_timer(0.05, self._loop)
        self.get_logger().info(
            'QR Aligner v11\n'
            f'  STOP={STOP_DIST*100:.0f}cm  BACK={BACK_DIST*100:.0f}cm\n'
            '  FSM: SWEEP→SEND_LIFT_CMD→FACE_WALL→MOVE_CLOSE'
            '→SEND_HOLD→BACK_AWAY→DONE\n'
            '  Manual: ros2 topic pub --once /align/force_start '
            'std_msgs/msg/String "data: rack"')

    # ════════════════════════════════════════════════════════════
    # Callbacks
    # ════════════════════════════════════════════════════════════

    def _cb_odom(self, msg):
        p = msg.pose.pose
        self.rx, self.ry = p.position.x, p.position.y
        q = p.orientation
        self.rth = math.atan2(2*(q.w*q.z + q.x*q.y),
                               1 - 2*(q.y*q.y + q.z*q.z))

    def _cb_world_pos(self, msg: PointStamped):
        if self.state != self.SWEEP:
            return
        self._qr_x = msg.point.x
        self._qr_y = msg.point.y
        self._on_qr_found()

    def _cb_trigger(self, msg: String):
        val = msg.data.strip().lower()
        if val not in ('rack', 'conveyor'):
            return
        if self.state == self.SWEEP:
            self._trigger = val
            if self._qr_x is not None:
                self._on_qr_found()
            return
        if self.state != self.IDLE:
            return
        self._trigger_pending = val
        self.get_logger().info(f'[trigger] "{val}" almacenado')
        if self._goal_reached_valid():
            self._fire_sweep()

    def _cb_astar(self, msg: String):
        if msg.data.strip() in ('GOAL_REACHED', 'IDLE'):
            self._goal_reached_ts = time.monotonic()
            self.get_logger().info(f'[astar] {msg.data.strip()} — ts guardado')
            if self.state == self.IDLE and self._trigger_pending is not None:
                self._fire_sweep()

    def _cb_lift_done(self, msg: String):
        self._lift_label = msg.data.strip()

    def _cb_force(self, msg: String):
        val = msg.data.strip().lower()
        if val not in ('rack', 'conveyor') or self.state != self.IDLE:
            return
        self.get_logger().warn(f'[force_start] FORZADO trigger="{val}"')
        self._trigger_pending = val
        self._goal_reached_ts = time.monotonic()
        self._fire_sweep()

    # ════════════════════════════════════════════════════════════
    # Activación
    # ════════════════════════════════════════════════════════════

    def _goal_reached_valid(self):
        if self._goal_reached_ts is None:
            return False
        return (time.monotonic() - self._goal_reached_ts) < GOAL_REACHED_WINDOW_S

    def _fire_sweep(self):
        self._trigger = self._trigger_pending
        self._trigger_pending = None
        self._retry_count = 0
        self._qr_x = self._qr_y = None
        self.pub_active.publish(Bool(data=False))
        self._transition(self.SWEEP)

    # ════════════════════════════════════════════════════════════
    # QR encontrado → calcular geometría y tomar control
    # ════════════════════════════════════════════════════════════

    def _on_qr_found(self):
        if self._trigger is None or self._qr_x is None:
            return

        self._stop()
        self._cancel_state_timer()

        dx = self._qr_x - self.rx
        dy = self._qr_y - self.ry
        dist = math.hypot(dx, dy)

        if dist < 0.01:
            self.get_logger().warn('[QR] demasiado cerca — retry')
            self._retry(self.SWEEP)
            return

        # stop_point: desde el QR retroceder STOP_DIST hacia el robot
        ux = dx / dist
        uy = dy / dist
        self._stop_x = self._qr_x - ux * STOP_DIST
        self._stop_y = self._qr_y - uy * STOP_DIST

        # Yaw objetivo
        if self._trigger == 'rack':
            self._yaw_tgt = math.pi                          # siempre -X
        else:
            # conveyor: lado relativo al robot en el momento de detección
            if abs(dy) >= abs(dx):
                self._yaw_tgt = math.pi/2 if dy > 0 else -math.pi/2
            else:
                self._yaw_tgt = 0.0 if dx > 0 else math.pi

        self.get_logger().info(
            f'[QR found] trigger={self._trigger}\n'
            f'  QR=({self._qr_x:.3f},{self._qr_y:.3f})'
            f'  dist={dist:.3f}m  yaw={math.degrees(self._yaw_tgt):.0f}°\n'
            f'  stop=({self._stop_x:.3f},{self._stop_y:.3f})')

        # Tomar control total: pausar VFH+ y A*
        self.pub_active.publish(Bool(data=True))

        self._transition(self.SEND_LIFT_CMD)

    # ════════════════════════════════════════════════════════════
    # Loop 20 Hz
    # ════════════════════════════════════════════════════════════

    def _loop(self):
        if   self.state == self.SWEEP:      self._do_sweep()
        elif self.state == self.FACE_WALL:  self._do_face_wall()
        elif self.state == self.MOVE_CLOSE: self._do_move_close()
        elif self.state == self.BACK_AWAY:  self._do_back_away()

    # ── SWEEP ─────────────────────────────────────────────────
    def _do_sweep(self):
        if self._qr_x is not None and self._trigger is not None:
            self._on_qr_found()
            return

        dth = _norm(self.rth - self._sw_start)
        self._sw_swept += abs(dth)
        self._sw_start  = self.rth

        if self._sw_swept > 2.0 * SWEEP_MAX_RAD:
            self.get_logger().warn('[SWEEP] 360° sin QR → retry')
            self._qr_x = self._qr_y = None
            self._retry(self.SWEEP)
            return
        if self._sw_swept > SWEEP_MAX_RAD:
            self._sw_dir = -self._sw_dir

        tw = Twist()
        tw.angular.z = self._sw_dir * SWEEP_SPD
        self.pub_cmd.publish(tw)

    # ── FACE_WALL ─────────────────────────────────────────────
    def _do_face_wall(self):
        """Giro en sitio hasta yaw_tgt."""
        err_th = _norm(self._yaw_tgt - self.rth)
        if abs(err_th) < HEADING_TOL:
            self._stop()
            self.get_logger().info(
                f'[FACE_WALL] ✓  yaw={math.degrees(self.rth):.1f}°')
            self._cancel_state_timer()
            self._transition(self.MOVE_CLOSE)
            return
        tw = Twist()
        tw.angular.z = _clamp(KP_ANG * err_th, -MAX_ANG, MAX_ANG)
        self.pub_cmd.publish(tw)

    # ── MOVE_CLOSE ────────────────────────────────────────────
    def _do_move_close(self):
        """Avanza recto manteniendo yaw_tgt hasta stop_point."""
        dx = self._stop_x - self.rx
        dy = self._stop_y - self.ry
        dist = math.hypot(dx, dy)

        if dist < GOAL_TOL:
            self._stop()
            self.get_logger().info(
                f'[MOVE_CLOSE] ✓  pos=({self.rx:.3f},{self.ry:.3f})')
            self._cancel_state_timer()
            self._transition(self.SEND_HOLD)
            return

        # Mantener yaw_tgt durante el avance (no go-to-goal libre)
        err_th = _norm(self._yaw_tgt - self.rth)
        v_lin  = _clamp(KP_LIN * dist, MIN_LIN, MAX_LIN)
        v_ang  = _clamp(KP_ANG * err_th, -MAX_ANG, MAX_ANG)
        tw = Twist()
        tw.linear.x  = v_lin
        tw.angular.z = v_ang
        self.pub_cmd.publish(tw)

    # ── BACK_AWAY ─────────────────────────────────────────────
    def _do_back_away(self):
        if self._back_ox is None:
            self._back_ox, self._back_oy = self.rx, self.ry

        traveled = math.hypot(self.rx - self._back_ox,
                              self.ry - self._back_oy)
        if traveled >= BACK_DIST:
            self._stop()
            self.get_logger().info(f'[BACK_AWAY] ✓  {traveled:.3f}m')
            self._cancel_state_timer()
            self._transition(self.DONE)
            return

        err_th = _norm(self._yaw_tgt - self.rth)
        tw = Twist()
        tw.linear.x  = -BACK_SPD
        tw.angular.z = _clamp(KP_ANG * err_th, -MAX_ANG, MAX_ANG)
        self.pub_cmd.publish(tw)

    # ════════════════════════════════════════════════════════════
    # Transiciones
    # ════════════════════════════════════════════════════════════

    def _transition(self, new_state: str):
        self.get_logger().info(f'[FSM] {self.state} → {new_state}')
        self.state = new_state

        if new_state == self.SWEEP:
            self._sw_dir   = -self._sw_dir if self._retry_count > 0 else +1
            self._sw_start = self.rth
            self._sw_swept = 0.0
            self._set_state_timer(SWEEP_TIMEOUT_S, self.SWEEP,
                                  lambda: self._retry(self.SWEEP))

        elif new_state == self.SEND_LIFT_CMD:
            self._lift_label = None
            cmd = LIFT_CMD[self._trigger]
            self.pub_lift.publish(String(data=cmd))
            self.get_logger().info(f'[SEND_LIFT_CMD] /lift_auto="{cmd}"')
            self._lift_timer = self.create_timer(LIFT_TIMEOUT_S, self._lv_timeout)
            self._lift_poll  = self.create_timer(0.1,            self._lv_poll)

        elif new_state == self.FACE_WALL:
            self._set_state_timer(FACE_TIMEOUT_S, self.FACE_WALL,
                                  lambda: self._retry(self.FACE_WALL))

        elif new_state == self.MOVE_CLOSE:
            self._set_state_timer(MOVE_TIMEOUT_S, self.MOVE_CLOSE,
                                  lambda: self._retry(self.MOVE_CLOSE))

        elif new_state == self.SEND_HOLD:
            self._lift_label = None
            self.pub_lift.publish(String(data='hold'))
            self.get_logger().info('[SEND_HOLD] /lift_auto="hold"')
            self._lift_timer = self.create_timer(HOLD_TIMEOUT_S, self._ho_timeout)
            self._lift_poll  = self.create_timer(0.1,            self._ho_poll)

        elif new_state == self.BACK_AWAY:
            self._back_ox = self._back_oy = None
            self._set_state_timer(BACK_TIMEOUT_S, self.BACK_AWAY,
                                  lambda: self._retry(self.BACK_AWAY))

        elif new_state == self.DONE:
            self.pub_active.publish(Bool(data=False))
            self.pub_done.publish(Bool(data=True))
            self.get_logger().info(f'[DONE] ✓  trigger={self._trigger}')
            self._reset()

    # ── Lift nivel ────────────────────────────────────────────
    def _lv_poll(self):
        if self.state != self.SEND_LIFT_CMD:
            self._cancel_lift_timers(); return
        if self._lift_label and any(l in self._lift_label for l in LIFT_LV_LABELS):
            self.get_logger().info(f'[SEND_LIFT_CMD] ✓  "{self._lift_label}"')
            self._cancel_lift_timers()
            self._transition(self.FACE_WALL)

    def _lv_timeout(self):
        if self.state == self.SEND_LIFT_CMD:
            self.get_logger().warn('[SEND_LIFT_CMD] timeout → continuar')
            self._cancel_lift_timers()
            self._transition(self.FACE_WALL)

    # ── Lift hold ─────────────────────────────────────────────
    def _ho_poll(self):
        if self.state != self.SEND_HOLD:
            self._cancel_lift_timers(); return
        if self._lift_label and any(l in self._lift_label for l in LIFT_HO_LABELS):
            self.get_logger().info(f'[SEND_HOLD] ✓  "{self._lift_label}"')
            self._cancel_lift_timers()
            self._transition(self.BACK_AWAY)

    def _ho_timeout(self):
        if self.state == self.SEND_HOLD:
            self.get_logger().warn('[SEND_HOLD] timeout → continuar')
            self._cancel_lift_timers()
            self._transition(self.BACK_AWAY)

    # ── Timers ────────────────────────────────────────────────
    def _set_state_timer(self, duration, state_id, on_timeout=None):
        self._cancel_state_timer()
        def _cb():
            if self.state == state_id:
                (on_timeout or self._hard_abort)()
        self._state_timer = self.create_timer(duration, _cb)

    def _cancel_state_timer(self):
        if self._state_timer:
            self._state_timer.cancel()
            self._state_timer = None

    def _cancel_lift_timers(self):
        for t in (self._lift_timer, self._lift_poll):
            if t:
                try: t.cancel()
                except: pass
        self._lift_timer = self._lift_poll = None

    # ── Retry ─────────────────────────────────────────────────
    def _retry(self, fallback: str):
        self._retry_count += 1
        self.get_logger().warn(
            f'[RETRY {self._retry_count}/{MAX_RETRIES}] → {fallback}')
        if self._retry_count > MAX_RETRIES:
            self._hard_abort(); return
        self._stop()
        self._cancel_state_timer()
        self._cancel_lift_timers()
        if fallback == self.SWEEP:
            self._qr_x = self._qr_y = None
        self._transition(fallback)

    def _hard_abort(self):
        self.get_logger().error('[HARD ABORT] → IDLE')
        self._stop()
        self._cancel_state_timer()
        self._cancel_lift_timers()
        self.pub_active.publish(Bool(data=False))
        self._reset()

    def _reset(self):
        self._trigger = self._trigger_pending = None
        self._qr_x = self._qr_y = None
        self._stop_x = self._stop_y = None
        self._yaw_tgt = None
        self._back_ox = self._back_oy = None
        self._lift_label = None
        self._retry_count = 0
        self._sw_swept = 0.0
        self.state = self.IDLE

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
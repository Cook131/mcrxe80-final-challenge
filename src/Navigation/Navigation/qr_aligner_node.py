#!/usr/bin/env python3
"""
qr_aligner_node.py — Puzzlebot / Iolair  v8
════════════════════════════════════════════════════════════════
FSM:
  IDLE → SWEEP → FACE_QR(gtg) → SEND_LIFT_LV
       → APPROACH → SEND_LIFT_HO → BACK_AWAY → DONE → IDLE

CAMBIOS v8 (fix de alineación + offset de cámara)
────────────────────────────────────────────────────────────────
1. FUENTE ÚNICA: la posición del QR se calcula SOLO desde
   /qr/distance + /qr/angle (frame cámara). Se ignora /qr/world_pos
   para no tener dos fuentes pisándose (causa de aborts erráticos).

2. OFFSET DE CÁMARA CORRECTO: la cámara está en
   (CAM_OFFSET_X, CAM_OFFSET_Y) en el frame del robot. El offset se
   SUMA a la posición de la cámara en mundo ANTES de proyectar el QR:
       cam_wx = robot_x + OFF_X·cosθ − OFF_Y·sinθ
       cam_wy = robot_y + OFF_X·sinθ + OFF_Y·cosθ
       qr_x   = cam_wx + d·cos(θ+α)
       qr_y   = cam_wy + d·sin(θ+α)
   El aligner alinea el CENTRO del robot con ese QR.
   (La versión anterior restaba el offset al QR ya proyectado, lo cual
    aplica la corrección en el frame equivocado y rota con el robot.)

3. SWEEP solo detecta dentro del sweep — coords limpiadas en _fire_sweep.

4. SWEEP va directo a FACE_QR (go-to-goal al punto frente al QR).
   Se eliminó ALIGN_POS: el go-to-goal gira y avanza simultáneamente
   hacia face_goal sin depender de ejes cardinales ni reversa, lo que
   evitaba que el robot se moviera en el sentido equivocado cuando el
   sweep lo dejaba apuntando al lado opuesto del eje.

_yaw_target (de _WALL_REGIONS) se conserva SOLO para BACK_AWAY.

Activación:
  /collect/trigger ('rack'|'conveyor') + /astar/status=='GOAL_REACHED'

VFH+ (/align/active):
  True durante toda la maniobra (VFH+ desactivado) / False al terminar.

setup.py entry point:
  'qr_align_node = Navigation.qr_aligner_node:main'
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg      import Odometry
from std_msgs.msg      import Bool, String, Float32


# ════════════════════════════════════════════════════════════════
# Parámetros de tuning
# ════════════════════════════════════════════════════════════════

APPROACH_DIST = 0.20   # m — distancia final al QR
BACK_DIST     = 0.35   # m — retroceso tras HOLD

# Tolerancias
GOAL_TOL    = 0.03   # m — tolerancia llegada al goal (FACE_QR y APPROACH)
HEADING_TOL = 0.05   # rad

# Ganancias PD
KP_TH   = 1.2
KP_GTG  = 0.7;  KD_GTG  = 0.10
KP_HEAD = 1.0

# Velocidades
MAX_LIN    = 0.18   # m/s
MAX_ANG    = 0.8    # rad/s
MIN_LIN    = 0.04   # m/s
BACK_SPEED = 0.10   # m/s

# Sweep
SWEEP_SPEED   = 0.35      # rad/s
SWEEP_MAX_RAD = math.pi   # 180° por sentido antes de invertir

# Timeouts
SWEEP_TIMEOUT_S     = 20.0
FACE_TIMEOUT_S      = 12.0
LIFT_TIMEOUT_S      = 12.0
HOLD_TIMEOUT_S      = 8.0
APPROACH_TIMEOUT_S  = 15.0
BACK_TIMEOUT_S      = BACK_DIST / BACK_SPEED + 5.0

# ── Offset físico de la cámara respecto al CENTRO del robot ──
# Medido en el frame del robot: +X adelante, +Y a la izquierda.
# ¡AJUSTA estos valores con tu medición real en metros!
CAM_OFFSET_X = 0.00   # ← p.ej. cámara 8 cm adelante → 0.08
CAM_OFFSET_Y = 0.00   # ← p.ej. cámara 3 cm a la derecha → -0.03

# Detección
MIN_QR_DIST = 0.05    # m — descarta lecturas espurias < 5 cm

# Labels de /lift_done
LIFT_LV_LABELS = {'AT_N1', 'AT_N2'}
LIFT_HO_LABELS = {'HOLD'}
LIFT_LV_CMD    = {'rack': 'n1', 'conveyor': 'n2'}


# ════════════════════════════════════════════════════════════════
# Tabla de orientación de paredes por posición del QR
# Se usa SOLO para _yaw_target → BACK_AWAY (retroceso cardinal).
# ────────────────────────────────────────────────────────────────
_WALL_REGIONS = [
    # (x_min, x_max,  y_min,  y_max,  yaw_rad)
    (-2.81, -1.60,  -1.84,   1.84,   0.0),           # CONVEYOR  → mirar +X
    (-1.70, -0.35,  -1.00,  -0.40,   math.pi/2),     # RACK_AZ_SUR → mirar +Y
    (-1.70, -0.35,   0.20,   0.80,  -math.pi/2),     # RACK_AZ_NOR → mirar -Y
    ( 0.20,  0.70,  -1.84,   1.84,   math.pi),       # RACK_AMAR → mirar -X
]

def _wall_yaw(qx: float, qy: float) -> float | None:
    for x0, x1, y0, y1, yaw in _WALL_REGIONS:
        if x0 <= qx <= x1 and y0 <= qy <= y1:
            return yaw
    return None


# ════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════

def _norm(a: float) -> float:
    while a >  math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _qr_world_from_polar(d, alpha, rx, ry, rth):
    """
    Proyecta el QR a coordenadas de mundo a partir de la medición
    polar de la CÁMARA (d, alpha), compensando el offset físico de la
    cámara respecto al centro del robot.

    La cámara está en (CAM_OFFSET_X, CAM_OFFSET_Y) en el frame robot.
    Su posición en mundo:
        cam_wx = rx + OFF_X·cosθ − OFF_Y·sinθ
        cam_wy = ry + OFF_X·sinθ + OFF_Y·cosθ
    El QR se proyecta DESDE la cámara:
        qr_x = cam_wx + d·cos(θ+α)
        qr_y = cam_wy + d·sin(θ+α)
    """
    c, s = math.cos(rth), math.sin(rth)
    cam_wx = rx + CAM_OFFSET_X*c - CAM_OFFSET_Y*s
    cam_wy = ry + CAM_OFFSET_X*s + CAM_OFFSET_Y*c
    ang = rth + alpha
    return (cam_wx + d*math.cos(ang),
            cam_wy + d*math.sin(ang))


# ════════════════════════════════════════════════════════════════
class QRAligner(Node):

    IDLE         = 'IDLE'
    SWEEP        = 'SWEEP'
    FACE_QR      = 'FACE_QR'
    SEND_LIFT_LV = 'SEND_LIFT_LV'
    APPROACH     = 'APPROACH'
    SEND_LIFT_HO = 'SEND_LIFT_HO'
    BACK_AWAY    = 'BACK_AWAY'
    DONE         = 'DONE'

    def __init__(self):
        super().__init__('qr_aligner')

        # ── Pose robot ────────────────────────────────────────
        self.robot_x = self.robot_y = self.robot_theta = 0.0

        # ── QR — congelado al salir de SWEEP ──────────────────
        # adj_x/adj_y = posición del QR en mundo, ya con offset de
        # cámara compensado. El robot alinea su CENTRO con esto.
        self.adj_x = self.adj_y = None

        # ── Polares del checker (frame cámara) ────────────────
        self._qr_dist  = None
        self._qr_angle = None

        # ── Activación ────────────────────────────────────────
        self.trigger             = None   # 'rack' | 'conveyor'
        self._trigger_pending    = None
        self._astar_goal_reached = False

        # ── Geometría — calculada en _on_qr_detected ─────────
        self._yaw_target      = None
        self._face_goal_x     = self._face_goal_y     = None
        self._approach_goal_x = self._approach_goal_y = None

        # ── Sweep ─────────────────────────────────────────────
        self._sw_dir        = +1
        self._sw_prev_theta = None
        self._sw_swept      = 0.0

        # ── PD / back ─────────────────────────────────────────
        self._gtg_prev_dist = 0.0
        self._back_x = self._back_y = None

        # ── FSM ───────────────────────────────────────────────
        self.state        = self.IDLE
        self._state_timer = None
        self._lift_label  = None
        self._lift_timer  = None
        self._lift_poll   = None

        # ── Subs ──────────────────────────────────────────────
        # NOTA: /qr/world_pos YA NO se usa. Fuente única = polares.
        self.create_subscription(
            Float32,  '/qr/distance',     self._cb_qr_dist,   10)
        self.create_subscription(
            Float32,  '/qr/angle',        self._cb_qr_angle,  10)
        self.create_subscription(
            String,   '/collect/trigger', self._cb_trigger,   10)
        self.create_subscription(
            String,   '/astar/status',    self._cb_astar,     10)
        self.create_subscription(
            Odometry, '/odom',            self._cb_odom,      10)
        self.create_subscription(
            String,   '/lift_done',       self._cb_lift_done, 10)

        # ── Pubs ──────────────────────────────────────────────
        self.pub_cmd       = self.create_publisher(Twist,  '/cmd_vel',      10)
        self.pub_lift      = self.create_publisher(String, '/lift_auto',    10)
        self.pub_active    = self.create_publisher(Bool,   '/align/active', 10)
        self.pub_done      = self.create_publisher(Bool,   '/align/done',   10)

        self.create_timer(0.05, self._loop)

        self.get_logger().info(
            'QR Aligner v8\n'
            '  Fuente única: /qr/distance + /qr/angle (frame cámara)\n'
            '  Offset de cámara compensado ANTES de proyectar el QR\n'
            f'  CAM_OFFSET=({CAM_OFFSET_X:.3f},{CAM_OFFSET_Y:.3f})m')

    # ════════════════════════════════════════════════════════════
    # Callbacks
    # ════════════════════════════════════════════════════════════

    def _cb_odom(self, msg):
        p = msg.pose.pose
        self.robot_x, self.robot_y = p.position.x, p.position.y
        q = p.orientation
        self.robot_theta = math.atan2(2*(q.w*q.z+q.x*q.y),
                                       1-2*(q.y*q.y+q.z*q.z))

    def _cb_qr_dist(self, msg: Float32):
        self._qr_dist = msg.data
        self._try_polar()

    def _cb_qr_angle(self, msg: Float32):
        self._qr_angle = msg.data
        self._try_polar()

    def _try_polar(self):
        """
        Fuente única de posición del QR. Solo procesa en IDLE/SWEEP.
        Proyecta el QR a mundo compensando el offset de cámara.
        """
        if self.state not in (self.IDLE, self.SWEEP):
            return
        if self._qr_dist is None or self._qr_angle is None:
            return
        if self._qr_dist < MIN_QR_DIST:
            return

        self.adj_x, self.adj_y = _qr_world_from_polar(
            self._qr_dist, self._qr_angle,
            self.robot_x, self.robot_y, self.robot_theta)

        if self.state == self.SWEEP:
            self._on_qr_detected()

    def _cb_trigger(self, msg: String):
        val = msg.data.strip().lower()
        if val not in ('rack', 'conveyor'):
            self.get_logger().warn(f'trigger desconocido: "{val}"')
            return
        if self.state == self.SWEEP:
            self.trigger = val
            self._trigger_pending = None
            self.get_logger().info(f'[trigger] "{val}" durante SWEEP')
            if self.adj_x is not None:
                self._on_qr_detected()
            return
        if self.state != self.IDLE:
            self.get_logger().warn(
                f'trigger "{val}" ignorado — estado {self.state}')
            return
        self._trigger_pending = val
        self.get_logger().info(
            f'[trigger] "{val}" almacenado — esperando GOAL_REACHED')
        if self._astar_goal_reached:
            self._fire_sweep()

    def _cb_astar(self, msg: String):
        s = msg.data.strip()
        if s == 'GOAL_REACHED':
            self._astar_goal_reached = True
            # El robot llega a ciegas: arrancar SWEEP de inmediato y
            # buscar el QR durante el barrido. El trigger (rack/conveyor)
            # lo completará el zone_checker cuando vea el QR.
            if self.state == self.IDLE:
                self.get_logger().info('[astar] GOAL_REACHED → SWEEP')
                self._fire_sweep()
        elif s in ('EXECUTING', 'PLANNING'):
            self._astar_goal_reached = False

    def _cb_lift_done(self, msg: String):
        self._lift_label = msg.data.strip()
        self.get_logger().info(f'[lift_done] "{self._lift_label}"')

    # ════════════════════════════════════════════════════════════
    # Activación
    # ════════════════════════════════════════════════════════════

    def _fire_sweep(self):
        # El trigger puede no haber llegado todavía: el robot llega a ciegas
        # y el zone_checker publicará rack/conveyor cuando vea el QR durante
        # el barrido. Conservar _trigger_pending si ya existía.
        self.trigger = self._trigger_pending   # normalmente None aquí
        self._trigger_pending = None
        self._astar_goal_reached = False

        # Señal única de la maniobra: /align/active. VFH+, GoToGoal y A*
        # la respetan y se callan mientras el aligner controla /cmd_vel.
        # (Ya NO usamos /nav_pause — ese queda exclusivo para evasiones VFH+.)
        self.pub_active.publish(Bool(data=True))

        # Limpiar detecciones previas — la alineación solo arranca
        # si el QR se detecta DENTRO del sweep.
        self.adj_x = self.adj_y = None
        self._qr_dist = self._qr_angle = None

        self._sw_dir        = +1 if self.robot_x < 0 else -1
        self._sw_prev_theta = self.robot_theta
        self._sw_swept      = 0.0
        self._transition(self.SWEEP)

    # ════════════════════════════════════════════════════════════
    # Al detectar el QR durante SWEEP
    # ════════════════════════════════════════════════════════════

    def _on_qr_detected(self):
        """
        adj_x/adj_y = QR en mundo, offset de cámara ya compensado.
        Congela coords y calcula la geometría de la maniobra.
        face_goal aquí es provisional (se recalcula al entrar FACE_QR).
        """
        if self.trigger is None:
            return

        self._stop()
        self._cancel_state_timer()

        dx = self.adj_x - self.robot_x
        dy = self.adj_y - self.robot_y
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            self.get_logger().warn('[QR] demasiado cerca — abort')
            self._abort(); return

        # ── yaw cardinal de la pared (SOLO para BACK_AWAY) ────
        yaw = _wall_yaw(self.adj_x, self.adj_y)
        if yaw is None:
            self.get_logger().warn(
                f'[_on_qr_detected] QR ({self.adj_x:.3f},{self.adj_y:.3f})'
                f' fuera de regiones — heurística dx/dy')
            yaw = (math.pi if dx < 0 else 0.0) if abs(dx) >= abs(dy) \
                   else (math.pi/2 if dy > 0 else -math.pi/2)
        self._yaw_target = yaw   # SOLO para BACK_AWAY

        # face_goal: punto a APPROACH_DIST frente al QR, sobre la línea
        # robot→QR. Es provisional aquí; FACE_QR lo recalcula con la
        # posición actual del robot.
        dx_n = dx / dist
        dy_n = dy / dist
        self._face_goal_x     = self.adj_x - dx_n * APPROACH_DIST
        self._face_goal_y     = self.adj_y - dy_n * APPROACH_DIST
        self._approach_goal_x = self._face_goal_x
        self._approach_goal_y = self._face_goal_y

        self._gtg_prev_dist = 0.0
        self._lift_label    = None
        self._back_x = self._back_y = None

        self.get_logger().info(
            f'[QR detected]  trigger={self.trigger}\n'
            f'  QR(adj)=({self.adj_x:.3f},{self.adj_y:.3f})'
            f'  robot=({self.robot_x:.3f},{self.robot_y:.3f})\n'
            f'  face_goal=({self._face_goal_x:.3f},{self._face_goal_y:.3f})'
            f'  yaw_cardinal={math.degrees(self._yaw_target):.0f}°'
            f'  (solo BACK_AWAY)')

        self._transition(self.FACE_QR)

    # ════════════════════════════════════════════════════════════
    # Control loop 20 Hz
    # ════════════════════════════════════════════════════════════

    def _loop(self):
        if   self.state == self.SWEEP:        self._do_sweep()
        elif self.state == self.FACE_QR:      self._do_face_qr()
        elif self.state == self.APPROACH:     self._do_approach()
        elif self.state == self.BACK_AWAY:    self._do_back_away()

    # ── SWEEP ─────────────────────────────────────────────────
    def _do_sweep(self):
        if self.adj_x is not None and self.trigger is not None:
            self._on_qr_detected()
            return

        dth = _norm(self.robot_theta - self._sw_prev_theta)
        self._sw_swept     += abs(dth)
        self._sw_prev_theta = self.robot_theta

        if self._sw_swept > 2.0 * SWEEP_MAX_RAD:
            self.get_logger().error('[SWEEP] QR no encontrado → abort')
            self._abort(); return
        if self._sw_swept > SWEEP_MAX_RAD:
            self._sw_dir = -self._sw_dir

        twist = Twist()
        twist.angular.z = self._sw_dir * SWEEP_SPEED
        self.pub_cmd.publish(twist)

    # ── FACE_QR ───────────────────────────────────────────────
    def _do_face_qr(self):
        dx = self._face_goal_x - self.robot_x
        dy = self._face_goal_y - self.robot_y
        dist = math.hypot(dx, dy)

        derr = (dist - self._gtg_prev_dist) / 0.05
        self._gtg_prev_dist = dist

        if dist < GOAL_TOL:
            self._stop()
            self.get_logger().info(
                f'[FACE_QR] ✓  pos=({self.robot_x:.3f},{self.robot_y:.3f})'
                f'  QR=({self.adj_x:.3f},{self.adj_y:.3f})')
            self._cancel_state_timer()
            self._transition(self.SEND_LIFT_LV)
            return

        err_th = _norm(math.atan2(dy, dx) - self.robot_theta)
        v_lin  = _clamp(KP_GTG * dist + KD_GTG * derr, MIN_LIN, MAX_LIN)
        v_lin *= max(0.0, 1.0 - abs(err_th) / math.pi)
        v_ang  = _clamp(KP_HEAD * err_th, -MAX_ANG, MAX_ANG)

        twist = Twist()
        twist.linear.x  = v_lin
        twist.angular.z = v_ang
        self.pub_cmd.publish(twist)

    # ── APPROACH ──────────────────────────────────────────────
    def _do_approach(self):
        dx = self._approach_goal_x - self.robot_x
        dy = self._approach_goal_y - self.robot_y
        dist = math.hypot(dx, dy)

        derr = (dist - self._gtg_prev_dist) / 0.05
        self._gtg_prev_dist = dist

        if dist < GOAL_TOL:
            self._stop()
            self.get_logger().info(f'[APPROACH] ✓  dist_final={dist:.3f}m')
            self._cancel_state_timer()
            self._transition(self.SEND_LIFT_HO)
            return

        err_th = _norm(math.atan2(dy, dx) - self.robot_theta)
        v_lin  = _clamp(KP_GTG * dist + KD_GTG * derr, MIN_LIN, MAX_LIN)
        v_lin *= max(0.0, 1.0 - abs(err_th) / math.pi)
        v_ang  = _clamp(KP_HEAD * err_th, -MAX_ANG, MAX_ANG)

        twist = Twist()
        twist.linear.x  = v_lin
        twist.angular.z = v_ang
        self.pub_cmd.publish(twist)

    # ── BACK_AWAY ─────────────────────────────────────────────
    def _do_back_away(self):
        if self._back_x is None:
            self._back_x, self._back_y = self.robot_x, self.robot_y

        dist = math.hypot(self.robot_x - self._back_x,
                          self.robot_y - self._back_y)
        if dist >= BACK_DIST:
            self._stop()
            self.get_logger().info(f'[BACK_AWAY] ✓  {dist:.3f}m')
            self._cancel_state_timer()
            self._transition(self.DONE)
            return

        err_th = _norm(self._yaw_target - self.robot_theta)
        twist = Twist()
        twist.linear.x  = -BACK_SPEED
        twist.angular.z = _clamp(KP_HEAD * err_th, -MAX_ANG, MAX_ANG)
        self.pub_cmd.publish(twist)

    # ════════════════════════════════════════════════════════════
    # Transiciones FSM
    # ════════════════════════════════════════════════════════════

    def _transition(self, new_state: str):
        self.get_logger().info(f'[FSM] {self.state} → {new_state}')
        self.state = new_state

        if new_state == self.SWEEP:
            self._set_state_timer(SWEEP_TIMEOUT_S, self.SWEEP)

        elif new_state == self.FACE_QR:
            self._gtg_prev_dist = 0.0

            # Recalcular face_goal con posición ACTUAL (post-SWEEP)
            # para evitar goal detrás/encima → giro infinito.
            dx = self.adj_x - self.robot_x
            dy = self.adj_y - self.robot_y
            dist_to_qr = math.hypot(dx, dy)

            if dist_to_qr > APPROACH_DIST + 0.05:
                dx_n = dx / dist_to_qr
                dy_n = dy / dist_to_qr
                self._face_goal_x     = self.adj_x - dx_n * APPROACH_DIST
                self._face_goal_y     = self.adj_y - dy_n * APPROACH_DIST
                self._approach_goal_x = self._face_goal_x
                self._approach_goal_y = self._face_goal_y
                self.get_logger().info(
                    f'[FACE_QR] face_goal recalc → '
                    f'({self._face_goal_x:.3f},{self._face_goal_y:.3f})'
                    f'  dist_QR={dist_to_qr:.3f}m')
                self._set_state_timer(FACE_TIMEOUT_S, self.FACE_QR)
            else:
                self.get_logger().warn(
                    f'[FACE_QR] ya en zona approach (dist_QR={dist_to_qr:.3f}m)'
                    f' → SEND_LIFT_LV directo')
                self._transition(self.SEND_LIFT_LV)

        elif new_state == self.SEND_LIFT_LV:
            self._lift_label = None
            cmd = LIFT_LV_CMD.get(self.trigger, 'n1')
            self.pub_lift.publish(String(data=cmd))
            self.get_logger().info(f'[SEND_LIFT_LV] /lift_auto="{cmd}"')
            self._lift_timer = self.create_timer(LIFT_TIMEOUT_S, self._lv_timeout)
            self._lift_poll  = self.create_timer(0.1,            self._lv_poll)

        elif new_state == self.APPROACH:
            self._gtg_prev_dist = 0.0
            self._set_state_timer(APPROACH_TIMEOUT_S, self.APPROACH)

        elif new_state == self.SEND_LIFT_HO:
            self._lift_label = None
            self.pub_lift.publish(String(data='hold'))
            self.get_logger().info('[SEND_LIFT_HO] /lift_auto="hold"')
            self._lift_timer = self.create_timer(HOLD_TIMEOUT_S, self._ho_timeout)
            self._lift_poll  = self.create_timer(0.1,            self._ho_poll)

        elif new_state == self.BACK_AWAY:
            self._back_x = self._back_y = None
            self._set_state_timer(BACK_TIMEOUT_S, self.BACK_AWAY)

        elif new_state == self.DONE:
            # Orden: primero avisar que terminó (/align/done) para que el
            # mission_manager prepare el goal nuevo (id 15); luego soltar
            # /align/active para que A*/GoToGoal/VFH+ retomen. GoToGoal y A*
            # descartaron su ruta vieja, así que esperan el goal nuevo sin
            # moverse hacia el waypoint anterior.
            self.pub_done.publish(Bool(data=True))
            self.pub_active.publish(Bool(data=False))
            self.get_logger().info(f'[DONE] ✓  trigger={self.trigger}')
            self._reset()

    # ── Lift nivel ────────────────────────────────────────────
    def _lv_poll(self):
        if self.state != self.SEND_LIFT_LV:
            self._cancel_lift_timers(); return
        if self._lift_label and any(l in self._lift_label for l in LIFT_LV_LABELS):
            self.get_logger().info(f'[SEND_LIFT_LV] ✓  "{self._lift_label}"')
            self._cancel_lift_timers()
            self._transition(self.APPROACH)

    def _lv_timeout(self):
        if self.state == self.SEND_LIFT_LV:
            self.get_logger().warn('[SEND_LIFT_LV] timeout → continuando')
            self._cancel_lift_timers()
            self._transition(self.APPROACH)

    # ── Lift hold ─────────────────────────────────────────────
    def _ho_poll(self):
        if self.state != self.SEND_LIFT_HO:
            self._cancel_lift_timers(); return
        if self._lift_label and any(l in self._lift_label for l in LIFT_HO_LABELS):
            self.get_logger().info(f'[SEND_LIFT_HO] ✓  "{self._lift_label}"')
            self._cancel_lift_timers()
            self._transition(self.BACK_AWAY)

    def _ho_timeout(self):
        if self.state == self.SEND_LIFT_HO:
            self.get_logger().warn('[SEND_LIFT_HO] timeout → continuando')
            self._cancel_lift_timers()
            self._transition(self.BACK_AWAY)

    # ── Timers genéricos ──────────────────────────────────────
    def _set_state_timer(self, duration: float, state_id: str):
        self._cancel_state_timer()
        def _cb():
            if self.state == state_id:
                self.get_logger().error(
                    f'[TIMEOUT] {state_id} en {duration:.1f}s → abort')
                self._abort()
        self._state_timer = self.create_timer(duration, _cb)

    def _cancel_state_timer(self):
        if self._state_timer:
            self._state_timer.cancel()
            self._state_timer = None

    def _cancel_lift_timers(self):
        for t in [self._lift_timer, self._lift_poll]:
            if t:
                try: t.cancel()
                except: pass
        self._lift_timer = self._lift_poll = None

    # ── Abort / Reset ─────────────────────────────────────────
    def _abort(self):
        self.get_logger().error('[QRAligner] ABORT → IDLE')
        self._stop()
        self._cancel_lift_timers()
        self._cancel_state_timer()
        self.pub_active.publish(Bool(data=False))
        self._reset()

    def _reset(self):
        self.adj_x = self.adj_y = None
        self._qr_dist = self._qr_angle = None
        self.trigger = self._trigger_pending = None
        self._yaw_target = None
        self._face_goal_x = self._face_goal_y = None
        self._approach_goal_x = self._approach_goal_y = None
        self._sw_swept = 0.0
        self._sw_prev_theta = None
        self._gtg_prev_dist = 0.0
        self._back_x = self._back_y = None
        self._lift_label = None
        # _astar_goal_reached NO se resetea
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
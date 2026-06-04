#!/usr/bin/env python3
"""
bug_IBA.py  —  Capa Reflejo de Seguridad v4.1  (BUG2 + A* Replan)
==================================================================
Nodo intermedio entre nav_fsm/GoToGoal y puzzlebotController.

Pipeline:
  A* ──/goal──► GoToGoal ──/cmd_raw──► [bug_IBA] ──/cmd_vel──► Controller
                                            ▲  │
                                       /scan  └──/replan_trigger──► A*

Cambios v4.1 sobre v4.0 — "Wall-follow con control lateral"
──────────────────────────────────────────────────────────────
[FIX-1]  DISTANCIAS AUMENTADAS:
         warn_dist 0.55→0.65m, emergency_dist 0.22→0.35m, stop_dist 0.10→0.14m
         El robot ahora reacciona antes de estar pegado al obstáculo.

[FIX-2]  WALL-FOLLOW CON CONTROL LATERAL (P-controller):
         En lugar de girar con velocidad angular fija, el wall-follow
         ahora mantiene una distancia lateral deseada (wall_follow_dist)
         al obstáculo usando un P-controller sobre el sensor lateral.
         Esto elimina el comportamiento de "pasar rozando".
         La dirección de bordeo (izq/der) se fija al entrar al hit point
         y no cambia durante todo el wall-follow (evita oscilaciones).

[FIX-3]  VELOCIDAD LINEAL ADAPTATIVA en wall-follow:
         Se reduce la velocidad lineal cuando el frente se acerca al
         obstáculo, en lugar de mantenerla fija. Menos colisiones en
         esquinas.

[FIX-4]  CONDICIÓN DE SALIDA BUG2 REFORZADA:
         Se añade un timer mínimo (bug2_min_time_s) adicional a la
         distancia mínima recorrida. Evita salidas falsas por ruido
         de odometría en los primeros ciclos. Además la condición de
         frente libre sube a warn_dist (antes) para mayor margen.

[FIX-5]  SECTOR LATERAL CORRECTO SEGÚN DIRECCIÓN DE BORDEO:
         El sensor lateral que controla la distancia al muro usa
         el lado OPUESTO al giro de bordeo (el lado del muro real),
         no siempre el mismo lado.
"""

import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose2D
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, Bool
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


PASS_THROUGH     = "PASS"
PREDICTIVE_BRAKE = "BRAKE"
BUG2_WALL_FOLLOW = "BUG2_WALL"
REFLEX_STOP      = "REFLEX_STOP"


class BugReflex(Node):
    """Subsumption safety layer con BUG2, braking predictivo y replan A*."""

    def __init__(self):
        super().__init__('bug_reflex')

        # ── Parámetros ────────────────────────────────────────────────────
        # [FIX-1] Distancias aumentadas para reaccionar antes
        self.declare_parameter('warn_dist',           0.65)   # era 0.55
        self.declare_parameter('emergency_dist',      0.35)   # era 0.22
        self.declare_parameter('stop_dist',           0.14)   # era 0.10
        self.declare_parameter('reflex_v',            0.06)
        self.declare_parameter('reflex_w',            0.65)
        self.declare_parameter('reflex_hold_ms',      350)
        self.declare_parameter('front_half_deg',      30.0)
        self.declare_parameter('side_half_deg',       35.0)
        self.declare_parameter('hysteresis',          0.06)
        self.declare_parameter('lidar_yaw_offset',    math.pi)
        self.declare_parameter('replan_cooldown_s',   2.0)
        self.declare_parameter('m_line_tol',          0.15)   # era 0.12, un poco más tolerante
        self.declare_parameter('bug2_min_follow_m',   0.30)   # era 0.20

        # [FIX-2] Parámetros del P-controller lateral
        # Distancia deseada al muro durante el wall-follow [m]
        self.declare_parameter('wall_follow_dist',    0.40)
        # Ganancia proporcional del P-controller lateral
        self.declare_parameter('wall_follow_kp',      1.20)
        # Límite de corrección angular del P-controller [rad/s]
        self.declare_parameter('wall_follow_w_max',   0.80)

        # [FIX-4] Tiempo mínimo en wall-follow antes de chequear salida [s]
        self.declare_parameter('bug2_min_time_s',     1.0)

        self.warn_d       = float(self.get_parameter('warn_dist').value)
        self.emg_d        = float(self.get_parameter('emergency_dist').value)
        self.stop_d       = float(self.get_parameter('stop_dist').value)
        self.ref_v        = float(self.get_parameter('reflex_v').value)
        self.ref_w        = float(self.get_parameter('reflex_w').value)
        self.hold_s       = float(self.get_parameter('reflex_hold_ms').value) / 1000.0
        self.front_h      = math.radians(self.get_parameter('front_half_deg').value)
        self.side_h       = math.radians(self.get_parameter('side_half_deg').value)
        self.hyst         = float(self.get_parameter('hysteresis').value)
        self._lidar_yaw_offset = float(self.get_parameter('lidar_yaw_offset').value)
        self._replan_cooldown  = float(self.get_parameter('replan_cooldown_s').value)
        self._m_line_tol       = float(self.get_parameter('m_line_tol').value)
        self._bug2_min_follow  = float(self.get_parameter('bug2_min_follow_m').value)
        self._wall_follow_dist = float(self.get_parameter('wall_follow_dist').value)
        self._wall_follow_kp   = float(self.get_parameter('wall_follow_kp').value)
        self._wall_follow_w_max= float(self.get_parameter('wall_follow_w_max').value)
        self._bug2_min_time    = float(self.get_parameter('bug2_min_time_s').value)

        # ── Estado interno ────────────────────────────────────────────────
        self._mode        = PASS_THROUGH
        self._reflex_ts   = 0.0
        self._last_cmd    = Twist()
        self.scan: LaserScan | None = None

        self._robot_x     = 0.0
        self._robot_y     = 0.0

        self._last_replan_ts  = -999.0
        self._replanning      = False

        self._last_turn_sign  = +1.0  # desempate en corredor simétrico

        self._goal_x: float | None = None
        self._goal_y: float | None = None

        # Estado BUG2
        self._hit_x          = 0.0
        self._hit_y          = 0.0
        self._hit_dist_goal  = float('inf')
        self._bug2_traveled  = 0.0
        self._bug2_prev_x    = 0.0
        self._bug2_prev_y    = 0.0
        # [FIX-4] Timestamp de entrada al wall-follow
        self._bug2_enter_ts  = 0.0
        # [FIX-5] Dirección de bordeo fijada al entrar: +1=izq, -1=der
        self._bug2_turn_sign  = +1.0

        # ── QoS best-effort para LiDAR ────────────────────────────────────
        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5)

        # ── Suscriptores ──────────────────────────────────────────────────
        self.create_subscription(Twist,     '/cmd_raw',      self._cb_cmd,          10)
        self.create_subscription(LaserScan, '/scan',         self._cb_scan,         scan_qos)
        self.create_subscription(Odometry,  '/odom',         self._cb_odom,         10)
        self.create_subscription(String,    '/astar/status', self._cb_astar_status, 10)
        self.create_subscription(Pose2D,    '/astar/goal',   self._cb_goal,         10)

        # ── Publicadores ──────────────────────────────────────────────────
        self._pub_cmd       = self.create_publisher(Twist,  '/cmd_vel',        10)
        self._pub_status    = self.create_publisher(String, '/reflex_status',  10)
        self._pub_replan    = self.create_publisher(Pose2D, '/replan_trigger', 10)
        self._pub_nav_pause = self.create_publisher(Bool,   '/nav_pause',      10)

        self.create_timer(0.05, self._loop)   # 20 Hz

        self.get_logger().info(
            f'[BugReflex v4.1 — BUG2+LateralCtrl] Lista | '
            f'warn={self.warn_d:.2f}m | emg={self.emg_d:.2f}m | '
            f'stop={self.stop_d:.2f}m | wall_follow_dist={self._wall_follow_dist:.2f}m')

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _cb_cmd(self, msg: Twist):
        self._last_cmd = msg

    def _cb_scan(self, msg: LaserScan):
        self.scan = msg

    def _cb_odom(self, msg: Odometry):
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y

    def _cb_goal(self, msg: Pose2D):
        if msg.x != self._goal_x or msg.y != self._goal_y:
            self._goal_x = msg.x
            self._goal_y = msg.y
            self.get_logger().info(
                f'[BugReflex] Goal final actualizado: ({msg.x:.2f}, {msg.y:.2f})')

    def _cb_astar_status(self, msg: String):
        if msg.data in ('EXECUTING', 'GOAL_REACHED', 'NO_PATH'):
            if self._replanning:
                self.get_logger().info(
                    f'[BugReflex] Replan completado (A* status: {msg.data}) — '
                    f'retomando trayectoria planificada.')
            self._replanning = False

    # ── Loop principal ────────────────────────────────────────────────────

    def _loop(self):
        scan = self.scan
        if scan is None:
            self._publish(self._last_cmd, PASS_THROUGH)
            return

        front = self._sector_min(scan, 0.0,               self.front_h)
        left  = self._sector_min(scan, math.radians(90), self.side_h)
        right = self._sector_min(scan, math.radians(-90), self.side_h)

        now         = time.monotonic()
        hold_active = (now - self._reflex_ts) < self.hold_s

        # ── P1: REFLEX_STOP ───────────────────────────────────────────────
        in_stop  = (self._mode == REFLEX_STOP)
        stop_thr = self.stop_d + (self.hyst if in_stop else 0.0)

        in_wall_hold = hold_active and (self._mode == BUG2_WALL_FOLLOW)
        stop_triggered = (front <= stop_thr)
        turn_veto      = in_wall_hold and (front > self.stop_d)

        if stop_triggered and not turn_veto:
            if not in_stop:
                self._reflex_ts = now
            self._publish(Twist(), REFLEX_STOP)
            return

        # ── P2: BUG2_WALL_FOLLOW ──────────────────────────────────────────
        in_wall  = (self._mode == BUG2_WALL_FOLLOW)
        emg_thr  = self.emg_d + (self.hyst if in_wall else 0.0)

        if front <= emg_thr or (hold_active and in_wall):
            if not in_wall:
                # ── Entrada al wall-follow ──────────────────────────────
                self._reflex_ts      = now
                self._bug2_enter_ts  = now
                self._hit_x          = self._robot_x
                self._hit_y          = self._robot_y
                self._hit_dist_goal  = self._dist_to_goal(self._robot_x, self._robot_y)
                self._bug2_traveled  = 0.0
                self._bug2_prev_x    = self._robot_x
                self._bug2_prev_y    = self._robot_y

                # [FIX-5] Fijar dirección de bordeo: bordear por el lado más libre.
                # Si empate, usar memoria de giro anterior.
                if abs(left - right) > 0.05:
                    self._bug2_turn_sign = +1.0 if left > right else -1.0
                else:
                    self._bug2_turn_sign = self._last_turn_sign
                self._last_turn_sign = self._bug2_turn_sign

                pause_msg = Bool(); pause_msg.data = True
                self._pub_nav_pause.publish(pause_msg)
                self.get_logger().warn(
                    f'[BugReflex] BUG2 hit en ({self._hit_x:.2f}, {self._hit_y:.2f}) | '
                    f'dist_goal={self._hit_dist_goal:.2f}m | '
                    f'bordeo={"IZQ" if self._bug2_turn_sign > 0 else "DER"} — nav PAUSADA')
            else:
                # ── Dentro del wall-follow ──────────────────────────────
                step = math.hypot(self._robot_x - self._bug2_prev_x,
                                  self._robot_y - self._bug2_prev_y)
                self._bug2_traveled += step
                self._bug2_prev_x    = self._robot_x
                self._bug2_prev_y    = self._robot_y

                # [BUG2-3 + FIX-4] Chequear condición de salida
                if self._check_bug2_exit(front, now):
                    self.get_logger().info(
                        f'[BugReflex] BUG2 salida — sobre línea M, '
                        f'dist_goal={self._dist_to_goal(self._robot_x, self._robot_y):.2f}m '
                        f'< hit={self._hit_dist_goal:.2f}m | '
                        f'recorrido={self._bug2_traveled:.2f}m')
                    self._maybe_trigger_replan(now)
                    pause_msg = Bool(); pause_msg.data = False
                    self._pub_nav_pause.publish(pause_msg)
                    self._publish(self._last_cmd, PASS_THROUGH)
                    return

            # ── [FIX-2] Wall-follow con P-controller lateral ────────────
            cmd = self._wall_follow_cmd(front, left, right)
            self._publish(cmd, BUG2_WALL_FOLLOW)
            return

        # ── P3: PREDICTIVE_BRAKE ──────────────────────────────────────────
        in_brake   = (self._mode == PREDICTIVE_BRAKE)
        warn_thr   = self.warn_d + (self.hyst if in_brake else 0.0)
        incoming_v = self._last_cmd.linear.x

        if front <= warn_thr and incoming_v > 0.0:
            t = 1.0 - (front - self.emg_d) / max(self.warn_d - self.emg_d, 1e-6)
            t = max(0.0, min(1.0, t))
            scale = 1.0 - t

            cmd = Twist()
            cmd.linear.x  = incoming_v * scale
            cmd.angular.z = self._last_cmd.angular.z
            self._publish(cmd, PREDICTIVE_BRAKE)
            return

        # ── P4: PASS_THROUGH ──────────────────────────────────────────────
        if self._mode not in (PASS_THROUGH,):
            self.get_logger().info(
                f'[BugReflex] {self._mode} → PASS | frente libre ({front:.2f}m)')
        self._publish(self._last_cmd, PASS_THROUGH)

    # ── [FIX-2+3+5] Wall-follow con control lateral ───────────────────────

    def _wall_follow_cmd(self, front: float, left: float, right: float) -> Twist:
        """
        Genera el comando de wall-follow manteniendo una distancia lateral
        deseada al obstáculo con un P-controller.

        La dirección de bordeo (_bug2_turn_sign) está fijada desde la entrada
        al hit point y no cambia durante el bordeo.

        [FIX-2] P-controller lateral:
          - Si el robot está demasiado cerca al muro → gira alejándose.
          - Si el robot está demasiado lejos del muro → gira acercándose.
          - Si está a la distancia correcta → avanza recto.

        [FIX-3] Velocidad lineal adaptativa:
          - Se reduce cuando el frente se acerca, para manejar esquinas.
        """
        turn_sign = self._bug2_turn_sign  # +1=bordeo izquierda, -1=bordeo derecha

        # [FIX-5] El sensor lateral relevante es el lado del muro:
        #   Si bordeamos por la izquierda (giro +), el muro está a la DERECHA.
        #   Si bordeamos por la derecha (giro -), el muro está a la IZQUIERDA.
        wall_dist = right if turn_sign > 0 else left

        # [FIX-2] Error de distancia lateral: positivo = muy cerca, negativo = muy lejos
        lateral_error = self._wall_follow_dist - wall_dist

        # P-controller: si error>0 (muy cerca al muro) → alejar (giro opuesto al bordeo)
        #               si error<0 (muy lejos del muro) → acercar (giro en dirección de bordeo)
        w_lateral = -turn_sign * self._wall_follow_kp * lateral_error
        w_lateral = max(-self._wall_follow_w_max, min(self._wall_follow_w_max, w_lateral))

        # [FIX-3] Velocidad lineal adaptativa según distancia al frente
        # Rango: [emg_d, warn_d] → velocidad [ref_v*0.3, ref_v]
        front_ratio = (front - self.emg_d) / max(self.warn_d - self.emg_d, 1e-6)
        front_ratio = max(0.1, min(1.0, front_ratio))  # mínimo 10% de ref_v
        v_linear = self.ref_v * front_ratio

        # Si el frente está muy comprometido, forzar giro en dirección de bordeo
        if front < self.emg_d * 1.4:
            # Esquina detectada: parar avance y girar en la dirección de bordeo
            v_linear  = 0.0
            w_lateral = turn_sign * self.ref_w

        cmd = Twist()
        cmd.linear.x  = v_linear
        cmd.angular.z = w_lateral
        return cmd

    # ── BUG2: condición de salida ─────────────────────────────────────────

    def _check_bug2_exit(self, front: float, now: float) -> bool:
        """
        [BUG2-3 + FIX-4] Condición de salida BUG2:
          1. Tiempo mínimo transcurrido en wall-follow (anti-ruido odometría).
          2. Distancia mínima recorrida.
          3. Sobre la línea M (distancia perpendicular < m_line_tol).
          4. Más cerca al goal que en el hit point.
          5. Frente libre (>= warn_d) — no re-choca al retomar rumbo.
        """
        if self._goal_x is None:
            return False

        # 1. Tiempo mínimo [FIX-4]
        if (now - self._bug2_enter_ts) < self._bug2_min_time:
            return False

        # 2. Distancia mínima recorrida
        if self._bug2_traveled < self._bug2_min_follow:
            return False

        # 3. Sobre la línea M
        if not self._on_m_line(self._robot_x, self._robot_y):
            return False

        # 4. Más cerca al goal
        curr_dist = self._dist_to_goal(self._robot_x, self._robot_y)
        if curr_dist >= self._hit_dist_goal:
            return False

        # 5. Frente libre — umbral subido a warn_d para mayor margen [FIX-4]
        if front < self.warn_d:
            return False

        return True

    def _on_m_line(self, px: float, py: float) -> bool:
        """
        Distancia perpendicular del punto (px, py) a la línea M
        (hit_point → goal). Devuelve True si < m_line_tol y la proyección
        cae ENTRE hit y goal (no detrás del hit).
        """
        if self._goal_x is None:
            return False

        ax, ay = self._hit_x, self._hit_y
        bx, by = self._goal_x, self._goal_y

        dx, dy = bx - ax, by - ay
        seg_len = math.hypot(dx, dy)

        if seg_len < 1e-6:
            return True

        cross    = abs(dx * (ay - py) - dy * (ax - px))
        dist_perp = cross / seg_len

        t = ((px - ax) * dx + (py - ay) * dy) / (seg_len * seg_len)

        return dist_perp < self._m_line_tol and t > 0.0

    def _dist_to_goal(self, x: float, y: float) -> float:
        if self._goal_x is None:
            return float('inf')
        return math.hypot(self._goal_x - x, self._goal_y - y)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _maybe_trigger_replan(self, now: float):
        if self._replanning:
            return
        if (now - self._last_replan_ts) < self._replan_cooldown:
            return

        msg = Pose2D()
        msg.x = self._robot_x
        msg.y = self._robot_y
        self._pub_replan.publish(msg)

        self._replanning     = True
        self._last_replan_ts = now

        self.get_logger().warn(
            f'[BugReflex] Replan disparado desde '
            f'({self._robot_x:.2f}, {self._robot_y:.2f})')

    def _sector_min(self, scan: LaserScan, center_rad: float, half_rad: float) -> float:
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        angles = (scan.angle_min
                  + np.arange(len(ranges), dtype=np.float32) * scan.angle_increment
                  + self._lidar_yaw_offset)

        diff = np.arctan2(np.sin(angles - center_rad),
                          np.cos(angles - center_rad))

        mask = (
            (np.abs(diff) <= half_rad) &
            (ranges > scan.range_min) &
            (ranges < scan.range_max)
        )

        valid = ranges[mask]
        return float(np.min(valid)) if valid.size > 0 else float('inf')

    def _publish(self, cmd: Twist, mode: str):
        if mode != self._mode:
            if mode not in (PASS_THROUGH, PREDICTIVE_BRAKE):
                self.get_logger().warn(
                    f'[BugReflex] {self._mode} → {mode}',
                    throttle_duration_sec=0.4)
            if self._mode == BUG2_WALL_FOLLOW and mode == PASS_THROUGH:
                pause_msg = Bool(); pause_msg.data = False
                self._pub_nav_pause.publish(pause_msg)
            self._mode = mode

        self._pub_cmd.publish(cmd)
        s = String(); s.data = mode
        self._pub_status.publish(s)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = BugReflex()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._pub_cmd.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
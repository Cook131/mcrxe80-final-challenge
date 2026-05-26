#!/usr/bin/env python3
"""
bug2_navigator.py  —  Task 2: Bug 2 Reactive Navigation
=========================================================
Algoritmo Bug 2:
  Define la M-Line: recta desde la posición inicial hasta el goal.

  - GO_TO_GOAL  : Avanza directo al goal mientras no haya obstáculo.
      → Si detecta obstáculo: registra hit_point, pasa a FOLLOW_WALL.

  - FOLLOW_WALL : Sigue la pared derecha (right-wall follower).
      → Regresa a GO_TO_GOAL cuando:
          a) la distancia perpendicular a la M-Line < mline_tol, Y
          b) la distancia al goal < distancia en hit_point (más cerca).

Covarianza / dead reckoning:
  El robot integra velocidades → el error de posición crece con la
  distancia. En FOLLOW_WALL la odometría puede desviarse hasta ~5–10 cm/m.
  Por eso mline_tol=0.15 m (generoso) y hit_dist_margin=0.10 m.
  Si el robot no retoma la M-Line correctamente, aumentar mline_tol.

Suscribe:  /scan  (sensor_msgs/LaserScan)
           /odom  (nav_msgs/Odometry)
Publica:   /cmd_vel (geometry_msgs/Twist)

Sin librerías externas. Solo stdlib + math.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import math


class Bug2Navigator(Node):

    # ── Estados ───────────────────────────────────────────────────────────
    GO_TO_GOAL   = 0
    FOLLOW_WALL  = 1
    GOAL_REACHED = 2

    STATE_NAMES = {0: 'GO_TO_GOAL', 1: 'FOLLOW_WALL', 2: 'GOAL_REACHED'}

    def __init__(self):
        super().__init__('bug2_navigator')

        # ── Parámetros ────────────────────────────────────────────────────
        self.declare_parameter('goal_x',           5.0)
        self.declare_parameter('goal_y',           0.0)
        self.declare_parameter('start_x',          0.0)    # origen de la M-Line
        self.declare_parameter('start_y',          0.0)
        self.declare_parameter('obstacle_dist',    0.40)   # [m] umbral de detección
        self.declare_parameter('dist_tol',         0.15)   # [m] tolerancia de llegada
        self.declare_parameter('mline_tol',        0.15)   # [m] tolerancia sobre M-Line
        self.declare_parameter('wall_dist_target', 0.30)   # [m] distancia deseada a la pared
        self.declare_parameter('v_max',            0.16)   # [m/s] GO_TO_GOAL
        self.declare_parameter('v_wall',           0.10)   # [m/s] FOLLOW_WALL
        self.declare_parameter('w_max',            0.70)   # [rad/s] máximo giro
        self.declare_parameter('kp_w',             1.4)    # ganancia angular
        self.declare_parameter('wall_min_steps',   50)     # pasos mínimos en FOLLOW_WALL
                                                           # (~2.5 s a 20 Hz)

        self.goal_x      = float(self.get_parameter('goal_x').value)
        self.goal_y      = float(self.get_parameter('goal_y').value)
        start_x          = float(self.get_parameter('start_x').value)
        start_y          = float(self.get_parameter('start_y').value)
        self.obs_d       = float(self.get_parameter('obstacle_dist').value)
        self.dist_tol    = float(self.get_parameter('dist_tol').value)
        self.mline_tol   = float(self.get_parameter('mline_tol').value)
        self.wall_tgt    = float(self.get_parameter('wall_dist_target').value)
        self.v_max       = float(self.get_parameter('v_max').value)
        self.v_wall      = float(self.get_parameter('v_wall').value)
        self.w_max       = float(self.get_parameter('w_max').value)
        self.kp_w        = float(self.get_parameter('kp_w').value)
        self.wall_min_st = int(self.get_parameter('wall_min_steps').value)

        # ── M-Line (segmento start → goal) ───────────────────────────────
        self._ml_x1, self._ml_y1 = start_x,       start_y
        self._ml_x2, self._ml_y2 = self.goal_x,   self.goal_y
        self._ml_len = math.hypot(self.goal_x - start_x, self.goal_y - start_y)

        # ── Estado interno ────────────────────────────────────────────────
        self.state  = self.GO_TO_GOAL
        self.x      = 0.0
        self.y      = 0.0
        self.theta  = 0.0
        self.scan   = None

        # hit_point: posición donde el robot chocó y entró a FOLLOW_WALL
        self.hit_x          = 0.0
        self.hit_y          = 0.0
        self.hit_dist_goal  = float('inf')   # distancia al goal desde hit_point
        self._wall_steps    = 0              # pasos en FOLLOW_WALL (anti-flicker)

        # Seguimiento de covarianza
        self._dist_traveled = 0.0
        self._prev_x        = 0.0
        self._prev_y        = 0.0

        # ── Suscriptores y publicadores ───────────────────────────────────
        self.create_subscription(LaserScan, 'scan', self._scan_cb, 10)
        self.create_subscription(Odometry,  'odom', self._odom_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.create_timer(0.05, self._loop)   # 20 Hz

        self.get_logger().info(
            f'[Bug2] Iniciado | goal=({self.goal_x:.2f},{self.goal_y:.2f}) | '
            f'M-Line: ({start_x:.2f},{start_y:.2f})→({self.goal_x:.2f},{self.goal_y:.2f}) | '
            f'len={self._ml_len:.2f} m')

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _scan_cb(self, msg: LaserScan):
        self.scan = msg

    def _odom_cb(self, msg: Odometry):
        self.x     = msg.pose.pose.position.x
        self.y     = msg.pose.pose.position.y
        qz         = msg.pose.pose.orientation.z
        qw         = msg.pose.pose.orientation.w
        self.theta = 2.0 * math.atan2(qz, qw)

        d = math.hypot(self.x - self._prev_x, self.y - self._prev_y)
        self._dist_traveled += d
        self._prev_x, self._prev_y = self.x, self.y

        if self._dist_traveled > 10.0:
            self.get_logger().warn(
                f'[COVARIANCE] {self._dist_traveled:.1f} m recorridos. '
                'Covarianza creciente: si el robot no retoma la M-Line, '
                'aumentar el parámetro mline_tol.',
                throttle_duration_sec=6.0)

    # ── Geometría ─────────────────────────────────────────────────────────

    def _dist_to_mline(self, px: float, py: float) -> float:
        """Distancia perpendicular del punto (px,py) a la M-Line (segmento infinito)."""
        x1, y1 = self._ml_x1, self._ml_y1
        x2, y2 = self._ml_x2, self._ml_y2
        if self._ml_len < 1e-9:
            return math.hypot(px - x1, py - y1)
        # |cross product| / length
        num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
        return num / self._ml_len

    def _is_on_mline_segment(self, px: float, py: float) -> bool:
        """
        True si (px,py) está dentro de mline_tol de la M-Line Y
        su proyección cae entre start y goal (no en extensiones).
        """
        if self._dist_to_mline(px, py) > self.mline_tol:
            return False
        # Parámetro t de proyección sobre el segmento
        dx = self._ml_x2 - self._ml_x1
        dy = self._ml_y2 - self._ml_y1
        denom = dx * dx + dy * dy
        if denom < 1e-9:
            return False
        t = ((px - self._ml_x1) * dx + (py - self._ml_y1) * dy) / denom
        return 0.05 <= t <= 1.0    # 0.05: ignora los primeros cm (zona de start)

    def _dist_to_goal(self, px: float = None, py: float = None) -> float:
        if px is None:
            px, py = self.x, self.y
        return math.hypot(self.goal_x - px, self.goal_y - py)

    # ── Utilidades de sensor ──────────────────────────────────────────────

    def _sector_min(self, center_rad: float, half_rad: float) -> float:
        """Mínimo rango válido en sector [center±half] rad."""
        if self.scan is None:
            return float('inf')
        min_r = float('inf')
        for i, r in enumerate(self.scan.ranges):
            a = self.scan.angle_min + i * self.scan.angle_increment
            diff = math.atan2(math.sin(a - center_rad), math.cos(a - center_rad))
            if abs(diff) <= half_rad:
                if self.scan.range_min < r < self.scan.range_max:
                    min_r = min(min_r, r)
        return min_r

    @staticmethod
    def _wrap(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _stop(self):
        self.cmd_pub.publish(Twist())

    # ── Bucle de control ──────────────────────────────────────────────────

    def _loop(self):
        if self.scan is None:
            return

        dtg = self._dist_to_goal()
        cmd = Twist()

        # ── GOAL_REACHED ──────────────────────────────────────────────────
        if dtg < self.dist_tol:
            if self.state != self.GOAL_REACHED:
                self.state = self.GOAL_REACHED
                self.get_logger().info(
                    f'[Bug2] ¡GOAL ALCANZADO! pos=({self.x:.2f},{self.y:.2f}) '
                    f'dist_traveled={self._dist_traveled:.1f} m')
            self._stop()
            return

        # Lecturas de sectores del lidar
        #   Frente:         centro=0°,  half=±25°
        #   Frente-derecha: centro=-45°, half=±20°
        #   Derecha:        centro=-90°, half=±25°
        front       = self._sector_min(0.0,                  math.radians(25))
        front_right = self._sector_min(math.radians(-45),    math.radians(20))
        right       = self._sector_min(math.radians(-90),    math.radians(25))

        # ── GO_TO_GOAL ────────────────────────────────────────────────────
        if self.state == self.GO_TO_GOAL:
            if front < self.obs_d:
                # Registra hit_point y cambia a FOLLOW_WALL
                self.hit_x         = self.x
                self.hit_y         = self.y
                self.hit_dist_goal = dtg
                self._wall_steps   = 0
                self._transition(self.FOLLOW_WALL,
                                 f'obstáculo a {front:.2f} m | hit_dist={dtg:.2f} m')
            else:
                dx = self.goal_x - self.x
                dy = self.goal_y - self.y
                heading  = math.atan2(dy, dx)
                ang_err  = self._wrap(heading - self.theta)
                cmd.angular.z = max(-1.2, min(1.2, self.kp_w * ang_err))
                align = math.cos(ang_err)
                if align > 0.2:
                    cmd.linear.x = min(self.v_max, self.v_max * align)

        # ── FOLLOW_WALL ───────────────────────────────────────────────────
        elif self.state == self.FOLLOW_WALL:
            self._wall_steps += 1

            # Verificar condición de retorno a GO_TO_GOAL:
            #   a) Hemos dado suficientes pasos (evita flicker en el punto de entrada)
            #   b) Estamos sobre la M-Line (segmento)
            #   c) Estamos más cerca del goal que cuando chocamos
            if self._wall_steps > self.wall_min_st:
                closer   = dtg < self.hit_dist_goal - 0.10
                on_mline = self._is_on_mline_segment(self.x, self.y)

                if on_mline and closer:
                    self._transition(self.GO_TO_GOAL,
                                     f'retomó M-Line | dtg={dtg:.2f} m '
                                     f'(hit={self.hit_dist_goal:.2f} m)')
                    # Re-ejecuta GO_TO_GOAL en este mismo ciclo
                    dx = self.goal_x - self.x
                    dy = self.goal_y - self.y
                    heading = math.atan2(dy, dx)
                    ang_err = self._wrap(heading - self.theta)
                    cmd.angular.z = max(-1.2, min(1.2, self.kp_w * ang_err))
                    self.cmd_pub.publish(cmd)
                    return

            # Right-wall follower
            # ┌─────────────────────────────────────────────────────────┐
            # │ Prioridad (orden):                                      │
            # │  1. Obstáculo al frente → girar izquierda (CCW)        │
            # │  2. Obstáculo frente-derecha → girar izquierda suave   │
            # │  3. Pared derecha demasiado lejos → girar derecha (CW) │
            # │  4. En rango óptimo → avanzar recto                    │
            # └─────────────────────────────────────────────────────────┘
            if front < self.obs_d:
                cmd.angular.z = +self.w_max * 0.9          # giro brusco izquierda
                cmd.linear.x  = 0.0
            elif front_right < self.obs_d * 0.85:
                cmd.angular.z = +self.w_max * 0.45         # giro suave izquierda
                cmd.linear.x  = self.v_wall * 0.6
            elif right > self.wall_tgt + 0.12:
                cmd.angular.z = -self.w_max * 0.40         # giro suave derecha
                cmd.linear.x  = self.v_wall
            else:
                cmd.linear.x  = self.v_wall                # recto, pared en rango
                cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)

    def _transition(self, new_state: int, reason: str):
        old = self.STATE_NAMES[self.state]
        new = self.STATE_NAMES[new_state]
        self.state = new_state
        self.get_logger().info(
            f'[Bug2] {old} → {new} | {reason} | '
            f'pos=({self.x:.2f},{self.y:.2f}) θ={math.degrees(self.theta):.1f}°')


def main(args=None):
    rclpy.init(args=args)
    node = Bug2Navigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

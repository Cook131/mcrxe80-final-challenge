#!/usr/bin/env python3
"""
bug0_navigator.py  —  Task 1: Bug 0 Reactive Navigation
=========================================================
Algoritmo Bug 0:
  - GO_TO_GOAL   : Dirige el robot al goal en línea recta.
  - AVOID_OBSTACLE: Rota en el lugar hasta que el frente esté libre.
  - Repite hasta llegar al goal.

Nota sobre covarianza:
  El dead reckoning de Gazebo acumula error con la distancia recorrida,
  especialmente en giros. Si el robot pierde el goal a gran distancia,
  reducir dist_tol (tolerancia) o relanzar desde posición inicial.

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


class Bug0Navigator(Node):

    # ── Estados ───────────────────────────────────────────────────────────
    GO_TO_GOAL      = 0
    AVOID_OBSTACLE  = 1
    GOAL_REACHED    = 2

    STATE_NAMES = {0: 'GO_TO_GOAL', 1: 'AVOID_OBSTACLE', 2: 'GOAL_REACHED'}

    def __init__(self):
        super().__init__('bug0_navigator')

        # ── Parámetros ────────────────────────────────────────────────────
        self.declare_parameter('goal_x',            4.0)
        self.declare_parameter('goal_y',            0.0)
        self.declare_parameter('obstacle_dist',     0.40)   # [m] umbral de obstáculo
        self.declare_parameter('dist_tol',          0.15)   # [m] tolerancia de llegada
        self.declare_parameter('front_half_angle',  45.0)   # [deg] semiángulo del cono frontal
        self.declare_parameter('v_max',             0.18)   # [m/s] velocidad lineal máxima
        self.declare_parameter('w_avoid',           0.80)   # [rad/s] velocidad de giro al evitar
        self.declare_parameter('kp_w',              1.4)    # ganancia angular proporcional
        self.declare_parameter('turn_dir',          -1.0)    # +1.0=CCW, -1.0=CW

        self.goal_x   = float(self.get_parameter('goal_x').value)
        self.goal_y   = float(self.get_parameter('goal_y').value)
        self.obs_d    = float(self.get_parameter('obstacle_dist').value)
        self.dist_tol = float(self.get_parameter('dist_tol').value)
        self.front_h  = math.radians(self.get_parameter('front_half_angle').value)
        self.v_max    = float(self.get_parameter('v_max').value)
        self.w_avoid  = float(self.get_parameter('w_avoid').value)
        self.kp_w     = float(self.get_parameter('kp_w').value)
        self.turn_dir = float(self.get_parameter('turn_dir').value)

        # ── Estado interno ────────────────────────────────────────────────
        self.state  = self.GO_TO_GOAL
        self.x      = 0.0
        self.y      = 0.0
        self.theta  = 0.0
        self.scan   = None   # último LaserScan recibido
        self._avoid_steps   = 0
        self._AVOID_MIN_ST  = 15   # ~0.75 s a 20 Hz

        # Seguimiento de distancia recorrida (para advertencia de covarianza)
        self._dist_traveled = 0.0
        self._prev_x        = 0.0
        self._prev_y        = 0.0
        self._COV_WARN_M    = 8.0   # [m] umbral de advertencia

        # ── Suscriptores y publicadores ───────────────────────────────────
        self.create_subscription(LaserScan, 'scan', self._scan_cb, 10)
        self.create_subscription(Odometry,  'odom', self._odom_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.create_timer(0.05, self._loop)   # 20 Hz

        self.get_logger().info(
            f'[Bug0] Iniciado | goal=({self.goal_x:.2f}, {self.goal_y:.2f}) | '
            f'obs_threshold={self.obs_d:.2f} m')

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _scan_cb(self, msg: LaserScan):
        self.scan = msg

    def _odom_cb(self, msg: Odometry):
        self.x     = msg.pose.pose.position.x
        self.y     = msg.pose.pose.position.y
        qz         = msg.pose.pose.orientation.z
        qw         = msg.pose.pose.orientation.w
        self.theta = 2.0 * math.atan2(qz, qw)

        # Acumula distancia recorrida
        d = math.hypot(self.x - self._prev_x, self.y - self._prev_y)
        self._dist_traveled += d
        self._prev_x, self._prev_y = self.x, self.y

        # Advertencia de covarianza creciente
        if self._dist_traveled > self._COV_WARN_M:
            self.get_logger().warn(
                f'[COVARIANCE] {self._dist_traveled:.1f} m de dead reckoning acumulado. '
                'El error de posición crece — los umbrales de llegada pueden necesitar ajuste.',
                throttle_duration_sec=6.0)

    # ── Utilidades de sensor ──────────────────────────────────────────────

    def _sector_min(self, center_rad: float, half_rad: float) -> float:
        """Distancia mínima válida dentro del sector [center±half] radianes."""
        if self.scan is None:
            return float('inf')
        min_r = float('inf')
        for i, r in enumerate(self.scan.ranges):
            a = self.scan.angle_min + i * self.scan.angle_increment
            diff = math.atan2(math.sin(a - center_rad),
                              math.cos(a - center_rad))      # wrap a [-π,π]
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

        cmd = Twist()

        # Distancia al goal
        dx  = self.goal_x - self.x
        dy  = self.goal_y - self.y
        dtg = math.hypot(dx, dy)

        # ── GOAL_REACHED ──────────────────────────────────────────────────
        if dtg < self.dist_tol:
            if self.state != self.GOAL_REACHED:
                self.state = self.GOAL_REACHED
                self.get_logger().info(
                    f'[Bug0] ¡GOAL ALCANZADO! pos=({self.x:.2f},{self.y:.2f}) '
                    f'dist_traveled={self._dist_traveled:.1f} m')
            self._stop()
            return

        front_min = self._sector_min(0.0, self.front_h)

        # ── GO_TO_GOAL ────────────────────────────────────────────────────
        if self.state == self.GO_TO_GOAL:
            if front_min < self.obs_d:
                self._transition(self.AVOID_OBSTACLE,
                                 f'obstáculo a {front_min:.2f} m')
            else:
                heading = math.atan2(dy, dx)
                ang_err = self._wrap(heading - self.theta)
                cmd.angular.z = max(-1.2, min(1.2, self.kp_w * ang_err))
                # Velocidad lineal escalada por alineación (coseno del error)
                align = math.cos(ang_err)
                if align > 0.2:
                    cmd.linear.x = min(self.v_max, self.v_max * align)

        # ── AVOID_OBSTACLE ────────────────────────────────────────────────
        elif self.state == self.AVOID_OBSTACLE:
            self._avoid_steps += 1
            clear_threshold = self.obs_d + 0.12
            if self._avoid_steps > self._AVOID_MIN_ST and front_min > clear_threshold:
                self._avoid_steps = 0
                self._transition(self.GO_TO_GOAL, f'frente libre ({front_min:.2f} m)')
            else:
                cmd.angular.z = self.turn_dir * self.w_avoid

        self.cmd_pub.publish(cmd)

    def _transition(self, new_state: int, reason: str):
        old = self.STATE_NAMES[self.state]
        new = self.STATE_NAMES[new_state]
        self.state = new_state
        self.get_logger().info(
            f'[Bug0] {old} → {new} | {reason} | '
            f'pos=({self.x:.2f},{self.y:.2f}) θ={math.degrees(self.theta):.1f}°')


def main(args=None):
    rclpy.init(args=args)
    node = Bug0Navigator()
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

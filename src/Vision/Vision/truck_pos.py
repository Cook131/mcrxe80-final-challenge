#!/usr/bin/env python3
"""
YOLO World Position Node para Puzzlebot - Manchester Robotics

Calcula la posición global del logo detectado por yolo_detector_node
aplicando un offset de 20 cm a la derecha en el frame del robot.

El offset modela el punto de interés real (ej. entrada del camión)
que está desplazado lateralmente respecto al centro del logo.

Tópicos:
  Suscribe: /yolo/distance  (std_msgs/Float32)  — distancia en metros
            /yolo/angle     (std_msgs/Float32)   — ángulo horizontal en grados
            /odom           (nav_msgs/Odometry)

  Publica:  /yolo/world_pos (geometry_msgs/PointStamped) — posición global con offset
"""

import math

import rclpy
import rclpy.duration
from rclpy.node import Node
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped


# Offset lateral en metros — positivo = derecha en frame robot
LATERAL_OFFSET_M = 0.20


class YoloWorldPos(Node):

    def __init__(self):
        super().__init__('yolo_world_pos')

        # Estado del robot
        self.robot_x     = 0.0
        self.robot_y     = 0.0
        self.robot_theta = 0.0

        # Datos del detector
        self.yolo_distance  = None
        self.yolo_angle     = None
        self._dist_stamp    = None
        self._angle_stamp   = None
        self.STALE_SEC      = 0.5

        # Suscriptores
        self.create_subscription(Odometry, '/odom',          self.odom_callback,  10)
        self.create_subscription(Float32,  '/yolo/distance', self.dist_callback,  10)
        self.create_subscription(Float32,  '/yolo/angle',    self.angle_callback, 10)

        # Publicador
        self.pub_world_pos = self.create_publisher(
            PointStamped, '/yolo/world_pos', 10)

        self.get_logger().info(
            f'YOLO World Pos listo\n'
            f'  Offset lateral: {LATERAL_OFFSET_M*100:.0f} cm a la derecha\n'
            f'  Publica: /yolo/world_pos'
        )

    # ── Quaternion → yaw ──────────────────────────────────────
    def _get_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    # ── Callbacks ─────────────────────────────────────────────
    def odom_callback(self, msg):
        self.robot_x     = msg.pose.pose.position.x
        self.robot_y     = msg.pose.pose.position.y
        self.robot_theta = self._get_yaw(msg.pose.pose.orientation)

    def dist_callback(self, msg):
        self.yolo_distance = msg.data
        self._dist_stamp   = self.get_clock().now()

    def angle_callback(self, msg):
        self.yolo_angle    = msg.data
        self._angle_stamp  = self.get_clock().now()
        self._try_compute()

    # ── Lógica principal ──────────────────────────────────────
    def _try_compute(self):
        if self.yolo_distance is None or self.yolo_angle is None:
            return

        now   = self.get_clock().now()
        stale = rclpy.duration.Duration(seconds=self.STALE_SEC)
        if (now - self._dist_stamp) > stale or (now - self._angle_stamp) > stale:
            self.get_logger().warn('Datos YOLO stale, ignorando')
            return

        self._compute_and_publish(self.yolo_distance, self.yolo_angle)

    def _compute_and_publish(self, distance: float, angle_deg: float):
        # ── Posición del logo en frame global ─────────────────
        angle_rad     = math.radians(angle_deg)
        angulo_global = self.robot_theta + angle_rad

        logo_x = self.robot_x + distance * math.cos(angulo_global)
        logo_y = self.robot_y + distance * math.sin(angulo_global)

        # ── Offset lateral (20 cm a la derecha del robot) ─────
        # "Derecha del robot" = robot_theta - 90°
        # Vector unitario apuntando a la derecha del robot:
        #   right_x = cos(robot_theta - π/2) =  sin(robot_theta)
        #   right_y = sin(robot_theta - π/2) = -cos(robot_theta)
        right_x =  math.sin(self.robot_theta)
        right_y = -math.cos(self.robot_theta)

        target_x = logo_x + LATERAL_OFFSET_M * right_x
        target_y = logo_y + LATERAL_OFFSET_M * right_y

        # ── Publicar ──────────────────────────────────────────
        ps = PointStamped()
        ps.header.stamp    = self.get_clock().now().to_msg()
        ps.header.frame_id = 'map'
        ps.point.x = target_x
        ps.point.y = target_y
        ps.point.z = 0.0
        self.pub_world_pos.publish(ps)

        self.get_logger().info(
            f'Logo ({logo_x:.2f}, {logo_y:.2f}) '
            f'→ target con offset ({target_x:.2f}, {target_y:.2f})'
        )


def main(args=None):
    rclpy.init(args=args)
    node = YoloWorldPos()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
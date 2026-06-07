#!/usr/bin/env python3
"""
QR Zone Checker Node para Puzzlebot - Manchester Robotics

Determina si un QR detectado está dentro de una zona predefinida
y publica el trigger correspondiente ('conveyor' o 'rack').

Tópicos QR (estandarizados):
  Suscribe: /qr/distance  (std_msgs/Float32)  — distancia en metros
            /qr/angle     (std_msgs/Float32)   — ángulo horizontal en grados

  Suscribe: /odom         (nav_msgs/Odometry)

  Publica:  /collect/trigger (std_msgs/String)         — 'conveyor' | 'rack'
            /qr/world_pos  (geometry_msgs/PointStamped) — posición global del QR (x, y)
"""

import math

import rclpy
from rclpy.node import Node
import rclpy.duration
from std_msgs.msg import Float32, String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped


class QRZoneChecker(Node):

    def __init__(self):
        super().__init__('qr_zone_checker')

        # Límites de la zona
        self.zona_x_min = -2.81
        self.zona_x_max = -1.60
        self.zona_y_min = -1.84
        self.zona_y_max =  1.84

        # Estado del robot y sensor
        self.robot_x      = 0.0
        self.robot_y      = 0.0
        self.robot_theta  = 0.0
        self.qr_distance  = None
        self.qr_angle     = None
        self._dist_stamp  = None
        self._angle_stamp = None
        self._last_trigger = None
        self.STALE_SEC    = 0.5

        # Suscriptores
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        # Tópicos QR estandarizados (publicados por qr_detector_node)
        self.sub_dist = self.create_subscription(
            Float32, '/qr/distance', self.distance_callback, 10)
        self.sub_angle = self.create_subscription(
            Float32, '/qr/angle', self.angle_callback, 10)

        # Publicadores
        self.pub_trigger  = self.create_publisher(
            String,       '/collect/trigger', 10)
        self.pub_world_pos = self.create_publisher(
            PointStamped, '/qr/world_pos',    10)

        self.get_logger().info(
            'QR Zone Checker listo\n'
            '  QR topics: /qr/distance | /qr/angle\n'
            f'  Zona X=[{self.zona_x_min}, {self.zona_x_max}]  '
            f'Y=[{self.zona_y_min}, {self.zona_y_max}]\n'
            '  Publica: /collect/trigger | /qr/world_pos'
        )

    def get_yaw_from_quaternion(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg):
        self.robot_x     = msg.pose.pose.position.x
        self.robot_y     = msg.pose.pose.position.y
        self.robot_theta = self.get_yaw_from_quaternion(msg.pose.pose.orientation)

    def distance_callback(self, msg):
        self.qr_distance = msg.data
        self._dist_stamp = self.get_clock().now()

    def angle_callback(self, msg):
        self.qr_angle     = msg.data
        self._angle_stamp = self.get_clock().now()
        self._try_check_zone()

    def _try_check_zone(self):
        if self.qr_distance is None or self.qr_angle is None:
            return

        # Descartar datos stale
        now   = self.get_clock().now()
        stale = rclpy.duration.Duration(seconds=self.STALE_SEC)
        if (now - self._dist_stamp) > stale or (now - self._angle_stamp) > stale:
            self.get_logger().warn('Datos QR stale, ignorando')
            return

        self._check_zone(self.qr_distance, self.qr_angle)

    def _check_zone(self, distance: float, angle_deg: float):
        # Convertir ángulo a radianes y proyectar al frame global
        angulo_global = self.robot_theta + math.radians(angle_deg)
        qr_x = self.robot_x + distance * math.cos(angulo_global)
        qr_y = self.robot_y + distance * math.sin(angulo_global)

        # Publicar posición global del QR (siempre, independiente de zona)
        ps = PointStamped()
        ps.header.stamp    = self.get_clock().now().to_msg()
        ps.header.frame_id = 'map'
        ps.point.x = qr_x
        ps.point.y = qr_y
        ps.point.z = 0.0
        self.pub_world_pos.publish(ps)

        # Determinar zona y publicar trigger solo si cambia
        en_zona = (self.zona_x_min <= qr_x <= self.zona_x_max and
                   self.zona_y_min <= qr_y <= self.zona_y_max)
        trigger = 'conveyor' if en_zona else 'rack'

        if trigger != self._last_trigger:
            self.pub_trigger.publish(String(data=trigger))
            self.get_logger().info(
                f'QR global ({qr_x:.2f}, {qr_y:.2f}) → {trigger}')
            self._last_trigger = trigger


def main(args=None):
    rclpy.init(args=args)
    nodo = QRZoneChecker()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
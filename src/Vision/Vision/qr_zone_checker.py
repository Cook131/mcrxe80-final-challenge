#!/usr/bin/env python3
"""
qr_zone_checker.py — Puzzlebot / Iolair
Determina si un QR detectado está dentro de la zona conveyor o rack
y publica el trigger + la posición global del QR.

Tópicos:
  Sub:  /qr/distance      (std_msgs/Float32)   — distancia en metros
        /qr/angle         (std_msgs/Float32)   — ángulo horizontal en radianes
        /odom             (nav_msgs/Odometry)
  Pub:  /collect/trigger  (std_msgs/String)    — 'conveyor' | 'rack'
        /qr/world_pos     (geometry_msgs/PointStamped) — posición global del QR
"""
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped


class QRZoneChecker(Node):

    def __init__(self):
        super().__init__('qr_zone_checker')

        # Zona conveyor
        self.zona_x_min = -2.81
        self.zona_x_max = -1.60
        self.zona_y_min = -1.84
        self.zona_y_max =  1.84

        # Estado
        self.robot_x     = 0.0
        self.robot_y     = 0.0
        self.robot_theta = 0.0
        self.qr_distance = None

        # Subs
        self.create_subscription(Odometry, '/odom',         self.odom_callback,     10)
        self.create_subscription(Float32,  '/qr/distance',  self.distance_callback, 10)
        self.create_subscription(Float32,  '/qr/angle',     self.angle_callback,    10)

        # Pubs
        self.pub_trigger   = self.create_publisher(String,       '/collect/trigger', 10)
        self.pub_world_pos = self.create_publisher(PointStamped, '/qr/world_pos',    10)

        self.get_logger().info(
            'QR Zone Checker listo\n'
            '  Sub: /qr/distance | /qr/angle | /odom\n'
            '  Pub: /collect/trigger | /qr/world_pos\n'
            f'  Zona conveyor X=[{self.zona_x_min}, {self.zona_x_max}]'
            f'  Y=[{self.zona_y_min}, {self.zona_y_max}]')

    def odom_callback(self, msg):
        self.robot_x     = msg.pose.pose.position.x
        self.robot_y     = msg.pose.pose.position.y
        self.robot_theta = self._yaw_from_quat(msg.pose.pose.orientation)

    def distance_callback(self, msg):
        self.qr_distance = msg.data

    def angle_callback(self, msg):
        if self.qr_distance is None:
            return

        # Posición global del QR
        angulo_global = self.robot_theta + msg.data
        qr_x = self.robot_x + self.qr_distance * math.cos(angulo_global)
        qr_y = self.robot_y + self.qr_distance * math.sin(angulo_global)

        # Zona
        en_zona = (self.zona_x_min <= qr_x <= self.zona_x_max and
                   self.zona_y_min <= qr_y <= self.zona_y_max)
        label = 'conveyor' if en_zona else 'rack'

        # Publicar trigger
        self.pub_trigger.publish(String(data=label))

        # Publicar posición global
        ps = PointStamped()
        ps.header.stamp    = self.get_clock().now().to_msg()
        ps.header.frame_id = 'map'
        ps.point.x = qr_x
        ps.point.y = qr_y
        ps.point.z = 0.0
        self.pub_world_pos.publish(ps)

        self.get_logger().info(
            f'QR ({qr_x:.3f}, {qr_y:.3f}) → {label}')

    @staticmethod
    def _yaw_from_quat(q):
        return math.atan2(2*(q.w*q.z + q.x*q.y),
                          1 - 2*(q.y*q.y + q.z*q.z))


def main(args=None):
    rclpy.init(args=args)
    node = QRZoneChecker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
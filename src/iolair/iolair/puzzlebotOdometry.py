#!/usr/bin/env python3
"""
puzzlebotOdometry.py — Odometría por encoders de rueda
======================================================

Cinemática diferencial estándar (mano derecha, frame ROS):
  v = r * (wr + wl) / 2
  w = r * (wr - wl) / L

  x   += v * cos(th) * dt
  y   += v * sin(th) * dt
  th  += w * dt

La covarianza se propaga analíticamente con el Jacobiano del modelo
de movimiento (EKF de proceso), lo que le da al slam_node información
cuantitativa de cuánto confiar en la odometría vs. el ICP.

Publica:
  /odom  (nav_msgs/Odometry)  — pose + twist + covarianza

Suscribe:
  /VelocityEncL  (std_msgs/Float32) — velocidad angular rueda izquierda [rad/s]
  /VelocityEncR  (std_msgs/Float32) — velocidad angular rueda derecha   [rad/s]
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, TransformStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from tf2_ros import TransformBroadcaster


class PuzzlebotOdometry(Node):

    # Parámetros físicos del Puzzlebot
    WHEEL_RADIUS = 0.05   # metros
    WHEEL_BASE   = 0.19   # metros

    def __init__(self):
        super().__init__('puzzlebot_odom_node')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.sub_l = self.create_subscription(
            Float32, '/VelocityEncL', self.cb_l, qos_profile)
        self.sub_r = self.create_subscription(
            Float32, '/VelocityEncR', self.cb_r, qos_profile)
        self.pub_odom = self.create_publisher(Odometry, '/odom', 10)

        # Broadcaster TF odom → base_link
        # Necesario para que RViz y el slam_node puedan ubicar el robot
        # en el árbol de transformadas: map → odom → base_link
        self._tf_broadcaster = TransformBroadcaster(self)

        # Estado de la pose
        self.x  = 0.0
        self.y  = 0.0
        self.th = 0.0

        # Velocidades angulares de rueda [rad/s]
        self.wl = 0.0
        self.wr = 0.0

        # Covarianza de pose 3×3 [x, y, theta] — se propaga con EKF de proceso
        # Valores iniciales pequeños (posición de arranque conocida)
        self.P = np.diag([1e-6, 1e-6, 1e-6])

        # Ruido de proceso por unidad de tiempo [m²/s, m²/s, rad²/s]
        # Ajustar según el ruido real de los encoders
        self.Q_diag = np.array([0.005, 0.005, 0.01])

        self.rate = 50.0          # Hz — suficiente para esta cinemática
        self.last_time = self.get_clock().now()
        self.timer = self.create_timer(1.0 / self.rate, self.update_position)

        self.get_logger().info(
            f'Odometría iniciada — r={self.WHEEL_RADIUS} m, '
            f'L={self.WHEEL_BASE} m, rate={self.rate} Hz'
        )

    def cb_l(self, msg: Float32):
        self.wl = msg.data

    def cb_r(self, msg: Float32):
        self.wr = msg.data

    def update_position(self):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9

        # Descartar pasos demasiado cortos o demasiado largos
        # (p.ej. al inicio o tras una pausa del sistema)
        if dt < 0.001 or dt > 0.5:
            self.last_time = current_time
            return

        self.last_time = current_time

        r = self.WHEEL_RADIUS
        L = self.WHEEL_BASE

        # Cinemática diferencial
        # FIX: los encoders del Puzzlebot reportan wl y wr con convención
        # donde girar a la izquierda produce w negativo con (wr - wl).
        # El estándar ROS requiere w positivo para giro antihorario (izquierda),
        # por lo que se invierte: w = r*(wl - wr)/L
        v = r * (self.wr + self.wl) / 2.0
        w = r * (self.wl - self.wr) / L

        # Integración de Euler
        self.x  += v * math.cos(self.th) * dt
        self.y  += v * math.sin(self.th) * dt
        self.th += w * dt
        self.th  = math.atan2(math.sin(self.th), math.cos(self.th))

        # ── Propagación de covarianza (Jacobiano del modelo de movimiento) ─
        #
        # Estado q = [x, y, th].  Jacobiano F = dq'/dq:
        #   F = [[1, 0, -v*sin(th)*dt],
        #        [0, 1,  v*cos(th)*dt],
        #        [0, 0,  1           ]]
        #
        # Jacobiano de entrada G = dq'/d[v, w]:
        #   G = [[cos(th)*dt, 0     ],
        #        [sin(th)*dt, 0     ],
        #        [0,          dt    ]]
        #
        # Ruido de encoder proporcional a velocidad (modelo simple):
        #   sigma_v² = Q_diag[0] * |v|
        #   sigma_w² = Q_diag[2] * |w|
        #
        # P' = F @ P @ F.T + Q (simplificado como ruido aditivo diagonal)

        F = np.array([
            [1.0, 0.0, -v * math.sin(self.th) * dt],
            [0.0, 1.0,  v * math.cos(self.th) * dt],
            [0.0, 0.0,  1.0],
        ])
        Q = np.diag(self.Q_diag * dt)
        self.P = F @ self.P @ F.T + Q

        # ── Publicación ────────────────────────────────────────────────────

        odom_msg = Odometry()
        odom_msg.header.stamp    = current_time.to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id  = 'base_link'

        odom_msg.pose.pose.position.x  = self.x
        odom_msg.pose.pose.position.y  = self.y
        odom_msg.pose.pose.position.z  = 0.0
        odom_msg.pose.pose.orientation = self._euler_to_quaternion(
            0.0, 0.0, self.th)

        # Rellenar la covarianza 6×6 con la submatriz 3×3 relevante
        # (x, y, yaw) → índices 0, 7, 35 del vector aplanado row-major
        pose_cov = [0.0] * 36
        pose_cov[0]  = self.P[0, 0]   # var(x)
        pose_cov[1]  = self.P[0, 1]   # cov(x,y)
        pose_cov[5]  = self.P[0, 2]   # cov(x,th)
        pose_cov[6]  = self.P[1, 0]   # cov(y,x)
        pose_cov[7]  = self.P[1, 1]   # var(y)
        pose_cov[11] = self.P[1, 2]   # cov(y,th)
        pose_cov[30] = self.P[2, 0]   # cov(th,x)
        pose_cov[31] = self.P[2, 1]   # cov(th,y)
        pose_cov[35] = self.P[2, 2]   # var(th)
        odom_msg.pose.covariance = pose_cov

        odom_msg.twist.twist.linear.x  = v
        odom_msg.twist.twist.angular.z = w

        # Covarianza del twist: estimación simple basada en velocidad
        twist_cov = [0.0] * 36
        twist_cov[0]  = max(self.Q_diag[0], 0.001)  # var(vx)
        twist_cov[35] = max(self.Q_diag[2], 0.001)  # var(wz)
        odom_msg.twist.covariance = twist_cov

        self.pub_odom.publish(odom_msg)

        # Publicar TF odom → base_link
        # Esta transformada es la que completa la cadena map → odom → base_link
        tf_msg = TransformStamped()
        tf_msg.header.stamp    = current_time.to_msg()
        tf_msg.header.frame_id = 'odom'
        tf_msg.child_frame_id  = 'base_link'
        tf_msg.transform.translation.x = self.x
        tf_msg.transform.translation.y = self.y
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation      = self._euler_to_quaternion(
            0.0, 0.0, self.th)
        self._tf_broadcaster.sendTransform(tf_msg)

    @staticmethod
    def _euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        q = Quaternion()
        q.w = cr * cp * cy + sr * sp * sy
        q.x = sr * cp * cy - cr * sp * sy
        q.y = cr * sp * cy + sr * cp * sy
        q.z = cr * cp * sy - sr * sp * cy
        return q


def main(args=None):
    rclpy.init(args=args)
    node = PuzzlebotOdometry()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
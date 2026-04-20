import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, TransformStamped
import math
import time
from rclpy.qos import QoSProfile, ReliabilityPolicy

class PuzzlebotOdometry(Node):
    def __init__(self):
        super().__init__('puzzlebot_odom_node')
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, # Cambia a RELIABLE si persiste
            depth=10
        )

        self.sub_l = self.create_subscription(Float32, '/VelocityEncL', self.cb_l, qos_profile)
        self.sub_r = self.create_subscription(Float32, '/VelocityEncR', self.cb_r, qos_profile)
        # Publicador de Odometría estándar de ROS 2
        self.pub_odom = self.create_publisher(Odometry, '/odom', 10)

        # Dimensiones reales (Centroide basado en estas medidas)
        self.r = 0.05    # Radio de llanta (metros)
        self.L = 0.19    # Distancia entre llantas (metros)
        
        # Estado del robot (x, y en metros, theta en radianes)
        self.x = 0.0
        self.y = 0.0
        self.th = 0.0
        
        self.wl = 0.0
        self.wr = 0.0
        
        # Control de tiempo para integración precisa
        self.last_time = self.get_clock().now()
        
        # Timer para actualizar a 50Hz
        self.timer = self.create_timer(0.02, self.update_position)

    def cb_l(self, msg): self.wl = msg.data
    def cb_r(self, msg): self.wr = msg.data

    def update_position(self):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        # 1. Velocidad lineal y angular en el centroide
        v = self.r * (self.wr + self.wl) / 2.0
        w = self.r * (self.wr - self.wl) / self.L
        
        # 2. Diferenciales de movimiento
        delta_x = v * math.cos(self.th) * dt
        delta_y = v * math.sin(self.th) * dt
        delta_th = w * dt
        
        # 3. Actualización de la pose (Integración)
        self.x += delta_x
        self.y += delta_y
        self.th += delta_th

        # 4. Publicar el mensaje de Odometría
        odom_msg = Odometry()
        odom_msg.header.stamp = current_time.to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link" # El centroide del robot

        # Posición
        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        
        # Orientación (Convertir Euler a Cuaternión para ROS 2)
        odom_msg.pose.pose.orientation = self.euler_to_quaternion(0, 0, self.th)
        
        # Velocidades
        odom_msg.twist.twist.linear.x = v
        odom_msg.twist.twist.angular.z = w

        self.pub_odom.publish(odom_msg)

    def euler_to_quaternion(self, roll, pitch, yaw):
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        return Quaternion(x=qx, y=qy, z=qz, w=qw)

def main(args=None):
    rclpy.init(args=args)
    node = PuzzlebotOdometry()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
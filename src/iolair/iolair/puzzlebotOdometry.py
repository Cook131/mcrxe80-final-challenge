import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class PuzzlebotOdometry(Node):
    def __init__(self):
        super().__init__('puzzlebot_odom_node')

        # --- QoS de alto rendimiento para evitar lag ---
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.sub_l = self.create_subscription(Float32, '/VelocityEncL', self.cb_l, qos_profile)
        self.sub_r = self.create_subscription(Float32, '/VelocityEncR', self.cb_r, qos_profile)
        self.pub_odom = self.create_publisher(Odometry, '/odom', 10)

        # Parámetros físicos
        self.r = 0.05
        self.L = 0.19
        
        # Estado inicial
        self.x, self.y, self.th = 0.0, 0.0, 0.0
        self.wl, self.wr = 0.0, 0.0
        
        # Frecuencia a 200Hz (0.005s por ciclo)
        self.rate = 200.0
        self.last_time = self.get_clock().now()
        self.timer = self.create_timer(1.0 / self.rate, self.update_position)

        self.get_logger().info(f"🟢 Odometría iniciada (Rango Angular: -π a π rad)")

    def cb_l(self, msg): self.wl = msg.data
    def cb_r(self, msg): self.wr = msg.data

    def update_position(self):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        
        # Validación de dt para estabilidad ante lag
        if dt < 0.001 or dt > 0.1:
            self.last_time = current_time
            return
            
        self.last_time = current_time

        # 1. Cinemática Diferencial
        v = self.r * (self.wr + self.wl) / 2.0
        w = self.r * (self.wr - self.wl) / self.L
        
        # 2. Integración de Pose
        self.x += v * math.cos(self.th) * dt
        self.y += v * math.sin(self.th) * dt
        self.th += w * dt
        
        # --- MEJORA: Normalización de -PI a PI ---
        # atan2(sin, cos) es la forma más limpia de mantener el rango
        self.th = math.atan2(math.sin(self.th), math.cos(self.th))

        # 3. Publicación
        odom_msg = Odometry()
        odom_msg.header.stamp = current_time.to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"

        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        
        # Orientación en Cuaternión
        odom_msg.pose.pose.orientation = self.euler_to_quaternion(0, 0, self.th)
        
        # Velocidades en el mensaje
        odom_msg.twist.twist.linear.x = v
        odom_msg.twist.twist.angular.z = w

        self.pub_odom.publish(odom_msg)

    def euler_to_quaternion(self, roll, pitch, yaw):
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
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose2D
import math

class GoToGoalNode(Node):
    def __init__(self):
        super().__init__('go_to_goal_node')
        
        # Suscripción a la odometría que calculas en tu otro nodo
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Suscripción al objetivo (punto X, Y)
        self.sub_goal = self.create_subscription(Pose2D, '/goal', self.goal_callback, 10)
        
        # Publicador hacia tu controlador cinemático
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)

        # Estado actual y objetivo
        self.x, self.y, self.th = 0.0, 0.0, 0.0
        self.target_x, self.target_y = 0.0, 0.0
        self.active = False

        # Ganancias del controlador (puedes ajustarlas)
        self.kv = 0.5  # Ganancia velocidad lineal
        self.kw = 0.7  # Ganancia velocidad angular

        self.get_logger().info("Nodo Go-to-Goal listo. Esperando objetivo en /goal...")

    def odom_callback(self, msg):
        # Extraer posición del mensaje de odometría
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        
        # Convertir cuaternión a Euler (Yaw)
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.th = math.atan2(siny_cosp, cosy_cosp)

        if self.active:
            self.control_loop()

    def goal_callback(self, msg):
        self.target_x = msg.x
        self.target_y = msg.y
        self.active = True
        self.get_logger().info(f"Yendo al punto: x={self.target_x}, y={self.target_y}")

    def control_loop(self):
        # 1. Calcular errores
        dist = math.sqrt((self.target_x - self.x)**2 + (self.target_y - self.y)**2)
        angle_to_goal = math.atan2(self.target_y - self.y, self.target_x - self.x)
        error_angle = angle_to_goal - self.th
        
        # Normalizar ángulo entre -pi y pi
        error_angle = math.atan2(math.sin(error_angle), math.cos(error_angle))

        cmd = Twist()
        
        # 2. Condición de llegada (umbral de 5cm)
        if dist < 0.05:
            self.active = False
            self.get_logger().info("¡Llegamos al objetivo!")
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        else:
            # 3. Ley de control proporcional
            cmd.linear.x = min(self.kv * dist, 2.8) # Limitado a 5.0 m/s
            cmd.angular.z = self.kw * error_angle

        self.pub_cmd.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = GoToGoalNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
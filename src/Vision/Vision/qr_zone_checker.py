import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String  # Importamos String para enviar texto
from nav_msgs.msg import Odometry
import math

class QRZoneChecker(Node):
    def __init__(self):
        super().__init__('qr_zone_checker')

        # 1. Definir los límites de la zona
        self.zona_x_min = -2.81
        self.zona_x_max = -1.60
        self.zona_y_min = -1.84
        self.zona_y_max = 1.84

        # Variables de estado del robot y sensor
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.qr_distance = None

        # 2. Suscriptores
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
            
        self.sub_dist = self.create_subscription(
            Float32, '/aruco/qr/distance', self.distance_callback, 10)
            
        self.sub_angle = self.create_subscription(
            Float32, '/aruco/qr/angle', self.angle_callback, 10)

        # 3. NUEVO: Publicador para enviar la bandera
        self.pub_trigger = self.create_publisher(
            String, '/collect/trigger', 10)

        self.get_logger().info("Nodo QR Zone Checker con Trigger iniciado.")

    def get_yaw_from_quaternion(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_theta = self.get_yaw_from_quaternion(msg.pose.pose.orientation)

    def distance_callback(self, msg):
        self.qr_distance = msg.data

    def angle_callback(self, msg):
        if self.qr_distance is None:
            return

        qr_angle = msg.data

        # 1. Calcular coordenadas globales
        angulo_global = self.robot_theta + qr_angle
        qr_x = self.robot_x + self.qr_distance * math.cos(angulo_global)
        qr_y = self.robot_y + self.qr_distance * math.sin(angulo_global)

        # 2. Comprobar si está en la zona
        en_zona = (self.zona_x_min <= qr_x <= self.zona_x_max) and \
                  (self.zona_y_min <= qr_y <= self.zona_y_max)

        # 3. NUEVO: Lógica de publicación del Trigger
        trigger_msg = String()
        
        if en_zona:
            trigger_msg.data = "conveyor"
            self.get_logger().info(f"En zona -> Publicando: {trigger_msg.data}")
        else:
            trigger_msg.data = "rack"
            self.get_logger().info(f"Fuera de zona -> Publicando: {trigger_msg.data}")
            
        # Enviar el mensaje al tópico
        self.pub_trigger.publish(trigger_msg)


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
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32

class PuzzlebotController(Node):
    def __init__(self):
        super().__init__('puzzlebot_main_controller')
        
        # Publicadores para las dos llantas
        self.pub_left = self.create_publisher(Float32, '/VelocitySetL', 10)
        self.pub_right = self.create_publisher(Float32, '/VelocitySetR', 10)
        
        # Suscriptor a un tópico de mando (puedes usar este para teleoperación)
        self.sub_cmd = self.create_subscription(Twist, '/cmd_vel', self.callback_control, 10)
        
        self.get_logger().info("Controlador de Puzzlebot activado y listo.")

def callback_control(self, msg):
    # Parámetros físicos (ajústalos a tu Puzzlebot real)
    r = 0.05    # Radio de la rueda en metros (ej. 5cm)
    L = 0.19    # Distancia entre ejes en metros (ej. 19cm)

    v = msg.linear.x
    w = msg.angular.z

    # Aplicamos el modelo cinemático
    # Calculamos rad/s para cada rueda
    vel_r = (2 * v + w * L) / (2 * r)
    vel_l = (2 * v - w * L) / (2 * r)

    # Publicamos en los tópicos que ya conocemos
    msg_r = Float32()
    msg_l = Float32()
    
    msg_r.data = float(vel_r)
    msg_l.data = float(vel_l)

    self.pub_right.publish(msg_r)
    self.pub_left.publish(msg_l)

def main(args=None):
    rclpy.init(args=args)
    node = PuzzlebotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
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
        # Lógica simple: avanzar recto
        # En un robot diferencial: v_rueda = v_lineal +/- (v_angular * distancia_ejes/2)
        v_lineal = msg.linear.x
        v_angular = msg.angular.z
        
        # Por ahora, hagamos que avance parejo
        msg_l = Float32()
        msg_r = Float32()
        
        msg_l.data = v_lineal - v_angular
        msg_r.data = v_lineal + v_angular
        
        self.pub_left.publish(msg_l)
        self.pub_right.publish(msg_r)

def main(args=None):
    rclpy.init(args=args)
    node = PuzzlebotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
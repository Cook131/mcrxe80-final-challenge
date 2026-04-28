import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose2D
import math

class GoToGoalNode(Node):
    def __init__(self):
        super().__init__('go_to_goal_node')
        
        # --- COMUNICACIÓN ROS 2 ---
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.sub_goal = self.create_subscription(Pose2D, '/goal', self.goal_callback, 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)

        # --- ESTADO DEL ROBOT ---
        self.x, self.y, self.th = 0.0, 0.0, 0.0
        self.target_x, self.target_y = 0.0, 0.0
        self.active = False

        # --- PARÁMETROS PID (AJUSTADOS PARA LINEAL RÁPIDO / GIRO LENTO) ---
        # PID para Velocidad Lineal (Agresivo)
        self.kp_v, self.ki_v, self.kd_v = 1.8, 0.02, 0.1
        
        # PID para Velocidad Angular (Suave)
        self.kp_w, self.ki_w, self.kd_w = 1.0, 0.0, 0.0

        # --- LÍMITES DE SATURACIÓN ---
        self.max_linear_velocity = 4.0  # m/s (Más rápido en recta)
        self.max_angular_velocity = 1.0  # rad/s (Giro lento y controlado)
        self.angle_threshold = 0.05      # Radianes (~17°). Si el error es mayor, prioriza giro.

        # --- VARIABLES DE CONTROL ---
        self.error_dist_prev = 0.0
        self.error_angle_prev = 0.0
        self.integral_dist = 0.0
        self.integral_angle = 0.0
        
        self.last_time = self.get_clock().now()

        self.get_logger().info("Nodo Go-to-Goal PID iniciado.")
        self.get_logger().info(f"Configuración: V_MAX={self.max_linear_velocity}, W_MAX={self.max_angular_velocity}")

    def odom_callback(self, msg):
        # 1. Extraer posición
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        
        # 2. Convertir Cuaternión a Euler (Yaw)
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
        
        # Reiniciar errores acumulados para el nuevo trayecto
        self.integral_dist = 0.0
        self.integral_angle = 0.0
        self.error_dist_prev = 0.0
        self.error_angle_prev = 0.0
        
        self.get_logger().info(f"📍 Nuevo objetivo: x={self.target_x}, y={self.target_y}")

    def control_loop(self):
        # Cálculo de tiempo (Delta Time)
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        if dt <= 0: return
        self.last_time = current_time

        # 1. Calcular errores actuales
        dist = math.sqrt((self.target_x - self.x)**2 + (self.target_y - self.y)**2)
        angle_to_goal = math.atan2(self.target_y - self.y, self.target_x - self.x)
        error_angle = angle_to_goal - self.th
        
        # Normalizar ángulo entre -pi y pi
        error_angle = math.atan2(math.sin(error_angle), math.cos(error_angle))

        cmd = Twist()

        # Condición de parada (Llegada al punto)
        if dist < 0.02:
            self.active = False
            self.get_logger().info("✅ Objetivo alcanzado")
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        else:
            # 2. PID Lineal (Distancia)
            self.integral_dist += dist * dt
            derivative_dist = (dist - self.error_dist_prev) / dt
            v_out = (self.kp_v * dist) + (self.ki_v * self.integral_dist) + (self.kd_v * derivative_dist)
            
            # 3. PID Angular (Orientación)
            self.integral_angle += error_angle * dt
            derivative_angle = (error_angle - self.error_angle_prev) / dt
            w_out = (self.kp_w * error_angle) + (self.ki_w * self.integral_angle) + (self.kd_w * derivative_angle)

            # 4. Lógica de Movimiento y Saturación
            # Primero: Limitar la rotación para que sea lenta
            cmd.angular.z = max(min(w_out, self.max_angular_velocity), -self.max_angular_velocity)
            
            # Segundo: Controlar el avance lineal según la alineación
            if abs(error_angle) > self.angle_threshold:
                # Si está muy desalineado, casi no avanza (solo gira)
                cmd.linear.x = 0.02
            else:
                # Si está alineado, aplica la velocidad lineal (hasta 0.7 m/s)
                cmd.linear.x = min(v_out, self.max_linear_velocity)

            # Guardar errores para el siguiente ciclo
            self.error_dist_prev = dist
            self.error_angle_prev = error_angle

        self.pub_cmd.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = GoToGoalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
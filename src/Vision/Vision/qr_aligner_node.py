#!/usr/bin/env python3
import math
import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist

# Función auxiliar para cargar calibración Fisheye (la misma que ya usamos)
# Función auxiliar para cargar calibración Fisheye de forma segura
def load_calibration(script_dir):
    search = [script_dir, os.path.join(script_dir, '..', 'puzzlebot')]
    for d in search:
        for name in ["fisheye_params.npz", "camera_params.npz"]:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                data = dict(np.load(p))
                
                # Buscar llaves sin importar cómo las haya guardado tu script de calibración
                K = data.get("camera_matrix", data.get("K", data.get("mtx")))
                D = data.get("dist_coeffs", data.get("D", data.get("dist")))
                
                if K is None or D is None:
                    continue  # Faltan datos, intenta con otro archivo si existe
                
                K_arr = np.array(K, dtype=np.float64).reshape(3, 3)
                D_arr = np.array(D, dtype=np.float64).flatten()
                
                # Fisheye necesita exactamente 4 coeficientes. Si hay menos, rellenamos con 0.
                if D_arr.size < 4:
                    D_padded = np.zeros(4, dtype=np.float64)
                    D_padded[:D_arr.size] = D_arr
                    D_arr = D_padded
                    
                return K_arr, D_arr[:4].reshape(1, 4)
                
    return None, None

class QRAligner(Node):
    def __init__(self):
        super().__init__('qr_aligner')
        
        # ─── PARÁMETROS DEL ROBOT ──────────────────────────────────────
        self.declare_parameter('qr_size', 0.09)           # Tamaño del QR (9 cm)
        self.declare_parameter('target_distance', 0.30)   # Distancia meta (20 cm)
        
        # Desplazamiento de la cámara respecto al centro (base_link)
        # Eje X del robot: Apunta al frente. Si la cámara está atrás, es negativo.
        # Eje Y del robot: Apunta a la izquierda. Si la cámara está a la izquierda, es positivo.
        self.declare_parameter('cam_offset_x', -0.05)     # Ej: 5 cm hacia atrás
        self.declare_parameter('cam_offset_y', 0.08)      # Ej: 8 cm a la izquierda
        
        # Ganancias del controlador
        self.declare_parameter('kp_linear', 0.8)
        self.declare_parameter('kp_angular', 1.5)

        # ─── SETUP DE OPENCV Y ROS ─────────────────────────────────────
        self.bridge = CvBridge()
        self.qr_detector = cv2.QRCodeDetector()
        
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(CompressedImage, '/camera_raw/compressed', self.image_callback, 1)

        # Cargar calibración
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.camera_matrix, self.dist_coeffs = load_calibration(script_dir)
        
        self.map1, self.map2, self.new_camera_matrix = None, None, None
        self.last_qr_time = self.get_clock().now()

        # Timer de seguridad: Si no vemos el QR en 0.5s, nos detenemos
        self.create_timer(0.5, self.safety_stop_callback)

        self.get_logger().info("Nodo de Alineación QR listo. Esperando código...")

    def image_callback(self, msg):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            return

        if self.camera_matrix is None:
            self.get_logger().error("¡Falta calibración fisheye! No puedo calcular distancias.")
            return

        # 1. Undistort (Aplanar imagen fisheye)
        h, w = frame.shape[:2]
        if self.map1 is None:
            self.new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.camera_matrix, self.dist_coeffs, (w, h), np.eye(3), balance=0.5)
            self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
                self.camera_matrix, self.dist_coeffs, np.eye(3), self.new_camera_matrix, (w, h), cv2.CV_16SC2)
        
        frame_rect = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame_rect, cv2.COLOR_BGR2GRAY)

        # 2. Detección
        qr_data, qr_points, _ = self.qr_detector.detectAndDecode(gray)
        
        if qr_points is not None and len(qr_points) > 0:
            self.last_qr_time = self.get_clock().now()
            
            # Puntos del QR real en metros
            half_qr = self.get_parameter('qr_size').value / 2.0
            obj_pts = np.array([
                [-half_qr,  half_qr, 0], [ half_qr,  half_qr, 0], 
                [ half_qr, -half_qr, 0], [-half_qr, -half_qr, 0]
            ], dtype=np.float32)
            
            # 3. Calcular la pose del QR respecto a la cámara
            ok, _, tvec = cv2.solvePnP(
                obj_pts, qr_points[0].astype(np.float32), 
                self.new_camera_matrix, np.zeros((1, 4), dtype=np.float64), 
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            
            if ok:
                self.align_robot(tvec.flatten())

    def align_robot(self, tvec):
        """Calcula el error y publica cmd_vel para alinear el robot."""
        
        # 1. Extraer ejes de la cámara (Ópticos: Z es profundidad, X es derecha)
        cam_x = float(tvec[0]) 
        cam_z = float(tvec[2])

        # 2. Obtener los offsets paramétricos
        off_x = self.get_parameter('cam_offset_x').value
        off_y = self.get_parameter('cam_offset_y').value

        # 3. Transformación al centro del robot (base_link)
        # El frente del robot (X) depende de la profundidad de la cámara + offset
        x_robot = cam_z + off_x 
        
        # La izquierda del robot (Y) depende de lo que la cámara ve a la izquierda (-X_cam) + offset
        y_robot = -cam_x + off_y 

        # 4. Calcular errores
        target_dist = self.get_parameter('target_distance').value
        
        # Error lineal: Cuánto nos falta para los 20 cm
        error_dist = x_robot - target_dist
        
        # Error angular: Qué tan desalineado está el objetivo respecto al centro del robot
        error_angle = math.atan2(y_robot, x_robot)

        # 5. Ley de Control Proporcional
        kp_v = self.get_parameter('kp_linear').value
        kp_w = self.get_parameter('kp_angular').value

        cmd = Twist()

        # Tolerancias: Si estamos a menos de 2 cm y menos de 3 grados, ya llegamos.
        if abs(error_dist) < 0.02 and abs(error_angle) < 0.05:
            self.get_logger().info(f"¡ALINEADO! a {x_robot:.2f}m")
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        else:
            # Si está MUY chueco (más de 15 grados), priorizamos girar antes que avanzar
            if abs(error_angle) > 0.26: 
                cmd.linear.x = 0.0  
                cmd.angular.z = max(-0.5, min(0.5, error_angle * kp_w))
            else:
                # Nos movemos y giramos al mismo tiempo
                cmd.linear.x = max(-0.15, min(0.15, error_dist * kp_v))
                cmd.angular.z = max(-0.3, min(0.3, error_angle * kp_w))
                
        self.pub_cmd.publish(cmd)

    def safety_stop_callback(self):
        # Si pasó más de 0.5s sin ver el QR, detiene los motores por seguridad
        time_since_last_qr = (self.get_clock().now() - self.last_qr_time).nanoseconds / 1e9
        if time_since_last_qr > 0.5:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.pub_cmd.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = QRAligner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        cmd = Twist()
        node.pub_cmd.publish(cmd)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
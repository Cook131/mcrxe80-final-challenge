#!/usr/bin/env python3
"""
QR Detector Node Robusto para Puzzlebot (FISHEYE)
Corrige la distorsión de la imagen antes de detectar para evitar pérdidas.
"""

import json
import math
import os
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32, String

# ─────────────────────────────────────────────────────────────────────
# Calibración Fisheye
# ─────────────────────────────────────────────────────────────────────
_KEY_K = ["camera_matrix", "K", "mtx", "cameraMatrix", "intrinsic"]
_KEY_D = ["dist_coeffs",   "D", "dist", "distCoeffs",  "distortion"]

def _find_key(data, aliases):
    for k in aliases:
        if k in data: return data[k]
    return None

def _load_calibration(path: str):
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".npz": data = dict(np.load(path))
    elif ext == ".json":
        with open(path) as f: data = json.load(f)
    else: raise ValueError(f"Formato no soportado: '{ext}'")
    K, D = _find_key(data, _KEY_K), _find_key(data, _KEY_D)
    if K is None or D is None: raise KeyError("Claves no encontradas")
    K_arr = np.array(K, dtype=np.float64).reshape(3, 3)
    D_arr = np.array(D, dtype=np.float64).flatten()
    return K_arr, D_arr[:4].reshape(1, 4)

def _auto_find_calib(script_dir: str):
    search = [script_dir, os.path.join(script_dir, '..', 'puzzlebot')]
    for d in search:
        for name in ["fisheye_params.npz", "fisheye_params.json"]:
            p = os.path.normpath(os.path.join(d, name))
            if os.path.isfile(p): return p
    return None

# ─────────────────────────────────────────────────────────────────────
class QRDetectorNode(Node):

    QR_SIZE = 0.09  # metros

    def __init__(self):
        super().__init__('qr_detector')

        self.declare_parameter('camera_topic', '/camera_raw/compressed')
        self.declare_parameter('publish_image', True)
        self.declare_parameter('calib_file',    '')
        self.declare_parameter('qr_size',       self.QR_SIZE)
        self.declare_parameter('cam_offset', [-0.07, -0.08, 0.15])

        camera_topic = self.get_parameter('camera_topic').value
        self.qr_size = float(self.get_parameter('qr_size').value)

        self.camera_matrix = None
        self.dist_coeffs   = None
        self.pose_ready    = False
        
        # Variables para acelerar el undistort
        self.map1 = None
        self.map2 = None
        self.new_camera_matrix = None

        script_dir = os.path.dirname(os.path.abspath(__file__))
        calib_file = self.get_parameter('calib_file').value or _auto_find_calib(script_dir)

        if calib_file:
            try:
                self.camera_matrix, self.dist_coeffs = _load_calibration(calib_file)
                self.pose_ready = True
                self.get_logger().info(f"Calibración FISHEYE OK: '{calib_file}'")
            except Exception as e:
                self.get_logger().warn(f"Calibración fallida: {e}")

        self.bridge = CvBridge()
        self.qr_detector = cv2.QRCodeDetector()
        
        # CLAHE para robustez ante cambios de luz
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
        qos_cam = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(CompressedImage, camera_topic, self.image_callback, qos_cam)

        self.pub_qr          = self.create_publisher(String,  '/qr/data',     10)
        self.pub_qr_distance = self.create_publisher(Float32, '/qr/distance', 10)
        self.pub_qr_angle    = self.create_publisher(Float32, '/qr/angle',    10)
        self.pub_image       = self.create_publisher(Image,   '/qr/imagen',   10)

        self._prev_qr = ""
        self.get_logger().info("Nodo QR FISHEYE (Avanzado) listo.")

    def _angle_distance(self, tvec):
        offset = self.get_parameter('cam_offset').value
        tx, ty = float(tvec[0]) - float(offset[0]), float(tvec[1]) - float(offset[1])
        tz = float(tvec[2]) - float(offset[2])
        return math.sqrt(tx*tx + ty*ty + tz*tz), math.sqrt(tx*tx + tz*tz), math.degrees(math.atan2(tx, tz)), math.degrees(math.atan2(-ty, tz))

    def _draw_qr(self, out, qr_data: str, qr_points):
        if qr_points is None: return
        pts = qr_points[0].astype(int)
        cv2.polylines(out, [pts], True, (255, 0, 255), 2)
        cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
        label = f"QR: {qr_data[:30]}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (cx-5, cy-th-8), (cx+tw+5, cy+4), (0, 0, 0), -1)
        cv2.putText(out, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)

    def image_callback(self, msg: CompressedImage):
        try: frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception: return

        # 1. Undistort de LA IMAGEN COMPLETA para reparar líneas curvas del fisheye
        if self.pose_ready:
            h, w = frame.shape[:2]
            if self.map1 is None:
                # Calculamos el mapa de rectificación solo la primera vez para ahorrar CPU
                # balance=0.5 equilibra recortes y bordes negros
                self.new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    self.camera_matrix, self.dist_coeffs, (w, h), np.eye(3), balance=0.5
                )
                self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
                    self.camera_matrix, self.dist_coeffs, np.eye(3), 
                    self.new_camera_matrix, (w, h), cv2.CV_16SC2
                )
            
            # Aplicar remap
            frame_process = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
        else:
            frame_process = frame
            self.new_camera_matrix = self.camera_matrix

        # 2. Preprocesamiento de luz (CLAHE)
        gray = cv2.cvtColor(frame_process, cv2.COLOR_BGR2GRAY)
        gray_enhanced = self.clahe.apply(gray)

        # 3. Detectar sobre la imagen ya aplanada y mejorada
        qr_data, qr_points, _ = self.qr_detector.detectAndDecode(gray_enhanced)
        
        if qr_data and qr_points is not None:
            if qr_data != self._prev_qr:
                self.pub_qr.publish(String(data=qr_data))
                self.get_logger().info(f"  [QR] {qr_data}")
                self._prev_qr = qr_data

            if self.pose_ready:
                half_qr = self.qr_size / 2.0
                qr_obj_pts = np.array([[-half_qr, half_qr, 0], [half_qr, half_qr, 0], 
                                       [half_qr, -half_qr, 0], [-half_qr, -half_qr, 0]], dtype=np.float32)
                
                # OJO: Ya NO hacemos undistort_points. Los puntos ya están rectos.
                # Directo a solvePnP usando la NUEVA matriz de cámara.
                ok_qr, _, qr_tvec = cv2.solvePnP(
                    qr_obj_pts, 
                    qr_points[0].astype(np.float32), 
                    self.new_camera_matrix,
                    np.zeros((1, 4), dtype=np.float64), # Distorsión = 0 porque ya remapeamos
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                
                if ok_qr:
                    _, dist_xz_qr, angle_h_qr, _ = self._angle_distance(qr_tvec.flatten())
                    self.pub_qr_distance.publish(Float32(data=float(dist_xz_qr)))
                    self.pub_qr_angle.publish(Float32(data=float(angle_h_qr)))
        elif not qr_data:
            self._prev_qr = ""

        # 4. Publicar la imagen aplanada y anotada
        if self.get_parameter('publish_image').value:
            out = frame_process.copy()
            if qr_data: self._draw_qr(out, qr_data, qr_points)
            ann_msg = self.bridge.cv2_to_imgmsg(out, encoding='bgr8')
            ann_msg.header = msg.header
            self.pub_image.publish(ann_msg)

def main(args=None):
    rclpy.init(args=args)
    node = QRDetectorNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
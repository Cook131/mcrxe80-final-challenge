#!/usr/bin/env python3
"""
QR Detector Node para Puzzlebot - Manchester Robotics

Detecta QR codes en la imagen de la cámara y publica:
  - El contenido del QR decodificado
  - Distancia al QR en metros (plano XZ)
  - Ángulo horizontal al QR en grados (+ = derecha)

Tópicos:
  Suscribe:  /camera_raw/compressed     (sensor_msgs/CompressedImage)

  Publica:
             /qr/data                   (std_msgs/msg/String)   contenido del QR
             /qr/distance               (std_msgs/msg/Float32)  metros en plano XZ
             /qr/angle                  (std_msgs/msg/Float32)  grados, + = derecha
             /qr/imagen                 (sensor_msgs/msg/Image)  imagen anotada

Calibración FISHEYE:
  Busca automáticamente fisheye_params.npz o fisheye_params.json
  en la misma carpeta que este script.
  Usa el modelo fisheye de OpenCV (cv2.fisheye.*) con 4 coeficientes (k1,k2,k3,k4).
  Flujo correcto:
    1. cv2.fisheye.undistortPoints()  → puntos corregidos
    2. cv2.solvePnP(..., distCoeffs=zeros)  → pose sin distorsión
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
# Calibración
# ─────────────────────────────────────────────────────────────────────
_KEY_K = ["camera_matrix", "K", "mtx", "cameraMatrix", "intrinsic"]
_KEY_D = ["dist_coeffs",   "D", "dist", "distCoeffs",  "distortion"]

def _find_key(data, aliases):
    for k in aliases:
        if k in data:
            return data[k]
    return None

def _load_calibration(path: str):
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".npz":
        data = dict(np.load(path))
    elif ext == ".json":
        with open(path) as f:
            data = json.load(f)
    else:
        raise ValueError(f"Formato no soportado: '{ext}'")
    K = _find_key(data, _KEY_K)
    D = _find_key(data, _KEY_D)
    if K is None or D is None:
        raise KeyError(f"Claves de calibración no encontradas en '{path}'")
    K_arr = np.array(K, dtype=np.float64).reshape(3, 3)
    D_arr = np.array(D, dtype=np.float64).flatten()
    if D_arr.size < 4:
        raise ValueError(
            f"Fisheye necesita 4 coeficientes de distorsión, "
            f"se encontraron {D_arr.size} en '{path}'"
        )
    return K_arr, D_arr[:4].reshape(1, 4)

def _auto_find_calib(script_dir: str):
    search = [script_dir, os.path.join(script_dir, '..', 'puzzlebot')]
    for d in search:
        for name in ["fisheye_params.npz", "fisheye_params.json",
                     "camera_params.npz",  "camera_params.json"]:
            p = os.path.normpath(os.path.join(d, name))
            if os.path.isfile(p):
                return p
    return None

# ─────────────────────────────────────────────────────────────────────
class QRDetectorNode(Node):

    QR_SIZE = 0.09   # metros — medir el QR físico con regla

    def __init__(self):
        super().__init__('qr_detector')

        # ── Parámetros ────────────────────────────────────────────────
        self.declare_parameter('camera_topic', '/camera_raw/compressed')
        self.declare_parameter('publish_image', True)
        self.declare_parameter('calib_file',    '')
        self.declare_parameter('qr_size',       self.QR_SIZE)
        self.declare_parameter('cam_offset',    [0.07, 0.08, 0.15])

        camera_topic  = self.get_parameter('camera_topic').value
        self.qr_size  = float(self.get_parameter('qr_size').value)

        # ── Calibración ───────────────────────────────────────────────
        self.camera_matrix = None
        self.dist_coeffs   = None
        self.pose_ready    = False

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.get_logger().info(f"Buscando calibración en: {script_dir}")
        calib_file = self.get_parameter('calib_file').value or _auto_find_calib(script_dir)

        if calib_file:
            try:
                self.camera_matrix, self.dist_coeffs = _load_calibration(calib_file)
                self.pose_ready = True
                K = self.camera_matrix
                self.get_logger().info(
                    f"Calibración FISHEYE OK: '{calib_file}' | qr_size={self.qr_size}m\n"
                    f"  fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
                    f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}\n"
                    f"  dist(k1..k4)={self.dist_coeffs.flatten()}"
                )
            except Exception as e:
                self.get_logger().warn(f"Calibración fallida: {e} → Pose DESACTIVADA")
        else:
            self.get_logger().warn("Sin calibración → Pose DESACTIVADA (solo contenido QR)")

        # ── Detector QR ───────────────────────────────────────────────
        self.bridge      = CvBridge()
        self.qr_detector = cv2.QRCodeDetector()

        # ── Suscriptor ────────────────────────────────────────────────
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
        qos_cam = QoSProfile(
            reliability = QoSReliabilityPolicy.BEST_EFFORT,
            history     = QoSHistoryPolicy.KEEP_LAST,
            depth       = 1,
        )
        self.create_subscription(
            CompressedImage, camera_topic, self.image_callback, qos_cam)

        # ── Publicadores ──────────────────────────────────────────────
        self.pub_qr       = self.create_publisher(String,  '/qr/data',     10)
        self.pub_distance = self.create_publisher(Float32, '/qr/distance', 10)
        self.pub_angle    = self.create_publisher(Float32, '/qr/angle',    10)
        self.pub_image    = self.create_publisher(Image,   '/qr/imagen',   10)

        self._prev_qr = ""

        self.get_logger().info(
            f"QR Detector listo [MODO FISHEYE] | topic: {camera_topic}\n"
            f"  Publica: /qr/data | /qr/distance | /qr/angle | /qr/imagen"
        )

    # ─────────────────────────────────────────────────────────────────
    def _undistort_points(self, pts_2d: np.ndarray) -> np.ndarray:
        pts = pts_2d.reshape(-1, 1, 2).astype(np.float32)
        undist = cv2.fisheye.undistortPoints(
            pts,
            self.camera_matrix,
            self.dist_coeffs,
            R=np.eye(3),
            P=self.camera_matrix,
        )
        return undist.reshape(-1, 2).astype(np.float32)

    def _angle_distance(self, tvec):
        offset = self.get_parameter('cam_offset').value
        tx = float(tvec[0]) - float(offset[0])
        ty = float(tvec[1]) - float(offset[1])
        tz = float(tvec[2]) - float(offset[2])

        dist_3d = math.sqrt(tx*tx + ty*ty + tz*tz)
        dist_xz = math.sqrt(tx*tx + tz*tz)
        angle_h = math.degrees(math.atan2(tx,  tz))
        angle_v = math.degrees(math.atan2(-ty, tz))

        return dist_3d, dist_xz, angle_h, angle_v

    # ─────────────────────────────────────────────────────────────────
    # Anotación visual
    # ─────────────────────────────────────────────────────────────────
    def _draw_qr(self, out, qr_data: str, qr_points,
                 dist_xz: float = None, angle_h: float = None):
        if qr_points is None:
            return
        pts = qr_points[0].astype(int)
        cv2.polylines(out, [pts], True, (255, 0, 255), 2)
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())

        info_lines = [f"QR: {qr_data[:30]}"]
        if dist_xz is not None:
            info_lines.append(f"dist  {dist_xz:.3f} m")
            info_lines.append(f"az  {angle_h:+.1f} deg")

        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        line_h = 20
        box_w  = max(cv2.getTextSize(l, font, scale, thick)[0][0] for l in info_lines) + 10
        box_h  = line_h * len(info_lines) + 6
        cv2.rectangle(out, (cx - 5, cy - box_h), (cx + box_w, cy + 4), (0, 0, 0), -1)
        for i, line in enumerate(info_lines):
            cv2.putText(out, line, (cx, cy - box_h + line_h * (i + 1)),
                        font, scale, (255, 0, 255), thick, cv2.LINE_AA)

    def _draw_crosshair(self, out):
        h, w = out.shape[:2]
        cx, cy = w // 2, h // 2
        color  = (0, 255, 255)
        arm, gap = 12, 5
        cv2.line(out, (cx - gap - arm, cy), (cx - gap,       cy), color, 1, cv2.LINE_AA)
        cv2.line(out, (cx + gap,       cy), (cx + gap + arm, cy), color, 1, cv2.LINE_AA)
        cv2.line(out, (cx, cy - gap - arm), (cx, cy - gap),       color, 1, cv2.LINE_AA)
        cv2.line(out, (cx, cy + gap),       (cx, cy + gap + arm), color, 1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 2, color, -1, cv2.LINE_AA)

    # ─────────────────────────────────────────────────────────────────
    # Callback principal
    # ─────────────────────────────────────────────────────────────────
    def image_callback(self, msg: CompressedImage):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Error decodificando imagen comprimida: {e}")
            return

        qr_data, qr_points, _ = self.qr_detector.detectAndDecode(frame)
        qr_data   = qr_data or ""
        qr_points = qr_points if (qr_points is not None and qr_data) else None

        dist_xz_qr = None
        angle_h_qr = None

        if qr_data and qr_points is not None:
            # Publicar contenido solo si cambió
            if qr_data != self._prev_qr:
                self.pub_qr.publish(String(data=qr_data))
                self.get_logger().info(f"  [QR] {qr_data}")
                self._prev_qr = qr_data

            # Estimar pose si hay calibración
            if self.pose_ready:
                half_qr = self.qr_size / 2.0
                qr_obj_pts = np.array([
                    [-half_qr,  half_qr, 0],
                    [ half_qr,  half_qr, 0],
                    [ half_qr, -half_qr, 0],
                    [-half_qr, -half_qr, 0],
                ], dtype=np.float32)

                qr_pts_undist = self._undistort_points(
                    qr_points[0].astype(np.float32)
                )
                ok_qr, _, qr_tvec = cv2.solvePnP(
                    qr_obj_pts,
                    qr_pts_undist,
                    self.camera_matrix,
                    np.zeros((1, 4), dtype=np.float64),
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
                if ok_qr:
                    _, dist_xz_qr, angle_h_qr, _ = self._angle_distance(
                        qr_tvec.flatten())
                    self.pub_distance.publish(Float32(data=float(dist_xz_qr)))
                    self.pub_angle.publish(Float32(data=float(angle_h_qr)))

        elif not qr_data:
            self._prev_qr = ""

        # ── Imagen anotada ────────────────────────────────────────────
        if self.get_parameter('publish_image').value:
            out = frame.copy()
            if qr_data:
                self._draw_qr(out, qr_data, qr_points, dist_xz_qr, angle_h_qr)
            self._draw_crosshair(out)
            ann_msg = self.bridge.cv2_to_imgmsg(out, encoding='bgr8')
            ann_msg.header = msg.header
            self.pub_image.publish(ann_msg)


# ─────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = QRDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
"""
yolo_detector_node.py
───────────────────────────────────────────────────────────────
YOLOv8 Detector Node  —  Cookie / E80 Group R&D
Se suscribe a /camera/image_raw/compressed y publica detecciones
+ imagen anotada (mismo estilo que aruco_detector).

Topics:
  SUB  /camera_raw/compressed         sensor_msgs/CompressedImage
  PUB  /yolo/imagen                   sensor_msgs/Image    (visualizacion)
  PUB  /yolo/detecciones              std_msgs/String      (JSON con dist/angle)
  PUB  /yolo/distance                 std_msgs/Float32     (det. más cercana, metros)
  PUB  /yolo/angle                    std_msgs/Float32     (det. más cercana, grados)

Calibración FISHEYE:
  Busca fisheye_params.npz / fisheye_params.json en la misma carpeta.
  Mismo flujo que qr_detector.py:
    1. cv2.fisheye.undistortPoints() en las 4 esquinas del bbox
    2. cv2.solvePnP(..., distCoeffs=zeros)
    3. dist/angle en frame cámara — el offset cam→base_link lo maneja truck_align_node

Logo size: 0.10 x 0.10 m (cuadrado)

Uso:
  ros2 run <paquete> yolo_detector_node
───────────────────────────────────────────────────────────────
"""

# =============================================================
# CONFIGURACION — edita solo este bloque
# =============================================================
WEIGHTS      = "src/Vision/weights/best.pt"
CAMERA_TOPIC = "/camera_raw/compressed"
CONF         = 0.65
DEVICE       = "0"          # "0" = GPU, "cpu" = CPU
IMGSZ        = 320
LOGO_SIZE    = 0.10         # metros — lado del logo cuadrado
# =============================================================

import json
import math
import os

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos  import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg    import String, Float32
from cv_bridge       import CvBridge
from ultralytics     import YOLO

# ── Colores por clase (BGR) ────────────────────────────────────
CLASS_COLORS = {
    "nalmart" : (0,   255, 255),
    "nemezon" : (255, 165, 0  ),
    "nepsi"   : (0,   255, 0  ),
}
DEFAULT_COLOR = (0, 200, 255)

# ─────────────────────────────────────────────────────────────
# Calibración fisheye — igual que qr_detector.py
# ─────────────────────────────────────────────────────────────
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
        import json as _json
        with open(path) as f:
            data = _json.load(f)
    else:
        raise ValueError(f"Formato no soportado: '{ext}'")

    K = _find_key(data, _KEY_K)
    D = _find_key(data, _KEY_D)
    if K is None or D is None:
        raise KeyError(f"Claves de calibración no encontradas en '{path}'")

    K_arr = np.array(K, dtype=np.float64).reshape(3, 3)
    D_arr = np.array(D, dtype=np.float64).flatten()
    if D_arr.size < 4:
        raise ValueError(f"Fisheye necesita 4 coef., se encontraron {D_arr.size}")
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


# ─────────────────────────────────────────────────────────────
class YoloDetectorNode(Node):

    # Puntos 3D del logo cuadrado (igual orden que bbox: TL TR BR BL)
    #   x apunta derecha, y apunta arriba, z=0 (plano del logo)
    _LOGO_OBJ_PTS = None   # se construye en __init__ con LOGO_SIZE

    def __init__(self):
        super().__init__("yolo_detector")

        self.model   = YOLO(WEIGHTS)
        self.conf    = CONF
        self.device  = DEVICE
        self.imgsz   = IMGSZ
        self.bridge  = CvBridge()

        half = LOGO_SIZE / 2.0
        self._LOGO_OBJ_PTS = np.array([
            [-half,  half, 0],   # TL
            [ half,  half, 0],   # TR
            [ half, -half, 0],   # BR
            [-half, -half, 0],   # BL
        ], dtype=np.float32)

        # ── Calibración fisheye ───────────────────────────────
        self.camera_matrix = None
        self.dist_coeffs   = None
        self.pose_ready    = False

        script_dir = os.path.dirname(os.path.abspath(__file__))
        calib_file = _auto_find_calib(script_dir)
        if calib_file:
            try:
                self.camera_matrix, self.dist_coeffs = _load_calibration(calib_file)
                self.pose_ready = True
                K = self.camera_matrix
                self.get_logger().info(
                    f"Calibración FISHEYE OK: '{calib_file}'\n"
                    f"  fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
                    f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}\n"
                    f"  dist(k1..k4)={self.dist_coeffs.flatten()}"
                )
            except Exception as e:
                self.get_logger().warn(f"Calibración fallida: {e} → Pose DESACTIVADA")
        else:
            self.get_logger().warn("Sin calibración → dist/angle DESACTIVADOS")

        # ── QoS ───────────────────────────────────────────────
        qos_cam = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Sub / Pub ─────────────────────────────────────────
        self.sub = self.create_subscription(
            CompressedImage, CAMERA_TOPIC, self.image_callback, qos_cam)

        self.img_pub  = self.create_publisher(Image,   "/yolo/imagen",      10)
        self.det_pub  = self.create_publisher(String,  "/yolo/detecciones", 10)
        self.dist_pub = self.create_publisher(Float32, "/yolo/distance",    10)
        self.ang_pub  = self.create_publisher(Float32, "/yolo/angle",       10)

        self.get_logger().info(
            f"YoloDetector listo | topic: {CAMERA_TOPIC} | "
            f"modelo: {WEIGHTS} | conf: {CONF} | logo: {LOGO_SIZE}m\n"
            f"  Publica: /yolo/detecciones | /yolo/distance | /yolo/angle"
        )

    # ── Fisheye helpers ───────────────────────────────────────
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

    def _estimate_pose(self, x1, y1, x2, y2):
        """
        Estima distancia y ángulo horizontal del logo usando solvePnP.
        Las 4 esquinas del bbox se mapean a los puntos 3D del logo cuadrado.

        Returns (dist_xz, angle_h) en metros y grados, o (None, None) si falla.
        """
        if not self.pose_ready:
            return None, None

        # Esquinas bbox en orden TL TR BR BL
        corners_2d = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ], dtype=np.float32)

        # Paso 1: corregir distorsión fisheye
        corners_undist = self._undistort_points(corners_2d)

        # Paso 2: solvePnP con distCoeffs=zeros (puntos ya rectificados)
        ok, _, tvec = cv2.solvePnP(
            self._LOGO_OBJ_PTS,
            corners_undist,
            self.camera_matrix,
            np.zeros((1, 4), dtype=np.float64),
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            return None, None

        # Paso 3: dist/angle en frame cámara (sin offset cam→base_link)
        tx = float(tvec[0])
        tz = float(tvec[2])
        dist_xz = math.sqrt(tx * tx + tz * tz)
        angle_h = math.degrees(math.atan2(tx, tz))   # + = derecha
        return dist_xz, angle_h

    # ── Draw helpers ──────────────────────────────────────────
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

    def _draw_alignment(self, out, x1, y1, x2, y2, color,
                        dist_xz=None, angle_h=None):
        h, w = out.shape[:2]
        img_cx, img_cy = w // 2, h // 2
        box_cx = (x1 + x2) // 2
        box_cy = (y1 + y2) // 2

        cv2.line(out, (img_cx, img_cy), (box_cx, box_cy), color, 1, cv2.LINE_AA)
        cv2.circle(out, (box_cx, box_cy), 3, color, -1, cv2.LINE_AA)

        dx = box_cx - img_cx
        dy = box_cy - img_cy
        lines = [f"dx:{dx:+d} dy:{dy:+d}"]
        if dist_xz is not None:
            lines.append(f"dist {dist_xz:.3f}m  az {angle_h:+.1f}deg")

        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
        lh = 16
        mid_x = (img_cx + box_cx) // 2
        mid_y = (img_cy + box_cy) // 2
        box_w = max(cv2.getTextSize(l, font, scale, thick)[0][0] for l in lines) + 6
        cv2.rectangle(out,
                      (mid_x - 2, mid_y - lh * len(lines) - 2),
                      (mid_x + box_w, mid_y + 2), (0, 0, 0), -1)
        for i, line in enumerate(lines):
            cv2.putText(out, line,
                        (mid_x, mid_y - lh * (len(lines) - 1 - i)),
                        font, scale, color, thick, cv2.LINE_AA)

    # ── Callback principal ────────────────────────────────────
    def image_callback(self, msg: CompressedImage):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Error decodificando imagen: {e}")
            return

        results = self.model.predict(
            frame, conf=self.conf, device=self.device,
            verbose=False, imgsz=self.imgsz,
        )

        annotated  = frame.copy()
        detections = []
        best_dist  = None   # detección más cercana para /yolo/distance|angle
        best_angle = None

        for r in results:
            for box in r.boxes:
                cls_name = self.model.names[int(box.cls)]
                conf_val = float(box.conf)
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)

                # ── Pose ──────────────────────────────────────
                dist_xz, angle_h = self._estimate_pose(x1, y1, x2, y2)

                # Actualizar detección más cercana
                if dist_xz is not None:
                    if best_dist is None or dist_xz < best_dist:
                        best_dist  = dist_xz
                        best_angle = angle_h

                # ── Bounding box + label ───────────────────────
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name} {conf_val:.2f}"
                if dist_xz is not None:
                    label += f" | {dist_xz:.2f}m"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(annotated,
                              (x1, y1 - th - 6), (x1 + tw + 4, y1),
                              color, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 0, 0), 1, cv2.LINE_AA)

                detections.append({
                    "class"    : cls_name,
                    "conf"     : round(conf_val, 3),
                    "bbox"     : [x1, y1, x2, y2],
                    "distance" : round(dist_xz, 4) if dist_xz is not None else None,
                    "angle"    : round(angle_h, 2) if angle_h is not None else None,
                })

        self._draw_crosshair(annotated)
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            color = CLASS_COLORS.get(d["class"], DEFAULT_COLOR)
            self._draw_alignment(annotated, x1, y1, x2, y2, color,
                                 d["distance"], d["angle"])

        # ── Publicar imagen ───────────────────────────────────
        try:
            img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            img_msg.header = msg.header
            self.img_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f"Error publicando imagen: {e}")

        # ── Publicar JSON ─────────────────────────────────────
        self.det_pub.publish(String(data=json.dumps(detections)))

        # ── Publicar dist/angle de la detección más cercana ───
        if best_dist is not None:
            self.dist_pub.publish(Float32(data=float(best_dist)))
            self.ang_pub.publish(Float32(data=float(best_angle)))
            self.get_logger().debug(
                f"[YOLO] closest dist={best_dist:.3f}m  angle={best_angle:+.1f}°")

        if detections:
            self.get_logger().info(
                f"Detectado: {[d['class'] for d in detections]}")


# ─────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
ArUco Detector Node para Puzzlebot - Manchester Robotics

Detecta marcadores ArUco usando DICT_4X4_50:
  - IDs 0-4   → External WP 1…5  (paredes/esquinas)
  - IDs 5-10  → Internal WP 1…6  (objetivos internos)

Mapeo de IDs publicados:
  20…24  → External WPs  (4X4_50 IDs 0-4  → pub 20-24)
  10…15  → Internal WPs  (4X4_50 IDs 5-10 → pub 10-15)

Tópicos:
  Suscribe:  /camera_raw/compressed   (sensor_msgs/CompressedImage)

  Publica:
             /aruco/id                (std_msgs/Int32)
             /aruco/label             (std_msgs/String)
             /aruco/imagen            (sensor_msgs/Image)
             /aruco/waypoint          (geometry_msgs/PoseStamped)
             /aruco/distance          (std_msgs/Float32)  metros en plano XZ
             /aruco/angle             (std_msgs/Float32)  grados, + = derecha

Calibración FISHEYE:
  Busca automáticamente fisheye_params.npz o fisheye_params.json
  en la misma carpeta que este script.
  Usa el modelo fisheye de OpenCV (cv2.fisheye.*) con 4 coeficientes (k1,k2,k3,k4).

  Flujo correcto:
    1. cv2.fisheye.undistortPoints()  → puntos corregidos
    2. cv2.solvePnP(..., distCoeffs=zeros) → pose sin distorsión
    3. tvec está en frame cámara → se publica directo sin compensar offsets

  NOTA: el offset cámara→base_link lo maneja aruco_localizer_node.
  NO se aplica aquí para evitar doble compensación.
"""

import json
import math
import os

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32, Int32, String

# ─────────────────────────────────────────────────────────────────────
# Mapas de IDs y etiquetas
# ─────────────────────────────────────────────────────────────────────
EXTERNAL_4X4_IDS = {0, 1, 2, 3, 4}
INTERNAL_4X4_IDS = {5, 6, 7, 8, 9, 10}

LABEL_EXTERNAL = {
    0: "External WP 1", 1: "External WP 2", 2: "External WP 3",
    3: "External WP 4", 4: "External WP 5",
}
LABEL_INTERNAL = {
    5: "Internal WP 1", 6: "Internal WP 2",  7: "Internal WP 3",
    8: "Internal WP 4", 9: "Internal WP 5", 10: "Internal WP 6",
}

def external_pub_id(mid: int) -> int: return 20 + mid
def internal_pub_id(mid: int) -> int: return 10 + (mid - 5)   # 5→10 … 10→15

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
# Conversión rvec/tvec → PoseStamped
# ─────────────────────────────────────────────────────────────────────
def _to_posestamped(rvec, tvec) -> PoseStamped:
    pose = PoseStamped()
    pose.pose.position.x = float(tvec[0])
    pose.pose.position.y = float(tvec[1])
    pose.pose.position.z = float(tvec[2])
    R, _ = cv2.Rodrigues(rvec)
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0); w = 0.25 / s
        x = (R[2,1]-R[1,2])*s; y = (R[0,2]-R[2,0])*s; z = (R[1,0]-R[0,1])*s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1]-R[1,2])/s; x = 0.25*s; y = (R[0,1]+R[1,0])/s; z = (R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2]-R[2,0])/s; x = (R[0,1]+R[1,0])/s; y = 0.25*s; z = (R[1,2]+R[2,1])/s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0]-R[0,1])/s; x = (R[0,2]+R[2,0])/s; y = (R[1,2]+R[2,1])/s; z = 0.25*s
    pose.pose.orientation.x = x
    pose.pose.orientation.y = y
    pose.pose.orientation.z = z
    pose.pose.orientation.w = w
    return pose

# ─────────────────────────────────────────────────────────────────────
class ArucoDetectorNode(Node):

    MARKER_SIZE = 0.095   # metros

    def __init__(self):
        super().__init__('aruco_detector')

        # ── Parámetros ────────────────────────────────────────────────
        self.declare_parameter('camera_topic', '/camera_raw/compressed')
        self.declare_parameter('publish_image', True)
        self.declare_parameter('unknown_id',    -1)
        self.declare_parameter('calib_file',    '')
        self.declare_parameter('marker_size',   self.MARKER_SIZE)

        camera_topic     = self.get_parameter('camera_topic').value
        self.marker_size = float(self.get_parameter('marker_size').value)

        # ── Calibración ───────────────────────────────────────────────
        self.camera_matrix = None
        self.dist_coeffs   = None
        self.pose_ready    = False

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.get_logger().info(f"Buscando calibración en: {script_dir}")
        calib_file = (self.get_parameter('calib_file').value
                      or _auto_find_calib(script_dir))

        if calib_file:
            try:
                self.camera_matrix, self.dist_coeffs = _load_calibration(calib_file)
                self.pose_ready = True
                K = self.camera_matrix
                self.get_logger().info(
                    f"Calibración FISHEYE OK: '{calib_file}' | marker={self.marker_size}m\n"
                    f"  fx={K[0,0]:.1f}  fy={K[1,1]:.1f} "
                    f"cx={K[0,2]:.1f}  cy={K[1,2]:.1f}\n"
                    f"  dist(k1..k4)={self.dist_coeffs.flatten()}"
                )
            except Exception as e:
                self.get_logger().warn(f"Calibración fallida: {e} → Pose DESACTIVADA")
        else:
            self.get_logger().warn("Sin calibración → Pose DESACTIVADA")

        # ── Detectores ────────────────────────────────────────────────
        self.bridge  = CvBridge()
        self.det_4x4 = self._build_4x4_detector()

        # ── Suscriptor ────────────────────────────────────────────────
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
        qos_cam = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            CompressedImage, camera_topic, self.image_callback, qos_cam)

        # ── Publicadores ──────────────────────────────────────────────
        self.pub_id       = self.create_publisher(Int32,       '/aruco/id',       10)
        self.pub_label    = self.create_publisher(String,      '/aruco/label',    10)
        self.pub_image    = self.create_publisher(Image,       '/aruco/imagen',   10)
        self.pub_waypoint = self.create_publisher(PoseStamped, '/aruco/waypoint', 10)
        self.pub_distance = self.create_publisher(Float32,     '/aruco/distance', 10)
        self.pub_angle    = self.create_publisher(Float32,     '/aruco/angle',    10)

        self._prev_key = None

        self.get_logger().info(
            f"ArUco Detector listo [MODO FISHEYE] | topic: {camera_topic}\n"
            f"  4X4_50 IDs 0-4   → External WPs (pub 20-24)\n"
            f"  4X4_50 IDs 5-10  → Internal WPs (pub 10-15)\n"
            f"  Publica: /aruco/id | /aruco/label | /aruco/distance | /aruco/angle\n"
            f"  Offset cam→base_link manejado por aruco_localizer (no aquí)"
        )

    # ─────────────────────────────────────────────────────────────────
    def _build_4x4_detector(self):
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        return cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())

    def _undistort_points(self, pts_2d: np.ndarray) -> np.ndarray:
        """Corrige distorsión fisheye. Devuelve puntos en frame rectificado."""
        pts = pts_2d.reshape(-1, 1, 2).astype(np.float32)
        undist = cv2.fisheye.undistortPoints(
            pts,
            self.camera_matrix,
            self.dist_coeffs,
            R=np.eye(3),
            P=self.camera_matrix,
        )
        return undist.reshape(-1, 2).astype(np.float32)

    def _detect_aruco(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_4x4, ids_4x4, _ = self.det_4x4.detectMarkers(gray)

        external_hits, internal_hits = [], []
        if ids_4x4 is not None:
            for i, mid in enumerate(ids_4x4.flatten()):
                mid = int(mid)
                if mid in EXTERNAL_4X4_IDS:
                    external_hits.append((mid, corners_4x4[i]))
                elif mid in INTERNAL_4X4_IDS:
                    internal_hits.append((mid, corners_4x4[i]))

        return external_hits, internal_hits

    def _estimate_pose(self, corner):
        if not self.pose_ready:
            return None, None

        half = self.marker_size / 2.0
        obj_pts = np.array([
            [-half,  half, 0], [ half,  half, 0],
            [ half, -half, 0], [-half, -half, 0],
        ], dtype=np.float32)

        img_pts_undist = self._undistort_points(corner[0])
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts_undist,
            self.camera_matrix,
            np.zeros((1, 4), dtype=np.float64),
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        return (rvec.flatten(), tvec.flatten()) if ok else (None, None)

    def _angle_distance(self, tvec):
        """
        Calcula distancia y ángulo horizontal desde el tvec de solvePnP.

        tvec está en frame cámara:
          tx — desplazamiento lateral  (+ = derecha)
          ty — desplazamiento vertical (+ = abajo)
          tz — profundidad             (+ = lejos)

        NO se aplica el offset cámara→base_link aquí.
        Ese offset lo maneja aruco_localizer con camera_to_base_x/y/z
        para evitar doble compensación.

        Returns:
          dist_xz  — distancia en plano horizontal (m)
          angle_h  — ángulo horizontal en grados   (+ = derecha)
        """
        tx = float(tvec[0])
        tz = float(tvec[2])

        dist_xz = math.sqrt(tx * tx + tz * tz)
        angle_h = math.degrees(math.atan2(tx, tz))   # + = derecha

        return dist_xz, angle_h

    # ─────────────────────────────────────────────────────────────────
    # Anotación visual
    # ─────────────────────────────────────────────────────────────────
    def _draw_marker(self, out, corner, label, color, rvec=None, tvec=None):
        pts = corner[0].astype(int)
        cx  = int(pts[:, 0].mean())
        cy  = int(pts[:, 1].mean())
        cv2.polylines(out, [pts], True, color, 2)

        if rvec is not None and self.pose_ready:
            cv2.drawFrameAxes(out, self.camera_matrix,
                              np.zeros((1, 4), dtype=np.float64),
                              rvec, tvec, self.marker_size * 0.5)
            dist_xz, angle_h = self._angle_distance(tvec)
            h, w = out.shape[:2]
            cv2.line(out, (w // 2, h // 2), (cx, cy), color, 1, cv2.LINE_AA)
            info_lines = [
                label,
                f"dist  {dist_xz:.3f} m",
                f"az    {angle_h:+.1f} deg",
            ]
        else:
            info_lines = [label]

        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1
        line_h = 18
        box_w  = max(cv2.getTextSize(l, font, scale, thick)[0][0]
                     for l in info_lines) + 10
        box_h  = line_h * len(info_lines) + 6
        cv2.rectangle(out,
                      (cx - 5, cy - box_h), (cx + box_w, cy + 4),
                      (0, 0, 0), -1)
        for i, line in enumerate(info_lines):
            cv2.putText(out, line,
                        (cx, cy - box_h + line_h * (i + 1)),
                        font, scale, color, thick, cv2.LINE_AA)

    def _draw_crosshair(self, out):
        h, w   = out.shape[:2]
        cx, cy = w // 2, h // 2
        color  = (0, 255, 255)
        arm, gap = 12, 5
        cv2.line(out, (cx - gap - arm, cy), (cx - gap,       cy), color, 1, cv2.LINE_AA)
        cv2.line(out, (cx + gap,       cy), (cx + gap + arm, cy), color, 1, cv2.LINE_AA)
        cv2.line(out, (cx, cy - gap - arm), (cx, cy - gap),       color, 1, cv2.LINE_AA)
        cv2.line(out, (cx, cy + gap),       (cx, cy + gap + arm), color, 1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 2, color, -1, cv2.LINE_AA)

    def _annotate(self, frame, external_hits, internal_hits, poses_ext, poses_int):
        out = frame.copy()
        for i, (mid, corner) in enumerate(external_hits):
            rv, tv = poses_ext[i]
            self._draw_marker(out, corner, LABEL_EXTERNAL[mid], (0, 215, 255), rv, tv)
        for i, (mid, corner) in enumerate(internal_hits):
            rv, tv = poses_int[i]
            self._draw_marker(out, corner, LABEL_INTERNAL[mid], (0, 255, 0),   rv, tv)
        self._draw_crosshair(out)
        return out

    # ─────────────────────────────────────────────────────────────────
    # Callback principal
    # ─────────────────────────────────────────────────────────────────
    def image_callback(self, msg: CompressedImage):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Error decodificando imagen: {e}")
            return

        external_hits, internal_hits = self._detect_aruco(frame)

        unknown_id = self.get_parameter('unknown_id').value

        poses_ext = [self._estimate_pose(c) for _, c in external_hits]
        poses_int = [self._estimate_pose(c) for _, c in internal_hits]

        any_aruco = external_hits or internal_hits

        if any_aruco:
            # Prioridad: internal > external
            if internal_hits:
                mid    = internal_hits[0][0]
                pub_id = internal_pub_id(mid)
                label  = LABEL_INTERNAL[mid]
            else:
                mid    = external_hits[0][0]
                pub_id = external_pub_id(mid)
                label  = LABEL_EXTERNAL[mid]

            self.pub_id.publish(Int32(data=pub_id))
            self.pub_label.publish(String(data=label))

            # PoseStamped para todos los marcadores detectados
            for poses, hits in [
                (poses_ext, external_hits),
                (poses_int, internal_hits),
            ]:
                for i, (_, _corner) in enumerate(hits):
                    rv, tv = poses[i]
                    if rv is not None:
                        pm = _to_posestamped(rv, tv)
                        pm.header.stamp    = msg.header.stamp
                        pm.header.frame_id = 'camera_optical_frame'
                        self.pub_waypoint.publish(pm)

            # Log solo cuando cambian los marcadores visibles
            curr_key = (
                tuple(sorted(h[0] for h in external_hits)),
                tuple(sorted(h[0] for h in internal_hits)),
            )
            if curr_key != self._prev_key:
                for hits, poses, label_fn, id_fn, tag in [
                    (external_hits, poses_ext,
                     lambda m: LABEL_EXTERNAL[m], external_pub_id, "EXT"),
                    (internal_hits, poses_int,
                     lambda m: LABEL_INTERNAL[m], internal_pub_id, "INT"),
                ]:
                    for i, (mid, _) in enumerate(hits):
                        rv, tv = poses[i]
                        if tv is not None:
                            dist_xz, angle_h = self._angle_distance(tv)
                            d = f" | dist_xz={dist_xz:.3f}m  az={angle_h:+.1f}°"
                        else:
                            d = ""
                        self.get_logger().info(
                            f"  [{tag}] {label_fn(mid)} → pub_id={id_fn(mid)}{d}"
                        )
                self._prev_key = curr_key

            # Publicar distancia/ángulo del marcador con mayor prioridad
            priority_tv = None
            if   internal_hits and poses_int[0][1] is not None:
                priority_tv = poses_int[0][1]
            elif external_hits and poses_ext[0][1] is not None:
                priority_tv = poses_ext[0][1]

            if priority_tv is not None:
                dist_xz, angle_h = self._angle_distance(priority_tv)
                self.pub_distance.publish(Float32(data=float(dist_xz)))
                self.pub_angle.publish(Float32(data=float(angle_h)))

        else:
            self.pub_id.publish(Int32(data=unknown_id))
            self.pub_label.publish(String(data=""))
            if self._prev_key not in (None, ((), ())):
                self.get_logger().info("  (sin marcadores ArUco)")
            self._prev_key = ((), ())

        # ── Imagen anotada ────────────────────────────────────────────
        if self.get_parameter('publish_image').value:
            annotated = self._annotate(
                frame, external_hits, internal_hits, poses_ext, poses_int)
            ann_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            ann_msg.header = msg.header
            self.pub_image.publish(ann_msg)


# ─────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
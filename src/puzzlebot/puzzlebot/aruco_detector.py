#!/usr/bin/env python3
"""
ArUco Detector Doble-Diccionario + QR Node para Puzzlebot - Manchester Robotics

Detecta marcadores usando DOS diccionarios en paralelo + QR codes:
  - Diccionario 4X4_50  IDs 0-2  → External WP 0, 1, 2   (paredes)
  - Diccionario 4X4_50  IDs 3-5  → Internal WP 1, 2, 3   (objetivos internos)
  - Diccionario 6X6_50  IDs 0-6  → Waypoint_0 … Waypoint_6
  - QR codes                     → contenido del QR como string

Tópicos:
  Suscribe:  /camera/image_raw/compressed  (sensor_msgs/msg/CompressedImage)
  Publica:   /aruco/id                     (std_msgs/msg/Int32)
             /aruco/label                  (std_msgs/msg/String)
             /aruco/imagen                 (sensor_msgs/msg/Image)
             /aruco/waypoint               (geometry_msgs/msg/PoseStamped)
             /aruco/qr                     (std_msgs/msg/String)

IDs publicados en /aruco/id:
  -1           → ningún marcador detectado
  11, 12, 13   → Internal WP 1/2/3   (4X4_50 IDs 3/4/5)
  20, 21, 22   → External WP 0/1/2   (4X4_50 IDs 0/1/2)
  30, 31 … 36  → Waypoint_0 … _6    (6X6_50 IDs 0-6)

Calibración:
  Busca automáticamente camera_params.npz o camera_params.json
  en la misma carpeta que este script.
"""

import json
import os

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Int32, String, Float32
import math

# ─────────────────────────────────────────────────────────────────────
# Mapas de IDs y etiquetas  (diccionario único 4X4_50)
# ─────────────────────────────────────────────────────────────────────
EXTERNAL_4X4_IDS = {0, 1, 2}   # paredes
INTERNAL_4X4_IDS = {3, 4, 5}   # objetivos internos

LABEL_EXTERNAL = {
    0: "External WP 1",
    1: "External WP 2",
    2: "External WP 3",
}

LABEL_INTERNAL = {
    3: "Internal WP 1",
    4: "Internal WP 2",
    5: "Internal WP 3",
}

# 6X6_50: IDs 0-6 → Waypoint_0 … Waypoint_6
WAYPOINT_6X6_IDS = set(range(7))

def label_6x6(mid: int) -> str:
    return f"Waypoint_{mid}"

# IDs únicos publicados en /aruco/id
def external_pub_id(marker_id: int) -> int: return 20 + marker_id       # 20-22
def internal_pub_id(marker_id: int) -> int: return 10 + (marker_id - 2) # 11-13  (IDs 3→11, 4→12, 5→13)
def waypoint_pub_id(mid: int) -> int:       return 30 + mid              # 30-36

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
        K = np.array(_find_key(data, _KEY_K), dtype=np.float64)
        D = np.array(_find_key(data, _KEY_D), dtype=np.float64)
    elif ext == ".json":
        with open(path) as f:
            data = json.load(f)
        K = np.array(_find_key(data, _KEY_K), dtype=np.float64)
        D = np.array(_find_key(data, _KEY_D), dtype=np.float64)
    else:
        raise ValueError(f"Formato no soportado: '{ext}'")
    if K is None or D is None:
        raise KeyError(f"Claves de calibración no encontradas en '{path}'")
    return K.reshape(3, 3), D.flatten()

def _auto_find_calib(script_dir: str):
    for name in ["camera_params.npz", "camera_params.json"]:
        p = os.path.join(script_dir, name)
        if os.path.isfile(p):
            return p
    return None

# ─────────────────────────────────────────────────────────────────────
# Pose
# ─────────────────────────────────────────────────────────────────────
def _to_posestamped(rvec, tvec) -> PoseStamped:
    pose = PoseStamped()
    pose.pose.position.x = float(tvec[0])
    pose.pose.position.y = float(tvec[1])
    pose.pose.position.z = float(tvec[2])
    R, _ = cv2.Rodrigues(rvec)
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5/np.sqrt(trace+1.0); w=0.25/s
        x=(R[2,1]-R[1,2])*s; y=(R[0,2]-R[2,0])*s; z=(R[1,0]-R[0,1])*s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s=2.0*np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])
        w=(R[2,1]-R[1,2])/s; x=0.25*s; y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s=2.0*np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])
        w=(R[0,2]-R[2,0])/s; x=(R[0,1]+R[1,0])/s; y=0.25*s; z=(R[1,2]+R[2,1])/s
    else:
        s=2.0*np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])
        w=(R[1,0]-R[0,1])/s; x=(R[0,2]+R[2,0])/s; y=(R[1,2]+R[2,1])/s; z=0.25*s
    pose.pose.orientation.x = x
    pose.pose.orientation.y = y
    pose.pose.orientation.z = z
    pose.pose.orientation.w = w
    return pose

# ─────────────────────────────────────────────────────────────────────
class ArucoDetectorNode(Node):

    MARKER_SIZE = 0.20  # metros

    def __init__(self):
        super().__init__('aruco_detector')

        # ── Parámetros ────────────────────────────────────────────────
        self.declare_parameter('camera_topic',  '/camera/image_raw/compressed')
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
        calib_file = self.get_parameter('calib_file').value or \
                     _auto_find_calib(script_dir)

        if calib_file:
            try:
                self.camera_matrix, self.dist_coeffs = _load_calibration(calib_file)
                self.pose_ready = True
                self.get_logger().info(
                    f"Calibración OK: '{calib_file}' | marker={self.marker_size}m"
                )
            except Exception as e:
                self.get_logger().warn(f"Calibración fallida: {e} → Pose DESACTIVADA")
        else:
            self.get_logger().warn("Sin calibración → Pose DESACTIVADA")

        # ── Detectores ────────────────────────────────────────────────
        self.bridge      = CvBridge()
        self.det_4x4     = self._build_4x4_detector()   # externo + interno
        self.det_6x6     = self._build_6x6_detector()   # waypoints
        self.qr_detector = cv2.QRCodeDetector()

        # ── QoS ───────────────────────────────────────────────────────
        qos_cam = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ── Suscriptor ────────────────────────────────────────────────
        self.create_subscription(CompressedImage, camera_topic,
                                 self.image_callback, qos_cam)

        # ── Publicadores ──────────────────────────────────────────────
        self.pub_id       = self.create_publisher(Int32,       '/aruco/id',       10)
        self.pub_label    = self.create_publisher(String,      '/aruco/label',    10)
        self.pub_image    = self.create_publisher(Image,       '/aruco/imagen',   10)
        self.pub_waypoint = self.create_publisher(PoseStamped, '/aruco/waypoint', 10)
        self.pub_qr       = self.create_publisher(String,      '/aruco/qr',       10)
        self.pub_distance = self.create_publisher(Float32, '/aruco/distance', 10)
        self.pub_angle    = self.create_publisher(Float32, '/aruco/angle',    10)

        self._prev_key = None
        self._prev_qr  = ""

        self.get_logger().info(
            f"ArUco Dual-Dict + QR listo | topic: {camera_topic}\n"
            f"  4X4_50 IDs 0-2 → External WPs (pub 20-22)\n"
            f"  4X4_50 IDs 3-5 → Internal WPs (pub 11-13)\n"
            f"  6X6_50 IDs 0-6 → Waypoints    (pub 30-36)\n"
            f"  QR → /aruco/qr"
        )

    # ─────────────────────────────────────────────────────────────────
    # Construcción de detectores
    # ─────────────────────────────────────────────────────────────────

    def _build_4x4_detector(self):
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        return cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())

    def _build_6x6_detector(self):
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
        return cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())

    # ─────────────────────────────────────────────────────────────────
    # Detección
    # ─────────────────────────────────────────────────────────────────

    def _detect_all(self, frame):
        """
        Retorna:
            external_hits : list of (marker_id, corner)   ← 4X4_50 IDs 0-2
            internal_hits : list of (marker_id, corner)   ← 4X4_50 IDs 3-5
            wp6x6_hits    : list of (marker_id, corner)   ← 6X6_50 IDs 0-6
            qr_data       : str                           ← vacío si no hay QR
            qr_points     : np.ndarray | None             ← bbox del QR para dibujar
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners_4x4, ids_4x4, _ = self.det_4x4.detectMarkers(gray)
        corners_6x6, ids_6x6, _ = self.det_6x6.detectMarkers(gray)

        # QR — una sola llamada; guardamos los puntos para reutilizarlos en _annotate
        qr_data, qr_points, _ = self.qr_detector.detectAndDecode(frame)
        qr_data   = qr_data or ""
        qr_points = qr_points if (qr_points is not None and qr_data) else None

        external_hits = []
        internal_hits = []
        if ids_4x4 is not None:
            for i, mid in enumerate(ids_4x4.flatten()):
                mid = int(mid)
                if mid in EXTERNAL_4X4_IDS:
                    external_hits.append((mid, corners_4x4[i]))
                elif mid in INTERNAL_4X4_IDS:
                    internal_hits.append((mid, corners_4x4[i]))

        wp6x6_hits = []
        if ids_6x6 is not None:
            for i, mid in enumerate(ids_6x6.flatten()):
                if int(mid) in WAYPOINT_6X6_IDS:
                    wp6x6_hits.append((int(mid), corners_6x6[i]))

        return external_hits, internal_hits, wp6x6_hits, qr_data, qr_points

    # ─────────────────────────────────────────────────────────────────
    # Pose
    # ─────────────────────────────────────────────────────────────────

    def _estimate_pose(self, corner):
        if not self.pose_ready:
            return None, None
        half = self.marker_size / 2.0
        obj_pts = np.array([
            [-half,  half, 0], [ half,  half, 0],
            [ half, -half, 0], [-half, -half, 0],
        ], dtype=np.float32)
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, corner[0].astype(np.float32),
            self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        return (rvec.flatten(), tvec.flatten()) if ok else (None, None)

    def _build_and_publish_pose(self, corner, stamp):
        rv, tv = self._estimate_pose(corner)
        if rv is not None:
            pm = _to_posestamped(rv, tv)
            pm.header.stamp    = stamp
            pm.header.frame_id = 'camera_optical_frame'
            self.pub_waypoint.publish(pm)
        return rv, tv
    

    # ─────────────────────────────────────────────────────────────────────
    # Ángulo y distancia desde tvec (frame de cámara)
    # ─────────────────────────────────────────────────────────────────────

    def _angle_distance(self, tvec):
        """
        tvec en frame óptico de cámara:
            X → derecha, Y → abajo, Z → profundidad (hacia el marcador)

        Retorna:
            distance  float  metros (norma euclidiana)
            angle_h   float  grados   positivo = marcador a la DERECHA
            angle_v   float  grados   positivo = marcador ABAJO
        """
        tx, ty, tz = float(tvec[0]), float(tvec[1]), float(tvec[2])
        distance = math.sqrt(tx*tx + ty*ty + tz*tz)
        angle_h  = math.degrees(math.atan2(tx,  tz))   # bearing horizontal
        angle_v  = math.degrees(math.atan2(-ty, tz))   # elevación (positivo = arriba)
        return distance, angle_h, angle_v

    # ─────────────────────────────────────────────────────────────────
    # Anotación visual
    # ─────────────────────────────────────────────────────────────────

    def _draw_marker(self, out, corner, label, color, rvec=None, tvec=None):
        pts = corner[0].astype(int)
        cx  = int(pts[:, 0].mean())
        cy  = int(pts[:, 1].mean())
        cv2.polylines(out, [pts], True, color, 2)

        if rvec is not None and self.pose_ready:
            cv2.drawFrameAxes(out, self.camera_matrix, self.dist_coeffs,
                            rvec, tvec, self.marker_size * 0.5)

            dist, angle_h, angle_v = self._angle_distance(tvec)
            # Línea de mira al marcador
            h, w = out.shape[:2]
            cv2.line(out, (w // 2, h // 2), (cx, cy), color, 1, cv2.LINE_AA)

            info_lines = [
                label,
                f"dist  {dist:.3f} m",
                f"az  {angle_h:+.1f} deg",
                f"el  {angle_v:+.1f} deg",
            ]
        else:
            info_lines = [label]

        # Caja de texto multilínea
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1
        line_h = 18
        box_w  = max(cv2.getTextSize(l, font, scale, thick)[0][0] for l in info_lines) + 10
        box_h  = line_h * len(info_lines) + 6
        cv2.rectangle(out,
                    (cx - 5, cy - box_h),
                    (cx + box_w, cy + 4),
                    (0, 0, 0), -1)
        for i, line in enumerate(info_lines):
            cv2.putText(out, line, (cx, cy - box_h + line_h * (i + 1)),
                        font, scale, color, thick, cv2.LINE_AA)

    def _draw_qr(self, out, qr_data: str, qr_points):
        """Dibuja el bbox del QR usando los puntos ya detectados en _detect_all."""
        if qr_points is None:
            return
        pts = qr_points[0].astype(int)
        cv2.polylines(out, [pts], True, (255, 0, 255), 2)
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        label = f"QR: {qr_data[:30]}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (cx-5, cy-th-8), (cx+tw+5, cy+4), (0,0,0), -1)
        cv2.putText(out, label, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)

    def _annotate(self, frame, external_hits, internal_hits,
                  wp6x6_hits, qr_data, qr_points, poses_ext, poses_int, poses_6x6):
        out = frame.copy()

        for i, (mid, corner) in enumerate(external_hits):
            rv, tv = poses_ext[i]
            self._draw_marker(out, corner, LABEL_EXTERNAL[mid],
                              (0, 215, 255), rv, tv)                  # amarillo

        for i, (mid, corner) in enumerate(internal_hits):
            rv, tv = poses_int[i]
            self._draw_marker(out, corner, LABEL_INTERNAL[mid],
                              (0, 255, 0), rv, tv)                    # verde

        for i, (mid, corner) in enumerate(wp6x6_hits):
            rv, tv = poses_6x6[i]
            self._draw_marker(out, corner, label_6x6(mid),
                              (255, 128, 0), rv, tv)                  # naranja

        if qr_data:
            self._draw_qr(out, qr_data, qr_points)

        self._draw_crosshair(out)

        return out
    
    # ─────────────────────────────────────────────────────────────────────
    # Crosshair / mira
    # ─────────────────────────────────────────────────────────────────────

    def _draw_crosshair(self, out):
        """
        Mira táctica centrada:
        - Cruz con gap central
        - Círculo interno y externo
        - Marcas de escala a 45 °
        """
        h, w = out.shape[:2]
        cx, cy = w // 2, h // 2

        color_outer = (200, 200, 200)   # gris claro
        color_inner = (0,   255, 255)   # cian
        alpha_layer = out.copy()

        R_outer = 40
        R_inner = 18
        gap     = 10   # píxeles sin línea al centro
        arm_len = 30

        # Círculos
        cv2.circle(alpha_layer, (cx, cy), R_outer, color_outer, 1, cv2.LINE_AA)
        cv2.circle(alpha_layer, (cx, cy), R_inner, color_inner, 1, cv2.LINE_AA)

        # Cruz — 4 brazos con gap
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            x0 = cx + dx * gap
            y0 = cy + dy * gap
            x1 = cx + dx * (gap + arm_len)
            y1 = cy + dy * (gap + arm_len)
            cv2.line(alpha_layer, (x0, y0), (x1, y1), color_inner, 1, cv2.LINE_AA)

        # Ticks diagonales a 45° (en el círculo exterior)
        for angle_deg in range(0, 360, 45):
            rad   = math.radians(angle_deg)
            tx0   = int(cx + R_outer       * math.cos(rad))
            ty0   = int(cy + R_outer       * math.sin(rad))
            tx1   = int(cx + (R_outer + 8) * math.cos(rad))
            ty1   = int(cy + (R_outer + 8) * math.sin(rad))
            cv2.line(alpha_layer, (tx0, ty0), (tx1, ty1), color_outer, 1, cv2.LINE_AA)

        # Blend semi-transparente
        cv2.addWeighted(alpha_layer, 0.75, out, 0.25, 0, out)

    # ─────────────────────────────────────────────────────────────────
    # Callback principal
    # ─────────────────────────────────────────────────────────────────

    def image_callback(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("imdecode devolvió None")
        except Exception as e:
            self.get_logger().error(f"Error decodificando imagen: {e}")
            return

        external_hits, internal_hits, wp6x6_hits, qr_data, qr_points = \
            self._detect_all(frame)

        unknown_id = self.get_parameter('unknown_id').value

        # ── Poses ─────────────────────────────────────────────────────
        poses_ext  = [self._estimate_pose(c) for _, c in external_hits]
        poses_int  = [self._estimate_pose(c) for _, c in internal_hits]
        poses_6x6  = [self._estimate_pose(c) for _, c in wp6x6_hits]

        # ── QR ────────────────────────────────────────────────────────
        if qr_data and qr_data != self._prev_qr:
            self.pub_qr.publish(String(data=qr_data))
            self.get_logger().info(f"  [QR] {qr_data}")
            self._prev_qr = qr_data
        elif not qr_data:
            self._prev_qr = ""

        # ── ArUcos ────────────────────────────────────────────────────
        any_aruco = external_hits or internal_hits or wp6x6_hits

        if any_aruco:
            # Prioridad: interno > externo > 6x6
            if internal_hits:
                mid    = internal_hits[0][0]
                pub_id = internal_pub_id(mid)
                label  = LABEL_INTERNAL[mid]
            elif external_hits:
                mid    = external_hits[0][0]
                pub_id = external_pub_id(mid)
                label  = LABEL_EXTERNAL[mid]
            else:
                mid    = wp6x6_hits[0][0]
                pub_id = waypoint_pub_id(mid)
                label  = label_6x6(mid)

            self.pub_id.publish(Int32(data=pub_id))
            self.pub_label.publish(String(data=label))

            # Publicar poses
            for i, (_, corner) in enumerate(external_hits):
                rv, tv = poses_ext[i]
                if rv is not None:
                    pm = _to_posestamped(rv, tv)
                    pm.header.stamp    = msg.header.stamp
                    pm.header.frame_id = 'camera_optical_frame'
                    self.pub_waypoint.publish(pm)

            for i, (_, corner) in enumerate(internal_hits):
                rv, tv = poses_int[i]
                if rv is not None:
                    pm = _to_posestamped(rv, tv)
                    pm.header.stamp    = msg.header.stamp
                    pm.header.frame_id = 'camera_optical_frame'
                    self.pub_waypoint.publish(pm)

            for i, (_, corner) in enumerate(wp6x6_hits):
                rv, tv = poses_6x6[i]
                if rv is not None:
                    pm = _to_posestamped(rv, tv)
                    pm.header.stamp    = msg.header.stamp
                    pm.header.frame_id = 'camera_optical_frame'
                    self.pub_waypoint.publish(pm)

            # Log (solo cuando cambia el conjunto visible)
            curr_key = (
                tuple(sorted(h[0] for h in external_hits)),
                tuple(sorted(h[0] for h in internal_hits)),
                tuple(sorted(h[0] for h in wp6x6_hits)),
            )
            if curr_key != self._prev_key:
                for i, (mid, _) in enumerate(external_hits):
                    rv, tv = poses_ext[i]
                    d = f" | dist={np.linalg.norm(tv):.3f}m" if tv is not None else ""
                    self.get_logger().info(
                        f"  [EXT] {LABEL_EXTERNAL[mid]}"
                        f" → pub_id={external_pub_id(mid)}{d}"
                    )
                for i, (mid, _) in enumerate(internal_hits):
                    rv, tv = poses_int[i]
                    d = f" | dist={np.linalg.norm(tv):.3f}m" if tv is not None else ""
                    self.get_logger().info(
                        f"  [INT] {LABEL_INTERNAL[mid]}"
                        f" → pub_id={internal_pub_id(mid)}{d}"
                    )
                for i, (mid, _) in enumerate(wp6x6_hits):
                    rv, tv = poses_6x6[i]
                    d = f" | dist={np.linalg.norm(tv):.3f}m" if tv is not None else ""
                    self.get_logger().info(
                        f"  [WP]  {label_6x6(mid)}"
                        f" → pub_id={waypoint_pub_id(mid)}{d}"
                    )
                self._prev_key = curr_key

            # ── Publica distancia y ángulo del marcador prioritario ───
            # (mismo orden de prioridad: interno > externo > 6x6)
            priority_tv = None
            if internal_hits:
                idx = 0
                rv, tv = poses_int[idx]
                if tv is not None:
                    priority_tv = tv
            elif external_hits:
                idx = 0
                rv, tv = poses_ext[idx]
                if tv is not None:
                    priority_tv = tv
            elif wp6x6_hits:
                idx = 0
                rv, tv = poses_6x6[idx]
                if tv is not None:
                    priority_tv = tv

            if priority_tv is not None:
                dist, angle_h, _ = self._angle_distance(priority_tv)
                self.pub_distance.publish(Float32(data=float(dist)))
                self.pub_angle.publish(Float32(data=float(angle_h)))

        else:
            self.pub_id.publish(Int32(data=unknown_id))
            self.pub_label.publish(String(data=""))
            if self._prev_key not in (None, ((), (), ())):
                self.get_logger().info("  (sin marcadores)")
            self._prev_key = ((), (), ())

        # ── Imagen anotada ────────────────────────────────────────────
        if self.get_parameter('publish_image').value:
            annotated = self._annotate(
                frame, external_hits, internal_hits, wp6x6_hits,
                qr_data, qr_points, poses_ext, poses_int, poses_6x6
            )
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
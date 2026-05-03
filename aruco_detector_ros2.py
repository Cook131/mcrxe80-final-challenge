#!/usr/bin/env python3
"""
ArUco Detector Dual-Diccionario Node para Puzzlebot - Manchester Robotics

Detecta 7 waypoints usando DOS diccionarios en paralelo:
  - Diccionario custom "16h3"  (3 marcadores) → Internal WP 1, 2, 3
  - Diccionario 4X4_50         (4 marcadores) → External WP 0, 1, 2, 3

Tópicos:
  Suscribe:  /camera/image_raw/compressed  (sensor_msgs/msg/CompressedImage)
  Publica:   /aruco/id                     (std_msgs/msg/Int32)
             /aruco/label                  (std_msgs/msg/String)
             /aruco/imagen                 (sensor_msgs/msg/Image)  ← frame anotado

Convención de IDs publicados en /aruco/id:
  -1          → ningún marcador detectado
  11, 12, 13  → Internal WP 1, 2, 3  (diccionario 16h3)
  20, 21, 22, 23 → External WP 0, 1, 2, 3  (diccionario 4X4_50)

Uso:
    ros2 run <tu_paquete> aruco_detector_ros2
    # o directo:
    python3 aruco_detector_ros2.py
"""

import os
import tempfile

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Int32, String

import cv2
import numpy as np
from cv_bridge import CvBridge

# ─────────────────────────────────────────────────────────────────────
# YAML embebido del diccionario custom "16h3"
# ─────────────────────────────────────────────────────────────────────
CUSTOM_16H3_YAML = """%YAML:1.0
---
nmarkers: 3
markersize: 4
maxCorrectionBits: 0
marker_0: "1000101100000011"
marker_1: "0010010100110111"
marker_2: "1011011011000111"
"""

# ─────────────────────────────────────────────────────────────────────
# Mapas de etiquetas
# ─────────────────────────────────────────────────────────────────────

# Índice OpenCV del dict 16h3 → ID original del usuario (1-3)
ID_ORDER_INTERNAL = [1, 2, 3]

LABEL_INTERNAL = {
    1: "Internal WP 1",
    2: "Internal WP 2",
    3: "Internal WP 3",
}

LABEL_EXTERNAL = {
    0: "External WP 0",
    1: "External WP 1",
    2: "External WP 2",
    3: "External WP 3",
}

# IDs únicos publicados en /aruco/id para que otros nodos los distingan:
#   Internal WP orig_id → 10 + orig_id  (11, 12, 13)
#   External WP marker_id → 20 + marker_id  (20, 21, 22, 23)
#   Sin detección → -1
def internal_pub_id(orig_id: int) -> int:
    return 10 + orig_id

def external_pub_id(marker_id: int) -> int:
    return 20 + marker_id


# ─────────────────────────────────────────────────────────────────────
# Nodo ROS2
# ─────────────────────────────────────────────────────────────────────

class ArucoDetectorNode(Node):

    def __init__(self):
        super().__init__('aruco_detector')

        # ── Parámetros ────────────────────────────────────────────────
        self.declare_parameter('camera_topic',  '/camera/image_raw/compressed')
        self.declare_parameter('publish_image', True)
        self.declare_parameter('unknown_id',    -1)

        camera_topic = self.get_parameter('camera_topic').value

        # ── Puentes y detectores ──────────────────────────────────────
        self.bridge       = CvBridge()
        self.det_internal = self._build_internal_detector()
        self.det_external = self._build_external_detector()

        # ── QoS ───────────────────────────────────────────────────────
        qos_camera = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ── Suscriptores ──────────────────────────────────────────────
        self.sub_img = self.create_subscription(
            CompressedImage,
            camera_topic,
            self.image_callback,
            qos_camera
        )

        # ── Publicadores ──────────────────────────────────────────────
        self.pub_id    = self.create_publisher(Int32,  '/aruco/id',     10)
        self.pub_label = self.create_publisher(String, '/aruco/label',  10)
        self.pub_image = self.create_publisher(Image,  '/aruco/imagen', 10)

        self._prev_key = None

        self.get_logger().info(
            f"ArUco Dual-Dict listo | 16h3 (interno) + 4X4_50 (externo) "
            f"| escuchando: {camera_topic}"
        )
        self.get_logger().info(
            "IDs publicados: Internal WP1/2/3 → 11/12/13  |  External WP0-3 → 20-23"
        )

    # ─────────────────────────────────────────────────────────────────
    # Construcción de detectores
    # ─────────────────────────────────────────────────────────────────

    def _load_custom_yaml(self) -> cv2.aruco.Dictionary:
        """Carga el diccionario 16h3 desde el YAML embebido."""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        tmp.write(CUSTOM_16H3_YAML)
        tmp.close()
        try:
            d = cv2.aruco.Dictionary()
            fs = cv2.FileStorage(tmp.name, cv2.FILE_STORAGE_READ)
            d.readDictionary(fs.root())
            fs.release()
        finally:
            os.unlink(tmp.name)
        return d

    def _build_internal_detector(self) -> cv2.aruco.ArucoDetector:
        d = self._load_custom_yaml()
        params = cv2.aruco.DetectorParameters()
        return cv2.aruco.ArucoDetector(d, params)

    def _build_external_detector(self) -> cv2.aruco.ArucoDetector:
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()
        return cv2.aruco.ArucoDetector(d, params)

    # ─────────────────────────────────────────────────────────────────
    # Detección en ambos diccionarios
    # ─────────────────────────────────────────────────────────────────

    def _detect_both(self, frame):
        """
        Retorna:
            internal_hits : list of (orig_id, corners)   ← IDs 1-3
            external_hits : list of (marker_id, corners) ← IDs 0-3
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        c_int, ids_int, _ = self.det_internal.detectMarkers(gray)
        c_ext, ids_ext, _ = self.det_external.detectMarkers(gray)

        internal_hits = []
        if ids_int is not None:
            for i, idx in enumerate(ids_int.flatten()):
                orig_id = ID_ORDER_INTERNAL[int(idx)]
                internal_hits.append((orig_id, c_int[i]))

        external_hits = []
        if ids_ext is not None:
            for i, mid in enumerate(ids_ext.flatten()):
                external_hits.append((int(mid), c_ext[i]))

        return internal_hits, external_hits

    # ─────────────────────────────────────────────────────────────────
    # Anotación visual
    # ─────────────────────────────────────────────────────────────────

    def _draw_marker(self, out, corner, label, color):
        pts = corner[0].astype(int)
        cx  = int(pts[:, 0].mean())
        cy  = int(pts[:, 1].mean())
        cv2.polylines(out, [pts], True, color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (cx - 5, cy - th - 8), (cx + tw + 5, cy + 4), (0, 0, 0), -1)
        cv2.putText(out, label, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    def _annotate(self, frame, internal_hits, external_hits):
        out = frame.copy()
        for orig_id, corner in internal_hits:
            self._draw_marker(out, corner, LABEL_INTERNAL[orig_id], (0, 255, 0))    # verde
        for mid, corner in external_hits:
            label = LABEL_EXTERNAL.get(mid, f"Ext ID {mid}")
            self._draw_marker(out, corner, label, (0, 215, 255))                     # amarillo
        return out

    # ─────────────────────────────────────────────────────────────────
    # Callback principal
    # ─────────────────────────────────────────────────────────────────

    def image_callback(self, msg: CompressedImage):
        # Decodificar CompressedImage → OpenCV BGR
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("cv2.imdecode devolvió None")
        except Exception as e:
            self.get_logger().error(f"Error decodificando imagen: {e}")
            return

        internal_hits, external_hits = self._detect_both(frame)
        unknown_id = self.get_parameter('unknown_id').value

        # ── Publicar ID y label ───────────────────────────────────────
        if internal_hits or external_hits:
            # Prioridad: el primer marcador interno detectado, luego externo
            if internal_hits:
                orig_id = internal_hits[0][0]
                pub_id  = internal_pub_id(orig_id)
                label   = LABEL_INTERNAL[orig_id]
            else:
                mid    = external_hits[0][0]
                pub_id = external_pub_id(mid)
                label  = LABEL_EXTERNAL.get(mid, f"Ext ID {mid}")

            self.pub_id.publish(Int32(data=pub_id))
            self.pub_label.publish(String(data=label))

            # Log solo cuando cambia lo detectado
            curr_key = (
                tuple(sorted(h[0] for h in internal_hits)),
                tuple(sorted(h[0] for h in external_hits))
            )
            if curr_key != self._prev_key:
                for orig_id, _ in sorted(internal_hits, key=lambda x: x[0]):
                    self.get_logger().info(
                        f"  [INT] {LABEL_INTERNAL[orig_id]}  → pub_id={internal_pub_id(orig_id)}"
                    )
                for mid, _ in sorted(external_hits, key=lambda x: x[0]):
                    lbl = LABEL_EXTERNAL.get(mid, f"Ext ID {mid}")
                    self.get_logger().info(
                        f"  [EXT] {lbl}  → pub_id={external_pub_id(mid)}"
                    )
                self._prev_key = curr_key

        else:
            self.pub_id.publish(Int32(data=unknown_id))
            self.pub_label.publish(String(data=""))
            if self._prev_key is not None and self._prev_key != ((), ()):
                self.get_logger().info("  (sin marcadores)")
            self._prev_key = ((), ())

        # ── Publicar imagen anotada (opcional) ────────────────────────
        if self.get_parameter('publish_image').value:
            annotated = self._annotate(frame, internal_hits, external_hits)
            ann_msg   = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
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

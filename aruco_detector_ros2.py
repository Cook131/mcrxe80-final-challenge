#!/usr/bin/env python3
"""
ArUco Detector Node para Puzzlebot - Manchester Robotics
Se suscribe al tópico ROS2 /camera/image_raw/compressed
(sensor_msgs/CompressedImage) y publica el ID y nombre de la señal.

Tópicos:
  Suscribe:  /camera/image_raw/compressed  (sensor_msgs/msg/CompressedImage)
  Publica:   /aruco/id                     (std_msgs/msg/Int32)
             /aruco/señal                  (std_msgs/msg/String)
             /aruco/imagen                 (sensor_msgs/msg/Image)  ← frame anotado

Uso:
    ros2 run <tu_paquete> aruco_detector_ros2
    # o directo:
    python3 aruco_detector_ros2.py
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Int32, String

import cv2
import numpy as np
from cv_bridge import CvBridge

# ─────────────────────────────────────────────────────────────
# MAPA DE ID → TIPO DE SEÑAL (Puzzlebot MCR)
# Ajusta según las imágenes del repo de Manchester Robotics.
# ─────────────────────────────────────────────────────────────
ARUCO_MAP = {
    0:  "Gira a la derecha",
    1:  "Gira a la izquierda",
    2:  "Sigue recto",
    3:  "Alto / Stop",
    4:  "Ceda el paso / Yield",
    5:  "Vuelta en U",
    6:  "Zona escolar",
    7:  "Velocidad máxima 20",
    8:  "Velocidad máxima 30",
    9:  "Velocidad máxima 40",
    10: "Estacionamiento",
    11: "No hay paso",
    12: "Semáforo",
    13: "Cruce peatonal",
    14: "Camino sinuoso",
    15: "Construcción / Obras",
}

ARUCO_DICTS = {
    "4X4_50":   cv2.aruco.DICT_4X4_50,
    "4X4_100":  cv2.aruco.DICT_4X4_100,
    "4X4_250":  cv2.aruco.DICT_4X4_250,
    "5X5_50":   cv2.aruco.DICT_5X5_50,
    "5X5_100":  cv2.aruco.DICT_5X5_100,
    "6X6_250":  cv2.aruco.DICT_6X6_250,
}


class ArucoDetectorNode(Node):

    def __init__(self):
        super().__init__('aruco_detector')

        # ── Parámetros declarables desde CLI / launch ──────────────────
        self.declare_parameter('dict_name',       '4X4_50')
        self.declare_parameter('camera_topic',    '/camera/image_raw/compressed')
        self.declare_parameter('publish_image',   True)       # False = menos CPU
        self.declare_parameter('unknown_id',      -1)         # ID a publicar si no hay detección

        dict_name    = self.get_parameter('dict_name').value
        camera_topic = self.get_parameter('camera_topic').value

        # ── Puentes y detector ─────────────────────────────────────────
        self.bridge   = CvBridge()
        self.detector, self.aruco_dict, self.use_new_api = self._build_detector(dict_name)

        # ── QoS: Best-Effort para tópicos de imagen comprimida ─────────
        # Los tópicos /compressed de cámaras suelen publicar con BEST_EFFORT.
        # Si tu publisher usa RELIABLE, cambia a ReliabilityPolicy.RELIABLE.
        qos_camera = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ── Suscriptores ───────────────────────────────────────────────
        self.sub_img = self.create_subscription(
            CompressedImage,
            camera_topic,
            self.image_callback,
            qos_camera
        )

        # ── Publicadores ───────────────────────────────────────────────
        self.pub_id     = self.create_publisher(Int32,  '/aruco/id',     10)
        self.pub_signal = self.create_publisher(String, '/aruco/senal',  10)
        self.pub_image  = self.create_publisher(Image,  '/aruco/imagen', 10)

        self._prev_ids: set = set()
        self.get_logger().info(
            f"ArUco Detector listo | dict=DICT_{dict_name.upper()} "
            f"| escuchando: {camera_topic}"
        )

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    def _build_detector(self, dict_name: str):
        """Crea el detector compatible con OpenCV 4.6+ y versiones anteriores."""
        dict_id = ARUCO_DICTS.get(dict_name.upper())
        if dict_id is None:
            self.get_logger().error(
                f"Diccionario '{dict_name}' no válido. Opciones: {list(ARUCO_DICTS.keys())}"
            )
            raise ValueError(f"Diccionario ArUco desconocido: {dict_name}")

        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        try:
            # OpenCV >= 4.7
            params   = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, params)
            use_new_api = True
        except AttributeError:
            # OpenCV < 4.7
            detector    = aruco_dict
            use_new_api = False

        return detector, aruco_dict, use_new_api

    def _detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.use_new_api:
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            params = cv2.aruco.DetectorParameters_create()
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=params
            )
        return corners, ids, rejected

    def _get_signal(self, marker_id: int) -> str:
        return ARUCO_MAP.get(marker_id, f"ID desconocido ({marker_id})")

    def _annotate(self, frame, corners, ids):
        """Dibuja marcadores y etiquetas sobre el frame."""
        if ids is None:
            return frame
        out = frame.copy()
        cv2.aruco.drawDetectedMarkers(out, corners, ids)
        for i, corner in enumerate(corners):
            mid   = int(ids[i][0])
            label = f"ID {mid}: {self._get_signal(mid)}"
            pts   = corner[0].astype(int)
            cx    = int(pts[:, 0].mean())
            cy    = int(pts[:, 1].mean())
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(out, (cx - 5, cy - h - 8), (cx + w + 5, cy + 4), (0, 0, 0), -1)
            cv2.putText(out, label, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        return out

    # ─────────────────────────────────────────────────────────────────
    # Callback principal
    # ─────────────────────────────────────────────────────────────────

    def image_callback(self, msg: CompressedImage):
        # Convertir CompressedImage → OpenCV BGR
        # cv_bridge NO soporta CompressedImage directamente: usamos numpy.
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)   # BGR
            if frame is None:
                raise ValueError("cv2.imdecode devolvió None — ¿formato incorrecto?")
        except Exception as e:
            self.get_logger().error(f"Error decodificando imagen comprimida: {e}")
            return

        corners, ids, _ = self._detect(frame)

        # ── Publicar ID y señal ────────────────────────────────────────
        unknown_id = self.get_parameter('unknown_id').value

        if ids is not None and len(ids) > 0:
            # Publicar el primer marcador detectado (el más prominente)
            # Si quieres publicar todos, itera y publica en un array.
            first_id     = int(ids[0][0])
            signal_name  = self._get_signal(first_id)

            self.pub_id.publish(Int32(data=first_id))
            self.pub_signal.publish(String(data=signal_name))

            current_ids = set(ids.flatten().tolist())
            if current_ids != self._prev_ids:
                for mid in sorted(current_ids):
                    self.get_logger().info(f"  • ID {mid:>3}  →  {self._get_signal(mid)}")
            self._prev_ids = current_ids
        else:
            # Ningún marcador → publicar sentinel
            self.pub_id.publish(Int32(data=unknown_id))
            self.pub_signal.publish(String(data=""))
            if self._prev_ids:
                self.get_logger().info("  (sin marcadores)")
            self._prev_ids = set()

        # ── Publicar imagen anotada (opcional) ─────────────────────────
        if self.get_parameter('publish_image').value:
            annotated = self._annotate(frame, corners, ids)
            ann_msg   = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            ann_msg.header = msg.header   # conservar timestamp y frame_id
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

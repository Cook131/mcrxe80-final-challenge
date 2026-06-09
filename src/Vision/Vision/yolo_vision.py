"""
yolo_detector_node.py
───────────────────────────────────────────────────────────────
YOLOv8 Detector Node  —  Cookie / E80 Group R&D
Se suscribe a /camera/image_raw/compressed y publica detecciones
+ imagen anotada (mismo estilo que aruco_detector).

Topics:
  SUB  /camera/image_raw/compressed   sensor_msgs/CompressedImage
  PUB  /yolo/imagen                   sensor_msgs/Image   (visualizacion)
  PUB  /yolo/detecciones              std_msgs/String     (JSON)

Uso:
  ros2 run <paquete> yolo_detector_node
───────────────────────────────────────────────────────────────
"""

# =============================================================
# CONFIGURACION — edita solo este bloque
# =============================================================
WEIGHTS      = "src/Vision/Vision/best.pt"
CAMERA_TOPIC = "/camera_raw/compressed"
CONF         = 0.65
DEVICE       = "0"                     # "0" = GPU, "cpu" = CPU
IMGSZ        = 320
# =============================================================

import json
import rclpy
from rclpy.node      import Node
from rclpy.qos       import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg    import String
from cv_bridge       import CvBridge
from ultralytics     import YOLO
import cv2

# ── Colores por clase (BGR) — mismo estilo que aruco detector ──
CLASS_COLORS = {
    "nalmart" : (0,   255, 255),   # amarillo
    "nemezon" : (255, 165, 0  ),   # naranja
    "nepsi"   : (0,   255, 0  ),   # verde
}
DEFAULT_COLOR = (0, 200, 255)      # cyan para clases no listadas


class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__("yolo_detector")

        # ── Modelo ────────────────────────────────────────────
        self.model  = YOLO(WEIGHTS)
        self.conf   = CONF
        self.device = DEVICE
        self.imgsz  = IMGSZ
        self.bridge = CvBridge()

        # ── QoS: BEST_EFFORT igual que el aruco detector ──────
        qos_cam = QoSProfile(
            reliability = QoSReliabilityPolicy.BEST_EFFORT,
            history     = QoSHistoryPolicy.KEEP_LAST,
            depth       = 1,
        )

        # ── Subscriber ────────────────────────────────────────
        self.sub = self.create_subscription(
            CompressedImage, CAMERA_TOPIC, self.image_callback, qos_cam)

        # ── Publishers ────────────────────────────────────────
        self.img_pub = self.create_publisher(Image,  "/yolo/imagen",      10)
        self.det_pub = self.create_publisher(String, "/yolo/detecciones", 10)

        self.get_logger().info(
            f"YoloDetector listo | topic: {CAMERA_TOPIC} | "
            f"modelo: {WEIGHTS} | conf: {CONF}"
        )

    # ─────────────────────────────────────────────────────────

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

    def _draw_alignment(self, out, x1, y1, x2, y2, color):
        """Linea desde centro de imagen al centro del bbox + offset text"""
        h, w   = out.shape[:2]
        img_cx, img_cy = w // 2, h // 2
        box_cx = (x1 + x2) // 2
        box_cy = (y1 + y2) // 2

        cv2.line(out, (img_cx, img_cy), (box_cx, box_cy), color, 1, cv2.LINE_AA)
        cv2.circle(out, (box_cx, box_cy), 3, color, -1, cv2.LINE_AA)

        dx = box_cx - img_cx
        dy = box_cy - img_cy
        offset_text = f"dx:{dx:+d} dy:{dy:+d}"
        (tw, th), _ = cv2.getTextSize(offset_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        mid_x = (img_cx + box_cx) // 2
        mid_y = (img_cy + box_cy) // 2
        cv2.rectangle(out, (mid_x - 2, mid_y - th - 4),
                        (mid_x + tw + 2, mid_y + 2), (0, 0, 0), -1)
        cv2.putText(out, offset_text, (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # ─────────────────────────────────────────────────────────
    def image_callback(self, msg: CompressedImage):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Error decodificando imagen comprimida: {e}")
            return

        results = self.model.predict(
            frame,
            conf    = self.conf,
            device  = self.device,
            verbose = False,
            imgsz   = self.imgsz,
        )

        annotated  = frame.copy()
        detections = []

        for r in results:
            for box in r.boxes:
                cls_name = self.model.names[int(box.cls)]
                conf_val = float(box.conf)
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)

                # Bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Label con fondo
                label = f"{cls_name} {conf_val:.2f}"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(
                    annotated,
                    (x1, y1 - th - 6),
                    (x1 + tw + 4, y1),
                    color, -1
                )
                cv2.putText(
                    annotated, label,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 1, cv2.LINE_AA
                )

                detections.append({
                    "class" : cls_name,
                    "conf"  : round(conf_val, 3),
                    "bbox"  : [x1, y1, x2, y2],
                })

        # Crosshair siempre visible
        self._draw_crosshair(annotated)

        # Linea de alineamiento a TODOS los objetos detectados
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            color = CLASS_COLORS.get(d["class"], DEFAULT_COLOR)
            self._draw_alignment(annotated, x1, y1, x2, y2, color)

        # Publicar imagen anotada
        try:
            img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            img_msg.header = msg.header
            self.img_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f"Error publicando imagen: {e}")

        # Publicar detecciones JSON
        det_msg      = String()
        det_msg.data = json.dumps(detections)
        self.det_pub.publish(det_msg)

        if detections:
            clases = [d["class"] for d in detections]
            self.get_logger().info(f"Detectado: {clases}")


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
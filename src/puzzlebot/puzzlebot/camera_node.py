import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
import threading


# LUT para el ajuste del canal 'a' en LAB (-8 a todos los píxeles)
_lut_a = np.clip(np.arange(256, dtype=np.int16) - 8, 0, 255).astype(np.uint8)


def build_color_lut() -> np.ndarray:
    """
    LUT BGR→BGR para ajuste de contraste/brillo.
    alpha=1.05, beta=+10 — sube el brillo sin aplastar las altas luces.
    """
    lut_1d = np.clip(
        np.round(np.arange(256) * 1.05 + 10).astype(np.int16), 0, 255
    ).astype(np.uint8)
    lut_3d = cv2.merge([lut_1d, lut_1d, lut_1d])
    return lut_3d


_SCALE_LUT = build_color_lut()


class JetsonFisheyeFiltered(Node):
    def __init__(self):
        super().__init__('jetson_camera_node')

        self.raw_pub = self.create_publisher(Image, 'camera/image_raw', 1)

        # wbmode=0 → balance de blancos automático (evita el tinte rosado/frío
        # que produce wbmode=1 en ambientes con luz fluorescente o natural).
        gst_pipeline = (
            "nvarguscamerasrc wbmode=0 ! "
            "video/x-raw(memory:NVMM), width=1280, height=720, framerate=60/1 ! "
            "nvvidconv flip-method=2 ! "
            "video/x-raw(memory:NVMM), width=320, height=240 ! "
            "nvvidconv ! "
            "video/x-raw, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )

        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.get_logger().error(
                "No se pudo abrir la cámara. Verifica el pipeline de GStreamer.")
            raise RuntimeError("Camera open failed")

        # Pre-alloca el mensaje para evitar alloc por frame
        self._raw_msg = Image()
        self._raw_msg.header.frame_id = 'camera'
        self._raw_msg.encoding = 'bgr8'
        self._raw_msg.is_bigendian = 0
        self._raw_msg.height = 240
        self._raw_msg.width = 320
        self._raw_msg.step = 320 * 3

        self._stop_event = threading.Event()
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        self.get_logger().info("Nodo activo — publicando RAW 320x240 @ 60 fps")

    def apply_color_filter(self, frame: np.ndarray) -> np.ndarray:
        # Corrección de tinte en espacio LAB (canal 'a')
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 1] = cv2.LUT(lab[:, :, 1], _lut_a)
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Ajuste de brillo/contraste con LUT pre-calculada (in-place)
        cv2.LUT(result, _SCALE_LUT, result)
        return result

    def _capture_loop(self):
        while not self._stop_event.is_set():
            if not self.cap.grab():
                continue

            ret, frame = self.cap.retrieve()
            if not ret or frame is None:
                continue

            # El pipeline ya entrega BGR — no se necesita conversión de color
            filtered = self.apply_color_filter(frame)
            stamp = self.get_clock().now().to_msg()

            self._raw_msg.header.stamp = stamp
            self._raw_msg.data = filtered.tobytes()
            self.raw_pub.publish(self._raw_msg)

    def destroy_node(self):
        self._stop_event.set()
        if self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = JetsonFisheyeFiltered()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
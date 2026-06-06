#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import threading

def gstreamer_pipeline(width=1280, height=720, fps=30, flip=2):
    return (
        f"nvarguscamerasrc sensor-id=0 sensor-mode=4 ! "
        f"video/x-raw(memory:NVMM), width={width}, height=(int){height}, "
        f"framerate={fps}/1, format=NV12 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw(memory:NVMM), format=I420 ! "
        f"nvjpegenc quality=60 ! "
        f"appsink max-buffers=1 drop=true sync=false"
    )

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.pub = self.create_publisher(CompressedImage, '/caera_raw/compressed', 10)

        self.cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.get_logger().error('No se pudo abrir la camara')
            return

        self.latest_jpeg = None
        self.lock = threading.Lock()

        self.reader = threading.Thread(target=self._read_loop, daemon=True)
        self.reader.start()

        self.publisher = threading.Thread(target=self._publish_loop, daemon=True)
        self.publisher.start()

        self.get_logger().info('Camara iniciada')

    def _read_loop(self):
        while rclpy.ok():
            ret, frame = self.cap.read()
            if not ret:
                continue
            # frame ya es JPEG raw en bytes, no BGR
            with self.lock:
                self.latest_jpeg = frame.tobytes()

    def _publish_loop(self):
        import time
        interval = 1.0 / 30.0
        while rclpy.ok():
            t0 = time.monotonic()

            with self.lock:
                jpeg = self.latest_jpeg
            if jpeg is None:
                time.sleep(0.001)
                continue

            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera'
            msg.format = 'jpeg'
            msg.data = jpeg
            self.pub.publish(msg)

            elapsed = time.monotonic() - t0
            sleep_t = interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
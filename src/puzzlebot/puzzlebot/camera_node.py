import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
import cv2
import numpy as np

class JetsonFisheyeFiltered(Node):
    def __init__(self):
        super().__init__('jetson_camera_node')
        
        self.comp_pub = self.create_publisher(CompressedImage, 'camera/image_raw/compressed', 5)
        self.raw_pub = self.create_publisher(Image, 'camera/image_raw', 5)

        gst_pipeline = (
            "nvarguscamerasrc wbmode=1 ! "
            "video/x-raw(memory:NVMM), width=320, height=240, framerate=30/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! appsink drop=True max-buffers=1"
        )

        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        self.timer = self.create_timer(1.0/30.0, self.timer_callback)
        self.get_logger().info("Nodo de cámara activo — raw y compressed")

    def apply_color_filter(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        a = cv2.subtract(a, 8)
        corrected_lab = cv2.merge((l, a, b))
        result = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
        result = cv2.convertScaleAbs(result, alpha=1.1, beta=-5)
        return result

    def frame_to_raw_msg(self, frame, stamp):
        """Construye sensor_msgs/Image sin cv_bridge."""
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = 'camera'
        msg.height = frame.shape[0]
        msg.width = frame.shape[1]
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = frame.shape[1] * 3          # ancho * 3 canales
        msg.data = frame.tobytes()
        return msg

    def timer_callback(self):
        if self.cap.grab():
            ret, frame = self.cap.retrieve()
            if ret:
                filtered_frame = self.apply_color_filter(frame)
                stamp = self.get_clock().now().to_msg()

                # --- Raw ---
                self.raw_pub.publish(self.frame_to_raw_msg(filtered_frame, stamp))

                # --- Compressed ---
                success, buffer = cv2.imencode(
                    '.jpg', filtered_frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                )
                if success:
                    comp_msg = CompressedImage()
                    comp_msg.header.stamp = stamp
                    comp_msg.header.frame_id = 'camera'
                    comp_msg.format = "jpeg"
                    comp_msg.data = buffer.tobytes()
                    self.comp_pub.publish(comp_msg)

def main(args=None):
    rclpy.init(args=args)
    node = JetsonFisheyeFiltered()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        node.destroy_node()
        rclpy.shutdown()

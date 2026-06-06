import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2, numpy as np, threading, os

os.makedirs('calib_imgs', exist_ok=True)

class Saver(Node):
    def __init__(self):
        super().__init__('img_saver')
        self.latest = None
        self.i = 0
        self.create_subscription(CompressedImage, '/camera_raw/compressed', self.cb, 10)

    def cb(self, msg):
        self.latest = msg

    def save(self):
        if self.latest is None:
            print('No hay imagen todavía')
            return
        arr = np.frombuffer(self.latest.data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        fname = f'calib_imgs/img{self.i:04d}.jpg'
        cv2.imwrite(fname, img)
        print(f'Saved {fname}')
        self.i += 1

rclpy.init()
node = Saver()

def input_loop():
    while True:
        input('Presiona Enter para capturar (Ctrl+C para salir)...')
        node.save()

t = threading.Thread(target=input_loop, daemon=True)
t.start()
rclpy.spin(node)
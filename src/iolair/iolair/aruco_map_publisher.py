#!/usr/bin/env python3
"""
aruco_map_publisher.py — Visualización de landmarks ArUco en RViz
=================================================================

Publica las posiciones predefinidas de los marcadores ArUco como
MarkerArray en el frame 'map', para visualizarlos en RViz.

Cada marcador se representa con:
  - Una esfera verde en su posición
  - Un texto flotante con su ID encima

Tópico publicado
----------------
  /aruco/markers  (visualization_msgs/MarkerArray)

Parámetros ROS
--------------
  landmarks_file   ruta al YAML con posiciones (igual que aruco_localizer)
  publish_rate     1.0  [Hz]  (estático, no necesita más)
  marker_color_r   0.0
  marker_color_g   1.0
  marker_color_b   0.0
  marker_alpha     0.9
  sphere_scale     0.08  [m]  radio de la esfera
  text_scale       0.12  [m]  altura del texto
"""

import os

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


class ArucoMapPublisher(Node):

    def __init__(self):
        super().__init__('aruco_map_publisher')

        # ── Parámetros ────────────────────────────────────────────────────
        self.declare_parameter('landmarks_file',  '')
        self.declare_parameter('publish_rate',    1.0)
        self.declare_parameter('marker_color_r',  0.0)
        self.declare_parameter('marker_color_g',  1.0)
        self.declare_parameter('marker_color_b',  0.0)
        self.declare_parameter('marker_alpha',    0.9)
        self.declare_parameter('sphere_scale',    0.08)
        self.declare_parameter('text_scale',      0.12)

        self._landmarks_file = self.get_parameter('landmarks_file').value
        rate        = self.get_parameter('publish_rate').value
        self._r     = self.get_parameter('marker_color_r').value
        self._g     = self.get_parameter('marker_color_g').value
        self._b     = self.get_parameter('marker_color_b').value
        self._a     = self.get_parameter('marker_alpha').value
        self._s_sph = self.get_parameter('sphere_scale').value
        self._s_txt = self.get_parameter('text_scale').value

        # ── Landmarks ─────────────────────────────────────────────────────
        # dict  id → (x, y)
        self._landmarks: dict[int, tuple[float, float]] = {}
        self._load_yaml()

        # ── Publicador ────────────────────────────────────────────────────
        self._pub = self.create_publisher(MarkerArray, '/aruco/markers', 10)
        self.create_timer(1.0 / rate, self._publish)

        self.get_logger().info(
            f'ArUco Map Publisher listo — '
            f'{len(self._landmarks)} landmarks → /aruco/markers'
        )

    # ── Carga YAML ────────────────────────────────────────────────────────

    def _load_yaml(self):
        path = self._landmarks_file
        if not path:
            self.get_logger().warn('landmarks_file vacío, no se publicará nada.')
            return
        if not _YAML_AVAILABLE:
            self.get_logger().error('PyYAML no disponible. pip install pyyaml')
            return
        if not os.path.isfile(path):
            self.get_logger().error(f'Archivo no encontrado: {path}')
            return
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict) or 'landmarks' not in data:
            self.get_logger().error('YAML no contiene clave "landmarks".')
            return
        for raw_id, coords in data['landmarks'].items():
            try:
                self._landmarks[int(raw_id)] = (
                    float(coords['x']),
                    float(coords['y']),
                )
            except (TypeError, KeyError, ValueError) as e:
                self.get_logger().warn(f'ID={raw_id} inválido: {e}')

    # ── Publicación ───────────────────────────────────────────────────────

    def _publish(self):
        if not self._landmarks:
            return

        array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        for aruco_id, (x, y) in self._landmarks.items():
            base_id = aruco_id * 2   # par = esfera, impar = texto

            # ── Esfera ────────────────────────────────────────────────────
            sphere = Marker()
            sphere.header.frame_id = 'map'
            sphere.header.stamp    = stamp
            sphere.ns              = 'aruco_spheres'
            sphere.id              = base_id
            sphere.type            = Marker.SPHERE
            sphere.action          = Marker.ADD
            sphere.pose.position.x = x
            sphere.pose.position.y = y
            sphere.pose.position.z = 0.0
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = self._s_sph
            sphere.scale.y = self._s_sph
            sphere.scale.z = self._s_sph
            sphere.color.r = self._r
            sphere.color.g = self._g
            sphere.color.b = self._b
            sphere.color.a = self._a
            sphere.lifetime = Duration(seconds=0).to_msg()  # 0 = permanente

            # ── Texto con ID ──────────────────────────────────────────────
            text = Marker()
            text.header.frame_id = 'map'
            text.header.stamp    = stamp
            text.ns              = 'aruco_labels'
            text.id              = base_id + 1
            text.type            = Marker.TEXT_VIEW_FACING
            text.action          = Marker.ADD
            text.pose.position.x = x
            text.pose.position.y = y
            text.pose.position.z = self._s_sph + 0.05   # justo encima de la esfera
            text.pose.orientation.w = 1.0
            text.scale.z         = self._s_txt
            text.color.r         = 1.0
            text.color.g         = 1.0
            text.color.b         = 1.0
            text.color.a         = 1.0
            text.text            = f'ID {aruco_id}'
            text.lifetime        = Duration(seconds=0).to_msg()

            array.markers.append(sphere)
            array.markers.append(text)

        self._pub.publish(array)


# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = ArucoMapPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
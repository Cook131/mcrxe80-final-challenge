#!/usr/bin/env python3
"""
Map Publisher Node — Puzzlebot SLAM
=====================================
Reads a pre-built map from a .pgm image + .yaml metadata file and
publishes it as a nav_msgs/OccupancyGrid on the /map topic.

The message is latched (TRANSIENT_LOCAL durability) so any subscriber
that connects after the first publish — including the MCL node and RViz —
will still receive the map immediately.

Subscribes to:  nothing
Publishes to:
    /map  (nav_msgs/OccupancyGrid, latched)

Parameters:
    map_yaml_path (string) — absolute path to the .yaml map descriptor.
                             Defaults to the maps/ directory inside the
                             installed package share.

YAML fields understood (standard ROS map_server format):
    image          — path to the .pgm file (relative to the yaml, or absolute)
    resolution     — metres per pixel
    origin         — [x, y, yaw]  world pose of the lower-left pixel
    negate         — 0 or 1 (inverts pixel values when 1)
    occupied_thresh  — pixels with p >= this are OCCUPIED  (0-1 scale)
    free_thresh      — pixels with p <= this are FREE       (0-1 scale)
    Pixels between the two thresholds → UNKNOWN (-1)
"""

import os
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose, Point, Quaternion
from ament_index_python.packages import get_package_share_directory

import yaml
import numpy as np
from PIL import Image


# ── Helpers ────────────────────────────────────────────────────────────────────

def yaw_to_quaternion(yaw: float) -> Quaternion:
    """Convert a yaw angle (radians) to a geometry_msgs/Quaternion."""
    q = Quaternion()
    q.w = math.cos(yaw * 0.5)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    return q


def pgm_to_occupancy(pgm_path: str,
                     occupied_thresh: float,
                     free_thresh: float,
                     negate: int) -> np.ndarray:
    """
    Load a .pgm (or any PIL-readable greyscale image) and convert
    each pixel to an OccupancyGrid cell value:
        100  → occupied
          0  → free
         -1  → unknown

    ROS convention:
        pixel value 0   = black  = obstacle  (p = 0.0  after normalisation)
        pixel value 255 = white  = free      (p = 1.0  after normalisation)
    When negate=1 the probability is 1 - p.
    """
    img = Image.open(pgm_path).convert('L')   # force 8-bit greyscale
    pixels = np.array(img, dtype=np.float32)  # shape (height, width)

    # Normalise to [0, 1]
    p = pixels / 255.0
    if negate:
        p = 1.0 - p

    # Map probability → OccupancyGrid convention
    # p close to 0 → definitely occupied → cell = 100
    # p close to 1 → definitely free     → cell = 0
    # In-between                          → cell = -1 (unknown)
    occupied_p = 1.0 - occupied_thresh   # pixel brightness threshold for obstacle
    free_p     = 1.0 - free_thresh       # pixel brightness threshold for free

    grid = np.full(pixels.shape, -1, dtype=np.int8)
    grid[p <= occupied_p] = 100    # obstacle
    grid[p >= free_p]     = 0      # free  (may overwrite some unknowns)

    # ROS stores the grid row-major with row 0 = bottom of the map
    # PIL stores row 0 = top of the image → flip vertically
    grid = np.flipud(grid)

    return grid   # shape: (height, width)


# ── Node ───────────────────────────────────────────────────────────────────────

class MapPublisherNode(Node):

    def __init__(self):
        super().__init__('map_publisher_node')

        # ── Parameter: path to the yaml descriptor ────────────────────────
        self.declare_parameter('map_yaml_path', '')
        yaml_path = self.get_parameter('map_yaml_path').value

        if not yaml_path:
            # Default: maps/puzzlebot_map.yaml inside the installed package
            pkg_share = get_package_share_directory('iolair')
            yaml_path = os.path.join(pkg_share, 'maps', 'puzzlebot_map.yaml')

        if not os.path.isfile(yaml_path):
            self.get_logger().fatal(
                f'Map YAML not found: {yaml_path}\n'
                f'Pass the correct path via --ros-args -p map_yaml_path:=<path>'
            )
            raise FileNotFoundError(yaml_path)

        self.get_logger().info(f'Loading map from: {yaml_path}')

        # ── Parse YAML ────────────────────────────────────────────────────
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)

        image_field      = cfg['image']
        resolution       = float(cfg.get('resolution', 0.05))
        origin           = cfg.get('origin', [0.0, 0.0, 0.0])
        negate           = int(cfg.get('negate', 0))
        occupied_thresh  = float(cfg.get('occupied_thresh', 0.65))
        free_thresh      = float(cfg.get('free_thresh', 0.196))

        # Resolve .pgm path (may be relative to the yaml file)
        if not os.path.isabs(image_field):
            pgm_path = os.path.join(os.path.dirname(yaml_path), image_field)
        else:
            pgm_path = image_field

        if not os.path.isfile(pgm_path):
            self.get_logger().fatal(f'Map image not found: {pgm_path}')
            raise FileNotFoundError(pgm_path)

        # ── Convert image → occupancy grid ────────────────────────────────
        grid = pgm_to_occupancy(pgm_path, occupied_thresh, free_thresh, negate)
        height, width = grid.shape

        self.get_logger().info(
            f'Map loaded: {width}x{height} cells, '
            f'res={resolution} m/cell, '
            f'origin=({origin[0]:.3f}, {origin[1]:.3f}, yaw={origin[2]:.3f})'
        )

        # ── Build OccupancyGrid message ────────────────────────────────────
        origin_pose = Pose()
        origin_pose.position    = Point(x=float(origin[0]),
                                        y=float(origin[1]),
                                        z=0.0)
        origin_pose.orientation = yaw_to_quaternion(float(origin[2]))

        meta = MapMetaData()
        meta.resolution = resolution
        meta.width      = width
        meta.height     = height
        meta.origin     = origin_pose

        self._map_msg            = OccupancyGrid()
        self._map_msg.header.frame_id = 'map'
        self._map_msg.info       = meta
        # Flatten row-major (row 0 = bottom of world after flipud above)
        self._map_msg.data       = grid.flatten().tolist()

        # ── Latched publisher (TRANSIENT_LOCAL) ───────────────────────────
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._pub = self.create_publisher(OccupancyGrid, '/map', latched_qos)

        # Publish once immediately, then every 5 s so late-joining nodes
        # (and re-started RViz sessions) receive it without restarting.
        self._publish()
        self.create_timer(5.0, self._publish)

        self.get_logger().info(
            '/map published (TRANSIENT_LOCAL).  '
            'MCL node and RViz should now display the occupancy grid.'
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _publish(self):
        self._map_msg.header.stamp = self.get_clock().now().to_msg()
        self._pub.publish(self._map_msg)


# ── Entry point ────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = MapPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
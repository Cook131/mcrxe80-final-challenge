#!/usr/bin/env python3
"""
Puzzlebot Online SLAM — Occupancy Grid Builder
===============================================
Builds a 2-D occupancy grid in real time from LiDAR scans and the
robot's estimated pose.  Publishes the growing map as a latched
nav_msgs/OccupancyGrid so RViz and any other node (including the MCL
node) can consume it.

Subscribes to:
    /lidar      (sensor_msgs/LaserScan)   — LiDAR measurements
    /mcl_pose   (geometry_msgs/PoseStamped, optional) — MCL best pose
    /odom       (nav_msgs/Odometry, fallback)          — odometry pose

Publishes to:
    /slam_map   (nav_msgs/OccupancyGrid, TRANSIENT_LOCAL latched)

Design decisions
----------------
* Log-odds update model (Thrun et al., Probabilistic Robotics, Ch. 9).
  Each cell stores a log-odds value; converting to probability only
  happens at publish time.  This avoids floating-point drift from
  repeated multiplication.

* The grid auto-expands when scan end-points land outside the current
  bounds.  It starts small (MAP_INIT_SIZE × MAP_INIT_SIZE cells) and
  grows in chunks of EXPAND_CELLS to keep reallocations infrequent.

* Bresenham ray-casting marks free cells between the robot and each
  beam end-point, and marks the end-point cell as occupied (unless
  the reading is at max range, in which case only free cells are marked).

* The node prefers /mcl_pose for the robot position because it is more
  accurate than dead-reckoning.  If /mcl_pose has never been received
  it falls back to /odom automatically.

Parameters (all have defaults; override via --ros-args -p name:=value)
----------
    resolution          float  0.05   m/cell
    map_frame           str    'map'  frame_id of the published grid
    publish_rate        float  2.0    Hz  (grid is published periodically)
    log_odds_occ        float  0.85   log-odds increment per occupied hit
    log_odds_free       float  0.40   log-odds decrement per free ray step
    log_odds_max        float  3.5    saturation cap (positive = occupied)
    log_odds_min        float -3.5    saturation cap (negative = free)
    lidar_max_range     float  10.0   m  (readings >= this are ignored as hits)
    beam_skip           int    2      use every Nth beam (1 = all beams)
    map_init_size       int    400    initial grid side length [cells]
    map_origin_x        float -10.0  world X of the bottom-left corner [m]
    map_origin_y        float -10.0  world Y of the bottom-left corner [m]
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg    import OccupancyGrid, MapMetaData, Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped

import numpy as np


# ── Helpers ────────────────────────────────────────────────────────────────────

def yaw_from_quaternion(q) -> float:
    """Extract yaw from a geometry_msgs/Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw * 0.5)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    return q


def bresenham(x0: int, y0: int, x1: int, y1: int):
    """
    Yield all integer (col, row) cells on the line from (x0,y0) to (x1,y1),
    exclusive of the end-point (so the caller can handle it separately).
    Uses the standard integer Bresenham algorithm.
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy

    while True:
        if x0 == x1 and y0 == y1:
            break
        yield x0, y0
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0  += sx
        if e2 < dx:
            err += dx
            y0  += sy


# ── SLAM Node ──────────────────────────────────────────────────────────────────

class SLAMNode(Node):

    def __init__(self):
        super().__init__('slam_node')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('resolution',      0.05)
        self.declare_parameter('map_frame',       'map')
        self.declare_parameter('publish_rate',    2.0)
        self.declare_parameter('log_odds_occ',    0.85)
        self.declare_parameter('log_odds_free',   0.40)
        self.declare_parameter('log_odds_max',    3.5)
        self.declare_parameter('log_odds_min',   -3.5)
        self.declare_parameter('lidar_max_range', 10.0)
        self.declare_parameter('beam_skip',       2)
        self.declare_parameter('map_init_size',   400)
        self.declare_parameter('map_origin_x',   -10.0)
        self.declare_parameter('map_origin_y',   -10.0)

        self.res        = self.get_parameter('resolution').value
        self.map_frame  = self.get_parameter('map_frame').value
        self.lo_occ     = self.get_parameter('log_odds_occ').value
        self.lo_free    = self.get_parameter('log_odds_free').value
        self.lo_max     = self.get_parameter('log_odds_max').value
        self.lo_min     = self.get_parameter('log_odds_min').value
        self.max_range  = self.get_parameter('lidar_max_range').value
        self.beam_skip  = self.get_parameter('beam_skip').value
        pub_rate        = self.get_parameter('publish_rate').value
        init_size       = self.get_parameter('map_init_size').value
        self.origin_x   = self.get_parameter('map_origin_x').value
        self.origin_y   = self.get_parameter('map_origin_y').value

        # ── Log-odds grid (float32 for precision) ─────────────────────────
        # Shape: (height, width) = (rows, cols); row 0 = world-south edge.
        self.grid_h = init_size   # rows  (Y axis)
        self.grid_w = init_size   # cols  (X axis)
        self.log_odds = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        # ── Robot pose (world frame) ──────────────────────────────────────
        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_yaw = 0.0
        self.mcl_pose_received = False   # prefer MCL over odom when available

        # ── QoS ───────────────────────────────────────────────────────────
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        sensor_qos = rclpy.qos.qos_profile_sensor_data

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(
            LaserScan, '/lidar', self._cb_scan, sensor_qos)
        self.create_subscription(
            PoseStamped, '/mcl_pose', self._cb_mcl_pose, 10)
        self.create_subscription(
            Odometry, '/odom', self._cb_odom, 10)

        # ── Publisher ─────────────────────────────────────────────────────
        self._map_pub = self.create_publisher(
            OccupancyGrid, '/slam_map', latched_qos)

        # ── Periodic publish timer ─────────────────────────────────────────
        self.create_timer(1.0 / pub_rate, self._publish_map)

        self.get_logger().info(
            f'SLAM node started — grid {self.grid_w}×{self.grid_h} cells, '
            f'res={self.res} m/cell, '
            f'origin=({self.origin_x}, {self.origin_y})'
        )

    # ── Pose callbacks ────────────────────────────────────────────────────────

    def _cb_mcl_pose(self, msg: PoseStamped):
        """Use the MCL best-pose estimate (preferred)."""
        self.robot_x   = msg.pose.position.x
        self.robot_y   = msg.pose.position.y
        self.robot_yaw = yaw_from_quaternion(msg.pose.orientation)
        self.mcl_pose_received = True

    def _cb_odom(self, msg: Odometry):
        """Fall back to odometry when MCL pose is not yet available."""
        if self.mcl_pose_received:
            return   # MCL is running — ignore raw odometry
        self.robot_x   = msg.pose.pose.position.x
        self.robot_y   = msg.pose.pose.position.y
        self.robot_yaw = yaw_from_quaternion(msg.pose.pose.orientation)

    # ── LiDAR callback — core mapping logic ──────────────────────────────────

    def _cb_scan(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        n_beams = len(ranges)

        # Robot cell in the grid
        rx, ry = self._world_to_cell(self.robot_x, self.robot_y)
        if not self._in_bounds(rx, ry):
            self._expand_to_fit(self.robot_x, self.robot_y)
            rx, ry = self._world_to_cell(self.robot_x, self.robot_y)

        for i in range(0, n_beams, self.beam_skip):
            r = ranges[i]
            angle = msg.angle_min + i * msg.angle_increment
            global_angle = self.robot_yaw + angle

            is_hit = math.isfinite(r) and msg.range_min < r < self.max_range

            # End-point in world frame
            if is_hit:
                ex = self.robot_x + r * math.cos(global_angle)
                ey = self.robot_y + r * math.sin(global_angle)
            else:
                # Max-range ray: mark free up to max_range, no obstacle hit
                ex = self.robot_x + self.max_range * math.cos(global_angle)
                ey = self.robot_y + self.max_range * math.sin(global_angle)

            # Expand grid if end-point is outside current bounds
            if not self._in_bounds(*self._world_to_cell(ex, ey)):
                self._expand_to_fit(ex, ey)
                rx, ry = self._world_to_cell(self.robot_x, self.robot_y)

            ex_c, ey_c = self._world_to_cell(ex, ey)
            ex_c = max(0, min(ex_c, self.grid_w - 1))
            ey_c = max(0, min(ey_c, self.grid_h - 1))

            # Mark free cells along the ray (Bresenham)
            for cx, cy in bresenham(rx, ry, ex_c, ey_c):
                if self._in_bounds(cx, cy):
                    self.log_odds[cy, cx] = max(
                        self.lo_min,
                        self.log_odds[cy, cx] - self.lo_free
                    )

            # Mark end-point as occupied (only if it was a real hit)
            if is_hit and self._in_bounds(ex_c, ey_c):
                self.log_odds[ey_c, ex_c] = min(
                    self.lo_max,
                    self.log_odds[ey_c, ex_c] + self.lo_occ
                )

    # ── Grid expansion ────────────────────────────────────────────────────────

    def _expand_to_fit(self, wx: float, wy: float, margin: int = 100):
        """
        Expand the grid so that world point (wx, wy) fits inside,
        plus a margin of cells on each side.
        The log-odds array is grown (zero-padded) and origin updated.
        """
        cx, cy = self._world_to_cell(wx, wy)

        # How many cells to add on each side
        pad_left  = max(0, margin - cx)
        pad_right = max(0, cx + margin - self.grid_w + 1)
        pad_bot   = max(0, margin - cy)
        pad_top   = max(0, cy + margin - self.grid_h + 1)

        if pad_left == 0 and pad_right == 0 and pad_bot == 0 and pad_top == 0:
            return   # already fits

        new_w = self.grid_w + pad_left + pad_right
        new_h = self.grid_h + pad_bot  + pad_top
        new_grid = np.zeros((new_h, new_w), dtype=np.float32)
        new_grid[pad_bot:pad_bot + self.grid_h,
                 pad_left:pad_left + self.grid_w] = self.log_odds

        self.log_odds = new_grid
        self.grid_w   = new_w
        self.grid_h   = new_h

        # Shift origin to account for the new padding
        self.origin_x -= pad_left * self.res
        self.origin_y -= pad_bot  * self.res

        self.get_logger().info(
            f'Grid expanded → {self.grid_w}×{self.grid_h} cells, '
            f'origin=({self.origin_x:.2f}, {self.origin_y:.2f})'
        )

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def _world_to_cell(self, wx: float, wy: float):
        """Convert world coordinates to integer grid cell (col, row)."""
        col = int((wx - self.origin_x) / self.res)
        row = int((wy - self.origin_y) / self.res)
        return col, row

    def _in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self.grid_w and 0 <= row < self.grid_h

    # ── Publisher ─────────────────────────────────────────────────────────────

    def _publish_map(self):
        """Convert log-odds grid → OccupancyGrid and publish."""
        # log-odds → probability → ROS convention [0, 100, -1]
        # p(occ) = 1 - 1/(1 + exp(lo))
        prob = 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))
        ros_grid = np.full(self.log_odds.shape, -1, dtype=np.int8)

        occ_thresh  = 0.65
        free_thresh = 0.35

        ros_grid[prob >= occ_thresh]  = 100
        ros_grid[prob <= free_thresh] = 0
        # cells between thresholds stay -1 (unknown)

        # Build message
        msg = OccupancyGrid()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame

        meta = MapMetaData()
        meta.resolution = self.res
        meta.width      = self.grid_w
        meta.height     = self.grid_h

        origin_pose = Pose()
        origin_pose.position    = Point(
            x=float(self.origin_x), y=float(self.origin_y), z=0.0)
        origin_pose.orientation = yaw_to_quaternion(0.0)
        meta.origin = origin_pose
        msg.info    = meta

        msg.data = ros_grid.flatten().tolist()
        self._map_pub.publish(msg)


# ── Entry point ────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = SLAMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

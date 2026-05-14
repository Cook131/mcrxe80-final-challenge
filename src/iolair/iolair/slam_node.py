#!/usr/bin/env python3
"""
Puzzlebot Online SLAM — Occupancy Grid Builder (No Nav2)
=========================================================
Builds a 2-D occupancy grid in real time from LiDAR scans and the
robot's estimated pose.  Publishes the growing map as a latched
nav_msgs/OccupancyGrid so RViz and any other node can consume it.

Key improvements over the original:
  - Publishes the map→odom TF transform directly (no MCL dependency).
  - Supports saving the map to disk (.pgm + .yaml) via a ROS service.
  - ICP-based scan matching corrects odometry drift between consecutive scans.
  - Thread-safe scan / odometry access with a Lock.
  - Configurable via ROS 2 parameters (all have sensible defaults).

Architecture (no nav2, no mcl_node required):
  ┌─────────────┐   /lidar (LaserScan)  ┌──────────────────────┐
  │  LiDAR      │ ──────────────────────►│                      │
  └─────────────┘                        │  slam_node           │
  ┌─────────────┐   /odom (Odometry)    │  • Log-odds grid     │──► /slam_map
  │  Odom node  │ ──────────────────────►│  • ICP refinement    │──► /slam_pose
  └─────────────┘                        │  • TF map→odom       │──► TF map→odom
                                         │  • Map save service  │
                                         └──────────────────────┘

Subscribes to:
    /lidar    (sensor_msgs/LaserScan)  — LiDAR measurements
    /odom     (nav_msgs/Odometry)      — wheel-odometry pose (dead reckoning)

Publishes:
    /slam_map   (nav_msgs/OccupancyGrid, TRANSIENT_LOCAL latched)
    /slam_pose  (geometry_msgs/PoseStamped) — SLAM-corrected robot pose

TF broadcast:
    map → odom   (dynamic, updated every scan via ICP correction)

Services:
    /slam/save_map  (std_srvs/Trigger) — saves .pgm + .yaml to /tmp/slam_map.*

Parameters (all have defaults; override via --ros-args -p name:=value):
    resolution          float  0.05   m/cell
    map_frame           str    'map'  frame_id of the published grid
    odom_frame          str    'odom'
    base_frame          str    'base_link'
    publish_rate        float  2.0    Hz  (grid published periodically)
    log_odds_occ        float  0.85   log-odds increment per occupied hit
    log_odds_free       float  0.40   log-odds decrement per free ray step
    log_odds_max        float  3.5    saturation cap (positive = occupied)
    log_odds_min        float -3.5    saturation cap (negative = free)
    lidar_max_range     float  10.0   m  (readings >= this are ignored as hits)
    beam_skip           int    3      use every Nth beam (1 = all beams)
    map_init_size       int    400    initial grid side length [cells]
    map_origin_x        float -10.0  world X of the bottom-left corner [m]
    map_origin_y        float -10.0  world Y of the bottom-left corner [m]
    use_icp             bool   True   enable ICP scan-to-scan correction
    icp_max_iter        int    20     ICP iteration limit
    icp_tolerance       float  1e-4   ICP convergence threshold
    save_map_path       str    '/tmp/slam_map'  base path for map saving
"""

import math
import threading
import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg        import OccupancyGrid, MapMetaData, Odometry
from sensor_msgs.msg     import LaserScan
from geometry_msgs.msg   import (
    Pose, Point, Quaternion, PoseStamped, TransformStamped
)
from std_srvs.srv        import Trigger
from tf2_ros             import TransformBroadcaster


# ── Quaternion helpers ─────────────────────────────────────────────────────────

def yaw_from_quaternion(q) -> float:
    """Extract yaw (radians) from a geometry_msgs/Quaternion."""
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


def normalize_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


# ── Bresenham ray casting ──────────────────────────────────────────────────────

def bresenham(x0: int, y0: int, x1: int, y1: int):
    """
    Yield integer (col, row) cells along a line from (x0,y0) to (x1,y1),
    NOT including the end-point so the caller handles it separately.
    """
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy

    while x0 != x1 or y0 != y1:
        yield x0, y0
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0  += sx
        if e2 < dx:
            err += dx
            y0  += sy


# ── ICP (2-D, point-to-point) ──────────────────────────────────────────────────

def scan_to_points(scan: LaserScan, max_range: float,
                   robot_x: float, robot_y: float, robot_yaw: float,
                   beam_skip: int = 1) -> np.ndarray:
    """
    Convert a LaserScan into a (N, 2) array of world-frame XY hit points.
    Returns an empty array when the scan has no valid beams.
    """
    pts = []
    ranges = scan.ranges
    for i in range(0, len(ranges), beam_skip):
        r = ranges[i]
        if not math.isfinite(r) or r < scan.range_min or r >= max_range:
            continue
        angle = scan.angle_min + i * scan.angle_increment + robot_yaw
        pts.append([
            robot_x + r * math.cos(angle),
            robot_y + r * math.sin(angle),
        ])
    return np.array(pts, dtype=np.float64) if pts else np.empty((0, 2))


def icp_2d(src: np.ndarray, dst: np.ndarray,
           max_iter: int = 20, tol: float = 1e-4):
    """
    Simple 2-D ICP (point-to-point, nearest-neighbour).
    Returns (dx, dy, dyaw) correction to apply to the source pose.
    Falls back to (0, 0, 0) when point clouds are too small or ICP diverges.
    """
    if len(src) < 10 or len(dst) < 10:
        return 0.0, 0.0, 0.0

    src_h = src.copy()
    dx_total = dy_total = dyaw_total = 0.0

    for _ in range(max_iter):
        # Nearest-neighbour matching (brute-force; fast enough for ~200 pts)
        dists = np.linalg.norm(
            src_h[:, None, :] - dst[None, :, :], axis=2)   # (Ns, Nd)
        nn_idx = np.argmin(dists, axis=1)
        nn_dist = dists[np.arange(len(src_h)), nn_idx]

        # Reject outliers beyond the 90th-percentile distance
        threshold = np.percentile(nn_dist, 90)
        mask = nn_dist < threshold
        if mask.sum() < 5:
            break

        S = src_h[mask]
        T = dst[nn_idx[mask]]

        # Compute optimal 2-D rigid transform (SVD)
        mu_s = S.mean(axis=0)
        mu_t = T.mean(axis=0)
        H = (S - mu_s).T @ (T - mu_t)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        # Ensure a proper rotation (det = +1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = mu_t - R @ mu_s

        dyaw = math.atan2(R[1, 0], R[0, 0])
        dx_total   += t[0]
        dy_total   += t[1]
        dyaw_total += dyaw

        # Apply transform to source for next iteration
        src_h = (R @ src_h.T).T + t

        if abs(t[0]) < tol and abs(t[1]) < tol and abs(dyaw) < tol:
            break

    return dx_total, dy_total, normalize_angle(dyaw_total)


# ── Map saver ─────────────────────────────────────────────────────────────────

def save_map_pgm_yaml(log_odds: np.ndarray, resolution: float,
                      origin_x: float, origin_y: float,
                      base_path: str) -> str:
    """
    Write an occupancy map to <base_path>.pgm and <base_path>.yaml.
    Returns a human-readable status string.
    """
    rows, cols = log_odds.shape

    # Convert log-odds → 8-bit PGM (ROS convention: 0=free, 205=unknown, 254=occupied)
    prob = 1.0 - 1.0 / (1.0 + np.exp(log_odds))
    pgm = np.full((rows, cols), 205, dtype=np.uint8)     # unknown
    pgm[prob >= 0.65] = 0                                 # occupied  → dark
    pgm[prob <= 0.35] = 254                               # free      → light

    # PGM is stored top-row = north (high Y); flip vertically
    pgm_img = np.flipud(pgm)

    pgm_path  = base_path + '.pgm'
    yaml_path = base_path + '.yaml'

    # Write PGM (P5 binary)
    with open(pgm_path, 'wb') as f:
        header = f'P5\n{cols} {rows}\n255\n'
        f.write(header.encode())
        f.write(pgm_img.tobytes())

    # Write YAML metadata (ROS map_server format)
    yaml_name = os.path.basename(pgm_path)
    with open(yaml_path, 'w') as f:
        f.write(f'image: {yaml_name}\n')
        f.write(f'resolution: {resolution}\n')
        f.write(f'origin: [{origin_x:.4f}, {origin_y:.4f}, 0.0]\n')
        f.write('negate: 0\n')
        f.write('occupied_thresh: 0.65\n')
        f.write('free_thresh: 0.35\n')

    return f'Map saved to {pgm_path} and {yaml_path} ({cols}×{rows} cells)'


# ── SLAM Node ──────────────────────────────────────────────────────────────────

class SLAMNode(Node):

    def __init__(self):
        super().__init__('slam_node')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('resolution',      0.05)
        self.declare_parameter('map_frame',       'map')
        self.declare_parameter('odom_frame',      'odom')
        self.declare_parameter('base_frame',      'base_link')
        self.declare_parameter('publish_rate',    2.0)
        self.declare_parameter('log_odds_occ',    0.85)
        self.declare_parameter('log_odds_free',   0.40)
        self.declare_parameter('log_odds_max',    3.5)
        self.declare_parameter('log_odds_min',   -3.5)
        self.declare_parameter('lidar_max_range', 10.0)
        self.declare_parameter('beam_skip',       3)
        self.declare_parameter('map_init_size',   400)
        self.declare_parameter('map_origin_x',   -10.0)
        self.declare_parameter('map_origin_y',   -10.0)
        self.declare_parameter('use_icp',         True)
        self.declare_parameter('icp_max_iter',    20)
        self.declare_parameter('icp_tolerance',   1e-4)
        self.declare_parameter('save_map_path',   '/tmp/slam_map')

        self.res        = self.get_parameter('resolution').value
        self.map_frame  = self.get_parameter('map_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.lo_occ     = self.get_parameter('log_odds_occ').value
        self.lo_free    = self.get_parameter('log_odds_free').value
        self.lo_max     = self.get_parameter('log_odds_max').value
        self.lo_min     = self.get_parameter('log_odds_min').value
        self.max_range  = self.get_parameter('lidar_max_range').value
        self.beam_skip  = self.get_parameter('beam_skip').value
        self.use_icp    = self.get_parameter('use_icp').value
        self.icp_iter   = self.get_parameter('icp_max_iter').value
        self.icp_tol    = self.get_parameter('icp_tolerance').value
        self.save_path  = self.get_parameter('save_map_path').value
        pub_rate        = self.get_parameter('publish_rate').value
        init_size       = self.get_parameter('map_init_size').value

        self.origin_x   = self.get_parameter('map_origin_x').value
        self.origin_y   = self.get_parameter('map_origin_y').value

        # ── Log-odds grid ─────────────────────────────────────────────────
        self.grid_h = init_size
        self.grid_w = init_size
        self.log_odds = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self._grid_lock = threading.Lock()

        # ── Robot pose in the MAP frame ───────────────────────────────────
        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_yaw = 0.0

        # ── ICP correction offset (map_frame vs odom_frame) ───────────────
        # map_pose = odom_pose + correction
        self._corr_x   = 0.0
        self._corr_y   = 0.0
        self._corr_yaw = 0.0

        # ── Previous scan (for ICP scan-to-scan matching) ─────────────────
        self._prev_scan_pts: np.ndarray = np.empty((0, 2))

        # ── QoS profiles ─────────────────────────────────────────────────
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        sensor_qos = rclpy.qos.qos_profile_sensor_data

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(
            LaserScan, '/scan', self._cb_scan, sensor_qos)
        self.create_subscription(
            Odometry, '/odom', self._cb_odom, 10)

        # ── Publishers ────────────────────────────────────────────────────
        self._map_pub  = self.create_publisher(
            OccupancyGrid, '/slam_map', latched_qos)
        self._pose_pub = self.create_publisher(
            PoseStamped, '/slam_pose', 10)

        # ── TF broadcaster (map → odom) ───────────────────────────────────
        self._tf_broadcaster = TransformBroadcaster(self)

        # ── Service: save map to disk ─────────────────────────────────────
        self.create_service(Trigger, '/slam/save_map', self._srv_save_map)

        # ── Timers ────────────────────────────────────────────────────────
        self.create_timer(1.0 / pub_rate, self._publish_map)
        # TF must be re-broadcast more frequently than the map publish rate
        self.create_timer(0.05, self._broadcast_tf)       # 20 Hz

        self.get_logger().info(
            f'SLAM node started — grid {self.grid_w}×{self.grid_h} cells, '
            f'res={self.res} m/cell, '
            f'origin=({self.origin_x}, {self.origin_y}), '
            f'ICP={"on" if self.use_icp else "off"}'
        )

    # ── Odometry callback ─────────────────────────────────────────────────────

    def _cb_odom(self, msg: Odometry):
        """Track the raw odometry pose; apply ICP correction to get map pose."""
        odom_x   = msg.pose.pose.position.x
        odom_y   = msg.pose.pose.position.y
        odom_yaw = yaw_from_quaternion(msg.pose.pose.orientation)

        # Compose odom pose with the accumulated ICP correction
        cos_c = math.cos(self._corr_yaw)
        sin_c = math.sin(self._corr_yaw)
        self.robot_x   = cos_c * odom_x - sin_c * odom_y + self._corr_x
        self.robot_y   = sin_c * odom_x + cos_c * odom_y + self._corr_y
        self.robot_yaw = normalize_angle(odom_yaw + self._corr_yaw)

    # ── LiDAR callback — core mapping + ICP ──────────────────────────────────

    def _cb_scan(self, msg: LaserScan):
        # ── ICP scan-matching: refine pose before updating the map ────────
        if self.use_icp:
            cur_pts = scan_to_points(
                msg, self.max_range,
                self.robot_x, self.robot_y, self.robot_yaw,
                beam_skip=self.beam_skip
            )
            if len(self._prev_scan_pts) > 10 and len(cur_pts) > 10:
                dx, dy, dyaw = icp_2d(
                    cur_pts, self._prev_scan_pts,
                    max_iter=self.icp_iter, tol=self.icp_tol
                )
                # Limit correction magnitude to avoid wild jumps
                if abs(dx) < 0.5 and abs(dy) < 0.5 and abs(dyaw) < 0.3:
                    self._corr_x   += dx
                    self._corr_y   += dy
                    self._corr_yaw  = normalize_angle(self._corr_yaw + dyaw)
                    # Re-apply updated correction to robot pose
                    self.robot_x   += dx
                    self.robot_y   += dy
                    self.robot_yaw  = normalize_angle(self.robot_yaw + dyaw)

            self._prev_scan_pts = scan_to_points(
                msg, self.max_range,
                self.robot_x, self.robot_y, self.robot_yaw,
                beam_skip=self.beam_skip
            )

        # ── Update occupancy grid ─────────────────────────────────────────
        ranges  = np.asarray(msg.ranges, dtype=np.float32)
        n_beams = len(ranges)

        with self._grid_lock:
            rx, ry = self._world_to_cell(self.robot_x, self.robot_y)
            if not self._in_bounds(rx, ry):
                self._expand_to_fit(self.robot_x, self.robot_y)
                rx, ry = self._world_to_cell(self.robot_x, self.robot_y)

            for i in range(0, n_beams, self.beam_skip):
                r = ranges[i]
                angle = msg.angle_min + i * msg.angle_increment
                global_angle = self.robot_yaw + angle

                is_hit = (math.isfinite(r)
                          and msg.range_min < r < self.max_range)

                if is_hit:
                    ex = self.robot_x + r * math.cos(global_angle)
                    ey = self.robot_y + r * math.sin(global_angle)
                else:
                    ex = self.robot_x + self.max_range * math.cos(global_angle)
                    ey = self.robot_y + self.max_range * math.sin(global_angle)

                if not self._in_bounds(*self._world_to_cell(ex, ey)):
                    self._expand_to_fit(ex, ey)
                    rx, ry = self._world_to_cell(self.robot_x, self.robot_y)

                ex_c, ey_c = self._world_to_cell(ex, ey)
                ex_c = max(0, min(ex_c, self.grid_w - 1))
                ey_c = max(0, min(ey_c, self.grid_h - 1))

                for cx, cy in bresenham(rx, ry, ex_c, ey_c):
                    if self._in_bounds(cx, cy):
                        self.log_odds[cy, cx] = max(
                            self.lo_min,
                            self.log_odds[cy, cx] - self.lo_free
                        )

                if is_hit and self._in_bounds(ex_c, ey_c):
                    self.log_odds[ey_c, ex_c] = min(
                        self.lo_max,
                        self.log_odds[ey_c, ex_c] + self.lo_occ
                    )

    # ── TF broadcaster ────────────────────────────────────────────────────────

    def _broadcast_tf(self):
        """
        Broadcast map → odom TF.

        The map→odom transform encodes the ICP correction accumulated
        since startup.  Combined with odom→base_link (from the odometry
        node), this gives a full map→base_link chain in RViz without nav2.
        """
        t = TransformStamped()
        t.header.stamp    = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame
        t.child_frame_id  = self.odom_frame
        t.transform.translation.x = self._corr_x
        t.transform.translation.y = self._corr_y
        t.transform.translation.z = 0.0
        t.transform.rotation      = yaw_to_quaternion(self._corr_yaw)
        self._tf_broadcaster.sendTransform(t)

    # ── Grid expansion ────────────────────────────────────────────────────────

    def _expand_to_fit(self, wx: float, wy: float, margin: int = 100):
        cx, cy = self._world_to_cell(wx, wy)

        pad_left  = max(0, margin - cx)
        pad_right = max(0, cx + margin - self.grid_w + 1)
        pad_bot   = max(0, margin - cy)
        pad_top   = max(0, cy + margin - self.grid_h + 1)

        if not any([pad_left, pad_right, pad_bot, pad_top]):
            return

        new_w = self.grid_w + pad_left + pad_right
        new_h = self.grid_h + pad_bot  + pad_top
        new_grid = np.zeros((new_h, new_w), dtype=np.float32)
        new_grid[pad_bot:pad_bot + self.grid_h,
                 pad_left:pad_left + self.grid_w] = self.log_odds

        self.log_odds = new_grid
        self.grid_w   = new_w
        self.grid_h   = new_h
        self.origin_x -= pad_left * self.res
        self.origin_y -= pad_bot  * self.res

        self.get_logger().info(
            f'Grid expanded → {self.grid_w}×{self.grid_h} cells, '
            f'origin=({self.origin_x:.2f}, {self.origin_y:.2f})'
        )

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def _world_to_cell(self, wx: float, wy: float):
        col = int((wx - self.origin_x) / self.res)
        row = int((wy - self.origin_y) / self.res)
        return col, row

    def _in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self.grid_w and 0 <= row < self.grid_h

    # ── Map publisher ─────────────────────────────────────────────────────────

    def _publish_map(self):
        with self._grid_lock:
            lo_copy    = self.log_odds.copy()
            origin_x   = self.origin_x
            origin_y   = self.origin_y
            grid_w     = self.grid_w
            grid_h     = self.grid_h

        prob = 1.0 - 1.0 / (1.0 + np.exp(lo_copy))
        ros_grid = np.full(lo_copy.shape, -1, dtype=np.int8)
        ros_grid[prob >= 0.65] = 100
        ros_grid[prob <= 0.35] = 0

        msg = OccupancyGrid()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame

        meta = MapMetaData()
        meta.resolution = self.res
        meta.width      = grid_w
        meta.height     = grid_h
        meta.origin     = Pose(
            position=Point(x=float(origin_x), y=float(origin_y), z=0.0),
            orientation=yaw_to_quaternion(0.0)
        )
        msg.info = meta
        msg.data = ros_grid.flatten().tolist()
        self._map_pub.publish(msg)

        # Also publish the current SLAM pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp    = msg.header.stamp
        pose_msg.header.frame_id = self.map_frame
        pose_msg.pose.position   = Point(
            x=self.robot_x, y=self.robot_y, z=0.0)
        pose_msg.pose.orientation = yaw_to_quaternion(self.robot_yaw)
        self._pose_pub.publish(pose_msg)

    # ── Save-map service ─────────────────────────────────────────────────────

    def _srv_save_map(self, _request, response):
        try:
            with self._grid_lock:
                lo_copy  = self.log_odds.copy()
                origin_x = self.origin_x
                origin_y = self.origin_y

            msg = save_map_pgm_yaml(
                lo_copy, self.res, origin_x, origin_y, self.save_path)
            self.get_logger().info(msg)
            response.success = True
            response.message = msg
        except Exception as exc:  # noqa: BLE001
            response.success = False
            response.message = str(exc)
        return response


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
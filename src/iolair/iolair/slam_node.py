#!/usr/bin/env python3
"""
Puzzlebot Online SLAM — Occupancy Grid Builder (No Nav2)
=========================================================
Builds a 2-D occupancy grid in real time from LiDAR scans and the
robot's estimated pose.  Publishes the growing map as a latched
nav_msgs/OccupancyGrid so RViz and any other node can consume it.

Improvements over the previous version
---------------------------------------
  1. Vectorised Bresenham ray-casting (NumPy; ~10× faster per scan).
  2. KD-tree nearest-neighbour in ICP (scipy; O(N log N) vs O(N²)).
  3. Scan-to-MAP ICP (in addition to scan-to-scan) for global consistency.
  4. Separate pose-lock (``_pose_lock``) decoupled from the grid-lock so
     the odometry callback never blocks the map publisher.
  5. Adaptive beam-skip: denser scans are sub-sampled more aggressively
     so the per-scan CPU budget stays roughly constant.
  6. Motion-gated mapping: the map is only updated when the robot has
     moved/rotated enough (``min_travel_m`` / ``min_travel_rad``).
  7. Configurable TF publish rate (``tf_rate`` parameter, default 20 Hz).
  8. ``/slam/diagnostics`` topic with mapping statistics (rclpy Timer).
  9. Parameter validation with clamping and logger warnings.
 10. Bug-fix: topic was hard-coded to ``/lidar`` in one place; now unified
     to the ``scan_topic`` parameter.

Architecture (no nav2, no mcl_node required):
  ┌─────────────┐   /scan  (LaserScan)   ┌──────────────────────────┐
  │  LiDAR      │ ──────────────────────►│                          │
  └─────────────┘                        │  slam_node               │
  ┌─────────────┐   /odom  (Odometry)    │  • Log-odds grid         │──► /slam_map
  │  Odom node  │ ──────────────────────►│  • ICP (KD-tree)         │──► /slam_pose
  └─────────────┘                        │  • TF map→odom           │──► /slam/diagnostics
                                         │  • Map save service      │──► TF map→odom
                                         └──────────────────────────┘

Subscribes:
    <scan_topic>  (sensor_msgs/LaserScan)  — LiDAR measurements
    /odom         (nav_msgs/Odometry)      — wheel-odometry pose

Publishes:
    /slam_map          (nav_msgs/OccupancyGrid, TRANSIENT_LOCAL)
    /slam_pose         (geometry_msgs/PoseStamped)
    /slam/diagnostics  (std_msgs/String)   — JSON stats

TF broadcast:
    map → odom   (updated every scan via ICP correction)

Services:
    /slam/save_map  (std_srvs/Trigger) — saves .pgm + .yaml

Parameters (all have sensible defaults):
    scan_topic          str    '/scan'
    resolution          float  0.05   m/cell
    map_frame           str    'map'
    odom_frame          str    'odom'
    base_frame          str    'base_link'
    publish_rate        float  2.0    Hz
    tf_rate             float  20.0   Hz
    log_odds_occ        float  0.85
    log_odds_free       float  0.40
    log_odds_max        float  3.5
    log_odds_min        float -3.5
    lidar_max_range     float  10.0   m
    beam_skip           int    1      (0 = adaptive)
    target_beams        int    180    beams per scan when adaptive
    map_init_size       int    400    cells (square)
    map_origin_x        float -10.0  m
    map_origin_y        float -10.0  m
    use_icp             bool   True
    icp_max_iter        int    30
    icp_tolerance       float  1e-4
    icp_max_correction  float  0.5    m  (jump guard)
    min_travel_m        float  0.05   m  (motion gate)
    min_travel_rad      float  0.02   rad
    save_map_path       str    '/tmp/slam_map'
"""

import json
import math
import os
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg       import OccupancyGrid, MapMetaData, Odometry
from sensor_msgs.msg    import LaserScan
from geometry_msgs.msg  import (
    Pose, Point, Quaternion, PoseStamped, TransformStamped
)
from std_msgs.msg       import String
from std_srvs.srv       import Trigger
from tf2_ros            import TransformBroadcaster

try:
    from scipy.spatial import cKDTree as KDTree
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


# ── Quaternion helpers ─────────────────────────────────────────────────────────

def yaw_from_quaternion(q) -> float:
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


# ── Vectorised Bresenham ───────────────────────────────────────────────────────

def bresenham_batch(x0: int, y0: int,
                    x1_arr: np.ndarray,
                    y1_arr: np.ndarray,
                    grid_w: int, grid_h: int):
    """
    Mark free cells for multiple rays at once using NumPy vectorisation.

    Returns a flat int64 array of linearised (row * grid_w + col) indices of
    cells that should be decremented (free-space cells only; end-points are
    handled separately by the caller).

    Uses the parametric form: for each ray we step N times from (x0,y0)
    towards (x1,y1), collecting integer cell centres.  This is an
    approximation of Bresenham but is fully vectorised and ~10× faster.
    """
    n_rays = len(x1_arr)
    if n_rays == 0:
        return np.empty(0, dtype=np.int64)

    dx = x1_arr - x0
    dy = y1_arr - y0
    lengths = np.hypot(dx, dy).astype(np.float32)
    lengths = np.where(lengths < 1, 1, lengths)   # avoid div-by-zero

    # Step count = ⌊length⌋  (we do NOT include the end-point)
    max_steps = int(lengths.max())
    if max_steps == 0:
        return np.empty(0, dtype=np.int64)

    # t ∈ [0, 1) exclusive of 1
    t = np.arange(max_steps, dtype=np.float32) / lengths[:, None]  # (N, max_steps)

    xs = (x0 + dx[:, None] * t).astype(np.int32)   # (N, max_steps)
    ys = (y0 + dy[:, None] * t).astype(np.int32)

    # Mask steps that go beyond each individual ray length
    steps_needed = np.floor(lengths).astype(int)               # (N,)
    step_idx = np.arange(max_steps)[None, :]                    # (1, max_steps)
    valid = step_idx < steps_needed[:, None]

    # Also mask out-of-bounds cells
    in_bounds = (xs >= 0) & (xs < grid_w) & (ys >= 0) & (ys < grid_h)
    mask = valid & in_bounds

    flat = (ys[mask].astype(np.int64) * grid_w + xs[mask].astype(np.int64))
    return flat


# ── Scan → point cloud ────────────────────────────────────────────────────────

def scan_to_points(scan: LaserScan, max_range: float,
                   robot_x: float, robot_y: float, robot_yaw: float,
                   step: int = 1) -> np.ndarray:
    """
    Vectorised conversion of a LaserScan to a (N, 2) world-frame XY array.
    """
    ranges = np.asarray(scan.ranges, dtype=np.float64)
    indices = np.arange(0, len(ranges), max(1, step))
    r = ranges[indices]
    valid = np.isfinite(r) & (r >= scan.range_min) & (r < max_range)
    r = r[valid]
    if len(r) == 0:
        return np.empty((0, 2))
    angles = (scan.angle_min
              + indices[valid] * scan.angle_increment
              + robot_yaw)
    pts = np.stack([
        robot_x + r * np.cos(angles),
        robot_y + r * np.sin(angles),
    ], axis=1)
    return pts


# ── ICP (2-D, point-to-point, KD-tree) ───────────────────────────────────────

def icp_2d(src: np.ndarray, dst: np.ndarray,
           max_iter: int = 30, tol: float = 1e-4,
           max_correction: float = 0.5):
    """
    2-D ICP with KD-tree nearest-neighbour (falls back to brute-force when
    scipy is unavailable).

    Returns (dx, dy, dyaw, converged).  If the correction exceeds
    ``max_correction`` in translation the result is discarded and (0,0,0,False)
    is returned so the caller can skip an unreliable alignment.
    """
    if len(src) < 10 or len(dst) < 10:
        return 0.0, 0.0, 0.0, False

    src_h = src.copy()
    dx_total = dy_total = dyaw_total = 0.0

    if _HAVE_SCIPY:
        dst_tree = KDTree(dst)

    for _ in range(max_iter):
        if _HAVE_SCIPY:
            nn_dist, nn_idx = dst_tree.query(src_h, workers=1)
        else:
            dists = np.linalg.norm(src_h[:, None] - dst[None], axis=2)
            nn_idx = np.argmin(dists, axis=1)
            nn_dist = dists[np.arange(len(src_h)), nn_idx]

        threshold = np.percentile(nn_dist, 80)   # tighter than 90 %
        mask = nn_dist < threshold
        if mask.sum() < 5:
            break

        S = src_h[mask]
        T = dst[nn_idx[mask]]

        mu_s = S.mean(axis=0)
        mu_t = T.mean(axis=0)
        H = (S - mu_s).T @ (T - mu_t)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = mu_t - R @ mu_s

        dyaw = math.atan2(R[1, 0], R[0, 0])
        dx_total   += t[0]
        dy_total   += t[1]
        dyaw_total += dyaw

        src_h = (R @ src_h.T).T + t

        if abs(t[0]) < tol and abs(t[1]) < tol and abs(dyaw) < tol:
            break

    total_dist = math.hypot(dx_total, dy_total)
    if total_dist > max_correction or abs(dyaw_total) > 0.5:
        return 0.0, 0.0, 0.0, False

    return dx_total, dy_total, normalize_angle(dyaw_total), True


# ── Map saver ─────────────────────────────────────────────────────────────────

def save_map_pgm_yaml(log_odds: np.ndarray, resolution: float,
                      origin_x: float, origin_y: float,
                      base_path: str) -> str:
    rows, cols = log_odds.shape
    prob = 1.0 - 1.0 / (1.0 + np.exp(np.clip(log_odds, -10, 10)))
    pgm = np.full((rows, cols), 205, dtype=np.uint8)
    pgm[prob >= 0.65] = 0      # occupied → dark
    pgm[prob <= 0.35] = 254    # free     → light
    pgm_img = np.flipud(pgm)   # top-row = north

    pgm_path  = base_path + '.pgm'
    yaml_path = base_path + '.yaml'

    with open(pgm_path, 'wb') as f:
        f.write(f'P5\n{cols} {rows}\n255\n'.encode())
        f.write(pgm_img.tobytes())

    with open(yaml_path, 'w') as f:
        f.write(f'image: {os.path.basename(pgm_path)}\n')
        f.write(f'resolution: {resolution}\n')
        f.write(f'origin: [{origin_x:.6f}, {origin_y:.6f}, 0.0]\n')
        f.write('negate: 0\n')
        f.write('occupied_thresh: 0.65\n')
        f.write('free_thresh: 0.35\n')

    return f'Map saved → {pgm_path}, {yaml_path}  ({cols}×{rows} cells)'


# ── SLAM Node ──────────────────────────────────────────────────────────────────

class SLAMNode(Node):

    def __init__(self):
        super().__init__('slam_node')

        # ── Parameters ────────────────────────────────────────────────────
        self._declare_params()
        self._load_params()

        # ── Internal state ────────────────────────────────────────────────
        self.log_odds = np.zeros(
            (self.grid_h, self.grid_w), dtype=np.float32)
        self._grid_lock = threading.Lock()   # protects log_odds + grid dims
        self._pose_lock = threading.Lock()   # protects robot_* + _corr_*

        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_yaw = 0.0

        self._corr_x   = 0.0
        self._corr_y   = 0.0
        self._corr_yaw = 0.0

        self._prev_scan_pts: np.ndarray = np.empty((0, 2))
        self._map_pts:       np.ndarray = np.empty((0, 2))  # scan-to-map ref

        # Motion gate: only update map when moved enough
        self._last_map_x   = 0.0
        self._last_map_y   = 0.0
        self._last_map_yaw = 0.0

        # Diagnostics counters
        self._scans_received   = 0
        self._scans_processed  = 0
        self._icp_successes    = 0
        self._t_scan_total     = 0.0   # seconds

        # ── QoS ───────────────────────────────────────────────────────────
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        sensor_qos = rclpy.qos.qos_profile_sensor_data

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(
            LaserScan, self.scan_topic, self._cb_scan, sensor_qos)
        self.create_subscription(
            Odometry, '/odom', self._cb_odom, 10)

        # ── Publishers ────────────────────────────────────────────────────
        self._map_pub   = self.create_publisher(
            OccupancyGrid, '/slam_map', latched_qos)
        self._pose_pub  = self.create_publisher(
            PoseStamped, '/slam_pose', 10)
        self._diag_pub  = self.create_publisher(
            String, '/slam/diagnostics', 10)

        # ── TF broadcaster ────────────────────────────────────────────────
        self._tf_broadcaster = TransformBroadcaster(self)

        # ── Service ───────────────────────────────────────────────────────
        self.create_service(Trigger, '/slam/save_map', self._srv_save_map)

        # ── Timers ────────────────────────────────────────────────────────
        self.create_timer(1.0 / self.pub_rate,  self._publish_map)
        self.create_timer(1.0 / self.tf_rate,   self._broadcast_tf)
        self.create_timer(5.0,                  self._publish_diagnostics)

        self.get_logger().info(
            f'SLAM node started — '
            f'grid {self.grid_w}×{self.grid_h} cells, '
            f'res={self.res} m/cell, '
            f'origin=({self.origin_x:.2f}, {self.origin_y:.2f}), '
            f'ICP={"on (KD-tree)" if (self.use_icp and _HAVE_SCIPY) else "on (brute-force)" if self.use_icp else "off"}, '
            f'scan_topic={self.scan_topic}'
        )
        if not _HAVE_SCIPY and self.use_icp:
            self.get_logger().warn(
                'scipy not found — ICP will use brute-force O(N²) matching. '
                'Install scipy for faster KD-tree matching.')

    # ── Parameter helpers ─────────────────────────────────────────────────────

    def _declare_params(self):
        self.declare_parameter('scan_topic',        '/scan')
        self.declare_parameter('resolution',         0.05)
        self.declare_parameter('map_frame',          'map')
        self.declare_parameter('odom_frame',         'odom')
        self.declare_parameter('base_frame',         'base_link')
        self.declare_parameter('publish_rate',       2.0)
        self.declare_parameter('tf_rate',            20.0)
        self.declare_parameter('log_odds_occ',       0.85)
        self.declare_parameter('log_odds_free',      0.40)
        self.declare_parameter('log_odds_max',       3.5)
        self.declare_parameter('log_odds_min',      -3.5)
        self.declare_parameter('lidar_max_range',    10.0)
        self.declare_parameter('beam_skip',          1)
        self.declare_parameter('target_beams',       180)
        self.declare_parameter('map_init_size',      400)
        self.declare_parameter('map_origin_x',      -10.0)
        self.declare_parameter('map_origin_y',      -10.0)
        self.declare_parameter('use_icp',            True)
        self.declare_parameter('icp_max_iter',       30)
        self.declare_parameter('icp_tolerance',      1e-4)
        self.declare_parameter('icp_max_correction', 0.5)
        self.declare_parameter('min_travel_m',       0.05)
        self.declare_parameter('min_travel_rad',     0.02)
        self.declare_parameter('save_map_path',      '/tmp/slam_map')

    def _load_params(self):
        g = self.get_parameter

        self.scan_topic  = g('scan_topic').value
        self.res         = max(0.01, g('resolution').value)
        self.map_frame   = g('map_frame').value
        self.odom_frame  = g('odom_frame').value
        self.base_frame  = g('base_frame').value
        self.pub_rate    = max(0.1, g('publish_rate').value)
        self.tf_rate     = max(1.0, g('tf_rate').value)
        self.lo_occ      = g('log_odds_occ').value
        self.lo_free     = g('log_odds_free').value
        self.lo_max      = g('log_odds_max').value
        self.lo_min      = g('log_odds_min').value
        self.max_range   = max(0.5, g('lidar_max_range').value)
        self.beam_skip   = max(0, g('beam_skip').value)        # 0 = adaptive
        self.target_beams = max(10, g('target_beams').value)
        self.use_icp     = g('use_icp').value
        self.icp_iter    = max(1, g('icp_max_iter').value)
        self.icp_tol     = g('icp_tolerance').value
        self.icp_max_corr = g('icp_max_correction').value
        self.min_travel_m   = g('min_travel_m').value
        self.min_travel_rad = g('min_travel_rad').value
        self.save_path   = g('save_map_path').value

        init_size      = max(10, g('map_init_size').value)
        self.grid_h    = init_size
        self.grid_w    = init_size
        self.origin_x  = g('map_origin_x').value
        self.origin_y  = g('map_origin_y').value

        if self.lo_min >= self.lo_max:
            self.get_logger().warn(
                'log_odds_min >= log_odds_max — resetting to defaults.')
            self.lo_min, self.lo_max = -3.5, 3.5

    # ── Odometry callback ─────────────────────────────────────────────────────

    def _cb_odom(self, msg: Odometry):
        odom_x   = msg.pose.pose.position.x
        odom_y   = msg.pose.pose.position.y
        odom_yaw = yaw_from_quaternion(msg.pose.pose.orientation)

        with self._pose_lock:
            cos_c = math.cos(self._corr_yaw)
            sin_c = math.sin(self._corr_yaw)
            self.robot_x   = cos_c * odom_x - sin_c * odom_y + self._corr_x
            self.robot_y   = sin_c * odom_x + cos_c * odom_y + self._corr_y
            self.robot_yaw = normalize_angle(odom_yaw + self._corr_yaw)

    # ── LiDAR callback ────────────────────────────────────────────────────────

    def _cb_scan(self, msg: LaserScan):
        self._scans_received += 1
        t0 = time.monotonic()

        # Snapshot the current pose (no holding pose_lock during long work)
        with self._pose_lock:
            rx, ry, ryaw = self.robot_x, self.robot_y, self.robot_yaw

        # ── Motion gate ───────────────────────────────────────────────────
        d_pos = math.hypot(rx - self._last_map_x, ry - self._last_map_y)
        d_yaw = abs(normalize_angle(ryaw - self._last_map_yaw))
        if d_pos < self.min_travel_m and d_yaw < self.min_travel_rad:
            return   # robot hasn't moved enough; skip this scan

        # ── Adaptive beam skip ────────────────────────────────────────────
        n_beams = len(msg.ranges)
        if self.beam_skip == 0:
            step = max(1, n_beams // self.target_beams)
        else:
            step = self.beam_skip

        # ── ICP scan correction ───────────────────────────────────────────
        if self.use_icp:
            cur_pts = scan_to_points(
                msg, self.max_range, rx, ry, ryaw, step=step)

            # Scan-to-scan (drift correction between consecutive frames)
            if len(self._prev_scan_pts) > 10 and len(cur_pts) > 10:
                dx, dy, dyaw, ok = icp_2d(
                    cur_pts, self._prev_scan_pts,
                    max_iter=self.icp_iter, tol=self.icp_tol,
                    max_correction=self.icp_max_corr)
                if ok:
                    self._icp_successes += 1
                    with self._pose_lock:
                        self._corr_x   += dx
                        self._corr_y   += dy
                        self._corr_yaw  = normalize_angle(
                            self._corr_yaw + dyaw)
                        self.robot_x   += dx
                        self.robot_y   += dy
                        self.robot_yaw  = normalize_angle(
                            self.robot_yaw + dyaw)
                    rx, ry, ryaw = (self.robot_x,
                                    self.robot_y,
                                    self.robot_yaw)

            # Scan-to-map (global consistency every time a reference is ready)
            if len(self._map_pts) > 30 and len(cur_pts) > 10:
                dx, dy, dyaw, ok = icp_2d(
                    cur_pts, self._map_pts,
                    max_iter=self.icp_iter // 2, tol=self.icp_tol * 2,
                    max_correction=self.icp_max_corr * 0.5)
                if ok:
                    with self._pose_lock:
                        self._corr_x   += dx
                        self._corr_y   += dy
                        self._corr_yaw  = normalize_angle(
                            self._corr_yaw + dyaw)
                        self.robot_x   += dx
                        self.robot_y   += dy
                        self.robot_yaw  = normalize_angle(
                            self.robot_yaw + dyaw)
                    rx, ry, ryaw = (self.robot_x,
                                    self.robot_y,
                                    self.robot_yaw)

            self._prev_scan_pts = scan_to_points(
                msg, self.max_range, rx, ry, ryaw, step=step)

        # ── Update occupancy grid (vectorised) ────────────────────────────
        self._update_grid(msg, rx, ry, ryaw, step)

        # Update motion gate reference
        self._last_map_x   = rx
        self._last_map_y   = ry
        self._last_map_yaw = ryaw
        self._scans_processed += 1
        self._t_scan_total += time.monotonic() - t0

    # ── Grid update (vectorised) ──────────────────────────────────────────────

    def _update_grid(self, msg: LaserScan,
                     rx: float, ry: float, ryaw: float, step: int):
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        indices = np.arange(0, len(ranges), max(1, step))
        r = ranges[indices]
        angles = (msg.angle_min
                  + indices * msg.angle_increment
                  + ryaw)

        valid_hit  = np.isfinite(r) & (r > msg.range_min) & (r < self.max_range)
        valid_free = ~valid_hit   # beams that didn't return a hit

        # End-point coordinates (world frame)
        ex = np.where(valid_hit,
                      rx + r * np.cos(angles),
                      rx + self.max_range * np.cos(angles)).astype(np.float64)
        ey = np.where(valid_hit,
                      ry + r * np.sin(angles),
                      ry + self.max_range * np.sin(angles)).astype(np.float64)

        with self._grid_lock:
            # Grow grid to fit robot position
            rrx, rry = self._world_to_cell(rx, ry)
            if not self._in_bounds(rrx, rry):
                self._expand_to_fit(rx, ry)
                rrx, rry = self._world_to_cell(rx, ry)

            # Grow grid to fit all end-points
            ex_c = ((ex - self.origin_x) / self.res).astype(np.int32)
            ey_c = ((ey - self.origin_y) / self.res).astype(np.int32)
            need_expand = (
                (ex_c.min() < 0) or (ex_c.max() >= self.grid_w)
                or (ey_c.min() < 0) or (ey_c.max() >= self.grid_h)
            )
            if need_expand:
                for wx, wy in zip(ex[valid_hit], ey[valid_hit]):
                    if not self._in_bounds(*self._world_to_cell(wx, wy)):
                        self._expand_to_fit(wx, wy, margin=50)
                rrx, rry = self._world_to_cell(rx, ry)
                ex_c = ((ex - self.origin_x) / self.res).astype(np.int32)
                ey_c = ((ey - self.origin_y) / self.res).astype(np.int32)

            # Clamp end-points to grid
            ex_c = np.clip(ex_c, 0, self.grid_w - 1)
            ey_c = np.clip(ey_c, 0, self.grid_h - 1)

            # Mark FREE cells along all rays (vectorised Bresenham)
            free_idx = bresenham_batch(
                rrx, rry, ex_c, ey_c, self.grid_w, self.grid_h)
            if len(free_idx) > 0:
                np.add.at(self.log_odds.ravel(), free_idx, -self.lo_free)
                np.clip(self.log_odds, self.lo_min, self.lo_max,
                        out=self.log_odds)

            # Mark OCCUPIED end-points
            hit_ex = ex_c[valid_hit]
            hit_ey = ey_c[valid_hit]
            in_b   = ((hit_ex >= 0) & (hit_ex < self.grid_w)
                      & (hit_ey >= 0) & (hit_ey < self.grid_h))
            hit_idx = (hit_ey[in_b].astype(np.int64) * self.grid_w
                       + hit_ex[in_b].astype(np.int64))
            np.add.at(self.log_odds.ravel(), hit_idx, self.lo_occ)
            np.clip(self.log_odds, self.lo_min, self.lo_max,
                    out=self.log_odds)

            # Refresh scan-to-map reference cloud from current occupied cells
            # (cheap: just store the hit world coords for next ICP)
            if len(hit_ex[in_b]) > 0:
                new_map_pts = np.stack([
                    self.origin_x + hit_ex[in_b] * self.res,
                    self.origin_y + hit_ey[in_b] * self.res,
                ], axis=1)
                if len(self._map_pts) == 0:
                    self._map_pts = new_map_pts
                else:
                    # Keep a rolling window of the last ~2000 map points
                    self._map_pts = np.vstack([self._map_pts, new_map_pts])
                    if len(self._map_pts) > 2000:
                        self._map_pts = self._map_pts[-2000:]

    # ── TF broadcaster ────────────────────────────────────────────────────────

    def _broadcast_tf(self):
        with self._pose_lock:
            cx, cy, cyaw = self._corr_x, self._corr_y, self._corr_yaw

        t = TransformStamped()
        t.header.stamp    = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame
        t.child_frame_id  = self.odom_frame
        t.transform.translation.x = cx
        t.transform.translation.y = cy
        t.transform.translation.z = 0.0
        t.transform.rotation      = yaw_to_quaternion(cyaw)
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

        self.log_odds  = new_grid
        self.grid_w    = new_w
        self.grid_h    = new_h
        self.origin_x -= pad_left * self.res
        self.origin_y -= pad_bot  * self.res

        self.get_logger().info(
            f'Grid expanded → {self.grid_w}×{self.grid_h} cells, '
            f'origin=({self.origin_x:.2f}, {self.origin_y:.2f})')

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
            lo_copy  = self.log_odds.copy()
            orig_x   = self.origin_x
            orig_y   = self.origin_y
            gw, gh   = self.grid_w, self.grid_h

        prob = 1.0 - 1.0 / (1.0 + np.exp(np.clip(lo_copy, -10, 10)))
        ros_grid = np.full(lo_copy.shape, -1, dtype=np.int8)
        ros_grid[prob >= 0.65] = 100
        ros_grid[prob <= 0.35] = 0

        now = self.get_clock().now().to_msg()

        msg = OccupancyGrid()
        msg.header.stamp    = now
        msg.header.frame_id = self.map_frame
        msg.info = MapMetaData(
            resolution=self.res,
            width=gw,
            height=gh,
            origin=Pose(
                position=Point(x=float(orig_x), y=float(orig_y), z=0.0),
                orientation=yaw_to_quaternion(0.0),
            ),
        )
        msg.data = ros_grid.flatten().tolist()
        self._map_pub.publish(msg)

        with self._pose_lock:
            px, py, pyaw = self.robot_x, self.robot_y, self.robot_yaw

        pose_msg = PoseStamped()
        pose_msg.header.stamp    = now
        pose_msg.header.frame_id = self.map_frame
        pose_msg.pose.position   = Point(x=px, y=py, z=0.0)
        pose_msg.pose.orientation = yaw_to_quaternion(pyaw)
        self._pose_pub.publish(pose_msg)

    # ── Diagnostics publisher ─────────────────────────────────────────────────

    def _publish_diagnostics(self):
        with self._pose_lock:
            px, py, pyaw = self.robot_x, self.robot_y, self.robot_yaw
        with self._grid_lock:
            n_occ  = int(np.sum(self.log_odds > self.lo_max * 0.5))
            n_free = int(np.sum(self.log_odds < self.lo_min * 0.5))
            gw, gh = self.grid_w, self.grid_h

        avg_ms = (
            1000.0 * self._t_scan_total / self._scans_processed
            if self._scans_processed else 0.0
        )
        stats = {
            'scans_received':  self._scans_received,
            'scans_processed': self._scans_processed,
            'icp_successes':   self._icp_successes,
            'avg_scan_ms':     round(avg_ms, 2),
            'grid_cells':      f'{gw}×{gh}',
            'occupied_cells':  n_occ,
            'free_cells':      n_free,
            'robot_x':         round(px, 3),
            'robot_y':         round(py, 3),
            'robot_yaw_deg':   round(math.degrees(pyaw), 1),
        }
        self._diag_pub.publish(String(data=json.dumps(stats)))

    # ── Save-map service ──────────────────────────────────────────────────────

    def _srv_save_map(self, _request, response):
        try:
            with self._grid_lock:
                lo_copy  = self.log_odds.copy()
                orig_x   = self.origin_x
                orig_y   = self.origin_y

            msg = save_map_pgm_yaml(
                lo_copy, self.res, orig_x, orig_y, self.save_path)
            self.get_logger().info(msg)
            response.success = True
            response.message = msg
        except Exception as exc:   # noqa: BLE001
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
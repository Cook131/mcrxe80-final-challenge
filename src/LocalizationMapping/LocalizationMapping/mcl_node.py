#!/usr/bin/env python3
"""
puzzlebotMCL.py — Monte Carlo Localization for the Puzzlebot
=============================================================

Architecture
------------
  EKF odometry (/odom)        → motion model  (velocity odometry, Thrun Alg. 5.3)
  LiDAR scan   (/scan)        → sensor model  (likelihood field, vectorised)
  nav2_map_server (/map)      → occupancy grid (TRANSIENT_LOCAL, started by launch)

Outputs
-------
  /mcl/pose  (PoseWithCovarianceStamped, map frame)  — publicado a pose_publish_hz
  TF: map → odom  at 20 Hz

Algorithm: SIR Particle Filter
  Predict  — full rot1/trans/rot2 decomposition + Gaussian noise
  Update   — likelihood-field sensor model (fully vectorised, O(N×B) numpy)
  Resample — low-variance (systematic) resampling
  Estimate — weighted circular mean

Performance notes
-----------------
  The sensor update is fully vectorised over particles AND beams using a
  precomputed EDT (Euclidean Distance Transform) of the occupancy map.
  The EDT is computed ONCE when /map arrives via scipy.ndimage.distance_transform_edt
  (fallback: cv2.distanceTransform).  Per-scan cost is a single (N,B) array
  lookup — no Python loops over particles or beams.

  Typical speedup vs. the old nested-loop _dist_to_obstacle: ~20-50×.

Pose publishing guarantee
-------------------------
  /mcl/pose se publica a pose_publish_hz Hz (default 20) sin importar si
  el robot se movió o no.  El timer usa siempre la última estimación
  disponible.  El threshold min_trans/min_rot solo controla si se corre
  el predict+update (actualiza la estimación); nunca bloquea el publish.

Integration (launch context)
-----------------------------
  nav2_map_server + lifecycle_manager → /map  (autostart, TRANSIENT_LOCAL)
  puzzlebotOdometry                   → /odom → MCL motion model
  puzzlebotMCL                        → /mcl/pose + map→odom TF
  puzzlebotController                 → consumes /mcl/pose
  TF tree: map ──(MCL)──> odom ──(odom node)──> base_link

TF note
-------
  The map→odom transform is computed from a snapshot of the odom pose taken
  at the same iteration that produced the MCL pose estimate.  The 20 Hz timer
  re-broadcasts the *last computed* transform — it does NOT read live odom
  state, so there is no race between the timer and _cb_odom.

Parameters (--ros-args -p name:=value)
--------------------------------------
  num_particles        300      trade quality vs CPU
  resample_interval    1        resample every N scans
  alpha1..4            0.05     odometry noise (Thrun Table 5.3)
  z_hit                0.80     Gaussian hit weight
  z_rand               0.15     uniform random weight
  z_max                0.05     max-range weight
  sigma_hit            0.2      std dev of beam hit [m]
  lidar_max_range      8.0      discard beams beyond this [m]
  beam_skip            6        use every Nth beam
  lidar_yaw_offset     0.0      static LiDAR mount correction [rad]
  min_trans            0.05     motion threshold for predict+update [m]
  min_rot              0.05     motion threshold for predict+update [rad]
  pose_publish_hz      20.0     pose publish rate (always-on timer)
  initial_x/y/yaw      0.0      initial pose mean
  initial_cov_x/y/yaw  0.5/0.5/0.2  initial spread (σ)
  map_frame            'map'
  odom_frame           'odom'
  base_frame           'base_link'

Dependencies
------------
  scipy  (pip install scipy)  — preferred for distance_transform_edt
  cv2    (opencv-python)      — fallback if scipy unavailable
"""

import math
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
)

from nav_msgs.msg      import Odometry, OccupancyGrid
from sensor_msgs.msg   import LaserScan
from geometry_msgs.msg import (
    PoseWithCovarianceStamped, TransformStamped, Quaternion
)
from tf2_ros import TransformBroadcaster

# ── EDT backend (scipy preferred, cv2 fallback) ────────────────────────────────

try:
    from scipy.ndimage import distance_transform_edt as _scipy_edt

    def _compute_edt(obstacle_mask: np.ndarray, res: float) -> np.ndarray:
        """Return per-cell distance (metres) to nearest obstacle. obstacle_mask is bool."""
        return (_scipy_edt(~obstacle_mask) * res).astype(np.float32)

    _EDT_BACKEND = 'scipy'

except ImportError:
    try:
        import cv2 as _cv2

        def _compute_edt(obstacle_mask: np.ndarray, res: float) -> np.ndarray:
            free_u8 = (~obstacle_mask).astype(np.uint8)
            dist_px = _cv2.distanceTransform(free_u8, _cv2.DIST_L2, 5)
            return (dist_px * res).astype(np.float32)

        _EDT_BACKEND = 'cv2'

    except ImportError:
        _compute_edt  = None
        _EDT_BACKEND  = 'none'


# ── helpers ────────────────────────────────────────────────────────────────────

def normalize_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def yaw_from_quaternion(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw * 0.5)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    return q


# ── occupancy map ──────────────────────────────────────────────────────────────

class OccupancyMap:
    """
    Wraps nav_msgs/OccupancyGrid.

    On every update() call the EDT (Euclidean Distance Transform) is recomputed
    so _update() can do O(1) per-beam distance lookups instead of an O(r²)
    neighbour search.

    dist_map[row, col] = distance in metres to the nearest obstacle cell.
    Cells that ARE obstacles have dist_map == 0.
    Unknown cells (value < 0) are treated as free for the EDT.
    """

    def __init__(self):
        self.grid      = None
        self.dist_map  = None   # (H, W) float32, metres to nearest obstacle
        self.width     = 0
        self.height    = 0
        self.res       = 0.05
        self.origin_x  = 0.0
        self.origin_y  = 0.0
        self._lock     = threading.Lock()

    def update(self, msg: OccupancyGrid):
        width    = msg.info.width
        height   = msg.info.height
        res      = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        grid     = np.array(msg.data, dtype=np.int8).reshape(height, width)

        # Obstacle mask: cells with value > 50 are occupied.
        # Unknown cells (value == -1) are treated as free.
        obstacle_mask = grid > 50

        if _compute_edt is not None:
            dist_map = _compute_edt(obstacle_mask, res)
        else:
            # No EDT backend available — fall back to a coarse approximation:
            # dist = 0 for obstacles, res for everything else.
            # This degrades sensor quality but keeps the node running.
            dist_map = np.where(obstacle_mask, 0.0, res).astype(np.float32)

        with self._lock:
            self.width    = width
            self.height   = height
            self.res      = res
            self.origin_x = origin_x
            self.origin_y = origin_y
            self.grid     = grid
            self.dist_map = dist_map

    def ready(self) -> bool:
        return self.dist_map is not None

    def ray_cast(self, ox: float, oy: float,
                 angle: float, max_range: float) -> float:
        """Retained for external callers / debugging. Not used by _update."""
        with self._lock:
            if self.grid is None:
                return max_range
            step = self.res * 0.7
            ca, sa = math.cos(angle), math.sin(angle)
            r = 0.0
            while r < max_range:
                r  += step
                col = int((ox + r * ca - self.origin_x) / self.res)
                row = int((oy + r * sa - self.origin_y) / self.res)
                if not (0 <= col < self.width and 0 <= row < self.height):
                    return r
                if self.grid[row, col] > 50:
                    return r
        return max_range


# ── MCL Node ───────────────────────────────────────────────────────────────────

class PuzzlebotMCL(Node):

    def __init__(self):
        super().__init__('puzzlebot_mcl')

        # ── parameters ────────────────────────────────────────────────────
        self.declare_parameter('num_particles',       300)
        self.declare_parameter('resample_interval',   1)
        self.declare_parameter('pose_publish_hz',     20.0)

        self.declare_parameter('alpha1', 0.05)
        self.declare_parameter('alpha2', 0.05)
        self.declare_parameter('alpha3', 0.05)
        self.declare_parameter('alpha4', 0.05)

        self.declare_parameter('z_hit',           0.80)
        self.declare_parameter('z_rand',          0.15)
        self.declare_parameter('z_max',           0.05)
        self.declare_parameter('sigma_hit',       0.2)
        self.declare_parameter('lidar_max_range', 8.0)
        self.declare_parameter('beam_skip',       6)
        self.declare_parameter('lidar_yaw_offset', 0.0)

        self.declare_parameter('min_trans', 0.05)
        self.declare_parameter('min_rot',   0.05)

        self.declare_parameter('initial_x',       0.0)
        self.declare_parameter('initial_y',       0.0)
        self.declare_parameter('initial_yaw',     0.0)
        self.declare_parameter('initial_cov_x',   0.5)
        self.declare_parameter('initial_cov_y',   0.5)
        self.declare_parameter('initial_cov_yaw', 0.2)

        self.declare_parameter('map_frame',  'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')

        # ── load params ───────────────────────────────────────────────────
        N  = self.get_parameter('num_particles').value
        self._resample_interval = self.get_parameter('resample_interval').value
        pose_hz = self.get_parameter('pose_publish_hz').value

        self._alpha = np.array([
            self.get_parameter('alpha1').value,
            self.get_parameter('alpha2').value,
            self.get_parameter('alpha3').value,
            self.get_parameter('alpha4').value,
        ])

        self._z_hit        = self.get_parameter('z_hit').value
        self._z_rand       = self.get_parameter('z_rand').value
        self._z_max        = self.get_parameter('z_max').value
        self._sigma_hit    = self.get_parameter('sigma_hit').value
        self._max_range    = self.get_parameter('lidar_max_range').value
        self._beam_skip    = self.get_parameter('beam_skip').value
        self._yaw_offset   = self.get_parameter('lidar_yaw_offset').value
        self._min_trans    = self.get_parameter('min_trans').value
        self._min_rot      = self.get_parameter('min_rot').value

        self._map_frame    = self.get_parameter('map_frame').value
        self._odom_frame   = self.get_parameter('odom_frame').value
        self._base_frame   = self.get_parameter('base_frame').value

        ix   = self.get_parameter('initial_x').value
        iy   = self.get_parameter('initial_y').value
        iya  = self.get_parameter('initial_yaw').value
        sx   = self.get_parameter('initial_cov_x').value
        sy   = self.get_parameter('initial_cov_y').value
        sya  = self.get_parameter('initial_cov_yaw').value

        # ── particle array: (N, 3) ────────────────────────────────────────
        rng = np.random.default_rng()
        self._particles = np.column_stack([
            rng.normal(ix,  sx,  N),
            rng.normal(iy,  sy,  N),
            rng.normal(iya, sya, N),
        ])
        self._weights = np.ones(N) / N
        self._N       = N
        self._rng     = rng

        # ── best-estimate pose (map frame) ────────────────────────────────
        self._pose_x   = ix
        self._pose_y   = iy
        self._pose_yaw = iya

        # ── pose publish state ────────────────────────────────────────────
        # Lock protects _pose_* + _odom_snap_for_pub together so the
        # publish timer always gets a consistent pair.
        self._pose_lock         = threading.Lock()
        self._odom_snap_for_pub = (0.0, 0.0, iya)

        # ── cached map→odom TF ────────────────────────────────────────────
        self._tf_lock  = threading.Lock()
        self._tf_tx    = 0.0
        self._tf_ty    = 0.0
        self._tf_yaw   = 0.0

        # ── odometry state ────────────────────────────────────────────────
        self._prev_x    = None
        self._prev_y    = None
        self._prev_yaw  = None

        self._accum_trans = 0.0
        self._accum_rot   = 0.0

        self._delta_dx   = 0.0
        self._delta_dy   = 0.0
        self._delta_dyaw = 0.0

        # ── map ───────────────────────────────────────────────────────────
        self._occ_map = OccupancyMap()

        # ── scan counter ──────────────────────────────────────────────────
        self._scan_count = 0

        # ── locks ─────────────────────────────────────────────────────────
        self._odom_lock = threading.Lock()
        self._scan_lock = threading.Lock()

        self._latest_scan: LaserScan | None = None

        # ── TF broadcaster ────────────────────────────────────────────────
        self._tf_broadcaster = TransformBroadcaster(self)

        # ── QoS ───────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── subscribers ───────────────────────────────────────────────────
        self.create_subscription(Odometry,      '/odom',        self._cb_odom,        10)
        self.create_subscription(LaserScan,     '/scan',        self._cb_scan,        sensor_qos)
        self.create_subscription(OccupancyGrid, '/map',         self._cb_map,         latched_qos)
        self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self._cb_initialpose, 10)

        # ── publishers ────────────────────────────────────────────────────
        self._pub_pose = self.create_publisher(
            PoseWithCovarianceStamped, '/mcl/pose', 10)

        # ── timers ────────────────────────────────────────────────────────
        # Pose timer — always-on, publishes last known estimate.
        self.create_timer(1.0 / pose_hz, self._timer_publish_pose)

        # TF timer at 20 Hz
        self.create_timer(0.05, self._broadcast_tf)

        self.get_logger().info(
            f'MCL ready — {N} particles | '
            f'EDT backend={_EDT_BACKEND} | '
            f'beam_skip={self._beam_skip} | '
            f'min_trans={self._min_trans} m | min_rot={self._min_rot} rad | '
            f'pose_hz={pose_hz}'
        )

        if _EDT_BACKEND == 'none':
            self.get_logger().warn(
                'Neither scipy nor cv2 found — EDT disabled. '
                'Install scipy (pip install scipy) for full performance.'
            )

    # ── callbacks ──────────────────────────────────────────────────────────────

    def _cb_map(self, msg: OccupancyGrid):
        self._occ_map.update(msg)
        self.get_logger().info(
            f'Map received: {msg.info.width}×{msg.info.height} cells, '
            f'res={msg.info.resolution} m/cell | '
            f'EDT computed (backend={_EDT_BACKEND})'
        )

    def _cb_odom(self, msg: Odometry):
        x   = msg.pose.pose.position.x
        y   = msg.pose.pose.position.y
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)

        with self._odom_lock:
            if self._prev_x is None:
                self._prev_x   = x
                self._prev_y   = y
                self._prev_yaw = yaw
                return

            dx   = x   - self._prev_x
            dy   = y   - self._prev_y
            dyaw = normalize_angle(yaw - self._prev_yaw)

            self._accum_trans  += math.hypot(dx, dy)
            self._accum_rot    += abs(dyaw)

            self._delta_dx   += dx
            self._delta_dy   += dy
            self._delta_dyaw  = normalize_angle(self._delta_dyaw + dyaw)

            self._prev_x   = x
            self._prev_y   = y
            self._prev_yaw = yaw

    def _cb_scan(self, msg: LaserScan):
        with self._scan_lock:
            self._latest_scan = msg
        self._mcl_update()

    def _cb_initialpose(self, msg: PoseWithCovarianceStamped):
        x   = msg.pose.pose.position.x
        y   = msg.pose.pose.position.y
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        cov = msg.pose.covariance

        sx  = math.sqrt(max(cov[0],  1e-6))
        sy  = math.sqrt(max(cov[7],  1e-6))
        sya = math.sqrt(max(cov[35], 1e-6))

        with self._pose_lock:
            self._particles[:, 0] = self._rng.normal(x,   sx,  self._N)
            self._particles[:, 1] = self._rng.normal(y,   sy,  self._N)
            self._particles[:, 2] = self._rng.normal(yaw, sya, self._N)
            self._weights[:]      = 1.0 / self._N
            self._pose_x   = x
            self._pose_y   = y
            self._pose_yaw = yaw

        self.get_logger().info(
            f'Particles re-initialized at ({x:.2f}, {y:.2f}, {math.degrees(yaw):.1f}°)'
        )

    # ── MCL core ───────────────────────────────────────────────────────────────

    def _mcl_update(self):
        """
        One full SIR iteration triggered by each incoming scan.

        Motion threshold controls whether predict+update runs, but the
        pose is ALWAYS published via the always-on timer — this method
        only updates the estimate when motion is sufficient.
        """
        # ── 0. Motion threshold ───────────────────────────────────────────
        with self._odom_lock:
            motion_ok = (
                self._accum_trans >= self._min_trans or
                self._accum_rot   >= self._min_rot   or
                self._scan_count  == 0
            )

            if not motion_ok:
                dx = dy = dyaw = prev_yaw_snap = 0.0
                odom_snap = (
                    self._prev_x   if self._prev_x   is not None else 0.0,
                    self._prev_y   if self._prev_y   is not None else 0.0,
                    self._prev_yaw if self._prev_yaw is not None else 0.0,
                )
            else:
                dx            = self._delta_dx
                dy            = self._delta_dy
                dyaw          = self._delta_dyaw
                prev_yaw_snap = self._prev_yaw if self._prev_yaw is not None else 0.0
                odom_snap = (
                    self._prev_x   if self._prev_x   is not None else 0.0,
                    self._prev_y   if self._prev_y   is not None else 0.0,
                    self._prev_yaw if self._prev_yaw is not None else 0.0,
                )
                self._accum_trans = 0.0
                self._accum_rot   = 0.0
                self._delta_dx    = 0.0
                self._delta_dy    = 0.0
                self._delta_dyaw  = 0.0

        with self._scan_lock:
            scan = self._latest_scan
        if scan is None:
            return

        # ── 1. Predict ────────────────────────────────────────────────────
        if motion_ok:
            self._predict(dx, dy, dyaw, prev_yaw_snap)

        # ── 2. Update weights ─────────────────────────────────────────────
        if motion_ok:
            if self._occ_map.ready():
                log_w = self._update(scan)
                log_w -= log_w.max()
                w      = np.exp(log_w)
            else:
                w = self._weights.copy()

            total = w.sum()
            if total < 1e-300:
                self.get_logger().warn('Weight collapse — reinitializing weights.')
                w[:] = 1.0
            self._weights = w / w.sum()

        # ── 3. Estimate pose ──────────────────────────────────────────────
        if motion_ok:
            px   = float(np.dot(self._weights, self._particles[:, 0]))
            py   = float(np.dot(self._weights, self._particles[:, 1]))
            ss   = float(np.dot(self._weights, np.sin(self._particles[:, 2])))
            cs   = float(np.dot(self._weights, np.cos(self._particles[:, 2])))
            pyaw = math.atan2(ss, cs)

            with self._pose_lock:
                self._pose_x            = px
                self._pose_y            = py
                self._pose_yaw          = pyaw
                self._odom_snap_for_pub = odom_snap

        # ── 4. Resample ───────────────────────────────────────────────────
        self._scan_count += 1
        if motion_ok and self._scan_count % self._resample_interval == 0:
            self._resample()

        # ── 5. Update cached TF ───────────────────────────────────────────
        self._update_tf_cache(odom_snap)

    # ── Step 1: Predict ───────────────────────────────────────────────────────

    def _predict(self, dx: float, dy: float, dyaw: float, prev_yaw: float):
        a1, a2, a3, a4 = self._alpha
        N = self._N

        trans = math.hypot(dx, dy)
        rot1  = (math.atan2(dy, dx) - prev_yaw) if trans > 1e-4 else 0.0
        rot2  = normalize_angle(dyaw - rot1)

        rot1_hat  = rot1  - self._rng.normal(
            0.0, math.sqrt(a1 * rot1**2  + a2 * trans**2), N)
        trans_hat = trans - self._rng.normal(
            0.0, math.sqrt(a3 * trans**2 + a4 * (rot1**2 + rot2**2)), N)
        rot2_hat  = rot2  - self._rng.normal(
            0.0, math.sqrt(a1 * rot2**2  + a2 * trans**2), N)

        heading = self._particles[:, 2]
        self._particles[:, 0] += trans_hat * np.cos(heading + rot1_hat)
        self._particles[:, 1] += trans_hat * np.sin(heading + rot1_hat)
        self._particles[:, 2]  = np.arctan2(
            np.sin(heading + rot1_hat + rot2_hat),
            np.cos(heading + rot1_hat + rot2_hat),
        )

    # ── Step 2: Update (vectorised over particles AND beams) ──────────────────

    def _update(self, scan: LaserScan) -> np.ndarray:
        """
        Vectorised likelihood-field sensor update.

        Key idea: dist_map[row, col] already holds the distance (metres) to
        the nearest obstacle for every cell in the map.  We compute beam
        endpoints for ALL particles at once — shape (N, B) — then do a
        single array index into dist_map.  No Python loops, no neighbour
        search, no ray casting.

        Returns log_weights (N,).
        """
        omap = self._occ_map
        with omap._lock:
            if omap.dist_map is None:
                return np.zeros(self._N)
            dist_map  = omap.dist_map     # (H, W) float32
            origin_x  = omap.origin_x
            origin_y  = omap.origin_y
            res       = omap.res
            width     = omap.width
            height    = omap.height

        # ── beam selection ────────────────────────────────────────────────
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        angles = (scan.angle_min
                  + np.arange(len(ranges), dtype=np.float32) * scan.angle_increment
                  + self._yaw_offset)

        idx    = np.arange(0, len(ranges), self._beam_skip)
        ranges = ranges[idx]
        angles = angles[idx]

        valid  = np.isfinite(ranges) & (ranges > scan.range_min) & (ranges < self._max_range)
        ranges = ranges[valid]   # (B,)
        angles = angles[valid]   # (B,)
        B = len(ranges)

        if B == 0:
            return np.zeros(self._N)

        # ── beam endpoints: (N, B) ────────────────────────────────────────
        # Broadcast: particles (N,1) × beams (B,)
        px   = self._particles[:, 0, np.newaxis]   # (N, 1)
        py   = self._particles[:, 1, np.newaxis]   # (N, 1)
        pyaw = self._particles[:, 2, np.newaxis]   # (N, 1)

        bx = px + ranges * np.cos(pyaw + angles)   # (N, B)
        by = py + ranges * np.sin(pyaw + angles)   # (N, B)

        # ── map cell indices ──────────────────────────────────────────────
        cx = ((bx - origin_x) / res).astype(np.int32)   # (N, B)
        cy = ((by - origin_y) / res).astype(np.int32)   # (N, B)

        in_bounds = (cx >= 0) & (cx < width) & (cy >= 0) & (cy < height)  # (N, B)

        cx_c = np.clip(cx, 0, width  - 1)
        cy_c = np.clip(cy, 0, height - 1)

        # ── O(1) distance lookup ──────────────────────────────────────────
        dist = dist_map[cy_c, cx_c]                          # (N, B)
        dist = np.where(in_bounds, dist, self._max_range)    # out-of-bounds → max

        # ── likelihood field model ────────────────────────────────────────
        sigma  = self._sigma_hit
        norm_k = 1.0 / (sigma * math.sqrt(2.0 * math.pi))

        p_hit  = self._z_hit  * norm_k * np.exp(-0.5 * (dist / sigma) ** 2)  # (N, B)
        p_rand = self._z_rand / self._max_range                                # scalar
        p_mxr  = np.where(ranges >= self._max_range - 0.05, self._z_max, 0.0) # (B,)

        # Mix: in-bounds beams get full model; out-of-bounds get p_rand only
        p = np.where(in_bounds, p_hit + p_rand + p_mxr, p_rand)   # (N, B)
        p = np.clip(p, 1e-300, None)

        log_weights = np.sum(np.log(p), axis=1)   # (N,)
        return log_weights

    # ── Step 3: Resample ──────────────────────────────────────────────────────

    def _resample(self):
        N   = self._N
        w   = self._weights
        cum = np.cumsum(w)
        r   = self._rng.uniform(0.0, 1.0 / N)
        idx = np.searchsorted(cum, r + np.arange(N) / N)
        idx = np.clip(idx, 0, N - 1)
        self._particles = self._particles[idx].copy()
        self._weights   = np.ones(N) / N

    # ── TF cache update ────────────────────────────────────────────────────────

    def _update_tf_cache(self, odom_snap: tuple):
        """
        Recompute and cache map→odom from the current pose estimate
        and the given odom snapshot.
        Called at the end of every _mcl_update() regardless of motion.
        """
        with self._pose_lock:
            px   = self._pose_x
            py   = self._pose_y
            pyaw = self._pose_yaw

        ox, oy, oyaw = odom_snap
        diff = normalize_angle(pyaw - oyaw)
        cd, sd = math.cos(diff), math.sin(diff)
        tx = px - (ox * cd - oy * sd)
        ty = py - (ox * sd + oy * cd)

        with self._tf_lock:
            self._tf_tx  = tx
            self._tf_ty  = ty
            self._tf_yaw = diff

    # ── publish helpers ────────────────────────────────────────────────────────

    def _timer_publish_pose(self):
        """
        Always-on timer: publishes the last known pose estimate at
        pose_publish_hz regardless of robot motion or scan availability.
        """
        with self._pose_lock:
            px   = self._pose_x
            py   = self._pose_y
            pyaw = self._pose_yaw
            w    = self._weights.copy()
            pts  = self._particles.copy()

        stamp = self.get_clock().now().to_msg()
        self._publish_pose_msg(stamp, px, py, pyaw, w, pts)

    def _publish_pose_msg(self, stamp, px, py, pyaw, w, pts):
        """Build and publish PoseWithCovarianceStamped from given state."""
        dx = pts[:, 0] - px
        dy = pts[:, 1] - py
        da = np.array([normalize_angle(a - pyaw) for a in pts[:, 2]])

        cxx = float(np.dot(w, dx * dx))
        cyy = float(np.dot(w, dy * dy))
        caa = float(np.dot(w, da * da))
        cxy = float(np.dot(w, dx * dy))
        cxa = float(np.dot(w, dx * da))
        cya = float(np.dot(w, dy * da))

        msg = PoseWithCovarianceStamped()
        msg.header.stamp    = stamp
        msg.header.frame_id = self._map_frame
        msg.pose.pose.position.x  = px
        msg.pose.pose.position.y  = py
        msg.pose.pose.position.z  = 0.0
        msg.pose.pose.orientation = yaw_to_quaternion(pyaw)

        c = [0.0] * 36
        c[0],  c[1],  c[5]  = cxx, cxy, cxa
        c[6],  c[7],  c[11] = cxy, cyy, cya
        c[30], c[31], c[35] = cxa, cya, caa
        msg.pose.covariance = c

        self._pub_pose.publish(msg)

    def _broadcast_tf(self):
        """Broadcast map → odom TF at 20 Hz from cached values."""
        with self._tf_lock:
            tx   = self._tf_tx
            ty   = self._tf_ty
            tyaw = self._tf_yaw

        t = TransformStamped()
        t.header.stamp    = self.get_clock().now().to_msg()
        t.header.frame_id = self._map_frame
        t.child_frame_id  = self._odom_frame
        t.transform.translation.x = tx
        t.transform.translation.y = ty
        t.transform.translation.z = 0.0
        t.transform.rotation      = yaw_to_quaternion(tyaw)

        self._tf_broadcaster.sendTransform(t)


# ── entry point ────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = PuzzlebotMCL()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
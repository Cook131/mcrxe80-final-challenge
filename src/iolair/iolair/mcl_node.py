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
  /mcl/pose  (PoseWithCovarianceStamped, map frame)
  TF: map → odom  at 20 Hz

Algorithm: SIR Particle Filter
  Predict  — full rot1/trans/rot2 decomposition + Gaussian noise
  Update   — likelihood-field sensor model (vectorised over particles)
  Resample — low-variance (systematic) resampling
  Estimate — weighted circular mean

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
  min_trans            0.05     motion threshold [m]
  min_rot              0.05     motion threshold [rad]
  initial_x/y/yaw      0.0      initial pose mean
  initial_cov_x/y/yaw  0.5/0.5/0.2  initial spread (σ)
  map_frame            'map'
  odom_frame           'odom'
  base_frame           'base_link'
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


# ── occupancy map (kept from puzzlebotMCL for ray-cast fallback) ──────────────

class OccupancyMap:
    """
    Wraps nav_msgs/OccupancyGrid.
    Exposes the 2-D grid array directly for the likelihood-field update
    and provides ray_cast() as a sensor-model fallback when needed.
    """

    def __init__(self):
        self.grid      = None          # np.ndarray int8, shape (H, W)
        self.width     = 0
        self.height    = 0
        self.res       = 0.05
        self.origin_x  = 0.0
        self.origin_y  = 0.0
        self._lock     = threading.Lock()

    def update(self, msg: OccupancyGrid):
        with self._lock:
            self.width    = msg.info.width
            self.height   = msg.info.height
            self.res      = msg.info.resolution
            self.origin_x = msg.info.origin.position.x
            self.origin_y = msg.info.origin.position.y
            self.grid     = np.array(msg.data, dtype=np.int8).reshape(
                self.height, self.width)

    def ready(self) -> bool:
        return self.grid is not None

    def ray_cast(self, ox: float, oy: float,
                 angle: float, max_range: float) -> float:
        """Step along the ray until hitting an occupied cell or max_range."""
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

        # Odometry noise — Thrun, Probabilistic Robotics Table 5.3
        self.declare_parameter('alpha1', 0.05)   # rot  noise from rot
        self.declare_parameter('alpha2', 0.05)   # rot  noise from trans
        self.declare_parameter('alpha3', 0.05)   # trans noise from trans
        self.declare_parameter('alpha4', 0.05)   # trans noise from rot

        # Sensor model
        self.declare_parameter('z_hit',           0.80)
        self.declare_parameter('z_rand',          0.15)
        self.declare_parameter('z_max',           0.05)
        self.declare_parameter('sigma_hit',       0.2)
        self.declare_parameter('lidar_max_range', 8.0)
        self.declare_parameter('beam_skip',       6)
        self.declare_parameter('lidar_yaw_offset', 0.0)

        # Motion thresholds
        self.declare_parameter('min_trans', 0.05)
        self.declare_parameter('min_rot',   0.05)

        # Initial pose
        self.declare_parameter('initial_x',       0.0)
        self.declare_parameter('initial_y',       0.0)
        self.declare_parameter('initial_yaw',     0.0)
        self.declare_parameter('initial_cov_x',   0.5)
        self.declare_parameter('initial_cov_y',   0.5)
        self.declare_parameter('initial_cov_yaw', 0.2)

        # Frames
        self.declare_parameter('map_frame',  'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')

        # ── load params ───────────────────────────────────────────────────
        N  = self.get_parameter('num_particles').value
        self._resample_interval = self.get_parameter('resample_interval').value

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

        # ── particle array: (N, 3) — columns: x, y, yaw ──────────────────
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

        # ── cached map→odom transform (updated atomically with pose) ──────
        # Snapshot of the odom pose that was current when _pose_* was last
        # computed.  _broadcast_tf() reads only these — never live _prev_*.
        self._tf_lock  = threading.Lock()
        self._tf_tx    = 0.0   # translation x of map→odom
        self._tf_ty    = 0.0   # translation y of map→odom
        self._tf_yaw   = 0.0   # rotation yaw  of map→odom

        # ── odometry state (EKF /odom feed) ──────────────────────────────
        # prev_* : last odometry reading used to compute the delta
        # These are in the odom frame — used only to derive (dx, dy, dyaw).
        self._prev_x    = None
        self._prev_y    = None
        self._prev_yaw  = None

        # Accumulated motion since the last MCL update cycle
        self._accum_trans = 0.0
        self._accum_rot   = 0.0

        # Delta snapshot consumed by _predict(); set in _cb_odom and
        # reset in _mcl_update().
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

        # ── latest scan ───────────────────────────────────────────────────
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

        # ── TF timer: broadcast at 20 Hz so RViz stays smooth ─────────────
        self.create_timer(0.05, self._broadcast_tf)

        self.get_logger().info(
            f'MCL ready — {N} particles | '
            f'beam_skip={self._beam_skip} | '
            f'min_trans={self._min_trans} m | min_rot={self._min_rot} rad'
        )

    # ── callbacks ──────────────────────────────────────────────────────────────

    def _cb_map(self, msg: OccupancyGrid):
        self._occ_map.update(msg)
        self.get_logger().info(
            f'Map received: {msg.info.width}×{msg.info.height} cells, '
            f'res={msg.info.resolution} m/cell'
        )

    def _cb_odom(self, msg: Odometry):
        """
        Accumulate the odometry delta (from EKF /odom) between MCL cycles.
        Stores the raw (dx, dy, dyaw) increment in the odom frame so that
        _predict() can apply the full Thrun rot1/trans/rot2 decomposition.
        """
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

            # Accumulate for threshold check
            self._accum_trans  += math.hypot(dx, dy)
            self._accum_rot    += abs(dyaw)

            # Accumulate Cartesian delta for predict step
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
        """Re-scatter particles around a user-supplied pose (RViz 2D Pose Estimate)."""
        x   = msg.pose.pose.position.x
        y   = msg.pose.pose.position.y
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        cov = msg.pose.covariance

        sx  = math.sqrt(max(cov[0],  1e-6))
        sy  = math.sqrt(max(cov[7],  1e-6))
        sya = math.sqrt(max(cov[35], 1e-6))

        self._particles[:, 0] = self._rng.normal(x,   sx,  self._N)
        self._particles[:, 1] = self._rng.normal(y,   sy,  self._N)
        self._particles[:, 2] = self._rng.normal(yaw, sya, self._N)
        self._weights[:]      = 1.0 / self._N

        self.get_logger().info(
            f'Particles re-initialized at ({x:.2f}, {y:.2f}, {math.degrees(yaw):.1f}°)'
        )

    # ── MCL core ───────────────────────────────────────────────────────────────

    def _mcl_update(self):
        """One full SIR iteration: predict → update → estimate → resample."""

        # ── 0. Motion threshold guard ─────────────────────────────────────
        with self._odom_lock:
            if (self._accum_trans < self._min_trans and
                    self._accum_rot   < self._min_rot and
                    self._scan_count  > 0):
                return

            # Snapshot and reset accumulators
            dx           = self._delta_dx
            dy           = self._delta_dy
            dyaw         = self._delta_dyaw
            prev_yaw_snap = self._prev_yaw if self._prev_yaw is not None else 0.0
            # Full odom pose snapshot for TF computation
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

        # ── 1. Predict (sample motion model) ─────────────────────────────
        self._predict(dx, dy, dyaw, prev_yaw_snap)

        # ── 2. Update weights (likelihood-field sensor model) ────────────
        if self._occ_map.ready():
            log_w = self._update(scan)
            log_w -= log_w.max()               # numerical stability
            w      = np.exp(log_w)
        else:
            # No map yet → keep uniform (pure odometry)
            w = self._weights.copy()

        total = w.sum()
        if total < 1e-300:
            self.get_logger().warn('Weight collapse — reinitializing weights.')
            w[:] = 1.0
        self._weights = w / w.sum()

        # ── 3. Estimate pose (weighted circular mean) ─────────────────────
        self._pose_x   = float(np.dot(self._weights, self._particles[:, 0]))
        self._pose_y   = float(np.dot(self._weights, self._particles[:, 1]))
        sin_s = float(np.dot(self._weights, np.sin(self._particles[:, 2])))
        cos_s = float(np.dot(self._weights, np.cos(self._particles[:, 2])))
        self._pose_yaw = math.atan2(sin_s, cos_s)

        # ── 4. Resample (systematic / low-variance) ───────────────────────
        self._scan_count += 1
        if self._scan_count % self._resample_interval == 0:
            self._resample()

        # ── 5. Publish ────────────────────────────────────────────────────
        self._publish_pose(scan.header.stamp, odom_snap)

    # ── Step 1: Predict ───────────────────────────────────────────────────────

    def _predict(self, dx: float, dy: float, dyaw: float, prev_yaw: float):
        """
        Velocity motion model — Thrun et al., Probabilistic Robotics Alg. 5.3.

        Decomposes the odometry increment (dx, dy, dyaw) into:
          rot1  — initial rotation to face the displacement direction
          trans — translation magnitude
          rot2  — residual rotation (dyaw - rot1)
        Then samples noisy versions for every particle simultaneously.
        """
        a1, a2, a3, a4 = self._alpha
        N = self._N

        trans = math.hypot(dx, dy)
        rot1  = (math.atan2(dy, dx) - prev_yaw) if trans > 1e-4 else 0.0
        rot2  = normalize_angle(dyaw - rot1)

        # Vectorised noise sampling
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

    # ── Step 2: Update ────────────────────────────────────────────────────────

    def _update(self, scan: LaserScan) -> np.ndarray:
        """
        Likelihood-field sensor model — vectorised over all particles.

        For each particle, projects beam endpoints into map coordinates and
        looks up the nearest-obstacle distance.  Returns an (N,) array of
        log-weights.
        """
        omap = self._occ_map
        with omap._lock:
            if omap.grid is None:
                return np.zeros(self._N)
            grid      = omap.grid
            origin_x  = omap.origin_x
            origin_y  = omap.origin_y
            res       = omap.res
            width     = omap.width
            height    = omap.height

        ranges = np.asarray(scan.ranges, dtype=np.float32)
        angles = (scan.angle_min
                  + np.arange(len(ranges), dtype=np.float32) * scan.angle_increment
                  + self._yaw_offset)

        # Sub-sample
        idx    = np.arange(0, len(ranges), self._beam_skip)
        ranges = ranges[idx]
        angles = angles[idx]

        # Valid beams
        valid  = np.isfinite(ranges) & (ranges > scan.range_min) & (ranges < self._max_range)
        ranges = ranges[valid]
        angles = angles[valid]

        if len(ranges) == 0:
            return np.zeros(self._N)

        sigma     = self._sigma_hit
        z_hit     = self._z_hit
        z_rand    = self._z_rand
        z_max     = self._z_max
        max_range = self._max_range
        norm_k    = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
        log_weights = np.zeros(self._N)

        for i, (px, py, pyaw) in enumerate(self._particles):
            # Beam endpoint coordinates in world frame
            bx = px + ranges * np.cos(pyaw + angles)
            by = py + ranges * np.sin(pyaw + angles)

            # Map cell indices
            cx = ((bx - origin_x) / res).astype(int)
            cy = ((by - origin_y) / res).astype(int)

            in_bounds = (
                (cx >= 0) & (cx < width) &
                (cy >= 0) & (cy < height)
            )
            cx = np.clip(cx, 0, width  - 1)
            cy = np.clip(cy, 0, height - 1)

            cell_vals = grid[cy, cx]

            # Nearest-obstacle distance (cell-based approx):
            #   cell ≥ 65 → occupied (d = 0)
            #   cell  < 0 → unknown  (d = max_range, low weight)
            #   otherwise → free     (brute-force small search)
            dist = np.where(
                cell_vals >= 65, 0.0,
                np.where(cell_vals < 0, max_range,
                         self._dist_to_obstacle(cx, cy, cell_vals,
                                                 grid, width, height, res))
            )

            p_hit  = z_hit  * norm_k * np.exp(-0.5 * (dist / sigma) ** 2)
            p_rand = z_rand / max_range
            p_mxr  = np.where(ranges >= max_range - 0.05, z_max, 0.0)
            p      = np.where(in_bounds, p_hit + p_rand + p_mxr, p_rand)
            p      = np.clip(p, 1e-300, None)

            log_weights[i] = np.sum(np.log(p))

        return log_weights

    @staticmethod
    def _dist_to_obstacle(cx, cy, cell_vals,
                           grid, width, height, res,
                           search_r: int = 5) -> np.ndarray:
        """
        Brute-force nearest-occupied-cell distance for free-space endpoints.
        Only called for cells that are neither occupied nor unknown.
        """
        dist = np.zeros(len(cx), dtype=np.float64)
        for k in range(len(cx)):
            if cell_vals[k] >= 65:
                continue
            best = float('inf')
            for dr in range(-search_r, search_r + 1):
                for dc in range(-search_r, search_r + 1):
                    r, c = cy[k] + dr, cx[k] + dc
                    if 0 <= r < height and 0 <= c < width:
                        if grid[r, c] >= 65:
                            d = math.sqrt(dr * dr + dc * dc) * res
                            if d < best:
                                best = d
            dist[k] = best if best != float('inf') else res * search_r
        return dist

    # ── Step 3: Resample ──────────────────────────────────────────────────────

    def _resample(self):
        """
        Low-variance (systematic) resampling — O(N), avoids degeneracy.
        Thrun et al., Probabilistic Robotics, Algorithm 4.4.
        """
        N   = self._N
        w   = self._weights
        cum = np.cumsum(w)
        r   = self._rng.uniform(0.0, 1.0 / N)
        idx = np.searchsorted(cum, r + np.arange(N) / N)
        idx = np.clip(idx, 0, N - 1)
        self._particles = self._particles[idx].copy()
        self._weights   = np.ones(N) / N

    # ── publish helpers ────────────────────────────────────────────────────────

    def _publish_pose(self, stamp, odom_snap: tuple):
        """
        Publish estimated pose with weighted covariance and cache the
        map→odom TF computed from the odom snapshot taken this cycle.

        odom_snap = (ox, oy, oyaw)  — odom pose at the time _pose_* was set.
        """
        w  = self._weights
        dx = self._particles[:, 0] - self._pose_x
        dy = self._particles[:, 1] - self._pose_y
        da = np.array([
            normalize_angle(a - self._pose_yaw)
            for a in self._particles[:, 2]
        ])

        cxx  = float(np.dot(w, dx * dx))
        cyy  = float(np.dot(w, dy * dy))
        caa  = float(np.dot(w, da * da))
        cxy  = float(np.dot(w, dx * dy))
        cxa  = float(np.dot(w, dx * da))
        cya  = float(np.dot(w, dy * da))

        msg = PoseWithCovarianceStamped()
        msg.header.stamp    = stamp
        msg.header.frame_id = self._map_frame
        msg.pose.pose.position.x  = self._pose_x
        msg.pose.pose.position.y  = self._pose_y
        msg.pose.pose.position.z  = 0.0
        msg.pose.pose.orientation = yaw_to_quaternion(self._pose_yaw)

        # 6×6 covariance (x, y, z, rx, ry, rz) — fill x, y, yaw slots
        c = [0.0] * 36
        c[0],  c[1],  c[5]  = cxx, cxy, cxa
        c[6],  c[7],  c[11] = cxy, cyy, cya
        c[30], c[31], c[35] = cxa, cya, caa
        msg.pose.covariance = c

        self._pub_pose.publish(msg)

        # ── compute and cache map→odom TF ─────────────────────────────────
        # T_map_odom = T_map_base * inv(T_odom_base)
        #   diff      = mcl_yaw − odom_yaw
        #   tx        = mcl_x − R(diff) * odom_xy  (x component)
        #   ty        = mcl_y − R(diff) * odom_xy  (y component)
        ox, oy, oyaw = odom_snap
        diff = normalize_angle(self._pose_yaw - oyaw)
        cd, sd = math.cos(diff), math.sin(diff)
        tx = self._pose_x - (ox * cd - oy * sd)
        ty = self._pose_y - (ox * sd + oy * cd)

        with self._tf_lock:
            self._tf_tx  = tx
            self._tf_ty  = ty
            self._tf_yaw = diff

    def _broadcast_tf(self):
        """
        Broadcast map → odom TF at 20 Hz.

        Re-sends the transform that was last computed by _publish_pose()
        using the odom snapshot from that MCL cycle.  Reads only the
        cached _tf_* values — never the live _prev_* odom state — so
        there is no race with _cb_odom.
        """
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
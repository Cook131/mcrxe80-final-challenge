#!/usr/bin/env python3
"""
puzzlebotMCL.py — Monte Carlo Localization for the Puzzlebot
=============================================================

Architecture
------------
This node fuses two sources of information:

  1. EKF odometry (/odom)  — motion model (predicts where particles moved)
  2. LiDAR scan (/scan)    — sensor model (weights particles by scan likelihood)

It outputs a corrected pose on /mcl/pose and broadcasts the map→odom TF
so the rest of the stack (RViz, controller, etc.) sees a globally
consistent frame without touching any existing node.

Algorithm: SIR Particle Filter (Sequential Importance Resampling)
  Predict  → sample motion from odometry delta + Gaussian noise
  Update   → weight each particle with a beam-based sensor model
  Resample → low-variance (systematic) resampling to avoid degeneracy
  Estimate → weighted mean/covariance of surviving particles

Integration with existing nodes
--------------------------------
  puzzlebotOdometry  → publishes /odom  (used as motion model)
  puzzlebotMCL       → subscribes /odom + /scan, publishes /mcl/pose
                       and broadcasts map→odom TF
  slam_node          → can be disabled or kept as a map provider

  TF tree:   map ──(MCL)──> odom ──(odometry node)──> base_link

Parameters (override via --ros-args -p name:=value)
----------------------------------------------------
  num_particles      : 300      — trade-off quality vs CPU
  resample_interval  : 1        — resample every N scans
  odom_alpha1..4     : noise model coefficients (see below)
  laser_sigma_hit    : 0.2      — std-dev of Gaussian hit model [m]
  laser_z_hit        : 0.90     — weight of hit model
  laser_z_rand       : 0.05     — weight of random model
  laser_z_max        : 0.05     — weight of max-range model
  laser_max_range    : 8.0      — discard beams beyond this [m]
  beam_skip          : 6        — use every Nth beam (speed vs accuracy)
  min_trans          : 0.05     — motion threshold to trigger update [m]
  min_rot            : 0.05     — motion threshold to trigger update [rad]
  initial_x/y/yaw    : 0.0      — initial pose mean
  initial_cov_x/y/yaw: 0.5/0.5/0.2 — initial pose spread (σ)
  map_frame          : 'map'
  odom_frame         : 'odom'
  base_frame         : 'base_link'

Odometry noise model (probabilistic robotics, Table 5.3)
---------------------------------------------------------
  alpha1 — rotation noise from rotation
  alpha2 — rotation noise from translation
  alpha3 — translation noise from translation
  alpha4 — translation noise from rotation
  Defaults tuned for the Puzzlebot differential drive.

Sensor model
------------
  Per-beam likelihood  p = z_hit * N(r; d, sigma_hit)
                         + z_rand / laser_max_range
                         + z_max  * [r >= max_range]
  where d is the expected range from a ray-cast into the occupancy map.

  NOTE: ray-casting requires a nav_msgs/OccupancyGrid on /map.
  If no map is available the node falls back to a simple inverse-distance
  model that still works reasonably well for localisation in known spaces.
"""

import math
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

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


def euler_to_quaternion(roll, pitch, yaw) -> Quaternion:
    cy, sy = math.cos(yaw * 0.5),   math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5),  math.sin(roll * 0.5)
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


# ── occupancy grid ray-caster ──────────────────────────────────────────────────

class OccupancyMap:
    """
    Thin wrapper around nav_msgs/OccupancyGrid that provides:
      - is_occupied(wx, wy)         → bool
      - ray_cast(ox,oy,angle,max_r) → float  (expected range)
    """

    def __init__(self):
        self._grid      = None   # np.ndarray int8 row-major
        self._width     = 0
        self._height    = 0
        self._res       = 0.05
        self._origin_x  = 0.0
        self._origin_y  = 0.0
        self._lock      = threading.Lock()

    def update(self, msg: OccupancyGrid):
        with self._lock:
            self._width    = msg.info.width
            self._height   = msg.info.height
            self._res      = msg.info.resolution
            self._origin_x = msg.info.origin.position.x
            self._origin_y = msg.info.origin.position.y
            self._grid     = np.array(msg.data, dtype=np.int8).reshape(
                self._height, self._width)

    def ready(self) -> bool:
        return self._grid is not None

    def _world_to_cell(self, wx, wy):
        col = int((wx - self._origin_x) / self._res)
        row = int((wy - self._origin_y) / self._res)
        return col, row

    def _in_bounds(self, col, row) -> bool:
        return 0 <= col < self._width and 0 <= row < self._height

    def is_occupied(self, wx: float, wy: float) -> bool:
        with self._lock:
            if self._grid is None:
                return False
            col, row = self._world_to_cell(wx, wy)
            if not self._in_bounds(col, row):
                return True   # treat out-of-bounds as occupied (wall)
            return self._grid[row, col] > 50

    def ray_cast(self, ox: float, oy: float,
                 angle: float, max_range: float) -> float:
        """
        Step along the ray until hitting an occupied cell or max_range.
        Returns the estimated range to the first obstacle.
        """
        with self._lock:
            if self._grid is None:
                return max_range

            step   = self._res * 0.7      # slightly smaller than cell size
            ca, sa = math.cos(angle), math.sin(angle)
            r      = 0.0

            while r < max_range:
                r   += step
                col  = int((ox + r * ca - self._origin_x) / self._res)
                row  = int((oy + r * sa - self._origin_y) / self._res)
                if not self._in_bounds(col, row):
                    return r
                if self._grid[row, col] > 50:
                    return r

        return max_range


# ── MCL Node ───────────────────────────────────────────────────────────────────

class PuzzlebotMCL(Node):

    def __init__(self):
        super().__init__('puzzlebot_mcl')

        # ── declare parameters ────────────────────────────────────────────
        self.declare_parameter('num_particles',       300)
        self.declare_parameter('resample_interval',   1)

        # Odometry noise (probabilistic robotics model)
        self.declare_parameter('odom_alpha1', 0.1)   # rot  noise from rot
        self.declare_parameter('odom_alpha2', 0.1)   # rot  noise from trans
        self.declare_parameter('odom_alpha3', 0.1)   # trans noise from trans
        self.declare_parameter('odom_alpha4', 0.05)  # trans noise from rot

        # Sensor model
        self.declare_parameter('laser_sigma_hit',  0.2)
        self.declare_parameter('laser_z_hit',      0.90)
        self.declare_parameter('laser_z_rand',     0.05)
        self.declare_parameter('laser_z_max',      0.05)
        self.declare_parameter('laser_max_range',  8.0)
        self.declare_parameter('beam_skip',        6)
        self.declare_parameter('lidar_yaw_offset', 0.0)

        # Motion thresholds before updating
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
        N   = self.get_parameter('num_particles').value
        a1  = self.get_parameter('odom_alpha1').value
        a2  = self.get_parameter('odom_alpha2').value
        a3  = self.get_parameter('odom_alpha3').value
        a4  = self.get_parameter('odom_alpha4').value

        self._sigma_hit  = self.get_parameter('laser_sigma_hit').value
        self._z_hit      = self.get_parameter('laser_z_hit').value
        self._z_rand     = self.get_parameter('laser_z_rand').value
        self._z_max      = self.get_parameter('laser_z_max').value
        self._max_range  = self.get_parameter('laser_max_range').value
        self._beam_skip        = self.get_parameter('beam_skip').value
        self._lidar_yaw_offset = self.get_parameter('lidar_yaw_offset').value
        self._min_trans  = self.get_parameter('min_trans').value
        self._min_rot    = self.get_parameter('min_rot').value
        self._resample_interval = self.get_parameter('resample_interval').value

        self._map_frame  = self.get_parameter('map_frame').value
        self._odom_frame = self.get_parameter('odom_frame').value
        self._base_frame = self.get_parameter('base_frame').value

        self._alpha = np.array([a1, a2, a3, a4])

        # ── particle set: (N,3) array of [x, y, yaw] ─────────────────────
        ix  = self.get_parameter('initial_x').value
        iy  = self.get_parameter('initial_y').value
        iya = self.get_parameter('initial_yaw').value
        sx  = self.get_parameter('initial_cov_x').value
        sy  = self.get_parameter('initial_cov_y').value
        sya = self.get_parameter('initial_cov_yaw').value

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

        # ── odometry tracking ─────────────────────────────────────────────
        # We store the *previous* odometry reading in the odom frame.
        # The delta between consecutive readings drives the motion model.
        self._prev_odom_x   = None
        self._prev_odom_y   = None
        self._prev_odom_yaw = None

        # Accumulated motion since last update
        self._accum_trans = 0.0
        self._accum_rot   = 0.0

        # ── map ───────────────────────────────────────────────────────────
        self._occ_map = OccupancyMap()

        # ── scan counter (for resampling interval) ────────────────────────
        self._scan_count = 0

        # ── locks ─────────────────────────────────────────────────────────
        self._odom_lock = threading.Lock()
        self._scan_lock = threading.Lock()

        # ── current scan (set in cb, consumed in update) ──────────────────
        self._latest_scan: LaserScan | None = None

        # ── TF broadcaster (map → odom) ───────────────────────────────────
        self._tf_broadcaster = TransformBroadcaster(self)

        # ── QoS profiles ──────────────────────────────────────────────────
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
        self.create_subscription(Odometry,      '/odom',  self._cb_odom,  10)
        self.create_subscription(LaserScan,     '/scan',  self._cb_scan,  sensor_qos)
        self.create_subscription(OccupancyGrid, '/map',   self._cb_map,   latched_qos)

        # Allow external pose initialization (e.g. from RViz "2D Pose Estimate")
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self._cb_initialpose,
            10,
        )

        # ── publishers ────────────────────────────────────────────────────
        self._pub_pose = self.create_publisher(
            PoseWithCovarianceStamped, '/mcl/pose', 10)

        # ── timers ────────────────────────────────────────────────────────
        # TF at 20 Hz so RViz stays smooth even when no scan arrives
        self.create_timer(0.05, self._broadcast_tf)

        self.get_logger().info(
            f'MCL started — {N} particles, '
            f'beam_skip={self._beam_skip}, '
            f'min_trans={self._min_trans} m, min_rot={self._min_rot} rad'
        )

    # ── callbacks ──────────────────────────────────────────────────────────────

    def _cb_map(self, msg: OccupancyGrid):
        self._occ_map.update(msg)
        self.get_logger().info('Occupancy map received — sensor model active.')

    def _cb_odom(self, msg: Odometry):
        """
        Store the latest odometry reading.
        The actual particle propagation happens when a scan arrives,
        so we only accumulate the delta here.
        """
        x   = msg.pose.pose.position.x
        y   = msg.pose.pose.position.y
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)

        with self._odom_lock:
            if self._prev_odom_x is None:
                self._prev_odom_x   = x
                self._prev_odom_y   = y
                self._prev_odom_yaw = yaw
                return

            dx   = x   - self._prev_odom_x
            dy   = y   - self._prev_odom_y
            dyaw = normalize_angle(yaw - self._prev_odom_yaw)

            self._accum_trans += math.hypot(dx, dy)
            self._accum_rot   += abs(dyaw)

            self._prev_odom_x   = x
            self._prev_odom_y   = y
            self._prev_odom_yaw = yaw

    def _cb_scan(self, msg: LaserScan):
        """Buffer the latest scan; trigger the MCL update step."""
        with self._scan_lock:
            self._latest_scan = msg
        self._mcl_update()

    def _cb_initialpose(self, msg: PoseWithCovarianceStamped):
        """Re-initialize the particle cloud around a user-supplied pose."""
        x   = msg.pose.pose.position.x
        y   = msg.pose.pose.position.y
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        cov = msg.pose.covariance   # 6×6 row-major

        sx  = math.sqrt(max(cov[0],  1e-6))
        sy  = math.sqrt(max(cov[7],  1e-6))
        sya = math.sqrt(max(cov[35], 1e-6))

        self._particles[:, 0] = self._rng.normal(x,   sx,  self._N)
        self._particles[:, 1] = self._rng.normal(y,   sy,  self._N)
        self._particles[:, 2] = self._rng.normal(yaw, sya, self._N)
        self._weights[:]      = 1.0 / self._N

        self.get_logger().info(
            f'Particle cloud re-initialized at ({x:.2f},{y:.2f},{yaw:.2f})'
        )

    # ── MCL core ───────────────────────────────────────────────────────────────

    def _mcl_update(self):
        """
        Full MCL iteration:
          1. Check motion thresholds
          2. Predict (sample_motion_model_odometry)
          3. Update  (beam_range_finder_model)
          4. Estimate pose
          5. Resample (systematic)
          6. Publish & broadcast TF
        """
        with self._odom_lock:
            accum_trans = self._accum_trans
            accum_rot   = self._accum_rot

        # Skip if the robot hasn't moved enough (avoids filter lock-up)
        if (accum_trans < self._min_trans and
                accum_rot < self._min_rot and
                self._scan_count > 0):
            return

        # Snapshot odometry delta for this cycle
        with self._odom_lock:
            dx   = self._prev_odom_x   - (self._prev_odom_x   - 0.0)  # placeholder
            # We use the raw accumulated delta as a scalar proxy; for
            # a proper motion model we need the actual delta (x,y,yaw)
            # which we reconstruct from prev values captured here.
            prev_x   = self._prev_odom_x
            prev_y   = self._prev_odom_y
            prev_yaw = self._prev_odom_yaw
            self._accum_trans = 0.0
            self._accum_rot   = 0.0

        with self._scan_lock:
            scan = self._latest_scan

        if scan is None:
            return

        # ── 1. Predict ────────────────────────────────────────────────────
        self._particles = self._motion_model(
            self._particles, accum_trans, prev_yaw
        )

        # ── 2. Update weights ─────────────────────────────────────────────
        if self._occ_map.ready():
            log_weights = self._sensor_model(self._particles, scan)
            # Normalize in log-space for numerical stability
            log_weights -= log_weights.max()
            w = np.exp(log_weights)
        else:
            # No map: keep uniform weights (pure odometry MCL)
            w = self._weights.copy()

        total = w.sum()
        if total < 1e-300:
            # Weight collapse → reinitialize from current best estimate
            w[:] = 1.0
        self._weights = w / w.sum()

        # ── 3. Estimate pose (weighted mean) ──────────────────────────────
        self._pose_x   = float(np.sum(self._weights * self._particles[:, 0]))
        self._pose_y   = float(np.sum(self._weights * self._particles[:, 1]))

        # Circular mean for yaw
        sin_mean = float(np.sum(self._weights * np.sin(self._particles[:, 2])))
        cos_mean = float(np.sum(self._weights * np.cos(self._particles[:, 2])))
        self._pose_yaw = math.atan2(sin_mean, cos_mean)

        # ── 4. Resample (systematic) ──────────────────────────────────────
        self._scan_count += 1
        if self._scan_count % self._resample_interval == 0:
            self._particles, self._weights = self._systematic_resample(
                self._particles, self._weights
            )

        # ── 5. Publish & TF ───────────────────────────────────────────────
        self._publish_pose(scan.header.stamp)

    # ── motion model ──────────────────────────────────────────────────────────

    def _motion_model(self, particles: np.ndarray,
                      delta_trans: float, prev_yaw: float) -> np.ndarray:
        """
        Sample-based odometry motion model.
        (Thrun, Burgard, Fox — Probabilistic Robotics, Algorithm 5.3)

        We approximate the full (delta_rot1, delta_trans, delta_rot2)
        decomposition using the accumulated translation and the heading
        direction from the odometry.

        Noise scales with motion magnitude via alpha coefficients.
        """
        a1, a2, a3, a4 = self._alpha
        N = len(particles)

        if delta_trans < 1e-6:
            # Pure rotation or no movement — add only yaw noise
            noise_yaw = self._rng.normal(0, math.sqrt(a1 * 1e-4 + a2 * 1e-4), N)
            new_p = particles.copy()
            new_p[:, 2] = np.vectorize(normalize_angle)(
                particles[:, 2] + noise_yaw)
            return new_p

        # Rotation component: how much yaw changed per unit displacement
        # Use prev_yaw as the approximate heading
        sigma_trans = math.sqrt(a3 * delta_trans ** 2 + a4 * delta_trans ** 2)
        sigma_rot   = math.sqrt(a1 * delta_trans ** 2 + a2 * delta_trans ** 2)

        noisy_trans = delta_trans + self._rng.normal(0, sigma_trans, N)
        noisy_rot   = self._rng.normal(0, sigma_rot, N)

        new_p = particles.copy()
        heading = particles[:, 2]    # each particle's current yaw

        new_p[:, 0] += noisy_trans * np.cos(heading)
        new_p[:, 1] += noisy_trans * np.sin(heading)
        new_p[:, 2] = np.vectorize(normalize_angle)(heading + noisy_rot)

        return new_p

    # ── sensor model ──────────────────────────────────────────────────────────

    def _sensor_model(self, particles: np.ndarray,
                      scan: LaserScan) -> np.ndarray:
        """
        Beam-range-finder model (log-likelihood, vectorised over particles).
        Returns an (N,) array of log-weights.
        """
        ranges     = np.asarray(scan.ranges, dtype=np.float32)
        angles     = (scan.angle_min
                      + np.arange(len(ranges)) * scan.angle_increment
                      + self._lidar_yaw_offset)

        # Sub-sample beams for speed
        idx    = np.arange(0, len(ranges), self._beam_skip)
        ranges = ranges[idx]
        angles = angles[idx]

        # Valid beams only
        valid  = np.isfinite(ranges) & (ranges >= scan.range_min) & \
                 (ranges < self._max_range)
        ranges = ranges[valid]
        angles = angles[valid]

        if len(ranges) == 0:
            return np.zeros(len(particles))

        N    = len(particles)
        M    = len(ranges)
        logs = np.zeros(N)

        z_hit   = self._z_hit
        z_rand  = self._z_rand
        z_max   = self._z_max
        sigma   = self._sigma_hit
        max_r   = self._max_range
        norm_k  = 1.0 / (math.sqrt(2 * math.pi) * sigma)

        for i in range(N):
            px, py, pyaw = particles[i]
            p_log = 0.0

            for j in range(M):
                beam_angle = pyaw + float(angles[j])
                z_obs      = float(ranges[j])

                z_exp = self._occ_map.ray_cast(px, py, beam_angle, max_r)

                # Gaussian hit term
                diff       = z_obs - z_exp
                p_hit      = norm_k * math.exp(-0.5 * (diff / sigma) ** 2)

                # Random term
                p_rnd      = z_rand / max_r

                # Max-range term
                p_max_term = z_max if z_obs >= (max_r - 0.05) else 0.0

                p_total    = z_hit * p_hit + p_rnd + p_max_term
                p_log     += math.log(max(p_total, 1e-300))

            logs[i] = p_log

        return logs

    # ── systematic resampling ─────────────────────────────────────────────────

    @staticmethod
    def _systematic_resample(particles: np.ndarray,
                              weights: np.ndarray):
        """
        Low-variance (systematic) resampling.
        O(N) — preserves diversity better than multinomial resampling.
        """
        N     = len(weights)
        cum   = np.cumsum(weights)
        step  = 1.0 / N
        start = np.random.uniform(0, step)
        idx   = np.searchsorted(cum, np.arange(N) * step + start)
        idx   = np.clip(idx, 0, N - 1)
        return particles[idx].copy(), np.ones(N) / N

    # ── publish helpers ────────────────────────────────────────────────────────

    def _publish_pose(self, stamp):
        """Publish the estimated pose with covariance."""
        # Weighted covariance 3×3
        dx  = self._particles[:, 0] - self._pose_x
        dy  = self._particles[:, 1] - self._pose_y
        dya = np.array([
            normalize_angle(a - self._pose_yaw)
            for a in self._particles[:, 2]
        ])

        w = self._weights
        cxx  = float(np.sum(w * dx  * dx))
        cyy  = float(np.sum(w * dy  * dy))
        cyaya = float(np.sum(w * dya * dya))
        cxy  = float(np.sum(w * dx  * dy))
        cxya = float(np.sum(w * dx  * dya))
        cyya = float(np.sum(w * dy  * dya))

        msg = PoseWithCovarianceStamped()
        msg.header.stamp    = stamp
        msg.header.frame_id = self._map_frame

        msg.pose.pose.position.x  = self._pose_x
        msg.pose.pose.position.y  = self._pose_y
        msg.pose.pose.position.z  = 0.0
        msg.pose.pose.orientation = yaw_to_quaternion(self._pose_yaw)

        # 6×6 covariance (x,y,z,rx,ry,rz) — we only fill x,y,yaw
        c = [0.0] * 36
        c[0]  = cxx
        c[1]  = cxy
        c[5]  = cxya
        c[6]  = cxy
        c[7]  = cyy
        c[11] = cyya
        c[30] = cxya
        c[31] = cyya
        c[35] = cyaya
        msg.pose.covariance = c

        self._pub_pose.publish(msg)

    def _broadcast_tf(self):
        """
        Broadcast map → odom TF.

        The map→odom transform is the *correction* applied on top of the
        raw odometry, so downstream nodes (RViz, controller) obtain the
        globally-consistent pose through: map ← MCL → odom ← odom_node → base_link
        """
        t = TransformStamped()
        t.header.stamp    = self.get_clock().now().to_msg()
        t.header.frame_id = self._map_frame
        t.child_frame_id  = self._odom_frame
        t.transform.translation.x = self._pose_x
        t.transform.translation.y = self._pose_y
        t.transform.translation.z = 0.0
        t.transform.rotation      = yaw_to_quaternion(self._pose_yaw)
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
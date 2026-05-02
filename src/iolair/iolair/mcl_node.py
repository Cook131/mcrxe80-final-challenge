#!/usr/bin/env python3
"""
Puzzlebot MCL (Monte Carlo Localization) Node
==============================================
Implements a full particle-filter localization algorithm.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster

import numpy as np
import math


# ── Helpers ────────────────────────────────────────────────────────────────────

def yaw_from_quaternion(q) -> float:
    """Extract yaw angle from a geometry_msgs/Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def euler_to_quaternion(yaw: float) -> Quaternion:
    """Build a geometry_msgs/Quaternion from a yaw angle (roll=pitch=0)."""
    q = Quaternion()
    q.w = math.cos(yaw * 0.5)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    return q


# ── MCL Node ───────────────────────────────────────────────────────────────────

class MCLNode(Node):

    def __init__(self):
        super().__init__('mcl_node')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('num_particles',     500)
        self.declare_parameter('init_x',            0.0)
        self.declare_parameter('init_y',            0.0)
        self.declare_parameter('init_yaw',          0.0)
        self.declare_parameter('init_spread_xy',    0.3)   # std dev [m]
        self.declare_parameter('init_spread_yaw',   0.2)   # std dev [rad]

        # Motion model noise (tuned for Puzzlebot)
        self.declare_parameter('alpha1', 0.05)  # rot  noise from rot
        self.declare_parameter('alpha2', 0.05)  # rot  noise from trans
        self.declare_parameter('alpha3', 0.05)  # trans noise from trans
        self.declare_parameter('alpha4', 0.05)  # trans noise from rot

        # Sensor model
        self.declare_parameter('z_hit',    0.8)   # weight for Gaussian hit
        self.declare_parameter('z_rand',   0.2)   # weight for random reading
        self.declare_parameter('sigma_hit', 0.3)  # std dev of beam hit [m]
        self.declare_parameter('lidar_max_range', 10.0)  # must match model.sdf

        # Only use every Nth beam to save CPU
        self.declare_parameter('beam_skip', 4)      # 360/4 = 90 beams → 4° spacing

        n = self.get_parameter('num_particles').value
        ix  = self.get_parameter('init_x').value
        iy  = self.get_parameter('init_y').value
        iyaw = self.get_parameter('init_yaw').value
        sx  = self.get_parameter('init_spread_xy').value
        syaw = self.get_parameter('init_spread_yaw').value

        self.alpha = [
            self.get_parameter('alpha1').value,
            self.get_parameter('alpha2').value,
            self.get_parameter('alpha3').value,
            self.get_parameter('alpha4').value,
        ]
        self.z_hit    = self.get_parameter('z_hit').value
        self.z_rand   = self.get_parameter('z_rand').value
        self.sigma_hit = self.get_parameter('sigma_hit').value
        self.max_range = self.get_parameter('lidar_max_range').value
        self.beam_skip = self.get_parameter('beam_skip').value

        # ── Particles: shape (N, 3) — columns: x, y, yaw ──────────────────
        self.particles = np.zeros((n, 3))
        self.particles[:, 0] = np.random.normal(ix,   sx,   n)
        self.particles[:, 1] = np.random.normal(iy,   sx,   n)
        self.particles[:, 2] = np.random.normal(iyaw, syaw, n)
        self.weights = np.ones(n) / n

        # ── Map (filled when /map arrives) ────────────────────────────────
        self.map_data      = None   # 2-D numpy array of occupancy [0-100, -1]
        self.map_origin_x  = 0.0
        self.map_origin_y  = 0.0
        self.map_res       = 0.05
        self.map_width     = 0
        self.map_height    = 0

        # ── Odometry tracking ─────────────────────────────────────────────
        self.prev_x   = ix
        self.prev_y   = iy
        self.prev_yaw = iyaw
        self.odom_ready = False

        # ── Best pose (initialise at start pose) ──────────────────────────
        self.best_x   = ix
        self.best_y   = iy
        self.best_yaw = iyaw

        # ── QoS: map topic uses transient-local (latched) ─────────────────
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(OccupancyGrid, '/map',   self.cb_map,  map_qos)
        self.create_subscription(Odometry,      '/odom',  self.cb_odom, 10)
        self.create_subscription(LaserScan,     '/lidar', self.cb_scan, 10)

        # ── Publishers ────────────────────────────────────────────────────
        self.pose_pub      = self.create_publisher(PoseStamped,  '/mcl_pose',  10)
        self.particle_pub  = self.create_publisher(MarkerArray,  '/particles', 10)

        # ── TF broadcaster (map → odom correction) ────────────────────────
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info(
            f'MCL node ready — {n} particles, '
            f'initial pose ({ix:.2f}, {iy:.2f}, {iyaw:.2f} rad)'
        )

    # ── Map callback ──────────────────────────────────────────────────────────

    def cb_map(self, msg: OccupancyGrid):
        self.map_res      = msg.info.resolution
        self.map_width    = msg.info.width
        self.map_height   = msg.info.height
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y

        # Reshape flat array to 2-D grid [row=y, col=x]
        raw = np.array(msg.data, dtype=np.int8)
        self.map_data = raw.reshape((self.map_height, self.map_width))

        self.get_logger().info(
            f'Map received: {self.map_width}x{self.map_height} cells, '
            f'res={self.map_res} m/cell, '
            f'origin=({self.map_origin_x:.2f}, {self.map_origin_y:.2f})'
        )

    # ── Odometry callback ─────────────────────────────────────────────────────

    def cb_odom(self, msg: Odometry):
        x   = msg.pose.pose.position.x
        y   = msg.pose.pose.position.y
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)

        if not self.odom_ready:
            self.prev_x   = x
            self.prev_y   = y
            self.prev_yaw = yaw
            self.odom_ready = True
            return

        # Compute relative motion in the previous robot frame (odometry delta)
        dx  = x   - self.prev_x
        dy  = y   - self.prev_y
        dyaw = yaw - self.prev_yaw
        dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))  # normalise

        # Only predict if the robot actually moved (avoids drift at rest)
        if abs(dx) > 1e-4 or abs(dy) > 1e-4 or abs(dyaw) > 1e-4:
            self._predict(dx, dy, dyaw)

        self.prev_x   = x
        self.prev_y   = y
        self.prev_yaw = yaw

    # ── Scan callback — main MCL loop ─────────────────────────────────────────

    def cb_scan(self, msg: LaserScan):
        if self.map_data is None:
            self.get_logger().warn('Waiting for map…', throttle_duration_sec=5.0)
            return

        self._update(msg)
        self._resample()
        self._publish_pose(msg.header.stamp)
        self._publish_particles(msg.header.stamp)

    # ── Step 1: Predict ───────────────────────────────────────────────────────

    def _predict(self, dx: float, dy: float, dyaw: float):
        """
        Velocity motion model (Thrun et al., Probabilistic Robotics Ch. 5).
        """
        a1, a2, a3, a4 = self.alpha
        n = len(self.particles)

        trans = math.sqrt(dx * dx + dy * dy)
        rot1  = math.atan2(dy, dx) - self.prev_yaw if trans > 1e-4 else 0.0
        rot2  = dyaw - rot1

        # Sample noisy motion for all particles at once
        rot1_hat  = rot1  - np.random.normal(0, math.sqrt(a1 * rot1**2  + a2 * trans**2), n)
        trans_hat = trans - np.random.normal(0, math.sqrt(a3 * trans**2 + a4 * (rot1**2 + rot2**2)), n)
        rot2_hat  = rot2  - np.random.normal(0, math.sqrt(a1 * rot2**2  + a2 * trans**2), n)

        self.particles[:, 0] += trans_hat * np.cos(self.particles[:, 2] + rot1_hat)
        self.particles[:, 1] += trans_hat * np.sin(self.particles[:, 2] + rot1_hat)
        self.particles[:, 2] += rot1_hat + rot2_hat
        self.particles[:, 2]  = np.arctan2(
            np.sin(self.particles[:, 2]),
            np.cos(self.particles[:, 2])
        )

    # ── Step 2: Update (sensor model) ─────────────────────────────────────────

    def _update(self, scan: LaserScan):
        """
        Beam-range sensor model: likelihood field.
        """
        ranges   = np.array(scan.ranges)
        angles   = np.arange(len(ranges)) * scan.angle_increment + scan.angle_min

        # Sub-sample beams to save CPU
        idx    = np.arange(0, len(ranges), self.beam_skip)
        ranges = ranges[idx]
        angles = angles[idx]

        # Mask out invalid readings
        valid = np.isfinite(ranges) & (ranges > scan.range_min) & (ranges < self.max_range)
        ranges = ranges[valid]
        angles = angles[valid]

        if len(ranges) == 0:
            return

        log_weights = np.zeros(len(self.particles))

        for i, (px, py, pyaw) in enumerate(self.particles):
            # Transform beam endpoints to map frame
            bx = px + ranges * np.cos(pyaw + angles)
            by = py + ranges * np.sin(pyaw + angles)

            # Convert to map cell indices
            cx = ((bx - self.map_origin_x) / self.map_res).astype(int)
            cy = ((by - self.map_origin_y) / self.map_res).astype(int)

            # Clip to map bounds
            in_bounds = (
                (cx >= 0) & (cx < self.map_width) &
                (cy >= 0) & (cy < self.map_height)
            )
            cx = np.clip(cx, 0, self.map_width  - 1)
            cy = np.clip(cy, 0, self.map_height - 1)

            cell_vals = self.map_data[cy, cx]

            dist = np.where(cell_vals >= 65, 0.0,
                   np.where(cell_vals < 0,  self.max_range,
                            self._dist_to_obstacle(cx, cy, cell_vals)))

            p_hit  = (self.z_hit  * np.exp(-0.5 * (dist / self.sigma_hit) ** 2)
                      / (self.sigma_hit * math.sqrt(2 * math.pi)))
            p_rand = self.z_rand / self.max_range

            p = np.where(in_bounds, p_hit + p_rand, p_rand)
            p = np.clip(p, 1e-300, None)

            log_weights[i] = np.sum(np.log(p))

        # Convert log weights → normalised weights
        log_weights -= np.max(log_weights)
        self.weights  = np.exp(log_weights)
        self.weights /= np.sum(self.weights)

    def _dist_to_obstacle(self, cx, cy, cell_vals):
        dist = np.zeros(len(cx), dtype=float)
        search_r = 5  # cells
        for k in range(len(cx)):
            if cell_vals[k] >= 65:
                dist[k] = 0.0
                continue
            best = float(self.max_range)
            for dr in range(-search_r, search_r + 1):
                for dc in range(-search_r, search_r + 1):
                    r = cy[k] + dr
                    c = cx[k] + dc
                    if 0 <= r < self.map_height and 0 <= c < self.map_width:
                        if self.map_data[r, c] >= 65:
                            d = math.sqrt(dr ** 2 + dc ** 2) * self.map_res
                            if d < best:
                                best = d
            dist[k] = best
        return dist

    # ── Step 3: Low-variance resample ─────────────────────────────────────────

    def _resample(self):
        n = len(self.particles)
        new_particles = np.empty_like(self.particles)
        r   = np.random.uniform(0, 1.0 / n)
        c   = self.weights[0]
        i   = 0
        u   = r
        for m in range(n):
            while u > c:
                i += 1
                c += self.weights[i]
            new_particles[m] = self.particles[i]
            u += 1.0 / n
        self.particles = new_particles
        self.weights   = np.ones(n) / n

    # ── Publish best pose ─────────────────────────────────────────────────────

    def _publish_pose(self, stamp):
        """Best pose = weighted mean of particles."""
        wx = np.sum(self.weights * self.particles[:, 0])
        wy = np.sum(self.weights * self.particles[:, 1])

        # Circular mean for yaw
        sin_sum = np.sum(self.weights * np.sin(self.particles[:, 2]))
        cos_sum = np.sum(self.weights * np.cos(self.particles[:, 2]))
        wyaw    = math.atan2(sin_sum, cos_sum)

        self.best_x   = wx
        self.best_y   = wy
        self.best_yaw = wyaw

        pose = PoseStamped()
        pose.header.stamp    = stamp
        pose.header.frame_id = 'map'
        pose.pose.position.x = wx
        pose.pose.position.y = wy
        pose.pose.position.z = 0.0
        pose.pose.orientation = euler_to_quaternion(wyaw)

        self.pose_pub.publish(pose)

        # ── Broadcast map → odom TF ───────────────────────────────────────
        t = TransformStamped()
        t.header.stamp    = stamp
        t.header.frame_id = 'map'
        t.child_frame_id  = 'odom'
        
        # Correct logic for map -> odom
        # T_map_odom = T_map_base * inverse(T_odom_base)
        diff_yaw = self.best_yaw - self.prev_yaw
        diff_yaw = math.atan2(math.sin(diff_yaw), math.cos(diff_yaw))

        # Rotate the odom position into the map frame correction
        t.transform.translation.x = self.best_x - (self.prev_x * math.cos(diff_yaw) - self.prev_y * math.sin(diff_yaw))
        t.transform.translation.y = self.best_y - (self.prev_x * math.sin(diff_yaw) + self.prev_y * math.cos(diff_yaw))
        t.transform.rotation      = euler_to_quaternion(diff_yaw)
        
        self.tf_broadcaster.sendTransform(t)

    # ── Publish particles for RViz ────────────────────────────────────────────

    def _publish_particles(self, stamp):
        arr = MarkerArray()
        del_marker = Marker()
        del_marker.action = Marker.DELETEALL
        arr.markers.append(del_marker)

        for i, (px, py, pyaw) in enumerate(self.particles):
            m = Marker()
            m.header.stamp    = stamp
            m.header.frame_id = 'map'
            m.ns, m.id, m.type, m.action = 'particles', i, Marker.ARROW, Marker.ADD
            m.pose.position.x, m.pose.position.y = px, py
            m.pose.orientation = euler_to_quaternion(pyaw)
            m.scale.x, m.scale.y, m.scale.z = 0.12, 0.04, 0.04
            w_norm = min(float(self.weights[i]) * len(self.particles), 1.0)
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0 - w_norm, w_norm, 0.0, 0.8
            arr.markers.append(m)
        self.particle_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = MCLNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
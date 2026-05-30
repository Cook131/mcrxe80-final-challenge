#!/usr/bin/env python3
"""
Map Merger Node
===============
Subscribes to the SLAM map topic (nav_msgs/OccupancyGrid) and accumulates
every incoming snapshot into a single, ever-growing merged map using a
log-odds Bayesian update.  The result is published as a latched
nav_msgs/OccupancyGrid on /merged_map and can be saved to disk (.pgm + .yaml)
via the /map_merger/save_map service (std_srvs/Trigger).

Design rationale
----------------
The SLAM node already does the hard work of building the map.  This node
just *integrates* successive published snapshots so that:
  - Cells confirmed occupied across many scans accumulate evidence.
  - Cells seen as free across many scans become more confidently free.
  - Unknown cells (-1 in ROS convention) contribute no evidence.
  - The merged grid expands automatically when incoming maps extend beyond
    the current merged-map boundary.

Topic / service summary
-----------------------
  Subscriptions:
    /slam_map        nav_msgs/OccupancyGrid   (input — any latched QoS)

  Publications:
    /merged_map      nav_msgs/OccupancyGrid   (latched, 1 Hz by default)

  Services:
    /map_merger/save_map   std_srvs/Trigger   (save .pgm + .yaml to disk)

Parameters
----------
  source_map_topic   str    '/slam_map'
  publish_rate       float   1.0          [Hz]
  map_frame          str    'map'
  resolution         float   0.05         [m/cell]  — used only for the very
                                           first map; subsequent maps must
                                           match this resolution.
  lo_occ             float   0.85         log-odds increment for occupied
  lo_free            float   0.40         log-odds decrement for free
  lo_max             float   5.0          log-odds saturation max
  lo_min             float  -5.0          log-odds saturation min
  occ_thresh         float   0.65         prob → 100 (occupied)
  free_thresh        float   0.35         prob → 0   (free)
  save_map_path      str    '/tmp/merged_map'
"""

import math
import os
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg      import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose, Point, Quaternion
from std_srvs.srv      import Trigger


# ── Quaternion helper ──────────────────────────────────────────────────────────

def _yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw * 0.5)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    return q


# ── PGM / YAML saver (mirrors slam_node.py convention exactly) ────────────────

def save_map_pgm_yaml(
    log_odds: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    base_path: str,
    occ_thresh: float = 0.65,
    free_thresh: float = 0.35,
) -> str:
    """
    Write the log-odds grid as a .pgm image + .yaml metadata file.

    Pixel encoding (ROS map_server convention):
        0   → occupied  (prob >= occ_thresh)
        254 → free      (prob <= free_thresh)
        205 → unknown
    Row 0 of the PGM = bottom of the map (positive-Y world direction).
    """
    rows, cols = log_odds.shape

    prob = 1.0 - 1.0 / (1.0 + np.exp(log_odds))

    pgm = np.full((rows, cols), 205, dtype=np.uint8)
    pgm[prob >= occ_thresh]  = 0
    pgm[prob <= free_thresh] = 254
    pgm_img = np.flipud(pgm)

    pgm_path  = base_path + '.pgm'
    yaml_path = base_path + '.yaml'

    os.makedirs(os.path.dirname(os.path.abspath(pgm_path)), exist_ok=True)

    with open(pgm_path, 'wb') as fh:
        fh.write(f'P5\n{cols} {rows}\n255\n'.encode('ascii'))
        fh.write(pgm_img.tobytes())

    yaml_name = os.path.basename(pgm_path)
    with open(yaml_path, 'w', encoding='utf-8') as fh:
        fh.write(f'image: {yaml_name}\n')
        fh.write(f'resolution: {resolution}\n')
        fh.write(f'origin: [{origin_x:.4f}, {origin_y:.4f}, 0.0]\n')
        fh.write('negate: 0\n')
        fh.write(f'occupied_thresh: {occ_thresh}\n')
        fh.write(f'free_thresh: {free_thresh}\n')

    return (
        f'Merged map saved → {pgm_path}  and  {yaml_path} '
        f'({cols}×{rows} cells)'
    )


# ── Map Merger Node ────────────────────────────────────────────────────────────

class MapMergerNode(Node):
    """
    Accumulates every OccupancyGrid received on *source_map_topic* into a
    single persistent log-odds grid and re-publishes the merged result.
    """

    def __init__(self):
        super().__init__('map_merger_node')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('source_map_topic', '/slam_map')
        self.declare_parameter('publish_rate',      1.0)
        self.declare_parameter('map_frame',        'map')
        self.declare_parameter('resolution',        0.05)
        self.declare_parameter('lo_occ',            0.85)
        self.declare_parameter('lo_free',           0.40)
        self.declare_parameter('lo_max',            5.0)
        self.declare_parameter('lo_min',           -5.0)
        self.declare_parameter('occ_thresh',        0.65)
        self.declare_parameter('free_thresh',       0.35)
        self.declare_parameter('save_map_path',    '/tmp/merged_map')

        self._source_topic = self.get_parameter('source_map_topic').value
        self._map_frame    = self.get_parameter('map_frame').value
        self._res          = self.get_parameter('resolution').value
        self._lo_occ       = self.get_parameter('lo_occ').value
        self._lo_free      = self.get_parameter('lo_free').value
        self._lo_max       = self.get_parameter('lo_max').value
        self._lo_min       = self.get_parameter('lo_min').value
        self._occ_thresh   = self.get_parameter('occ_thresh').value
        self._free_thresh  = self.get_parameter('free_thresh').value
        pub_rate           = self.get_parameter('publish_rate').value

        # ── Internal merged grid state ─────────────────────────────────────
        # Initialised lazily on the first incoming map.
        self._log_odds: np.ndarray | None = None
        self._origin_x: float = 0.0
        self._origin_y: float = 0.0
        self._grid_w:   int   = 0
        self._grid_h:   int   = 0
        self._map_count: int  = 0          # total snapshots merged so far
        self._lock = threading.Lock()

        # ── QoS profiles ──────────────────────────────────────────────────
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        # Accept latched maps (TRANSIENT_LOCAL) from the SLAM node.
        incoming_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ── Pub / Sub / Service ────────────────────────────────────────────
        self.create_subscription(
            OccupancyGrid, self._source_topic,
            self._cb_map, incoming_qos,
        )
        self._merged_pub = self.create_publisher(
            OccupancyGrid, '/merged_map', latched_qos,
        )
        self.create_service(
            Trigger, '/map_merger/save_map', self._srv_save_map,
        )
        self.create_timer(1.0 / pub_rate, self._publish_merged_map)

        self.get_logger().info(
            f'MapMergerNode ready — listening on "{self._source_topic}", '
            f'publishing merged map on "/merged_map" at {pub_rate} Hz.'
        )

    # ── Incoming map callback ─────────────────────────────────────────────

    def _cb_map(self, msg: OccupancyGrid):
        """
        Integrate a new OccupancyGrid snapshot into the merged log-odds grid.

        ROS OccupancyGrid convention:
            -1  → unknown
             0  → free
            100 → occupied
        """
        info = msg.info
        inc_w    = info.width
        inc_h    = info.height
        inc_res  = info.resolution
        inc_ox   = info.origin.position.x
        inc_oy   = info.origin.position.y

        # Sanity check resolution
        if abs(inc_res - self._res) > 1e-5:
            self.get_logger().warn(
                f'Incoming map resolution {inc_res:.4f} != expected '
                f'{self._res:.4f} — skipping snapshot.'
            )
            return

        data = np.asarray(msg.data, dtype=np.int8).reshape((inc_h, inc_w))

        with self._lock:
            # ── Lazy initialisation from first map ────────────────────────
            if self._log_odds is None:
                self._origin_x = inc_ox
                self._origin_y = inc_oy
                self._grid_w   = inc_w
                self._grid_h   = inc_h
                self._log_odds = np.zeros((inc_h, inc_w), dtype=np.float32)
                self.get_logger().info(
                    f'Merged grid initialised: {inc_w}×{inc_h} cells, '
                    f'origin=({inc_ox:.2f}, {inc_oy:.2f})'
                )

            # ── Expand merged grid if needed ──────────────────────────────
            # World extent of incoming map
            inc_right = inc_ox + inc_w * self._res
            inc_top   = inc_oy + inc_h * self._res

            # World extent of current merged grid
            cur_right = self._origin_x + self._grid_w * self._res
            cur_top   = self._origin_y + self._grid_h * self._res

            pad_left  = max(0, int(math.ceil(
                (self._origin_x - inc_ox) / self._res)) + 1)
            pad_bot   = max(0, int(math.ceil(
                (self._origin_y - inc_oy) / self._res)) + 1)
            pad_right = max(0, int(math.ceil(
                (inc_right - cur_right) / self._res)) + 1)
            pad_top   = max(0, int(math.ceil(
                (inc_top   - cur_top)   / self._res)) + 1)

            if any([pad_left, pad_bot, pad_right, pad_top]):
                self._expand_grid(pad_left, pad_bot, pad_right, pad_top)

            # ── Compute offset of incoming map inside merged grid ─────────
            col_off = int(round((inc_ox - self._origin_x) / self._res))
            row_off = int(round((inc_oy - self._origin_y) / self._res))

            # ── Apply Bayesian log-odds update cell by cell ───────────────
            # Vectorised for performance.
            #
            # Only cells that are NOT unknown (-1) contribute evidence.
            # occupied (100) → +lo_occ
            # free     (  0) → -lo_free
            known_mask = data != -1
            occ_mask   = known_mask & (data >= 50)
            free_mask  = known_mask & (data <  50)

            # Slice view into the merged grid where the incoming map lands
            r0, r1 = row_off, row_off + inc_h
            c0, c1 = col_off, col_off + inc_w

            # Clamp slice to actual merged-grid bounds (defensive)
            r0c = max(r0, 0);  r1c = min(r1, self._grid_h)
            c0c = max(c0, 0);  c1c = min(c1, self._grid_w)
            dr0 = r0c - r0;    dc0 = c0c - c0
            dr1 = dr0 + (r1c - r0c)
            dc1 = dc0 + (c1c - c0c)

            target = self._log_odds[r0c:r1c, c0c:c1c]
            occ    = occ_mask [dr0:dr1, dc0:dc1]
            free   = free_mask[dr0:dr1, dc0:dc1]

            target[occ]  = np.minimum(target[occ]  + self._lo_occ,  self._lo_max)
            target[free] = np.maximum(target[free] - self._lo_free, self._lo_min)

            self._map_count += 1

        self.get_logger().debug(
            f'Snapshot #{self._map_count} merged — '
            f'incoming size {inc_w}×{inc_h}, '
            f'merged grid now {self._grid_w}×{self._grid_h}.'
        )

    # ── Grid expansion ────────────────────────────────────────────────────

    def _expand_grid(self, pad_left: int, pad_bot: int,
                     pad_right: int, pad_top: int):
        """
        Grow the log-odds array by the requested number of cells on each side.
        Must be called while _lock is held.
        """
        new_w = self._grid_w + pad_left + pad_right
        new_h = self._grid_h + pad_bot  + pad_top
        new_grid = np.zeros((new_h, new_w), dtype=np.float32)
        new_grid[
            pad_bot : pad_bot + self._grid_h,
            pad_left: pad_left + self._grid_w
        ] = self._log_odds

        self._log_odds  = new_grid
        self._grid_w    = new_w
        self._grid_h    = new_h
        self._origin_x -= pad_left * self._res
        self._origin_y -= pad_bot  * self._res

        self.get_logger().debug(
            f'Merged grid expanded to {new_w}×{new_h} '
            f'(+{pad_left}L +{pad_right}R +{pad_bot}B +{pad_top}T).'
        )

    # ── Publish merged map ────────────────────────────────────────────────

    def _publish_merged_map(self):
        with self._lock:
            if self._log_odds is None:
                return  # Nothing received yet

            lo_copy  = self._log_odds.copy()
            origin_x = self._origin_x
            origin_y = self._origin_y
            grid_w   = self._grid_w
            grid_h   = self._grid_h

        # Convert log-odds → ROS OccupancyGrid values
        prob     = 1.0 - 1.0 / (1.0 + np.exp(lo_copy))
        ros_grid = np.full(lo_copy.shape, -1, dtype=np.int8)
        ros_grid[prob >= self._occ_thresh]  = 100
        ros_grid[prob <= self._free_thresh] = 0

        msg = OccupancyGrid()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self._map_frame

        meta            = MapMetaData()
        meta.resolution = self._res
        meta.width      = grid_w
        meta.height     = grid_h
        meta.origin     = Pose(
            position=Point(x=float(origin_x), y=float(origin_y), z=0.0),
            orientation=_yaw_to_quaternion(0.0),
        )
        msg.info = meta
        msg.data = ros_grid.flatten().tolist()

        self._merged_pub.publish(msg)

    # ── Save-map service ──────────────────────────────────────────────────

    def _srv_save_map(self, _request, response):
        """
        /map_merger/save_map  (std_srvs/Trigger)

        Snapshots the current merged log-odds grid and writes it to disk as
        <save_map_path>.pgm  +  <save_map_path>.yaml
        """
        save_path = self.get_parameter('save_map_path').value

        with self._lock:
            if self._log_odds is None:
                response.success = False
                response.message = 'No map data received yet — nothing to save.'
                self.get_logger().warn(response.message)
                return response

            lo_snap  = self._log_odds.copy()
            origin_x = self._origin_x
            origin_y = self._origin_y

        try:
            msg = save_map_pgm_yaml(
                lo_snap,
                self._res,
                origin_x,
                origin_y,
                save_path,
                occ_thresh=self._occ_thresh,
                free_thresh=self._free_thresh,
            )
            response.success = True
            response.message = msg
            self.get_logger().info(msg)
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            self.get_logger().error(f'save_map FAILED: {exc}')

        return response


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = MapMergerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
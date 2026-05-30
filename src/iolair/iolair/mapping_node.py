#!/usr/bin/env python3
"""
Map Merger Node
===============
Subscribes to the SLAM map topic (nav_msgs/OccupancyGrid) and accumulates
every incoming snapshot into a single, ever-growing persistent map.

Design rationale — max-occupancy strategy
------------------------------------------
Cells are NEVER un-occupied.  The merged grid stores the maximum occupancy
value ever seen for each cell across all incoming snapshots:

  - A cell seen as occupied (100) stays occupied forever.
  - A cell seen as free (0) is written only if it has never been occupied.
  - Unknown cells (-1) contribute no information.
  - The merged grid expands automatically when incoming maps grow beyond
    the current boundary.

This is the right strategy for map saving: the goal is a conservative,
complete record of every obstacle ever detected, not a live probabilistic
estimate that can erase walls when a scan misses them.

Topic / service summary
-----------------------
  Subscriptions:
    /slam_map        nav_msgs/OccupancyGrid   (input — latched QoS)

  Publications:
    /merged_map      nav_msgs/OccupancyGrid   (latched, 1 Hz by default)

  Services:
    /map_merger/save_map   std_srvs/Trigger   (save .pgm + .yaml to disk)

Parameters
----------
  source_map_topic   str    '/slam_map'
  publish_rate       float   1.0          [Hz]
  map_frame          str    'map'
  resolution         float   0.05         [m/cell]  — must match SLAM node
  occ_thresh         int     50           cells >= this value → occupied in .pgm
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
    grid: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    base_path: str,
    occ_thresh: int = 50,
) -> str:
    """
    Write the max-occupancy int8 grid as a .pgm image + .yaml metadata file.

    Input grid uses ROS OccupancyGrid convention:
        100 → occupied
          0 → free
         -1 → unknown

    Pixel encoding (ROS map_server convention):
          0 → occupied  (value >= occ_thresh)
        254 → free      (value == 0)
        205 → unknown   (value == -1)
    Row 0 of the PGM = bottom of the map (positive-Y world direction).
    """
    rows, cols = grid.shape

    pgm = np.full((rows, cols), 205, dtype=np.uint8)          # unknown
    pgm[grid == 0]             = 254                           # free
    pgm[grid >= occ_thresh]    = 0                             # occupied

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
        fh.write('occupied_thresh: 0.65\n')
        fh.write('free_thresh: 0.35\n')

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
        self.declare_parameter('occ_thresh',        50)           # int8 threshold
        self.declare_parameter('save_map_path',    '/tmp/merged_map')

        self._source_topic = self.get_parameter('source_map_topic').value
        self._map_frame    = self.get_parameter('map_frame').value
        self._res          = self.get_parameter('resolution').value
        self._occ_thresh   = self.get_parameter('occ_thresh').value
        pub_rate           = self.get_parameter('publish_rate').value

        # ── Internal merged grid state ─────────────────────────────────────
        # Stores the maximum occupancy value ever seen per cell.
        # dtype int8, same convention as ROS OccupancyGrid:
        #   -1 = unknown, 0 = free, 100 = occupied.
        # Initialised lazily on the first incoming map.
        self._grid: np.ndarray | None = None
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
        Merge a new OccupancyGrid snapshot using a max-occupancy strategy.

        ROS OccupancyGrid convention:
            -1  → unknown
             0  → free
            100 → occupied

        Rules applied per cell:
            incoming occupied  → cell becomes occupied, permanently
            incoming free      → cell becomes free only if currently unknown
            incoming unknown   → cell unchanged
        """
        info = msg.info
        inc_w    = info.width
        inc_h    = info.height
        inc_res  = info.resolution
        inc_ox   = info.origin.position.x
        inc_oy   = info.origin.position.y

        if abs(inc_res - self._res) > 1e-5:
            self.get_logger().warn(
                f'Incoming map resolution {inc_res:.4f} != expected '
                f'{self._res:.4f} — skipping snapshot.'
            )
            return

        data = np.asarray(msg.data, dtype=np.int8).reshape((inc_h, inc_w))

        with self._lock:
            # ── Lazy initialisation from first map ────────────────────────
            if self._grid is None:
                self._origin_x = inc_ox
                self._origin_y = inc_oy
                self._grid_w   = inc_w
                self._grid_h   = inc_h
                self._grid     = np.full((inc_h, inc_w), -1, dtype=np.int8)
                self.get_logger().info(
                    f'Merged grid initialised: {inc_w}×{inc_h} cells, '
                    f'origin=({inc_ox:.2f}, {inc_oy:.2f})'
                )

            # ── Expand merged grid if needed ──────────────────────────────
            inc_right = inc_ox + inc_w * self._res
            inc_top   = inc_oy + inc_h * self._res
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

            # ── Clamp slice to merged-grid bounds (defensive) ─────────────
            r0, r1 = row_off, row_off + inc_h
            c0, c1 = col_off, col_off + inc_w
            r0c = max(r0, 0);  r1c = min(r1, self._grid_h)
            c0c = max(c0, 0);  c1c = min(c1, self._grid_w)
            dr0 = r0c - r0;    dc0 = c0c - c0
            dr1 = dr0 + (r1c - r0c)
            dc1 = dc0 + (c1c - c0c)

            target = self._grid[r0c:r1c, c0c:c1c]
            patch  = data[dr0:dr1, dc0:dc1]

            # ── Max-occupancy merge ───────────────────────────────────────
            # 1. Mark free: only where merged cell is still unknown (-1)
            free_mask = (patch == 0) & (target == -1)
            target[free_mask] = 0

            # 2. Mark occupied: always — occupied cells never get erased
            occ_mask = patch >= self._occ_thresh
            target[occ_mask] = 100

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
        Grow the int8 grid by the requested number of cells on each side.
        New cells are initialised to -1 (unknown).
        Must be called while _lock is held.
        """
        new_w = self._grid_w + pad_left + pad_right
        new_h = self._grid_h + pad_bot  + pad_top
        new_grid = np.full((new_h, new_w), -1, dtype=np.int8)
        new_grid[
            pad_bot : pad_bot + self._grid_h,
            pad_left: pad_left + self._grid_w
        ] = self._grid

        self._grid      = new_grid
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
            if self._grid is None:
                return  # Nothing received yet

            grid_copy = self._grid.copy()
            origin_x  = self._origin_x
            origin_y  = self._origin_y
            grid_w    = self._grid_w
            grid_h    = self._grid_h

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
        msg.data = grid_copy.flatten().tolist()

        self._merged_pub.publish(msg)

    # ── Save-map service ──────────────────────────────────────────────────

    def _srv_save_map(self, _request, response):
        """
        /map_merger/save_map  (std_srvs/Trigger)

        Snapshots the current merged grid and writes it to disk as
        <save_map_path>.pgm  +  <save_map_path>.yaml
        """
        save_path = self.get_parameter('save_map_path').value

        with self._lock:
            if self._grid is None:
                response.success = False
                response.message = 'No map data received yet — nothing to save.'
                self.get_logger().warn(response.message)
                return response

            grid_snap = self._grid.copy()
            origin_x  = self._origin_x
            origin_y  = self._origin_y

        try:
            msg = save_map_pgm_yaml(
                grid_snap,
                self._res,
                origin_x,
                origin_y,
                save_path,
                occ_thresh=self._occ_thresh,
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
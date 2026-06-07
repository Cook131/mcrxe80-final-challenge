#!/usr/bin/env python3
import math
import time
import yaml  # Necesitas tener instalado python3-yaml
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D
from std_msgs.msg import String
import os

class MissionPlannerNode(Node):

    def __init__(self):
        super().__init__('mission_planner')

        # ── Parámetros ────────────────────────────────────────────────────
        self.declare_parameter('loop_mission', False)
        self.declare_parameter('start_delay', 2.0)
        self.declare_parameter('goal_timeout', 60.0)
        self.declare_parameter('waypoints_file', 'src/iolair/configs/waypoints.yaml') # Ruta al YAML

        self._loop        = self.get_parameter('loop_mission').value
        self._start_delay = self.get_parameter('start_delay').value
        self._timeout     = self.get_parameter('goal_timeout').value
        self._file_path   = self.get_parameter('waypoints_file').value

        # ── Cargar Waypoints ──────────────────────────────────────────────
        self._waypoints = self._load_waypoints_from_yaml()

        # ── Estado ────────────────────────────────────────────────────────
        self._current_idx = 0
        self._astar_status = 'IDLE'
        self._goal_sent_time: float = -1.0
        self._mission_done  = False
        self._started       = False

        # ── Suscriptores / Publicadores ───────────────────────────────────
        self._pub_goal = self.create_publisher(Pose2D, '/astar/goal', 10)
        self.create_subscription(String, '/astar/status', self._cb_status, 10)

        # ── Timer principal ───────────────────────────────────────────────
        self.create_timer(0.2, self._mission_loop)

        self.get_logger().info(f'Mission Planner iniciado con {len(self._waypoints)} waypoints.')

    def _load_waypoints_from_yaml(self):
        """Lee el archivo YAML y devuelve una lista de tuplas (x, y, desc)"""
        if not os.path.exists(self._file_path):
            self.get_logger().error(f'No se encuentra el archivo: {self._file_path}')
            return []

        with open(self._file_path, 'r') as f:
            data = yaml.safe_load(f)
            
        waypoints = []
        for wp in data.get('waypoints', []):
            # Asumimos estructura: id, x, y
            waypoints.append((float(wp['x']), float(wp['y']), f"Waypoint ID: {wp.get('id', 'N/A')}"))
        
        return waypoints

    # ... [El resto de los métodos _cb_status, _mission_loop y _send_current_goal se mantienen igual] ...

    def _cb_status(self, msg: String):
        self._astar_status = msg.data

    def _mission_loop(self):
        if self._mission_done or not self._waypoints:
            return

        if not self._started:
            if not hasattr(self, '_start_ts'):
                self._start_ts = time.monotonic()
                return
            if time.monotonic() - self._start_ts < self._start_delay:
                return
            self._started = True
            self._send_current_goal()
            return

        if self._current_idx >= len(self._waypoints):
            if self._loop:
                self._current_idx = 0
                self._send_current_goal()
            else:
                self._mission_done = True
            return

        if self._astar_status == 'GOAL_REACHED':
            self._current_idx += 1
            if self._current_idx < len(self._waypoints):
                self._send_current_goal()

        elif self._astar_status == 'NO_PATH':
            time.sleep(2.0)
            self._send_current_goal()

        elif (self._goal_sent_time > 0 and
              time.monotonic() - self._goal_sent_time > self._timeout):
            self._send_current_goal()

    def _send_current_goal(self):
        if self._current_idx >= len(self._waypoints):
            return

        x, y, desc = self._waypoints[self._current_idx]
        msg = Pose2D()
        msg.x = x
        msg.y = y
        self._pub_goal.publish(msg)
        self._goal_sent_time = time.monotonic()
        self._astar_status = 'PLANNING'
        self.get_logger().info(f'Enviando {desc}: ({x:.2f}, {y:.2f})')

def main(args=None):
    rclpy.init(args=args)
    node = MissionPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
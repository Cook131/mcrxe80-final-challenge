#!/usr/bin/env python3
"""
astar_planner.py — Planificador de Rutas A* para Puzzlebot
===========================================================
Nodo ROS 2 que combina:
  - Elección dinámica en caliente del mapa activo (/slam_map o /map)
  - Planificación de ruta con el algoritmo A* sobre la grilla
  - Ejecución del path waypoint a waypoint usando el PID de GoToGoal
"""

import heapq
import math
import threading
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import Pose2D, PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import String


# ── Helpers de quaternión ─────────────────────────────────────────────────────

def yaw_from_quat(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw * 0.5)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    return q


# ── Algoritmo A* ──────────────────────────────────────────────────────────────

class AStarGrid:
    """
    Planificador A* sobre un mapa de ocupación 2-D.
    """
    DIRS_4 = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
    DIRS_8 = DIRS_4 + [
        (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
        ( 1, -1, math.sqrt(2)), ( 1, 1, math.sqrt(2)),
    ]

    def __init__(self, grid: np.ndarray, allow_diagonal: bool = True):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.dirs = self.DIRS_8 if allow_diagonal else self.DIRS_4

    def heuristic(self, r0: int, c0: int, r1: int, c1: int) -> float:
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        return max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)

    def plan(self, start: tuple, goal: tuple):
        r0, c0 = start
        r1, c1 = goal

        if self.grid[r0, c0] or self.grid[r1, c1]:
            return []

        open_heap = []
        heapq.heappush(open_heap, (0.0, 0.0, r0, c0))

        came_from = {}
        g_score = {(r0, c0): 0.0}

        while open_heap:
            f, g, r, c = heapq.heappop(open_heap)

            if (r, c) == (r1, c1):
                return self._reconstruct(came_from, r1, c1)

            if g > g_score.get((r, c), float('inf')):
                continue

            for dr, dc, cost in self.dirs:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                if self.grid[nr, nc]:
                    continue

                ng = g + cost
                if ng < g_score.get((nr, nc), float('inf')):
                    g_score[(nr, nc)] = ng
                    h = self.heuristic(nr, nc, r1, c1)
                    came_from[(nr, nc)] = (r, c)
                    heapq.heappush(open_heap, (ng + h, ng, nr, nc))

        return []

    def _reconstruct(self, came_from: dict, r: int, c: int):
        path = []
        node = (r, c)
        while node in came_from:
            path.append(node)
            node = came_from[node]
        path.append(node)
        path.reverse()
        return path


# ── Nodo ROS 2 ────────────────────────────────────────────────────────────────

class AStarPlannerNode(Node):

    def __init__(self):
        super().__init__('astar_planner')

        # ── Parámetros ────────────────────────────────────────────────────
        self.declare_parameter('slam_map_topic',     '/slam_map')
        self.declare_parameter('static_map_topic',   '/map')
        self.declare_parameter('odom_topic',         '/odom')
        self.declare_parameter('goal_in_topic',      '/astar/goal')
        self.declare_parameter('goal_out_topic',     '/goal')
        self.declare_parameter('inflation_radius',   0.15)
        self.declare_parameter('waypoint_threshold', 0.10)
        self.declare_parameter('occupied_threshold', 65)
        self.declare_parameter('allow_diagonal',     True)

        self.slam_map_topic = self.get_parameter('slam_map_topic').value
        self.static_map_topic = self.get_parameter('static_map_topic').value
        odom_topic     = self.get_parameter('odom_topic').value
        goal_in_topic  = self.get_parameter('goal_in_topic').value
        goal_out_topic = self.get_parameter('goal_out_topic').value
        self.infl_r    = self.get_parameter('inflation_radius').value
        self.wp_thresh = self.get_parameter('waypoint_threshold').value
        self.occ_thresh= self.get_parameter('occupied_threshold').value
        self.diagonal  = self.get_parameter('allow_diagonal').value

        # ── Estado ────────────────────────────────────────────────────────
        self.map_msg   = None          
        self.map_lock  = threading.Lock()
        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_th  = 0.0

        # Ruta y Tópicos activos
        self.active_topic = None  # Almacena cuál es el tópico activo actualmente
        self.waypoints: deque = deque()
        self.active    = False
        self.status    = 'IDLE'

        # Guardar configuración de QoS para mapas
        self.map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # Inicializar el suscriptor dinámico
        self.sub_map = None

        # ── Suscriptores fijos ─────────────────────────────────────────────
        self.sub_odom = self.create_subscription(
            Odometry, odom_topic, self._cb_odom, 10)
        self.sub_goal = self.create_subscription(
            Pose2D, goal_in_topic, self._cb_goal, 10)

        # ── Publicadores ──────────────────────────────────────────────────
        self.pub_wp     = self.create_publisher(Pose2D,  goal_out_topic, 10)
        self.pub_path   = self.create_publisher(Path,    '/astar/path',  10)
        self.pub_status = self.create_publisher(String,  '/astar/status', 10)

        # ── Timers de Control y Descubrimiento Continúo ─────────────────────
        self.create_timer(0.1, self._control_loop)
        
        # Corre cada 1.5s comprobando la red para adaptarse a encendidos tardíos o cambios
        self.create_timer(1.5, self._check_map_sources)

        self.get_logger().info(
            f'[A*] Nodo iniciado. Monitoreando fuentes de mapas...\n'
            f' -> Opción A: «{self.slam_map_topic}»\n'
            f' -> Opción B: «{self.static_map_topic}»')

    # ── Gestión de Suscripción Dinámica en Caliente ───────────────────────────

    def _check_map_sources(self):
        """Verifica de forma continua cuál tópico tiene publicadores reales y conmuta el suscriptor."""
        slam_pubs = self.count_publishers(self.slam_map_topic)
        static_pubs = self.count_publishers(self.static_map_topic)

        target_topic = None

        # Definición de prioridades de escucha
        if slam_pubs > 0:
            target_topic = self.slam_map_topic
        elif static_pubs > 0:
            target_topic = self.static_map_topic

        # Si detectó un cambio respecto al tópico al que estábamos escuchando previamente
        if target_topic != self.active_topic:
            with self.map_lock:
                if target_topic is None:
                    self.get_logger().warn('[A*] Se han perdido todas las fuentes de mapas activas.')
                    if self.sub_map is not None:
                        self.destroy_subscription(self.sub_map)
                        self.sub_map = None
                else:
                    self.get_logger().info(f'[A*] Conmutando origen de mapa activo a: «{target_topic}»')
                    if self.sub_map is not None:
                        self.destroy_subscription(self.sub_map)
                    
                    # Generar la nueva suscripción bajo demanda
                    self.sub_map = self.create_subscription(
                        OccupancyGrid, target_topic, self._cb_map, self.map_qos)
                
                self.active_topic = target_topic

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cb_map(self, msg: OccupancyGrid):
        with self.map_lock:
            if self.map_msg is None:
                self.get_logger().info(f'[A*] ¡Mapa cargado! Resolución: {msg.info.resolution:.3f}m | Frame: "{msg.header.frame_id}"')
            self.map_msg = msg

    def _cb_odom(self, msg: Odometry):
        self.robot_x  = msg.pose.pose.position.x
        self.robot_y  = msg.pose.pose.position.y
        self.robot_th = yaw_from_quat(msg.pose.pose.orientation)

    def _cb_goal(self, msg: Pose2D):
        self.get_logger().info(
            f'[A*] Objetivo recibido: x={msg.x:.2f} y={msg.y:.2f}')
        self._plan_and_start(msg.x, msg.y)

    # ── Planificación ─────────────────────────────────────────────────────────

    def _plan_and_start(self, gx: float, gy: float):
        with self.map_lock:
            if self.map_msg is None:
                self.get_logger().warn('[A*] Sin mapa en memoria. Asegúrate de que el publicador de mapas esté activo.')
                return
            map_msg = self.map_msg

        meta   = map_msg.info
        res    = meta.resolution
        ox     = meta.origin.position.x
        oy     = meta.origin.position.y
        width  = meta.width
        height = meta.height

        raw = np.array(map_msg.data, dtype=np.int8).reshape((height, width))

        grid = np.zeros((height, width), dtype=np.uint8)
        grid[raw >= self.occ_thresh] = 1
        grid[raw == -1]              = 1

        infl_cells = max(1, int(math.ceil(self.infl_r / res)))
        grid = self._inflate(grid, infl_cells)

        def world_to_cell(wx, wy):
            col = int((wx - ox) / res)
            row = int((wy - oy) / res)
            return row, col

        def cell_to_world(row, col):
            wx = ox + (col + 0.5) * res
            wy = oy + (row + 0.5) * res
            return wx, wy

        sr, sc = world_to_cell(self.robot_x, self.robot_y)
        gr, gc = world_to_cell(gx, gy)

        sr = max(0, min(height - 1, sr))
        sc = max(0, min(width  - 1, sc))
        gr = max(0, min(height - 1, gr))
        gc = max(0, min(width  - 1, gc))

        planner = AStarGrid(grid, allow_diagonal=self.diagonal)
        cell_path = planner.plan((sr, sc), (gr, gc))

        if not cell_path:
            self.get_logger().error('[A*] No se encontró ruta al objetivo.')
            self._set_status('NO_PATH')
            return

        cell_path = self._shortcut(cell_path, grid)
        world_path = [cell_to_world(r, c) for r, c in cell_path]

        self.waypoints = deque(world_path[1:])
        self.active    = True
        self._set_status('PLANNING')

        self.get_logger().info(f'[A*] Ruta calculada con {len(world_path)} waypoints.')
        self._publish_path(world_path, map_msg.header.frame_id)

    # ── Loop de control ───────────────────────────────────────────────────────

    def _control_loop(self):
        if not self.active or not self.waypoints:
            return

        wx, wy = self.waypoints[0]
        dist = math.hypot(wx - self.robot_x, wy - self.robot_y)

        if dist < self.wp_thresh:
            self.waypoints.popleft()
            if not self.waypoints:
                self.active = False
                self._set_status('GOAL_REACHED')
                self.get_logger().info('[A*] ✅ Objetivo alcanzado.')
                return
            wx, wy = self.waypoints[0]
            self.get_logger().info(f'[A*] Waypoint alcanzado → siguiente ({wx:.2f}, {wy:.2f})')

        self._set_status('EXECUTING')
        wp_msg = Pose2D()
        wp_msg.x = wx
        wp_msg.y = wy
        self.pub_wp.publish(wp_msg)

    # ── Utilidades ────────────────────────────────────────────────────────────

    @staticmethod
    def _inflate(grid: np.ndarray, radius: int) -> np.ndarray:
        inflated = grid.copy()
        rows, cols = grid.shape
        obstacle_cells = list(zip(*np.where(grid == 1)))

        visited = set(obstacle_cells)
        frontier = deque(obstacle_cells)

        for _ in range(radius):
            next_frontier = deque()
            while frontier:
                r, c = frontier.popleft()
                for dr, dc in ((-1,0),(1,0),(0,-1),(0,1),
                               (-1,-1),(-1,1),(1,-1),(1,1)):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols and
                            (nr, nc) not in visited):
                        visited.add((nr, nc))
                        inflated[nr, nc] = 1
                        next_frontier.append((nr, nc))
            frontier = next_frontier

        return inflated

    @staticmethod
    def _shortcut(path: list, grid: np.ndarray) -> list:
        if len(path) <= 2:
            return path

        def los_clear(r0, c0, r1, c1) -> bool:
            dx = abs(c1 - c0); dy = abs(r1 - r0)
            sx = 1 if c1 > c0 else -1
            sy = 1 if r1 > r0 else -1
            err = dx - dy
            r, c = r0, c0
            while True:
                if grid[r, c]:
                    return False
                if r == r1 and c == c1:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy; c += sx
                if e2 < dx:
                    err += dx; r += sy
            return True

        pruned = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                r0, c0 = path[i]
                r1, c1 = path[j]
                if los_clear(r0, c0, r1, c1):
                    break
                j -= 1
            pruned.append(path[j])
            i = j

        return pruned

    def _publish_path(self, world_path: list, frame_id: str):
        path_msg = Path()
        path_msg.header.stamp    = self.get_clock().now().to_msg()
        path_msg.header.frame_id = frame_id if frame_id else 'map'

        for wx, wy in world_path:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x    = wx
            ps.pose.position.y    = wy
            ps.pose.position.z    = 0.0
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)

        self.pub_path.publish(path_msg)

    def _set_status(self, status: str):
        if status != self.status:
            self.status = status
            self.get_logger().info(f'[A*] Estado: {status}')
        msg = String()
        msg.data = status
        self.pub_status.publish(msg)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = AStarPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
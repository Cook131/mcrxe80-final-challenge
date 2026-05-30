#!/usr/bin/env python3
"""
iba_explorer.py — Explorador Autónomo por Fronteras (IBA-style)
================================================================
Nodo de exploración autónoma para mapeo SLAM sin teleop.

Estrategia (Frontier-Based Exploration):
  1. Escucha /slam_map (OccupancyGrid en tiempo real).
  2. Detecta celdas "frontera": libres (0) adyacentes a desconocidas (-1).
  3. Agrupa fronteras por conectividad y elige la más cercana al robot.
  4. Publica el centroide de esa frontera como objetivo en /goal (Pose2D).
  5. El GoToGoal + BugReflex navegan hacia allá mientras el SLAM actualiza.
  6. Cuando el robot llega (señal de /go_to_goal_status o timeout), elige
     la siguiente frontera.
  7. Al agotar fronteras, publica /slam/save_map y se detiene.

Pipeline de seguridad (subsumption):
  IBA Explorer → /goal → GoToGoal → /cmd_raw → BugReflex → /cmd_vel → Controller

Tópicos consumidos:
  /slam_map        (nav_msgs/OccupancyGrid)   mapa en construcción
  /odom            (nav_msgs/Odometry)         pose del robot
  /astar/status    (std_msgs/String)           estado del planner A*
  /reflex_status   (std_msgs/String)           estado del reflex layer

Tópicos publicados:
  /astar/goal      (geometry_msgs/Pose2D)      objetivo para A* planner
  /iba/status      (std_msgs/String)           estado del explorador
  /iba/frontier    (geometry_msgs/PoseArray)   fronteras detectadas (RViz)

Servicios llamados:
  /slam/save_map   (std_srvs/Trigger)          guardar mapa al terminar
"""

import math
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg      import OccupancyGrid, Odometry
from geometry_msgs.msg import Pose2D, PoseArray, Pose, Point, Quaternion
from std_msgs.msg      import String
from std_srvs.srv      import Trigger


# ── Estados del explorador ────────────────────────────────────────────────────

IDLE          = "IDLE"
DETECTING     = "DETECTING"
NAVIGATING    = "NAVIGATING"
WAITING       = "WAITING"
MAPPING_DONE  = "MAPPING_DONE"
SAVING_MAP    = "SAVING_MAP"


class IBAExplorer(Node):
    """
    Explorador de fronteras que dirige el SLAM autónomo del Puzzlebot.
    Compatible con el pipeline: GoToGoal → /cmd_raw → BugReflex → /cmd_vel.
    """

    def __init__(self):
        super().__init__('iba_explorer')

        # ── Parámetros ────────────────────────────────────────────────────
        self.declare_parameter('min_frontier_size',   8)      # celdas mín. para considerar frontera
        self.declare_parameter('goal_reached_dist',   0.25)   # [m] distancia para considerar llegada
        self.declare_parameter('nav_timeout_s',       30.0)   # timeout por waypoint [s]
        self.declare_parameter('replan_interval_s',   8.0)    # replanning periódico [s]
        self.declare_parameter('occupied_threshold',  65)     # umbral ocupación OccupancyGrid
        self.declare_parameter('map_done_ratio',      0.0)    # ratio libre/total para terminar (0=solo fronteras)
        self.declare_parameter('inflation_cells',     3)      # celdas de inflado para seguridad
        self.declare_parameter('auto_save',           True)   # guardar mapa al terminar

        self.min_frontier  = self.get_parameter('min_frontier_size').value
        self.goal_dist     = self.get_parameter('goal_reached_dist').value
        self.nav_timeout   = self.get_parameter('nav_timeout_s').value
        self.replan_iv     = self.get_parameter('replan_interval_s').value
        self.occ_thresh    = self.get_parameter('occupied_threshold').value
        self.auto_save     = self.get_parameter('auto_save').value

        # ── Estado interno ────────────────────────────────────────────────
        self._state          = IDLE
        self._map_msg        = None
        self._map_lock       = threading.Lock()
        self.robot_x         = 0.0
        self.robot_y         = 0.0
        self._current_goal   = None     # (gx, gy)
        self._goal_sent_time = 0.0
        self._last_replan    = 0.0
        self._visited_goals  = []       # evitar revisitar fronteras idénticas
        self._astar_status   = "IDLE"

        # ── QoS para mapa latched ─────────────────────────────────────────
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ── Subscriptores ─────────────────────────────────────────────────
        self.create_subscription(OccupancyGrid, '/slam_map',    self._cb_map,    map_qos)
        self.create_subscription(Odometry,      '/odom',        self._cb_odom,   10)
        self.create_subscription(String,        '/astar/status', self._cb_astar, 10)

        # ── Publicadores ──────────────────────────────────────────────────
        self._pub_goal     = self.create_publisher(Pose2D,    '/astar/goal',    10)
        self._pub_status   = self.create_publisher(String,    '/iba/status',    10)
        self._pub_frontier = self.create_publisher(PoseArray, '/iba/frontier',  10)

        # ── Cliente de servicio save_map ──────────────────────────────────
        self._save_client = self.create_client(Trigger, '/slam/save_map')

        # ── Timer principal ───────────────────────────────────────────────
        self.create_timer(1.0, self._loop)

        self.get_logger().info('[IBAExplorer] Listo. Esperando primer mapa en /slam_map...')
        self._set_state(IDLE)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cb_map(self, msg: OccupancyGrid):
        with self._map_lock:
            self._map_msg = msg
        if self._state == IDLE:
            self.get_logger().info('[IBAExplorer] Mapa recibido — iniciando exploración.')
            self._set_state(DETECTING)

    def _cb_odom(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

    def _cb_astar(self, msg: String):
        self._astar_status = msg.data

    # ── Loop principal ─────────────────────────────────────────────────────────

    def _loop(self):
        now = time.monotonic()

        if self._state == IDLE or self._state == MAPPING_DONE:
            return

        if self._state == SAVING_MAP:
            return

        # ── Comprobar si llegamos al objetivo ─────────────────────────────
        if self._state == NAVIGATING and self._current_goal:
            gx, gy = self._current_goal
            dist = math.hypot(gx - self.robot_x, gy - self.robot_y)

            goal_reached = (
                dist < self.goal_dist or
                self._astar_status == 'GOAL_REACHED'
            )
            timed_out = (now - self._goal_sent_time) > self.nav_timeout

            if goal_reached:
                self.get_logger().info(
                    f'[IBAExplorer] ✅ Frontera alcanzada ({gx:.2f}, {gy:.2f}). '
                    f'Buscando siguiente...')
                self._visited_goals.append(self._current_goal)
                self._current_goal = None
                self._set_state(DETECTING)

            elif timed_out:
                self.get_logger().warn(
                    f'[IBAExplorer] ⏱ Timeout navegando a ({gx:.2f}, {gy:.2f}). '
                    f'Marcando como visitada y replanificando.')
                self._visited_goals.append(self._current_goal)
                self._current_goal = None
                self._set_state(DETECTING)

        # ── Replanning periódico (mapa cambia → fronteras cambian) ────────
        if self._state == NAVIGATING:
            if (now - self._last_replan) > self.replan_iv:
                self.get_logger().info('[IBAExplorer] Replanificación periódica...')
                self._set_state(DETECTING)

        # ── Detectar y enviar nueva frontera ──────────────────────────────
        if self._state == DETECTING:
            self._detect_and_send()

    # ── Detección de fronteras ─────────────────────────────────────────────────

    def _detect_and_send(self):
        with self._map_lock:
            if self._map_msg is None:
                return
            map_msg = self._map_msg

        meta   = map_msg.info
        res    = meta.resolution
        ox     = meta.origin.position.x
        oy     = meta.origin.position.y
        w      = meta.width
        h      = meta.height

        grid = np.array(map_msg.data, dtype=np.int8).reshape((h, w))

        # ── 1. Máscara de celdas libres y desconocidas ────────────────────
        free    = (grid == 0)
        unknown = (grid == -1)
        occ     = (grid >= self.occ_thresh)

        # ── 2. Detectar fronteras: libre con vecino desconocido ───────────
        frontier_mask = np.zeros((h, w), dtype=bool)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            shifted_unk = np.roll(unknown, (dr, dc), axis=(0, 1))
            frontier_mask |= (free & shifted_unk)

        # Excluir celdas cerca de obstáculos (inflado de seguridad)
        infl = self.get_parameter('inflation_cells').value
        danger = np.zeros((h, w), dtype=bool)
        for dr in range(-infl, infl+1):
            for dc in range(-infl, infl+1):
                danger |= np.roll(occ, (dr, dc), axis=(0, 1))
        frontier_mask &= ~danger

        frontier_cells = list(zip(*np.where(frontier_mask)))
        if not frontier_cells:
            self.get_logger().info('[IBAExplorer] ✅ Sin fronteras — mapa completo.')
            self._set_state(MAPPING_DONE)
            if self.auto_save:
                self._trigger_save()
            return

        # ── 3. Agrupar fronteras por conectividad (BFS simple) ────────────
        visited  = set()
        clusters = []

        for seed in frontier_cells:
            if seed in visited:
                continue
            cluster = []
            queue   = [seed]
            visited.add(seed)
            while queue:
                r, c = queue.pop()
                cluster.append((r, c))
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),
                                (-1,-1),(-1,1),(1,-1),(1,1)]:
                    nr, nc = r+dr, c+dc
                    nb = (nr, nc)
                    if (0 <= nr < h and 0 <= nc < w and
                            nb not in visited and frontier_mask[nr, nc]):
                        visited.add(nb)
                        queue.append(nb)
            if len(cluster) >= self.min_frontier:
                clusters.append(cluster)

        if not clusters:
            self.get_logger().warn('[IBAExplorer] Fronteras muy pequeñas — posiblemente mapa casi completo.')
            self._set_state(MAPPING_DONE)
            if self.auto_save:
                self._trigger_save()
            return

        # ── 4. Calcular centroides y elegir la más cercana ────────────────
        robot_col = int((self.robot_x - ox) / res)
        robot_row = int((self.robot_y - oy) / res)

        best_goal  = None
        best_dist  = float('inf')
        best_size  = 0
        centroids  = []

        for cluster in clusters:
            rows = [r for r, c in cluster]
            cols = [c for r, c in cluster]
            cr   = int(np.mean(rows))
            cc   = int(np.mean(cols))

            gx = ox + (cc + 0.5) * res
            gy = oy + (cr + 0.5) * res
            centroids.append((gx, gy))

            # Saltar fronteras ya visitadas (radio de tolerancia)
            already_visited = any(
                math.hypot(gx - vx, gy - vy) < 0.40
                for vx, vy in self._visited_goals
            )
            if already_visited:
                continue

            d = math.hypot(cc - robot_col, cr - robot_row) * res
            # Prefiere fronteras grandes y cercanas
            score = d - 0.3 * len(cluster) * res
            if score < best_dist:
                best_dist  = score
                best_goal  = (gx, gy)
                best_size  = len(cluster)

        # ── 5. Publicar fronteras en RViz ─────────────────────────────────
        self._publish_frontiers(centroids, map_msg.header.frame_id)

        if best_goal is None:
            self.get_logger().info('[IBAExplorer] Todas las fronteras ya visitadas — mapa completo.')
            self._set_state(MAPPING_DONE)
            if self.auto_save:
                self._trigger_save()
            return

        gx, gy = best_goal
        self.get_logger().info(
            f'[IBAExplorer] → Frontera seleccionada: ({gx:.2f}, {gy:.2f}) '
            f'| tamaño={best_size} celdas | dist={best_dist:.2f}m')

        # ── 6. Enviar objetivo ────────────────────────────────────────────
        goal_msg = Pose2D()
        goal_msg.x = gx
        goal_msg.y = gy
        self._pub_goal.publish(goal_msg)

        self._current_goal   = best_goal
        self._goal_sent_time = time.monotonic()
        self._last_replan    = time.monotonic()
        self._set_state(NAVIGATING)

    # ── Guardar mapa ──────────────────────────────────────────────────────────

    def _trigger_save(self):
        self._set_state(SAVING_MAP)
        if not self._save_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('[IBAExplorer] Servicio /slam/save_map no disponible.')
            return

        req = Trigger.Request()
        future = self._save_client.call_async(req)
        future.add_done_callback(self._cb_save_done)

    def _cb_save_done(self, future):
        try:
            resp = future.result()
            if resp.success:
                self.get_logger().info(f'[IBAExplorer] 💾 Mapa guardado: {resp.message}')
            else:
                self.get_logger().error(f'[IBAExplorer] Error guardando mapa: {resp.message}')
        except Exception as e:
            self.get_logger().error(f'[IBAExplorer] Excepción en save_map: {e}')
        self._set_state(MAPPING_DONE)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _publish_frontiers(self, centroids, frame_id):
        msg = PoseArray()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id if frame_id else 'map'
        for gx, gy in centroids:
            p = Pose()
            p.position.x = gx
            p.position.y = gy
            p.orientation.w = 1.0
            msg.poses.append(p)
        self._pub_frontier.publish(msg)

    def _set_state(self, state: str):
        if state != self._state:
            self.get_logger().info(f'[IBAExplorer] Estado: {self._state} → {state}')
            self._state = state
        s = String()
        s.data = state
        self._pub_status.publish(s)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = IBAExplorer()
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
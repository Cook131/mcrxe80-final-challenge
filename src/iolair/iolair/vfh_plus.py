#!/usr/bin/env python3
"""
vfh_plus.py  —  Capa Reflejo de Seguridad v1.0  (VFH+ Obstacle Avoidance)
==========================================================================
Drop-in replacement de bug_tangent.py.

Misma interfaz de topics:
  SUB:  /scan            (sensor_msgs/LaserScan)
        /odom            (nav_msgs/Odometry)
        /cmd_raw         (geometry_msgs/Twist)   — comandos del GoToGoal
        /astar/goal      (geometry_msgs/Pose2D)
        /astar/status    (std_msgs/String)
        /map             (nav_msgs/OccupancyGrid) — opcional, para filtrar paredes

  PUB:  /cmd_vel         (geometry_msgs/Twist)
        /reflex_status   (std_msgs/String)        — PASS / BRAKE / VFH_STEER / REFLEX_STOP
        /replan_trigger  (geometry_msgs/Pose2D)
        /nav_pause       (std_msgs/Bool)

Algoritmo VFH+ (Vector Field Histogram Plus):
  1. Construye histograma polar de densidad de obstáculos (POD) desde /scan.
  2. Suaviza el POD con ventana gaussiana para reducir ruido LiDAR.
  3. Umbraliza: sectores con densidad < threshold son "libres", el resto "bloqueados".
  4. Agrupa sectores libres en valles candidatos.
  5. Selecciona el valle cuya dirección central minimiza costo(desviación_al_goal, cambio_de_heading).
  6. Genera cmd_vel con control proporcional hacia esa dirección.
  7. Si ningún valle viable → REFLEX_STOP.
"""

import math
import time
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Pose2D, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String

# ── Estados ───────────────────────────────────────────────────────────────────
PASS_THROUGH  = "PASS"
PREDICTIVE_BRAKE = "BRAKE"
VFH_STEER     = "TANGENT_WALL"   # mismo valor que bug_tangent para compatibilidad
REFLEX_STOP   = "REFLEX_STOP"


class VFHPlus(Node):
    """Nodo VFH+ que reemplaza TangentBugReflex manteniendo la misma interfaz."""

    def __init__(self):
        super().__init__('vfh_plus')

        # ── Geometría del robot ───────────────────────────────────────────
        self.declare_parameter('robot_radius_m',     0.18)   # [m]
        self.declare_parameter('safety_margin_m',    0.10)   # buffer extra sobre radio

        # ── Distancias de reacción ────────────────────────────────────────
        self.declare_parameter('warn_dist',          0.65)   # [m] empieza a frenar
        self.declare_parameter('emergency_dist',     0.35)   # [m] activa VFH
        self.declare_parameter('stop_dist',          0.14)   # [m] stop total
        self.declare_parameter('hysteresis',         0.06)   # [m] hyst de modos

        # ── VFH+ — histograma ─────────────────────────────────────────────
        self.declare_parameter('num_sectors',        180)    # resolución angular (2°/sector)
        self.declare_parameter('hist_threshold',     8.0)    # densidad umbral libre/bloqueado
        self.declare_parameter('smoothing_window',   5)      # sectores a cada lado para smooth
        self.declare_parameter('influence_radius_m', 1.20)   # solo lecturas dentro de este radio
        self.declare_parameter('a_weight',           1.0)    # peso distancia al obstáculo
        self.declare_parameter('b_weight',           1.0)    # decaimiento con distancia

        # ── VFH+ — selección de valle ─────────────────────────────────────
        self.declare_parameter('valley_min_width',   3)      # sectores mínimos para un valle
        self.declare_parameter('wide_valley_thresh', 12)     # sectores: ancho → selección de borde
        self.declare_parameter('cost_alpha',         5.0)    # peso desviación al goal
        self.declare_parameter('cost_beta',          2.0)    # peso cambio de heading actual
        self.declare_parameter('cost_gamma',         2.0)    # peso desviación al cmd_raw original

        # ── Control de salida ─────────────────────────────────────────────
        self.declare_parameter('max_v',              0.22)   # [m/s] vel lineal máxima VFH
        self.declare_parameter('min_v',              0.04)   # [m/s] vel lineal mínima
        self.declare_parameter('max_w',              1.20)   # [rad/s] vel angular máxima
        self.declare_parameter('kp_heading',         2.00)   # ganancia proporcional angular
        self.declare_parameter('speed_reduction',    0.60)   # factor de v al girar fuerte

        # ── LiDAR ─────────────────────────────────────────────────────────
        self.declare_parameter('lidar_yaw_offset',   math.pi)  # offset montaje LiDAR
        self.declare_parameter('replan_cooldown_s',  2.0)

        # ── Leer parámetros ───────────────────────────────────────────────
        self._robot_r      = float(self.get_parameter('robot_radius_m').value)
        self._safety_m     = float(self.get_parameter('safety_margin_m').value)
        self._d_safe       = self._robot_r + self._safety_m   # distancia total de seguridad

        self.warn_d        = float(self.get_parameter('warn_dist').value)
        self.emg_d         = float(self.get_parameter('emergency_dist').value)
        self.stop_d        = float(self.get_parameter('stop_dist').value)
        self.hyst          = float(self.get_parameter('hysteresis').value)

        self._N            = int(self.get_parameter('num_sectors').value)
        self._threshold    = float(self.get_parameter('hist_threshold').value)
        self._smooth_w     = int(self.get_parameter('smoothing_window').value)
        self._d_max        = float(self.get_parameter('influence_radius_m').value)
        self._a            = float(self.get_parameter('a_weight').value)
        self._b            = float(self.get_parameter('b_weight').value)

        self._valley_min   = int(self.get_parameter('valley_min_width').value)
        self._wide_thr     = int(self.get_parameter('wide_valley_thresh').value)
        self._c_alpha      = float(self.get_parameter('cost_alpha').value)
        self._c_beta       = float(self.get_parameter('cost_beta').value)
        self._c_gamma      = float(self.get_parameter('cost_gamma').value)

        self._max_v        = float(self.get_parameter('max_v').value)
        self._min_v        = float(self.get_parameter('min_v').value)
        self._max_w        = float(self.get_parameter('max_w').value)
        self._kp           = float(self.get_parameter('kp_heading').value)
        self._spd_red      = float(self.get_parameter('speed_reduction').value)

        self._lidar_yaw    = float(self.get_parameter('lidar_yaw_offset').value)
        self._replan_cd    = float(self.get_parameter('replan_cooldown_s').value)

        # ── Estado interno ────────────────────────────────────────────────
        self._mode           = PASS_THROUGH
        self._scan: Optional[LaserScan] = None
        self._last_cmd       = Twist()
        self._robot_x        = 0.0
        self._robot_y        = 0.0
        self._robot_yaw      = 0.0
        self._goal_x: Optional[float] = None
        self._goal_y: Optional[float] = None
        self._replanning     = False
        self._last_replan_ts = -999.0
        self._sector_rad     = (2.0 * math.pi) / self._N   # tamaño de cada sector [rad]
        self._collect_bypass = False   # True mientras qr_align esté en ALIGN/ADVANCE/etc.

        # Mapa estático (filtro opcional)
        self.grid_map     = None
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0
        self.map_res      = 0.05
        self.map_width    = 0
        self.map_height   = 0

        # ── Suscriptores ──────────────────────────────────────────────────
        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.create_subscription(LaserScan,     '/scan',           self._cb_scan,    scan_qos)
        self.create_subscription(Odometry,      '/odom',           self._cb_odom,    10)
        self.create_subscription(Twist,         '/cmd_raw',        self._cb_cmd,     10)
        self.create_subscription(Pose2D,        '/astar/goal',     self._cb_goal,    10)
        self.create_subscription(String,        '/astar/status',   self._cb_astar,   10)
        self.create_subscription(OccupancyGrid, '/map',            self._cb_map,     10)
        self.create_subscription(Bool,          '/align/active', self._cb_collect, 10)

        # ── Publicadores ──────────────────────────────────────────────────
        self._pub_cmd    = self.create_publisher(Twist,  '/cmd_vel',        10)
        self._pub_status = self.create_publisher(String, '/reflex_status',  10)
        self._pub_replan = self.create_publisher(Pose2D, '/replan_trigger', 10)
        self._pub_pause  = self.create_publisher(Bool,   '/nav_pause',      10)

        self.create_timer(0.05, self._loop)   # 20 Hz
        self.get_logger().info('[VFH+] Nodo iniciado — drop-in de bug_tangent')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cb_scan(self, msg: LaserScan):
        self._scan = msg

    def _cb_odom(self, msg: Odometry):
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._robot_yaw = math.atan2(siny, cosy)

    def _cb_cmd(self, msg: Twist):
        self._last_cmd = msg

    def _cb_goal(self, msg: Pose2D):
        self._goal_x = msg.x
        self._goal_y = msg.y

    def _cb_astar(self, msg: String):
        if msg.data in ('EXECUTING', 'GOAL_REACHED', 'NO_PATH'):
            self._replanning = False

    def _cb_map(self, msg: OccupancyGrid):
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y
        self.map_res      = msg.info.resolution
        self.map_width    = msg.info.width
        self.map_height   = msg.info.height
        self.grid_map     = np.array(msg.data, dtype=np.int8).reshape(
            (msg.info.height, msg.info.width))

    def _cb_collect(self, msg: Bool):
        """
        Recibe la flag de qr_align_node.
        True  → modo COLLECT_BYPASS: el VFH deja pasar cmd_raw sin intervenir.
        False → regresa al comportamiento normal de evasión.
        """
        prev = self._collect_bypass
        self._collect_bypass = msg.data
        if self._collect_bypass != prev:
            _bp_str = 'ON  — evasión inhibida para recolección' if self._collect_bypass else 'OFF — evasión reactiva normal'
            self.get_logger().warn(f'[VFH+] collect_bypass → {_bp_str}')

    # ── Filtro de mapa (ignora paredes conocidas igual que bug_tangent) ────────

    def _filter_scan(self, scan: LaserScan) -> np.ndarray:
        """Retorna rangos donde paredes conocidas del mapa se reemplazan por inf."""
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        if self.grid_map is None:
            return ranges

        angles = (scan.angle_min
                  + np.arange(len(ranges), dtype=np.float32) * scan.angle_increment
                  + self._lidar_yaw)
        x_w = self._robot_x + ranges * np.cos(angles + self._robot_yaw)
        y_w = self._robot_y + ranges * np.sin(angles + self._robot_yaw)

        ix = np.floor((x_w - self.map_origin_x) / self.map_res).astype(int)
        iy = np.floor((y_w - self.map_origin_y) / self.map_res).astype(int)

        valid = (
            (ix >= 0) & (ix < self.map_width) &
            (iy >= 0) & (iy < self.map_height) &
            (ranges > scan.range_min) & (ranges < scan.range_max)
        )
        is_known = np.zeros_like(ranges, dtype=bool)
        is_known[valid] = self.grid_map[iy[valid], ix[valid]] > 50

        filtered = ranges.copy()
        filtered[is_known] = np.inf
        return filtered

    # ── Construcción del histograma polar ─────────────────────────────────────

    def _build_histogram(self, scan: LaserScan, ranges: np.ndarray) -> np.ndarray:
        """
        Construye el Primary Polar Obstacle Density (POD).
        Cada lectura contribuye a su sector con peso  w = (a - b*d)^2
        donde d es la distancia al obstáculo.
        Solo lecturas dentro de _d_max con d > stop_d.
        """
        pod = np.zeros(self._N, dtype=np.float64)

        angles_raw = (scan.angle_min
                      + np.arange(len(ranges), dtype=np.float64) * scan.angle_increment
                      + self._lidar_yaw)

        # Solo lecturas válidas dentro de radio de influencia
        valid = (
            (ranges > self.stop_d) &
            (ranges <= self._d_max) &
            (ranges > scan.range_min) &
            np.isfinite(ranges)
        )
        if not np.any(valid):
            return pod

        d_valid     = ranges[valid].astype(np.float64)
        angles_valid = angles_raw[valid]

        # Ángulo en frame del robot: restar yaw para obtener ángulo relativo al frente
        angles_robot = angles_valid + self._robot_yaw
        # Normalizar a [0, 2π)
        angles_norm  = np.mod(angles_robot, 2.0 * math.pi)

        # Sector index
        sector_idx = (angles_norm / self._sector_rad).astype(int)
        sector_idx = np.clip(sector_idx, 0, self._N - 1)

        # Peso: (a - b*d)^2 — más alto cuanto más cerca
        weights = (self._a - self._b * d_valid) ** 2
        weights = np.maximum(weights, 0.0)

        np.add.at(pod, sector_idx, weights)
        return pod

    def _smooth_histogram(self, pod: np.ndarray) -> np.ndarray:
        """Suaviza con ventana gaussiana circular."""
        w = self._smooth_w
        if w <= 0:
            return pod.copy()
        # Kernel gaussiano normalizado
        k_range = np.arange(-w, w + 1)
        kernel  = np.exp(-0.5 * (k_range / max(w / 2.0, 0.5)) ** 2)
        kernel /= kernel.sum()
        # Convolución circular
        return np.convolve(
            np.tile(pod, 3), kernel, mode='same'
        )[self._N: 2 * self._N]

    # ── Selección del valle objetivo ──────────────────────────────────────────

    def _sector_of_angle(self, angle_world: float) -> int:
        """Convierte ángulo en frame mundo a índice de sector (frente = sector 0)."""
        rel = angle_world - self._robot_yaw
        rel = math.atan2(math.sin(rel), math.cos(rel))   # normaliza a (-π, π]
        norm = rel % (2.0 * math.pi)
        return int(norm / self._sector_rad) % self._N

    def _sector_to_world_angle(self, sector: int) -> float:
        """Convierte índice de sector a ángulo en frame mundo."""
        rel = sector * self._sector_rad
        if rel > math.pi:
            rel -= 2.0 * math.pi
        return self._robot_yaw + rel

    def _find_best_valley(self, smooth_pod: np.ndarray) -> Optional[float]:
        """
        Encuentra todos los valles libres y retorna la dirección (ángulo mundo)
        del valle con menor costo.

        Costo = α·Δgoal + β·Δheading_actual + γ·Δcmd_raw
        """
        # Máscara binaria: True = libre
        free = smooth_pod < self._threshold

        if not np.any(free):
            return None

        # Sector de la dirección objetivo (goal)
        goal_sector: Optional[int] = None
        if self._goal_x is not None:
            goal_ang    = math.atan2(self._goal_y - self._robot_y,
                                     self._goal_x - self._robot_x)
            goal_sector = self._sector_of_angle(goal_ang)

        # Sector del heading actual del robot (frente = sector 0)
        current_sector = 0  # "recto al frente" en el frame del robot

        # Sector del cmd_raw original de GoToGoal
        cmd_raw_sector = 0
        if abs(self._last_cmd.angular.z) > 0.01 or abs(self._last_cmd.linear.x) > 0.01:
            heading_cmd = math.atan2(self._last_cmd.angular.z,
                                     max(self._last_cmd.linear.x, 0.001))
            cmd_raw_sector = self._sector_of_angle(self._robot_yaw + heading_cmd)

        # Encontrar valles (grupos contiguos de sectores libres)
        valleys = []
        in_valley = False
        start = 0
        # Doble vuelta para capturar valles circulares
        doubled = np.concatenate([free, free])
        for i in range(2 * self._N):
            if doubled[i] and not in_valley:
                in_valley = True
                start = i
            elif not doubled[i] and in_valley:
                in_valley = False
                end = i - 1
                width = end - start + 1
                if width >= self._valley_min:
                    valleys.append((start % self._N, end % self._N, width))

        if not valleys:
            return None

        # Evaluar costo de cada valle
        best_cost    = float('inf')
        best_angle   = None

        def _sector_diff(s1: int, s2: int) -> float:
            """Diferencia circular mínima en sectores → ángulo [0, π]."""
            d = abs(s1 - s2) % self._N
            if d > self._N // 2:
                d = self._N - d
            return d * self._sector_rad

        for s_start, s_end, width in valleys:
            # Dirección candidata dentro del valle
            if width >= self._wide_thr:
                # Valle ancho: elige el borde más cercano al goal (VFH+ paper)
                if goal_sector is not None:
                    d_start = _sector_diff(s_start, goal_sector)
                    d_end   = _sector_diff(s_end,   goal_sector)
                    # Borde en dirección del goal + margen de seguridad
                    margin = max(1, int(self._valley_min // 2))
                    if d_start <= d_end:
                        candidate = (s_start + margin) % self._N
                    else:
                        candidate = (s_end   - margin) % self._N
                else:
                    candidate = int((s_start + s_end) / 2) % self._N
            else:
                # Valle estrecho: usa el centro
                candidate = int((s_start + s_end) / 2) % self._N

            # Calcular costo
            cost = 0.0
            if goal_sector is not None:
                cost += self._c_alpha * _sector_diff(candidate, goal_sector)
            cost += self._c_beta  * _sector_diff(candidate, current_sector)
            cost += self._c_gamma * _sector_diff(candidate, cmd_raw_sector)

            if cost < best_cost:
                best_cost  = cost
                best_angle = self._sector_to_world_angle(candidate)

        return best_angle

    # ── Generación del comando ────────────────────────────────────────────────

    def _cmd_toward(self, target_angle: float, front_dist: float) -> Twist:
        """Genera Twist para navegar hacia target_angle con velocidad adaptativa."""
        # Error angular
        err = math.atan2(math.sin(target_angle - self._robot_yaw),
                         math.cos(target_angle - self._robot_yaw))

        w = float(np.clip(self._kp * err, -self._max_w, self._max_w))

        # Reducir velocidad lineal si giro es pronunciado
        turn_factor  = 1.0 - self._spd_red * (abs(err) / math.pi)
        turn_factor  = max(0.0, turn_factor)

        # Reducir también si hay obstáculo cerca al frente
        if math.isfinite(front_dist):
            dist_factor = np.clip(
                (front_dist - self.stop_d) / max(self.emg_d - self.stop_d, 1e-3),
                0.1, 1.0
            )
        else:
            dist_factor = 1.0

        v_des = self._last_cmd.linear.x   # respeta velocidad pedida por GoToGoal
        v_des = float(np.clip(v_des, 0.0, self._max_v))
        v     = v_des * turn_factor * dist_factor
        v     = max(self._min_v, v) if v_des > 0.01 else 0.0

        cmd = Twist()
        cmd.linear.x  = v
        cmd.angular.z = w
        return cmd

    def _front_dist(self, scan: LaserScan, ranges: np.ndarray) -> float:
        """Distancia mínima en el sector frontal (±30°)."""
        angles = (scan.angle_min
                  + np.arange(len(ranges), dtype=np.float32) * scan.angle_increment
                  + self._lidar_yaw)
        diff = np.arctan2(np.sin(angles), np.cos(angles))
        mask = (np.abs(diff) <= math.radians(30.0)) & np.isfinite(ranges)
        valid = ranges[mask]
        return float(np.min(valid)) if valid.size > 0 else float('inf')

    # ── Loop principal ────────────────────────────────────────────────────────

    def _loop(self):
        scan = self._scan
        if scan is None:
            self._publish(self._last_cmd, PASS_THROUGH)
            return

        # ── COLLECT_BYPASS — qr_align tiene el control fino ───────────────
        # Mientras la operación de recolección esté activa, el VFH pasa
        # cmd_raw sin tocar nada. El robot ya está muy cerca del objetivo
        # y el QR (a ~28 cm) dispararía falsos positivos de obstáculo.
        if self._collect_bypass:
            if self._mode != PASS_THROUGH:
                self.get_logger().info(
                    '[VFH+] COLLECT_BYPASS activo — cediendo control a qr_align')
                self._mode = PASS_THROUGH
            self._pub_cmd.publish(self._last_cmd)
            s = String(); s.data = 'COLLECT_BYPASS'
            self._pub_status.publish(s)
            return

        ranges       = self._filter_scan(scan)
        front        = self._front_dist(scan, ranges)
        now          = time.monotonic()
        in_cur_mode  = lambda m: self._mode == m

        # ── P1: REFLEX_STOP ───────────────────────────────────────────────
        stop_thr = self.stop_d + (self.hyst if in_cur_mode(REFLEX_STOP) else 0.0)
        if front <= stop_thr:
            self._publish(Twist(), REFLEX_STOP)
            return

        # ── P2: VFH_STEER — evasión activa ───────────────────────────────
        emg_thr = self.emg_d + (self.hyst if in_cur_mode(VFH_STEER) else 0.0)

        if front <= emg_thr:
            if not in_cur_mode(VFH_STEER):
                self._pub_pause.publish(self._bool(True))
                self.get_logger().warn(
                    f'[VFH+] Obstáculo a {front:.2f}m — activando evasión VFH+')

            pod      = self._build_histogram(scan, ranges)
            smooth   = self._smooth_histogram(pod)
            best_dir = self._find_best_valley(smooth)

            if best_dir is None:
                # Sin valle disponible → stop
                self.get_logger().warn('[VFH+] Sin valle libre — REFLEX_STOP',
                                       throttle_duration_sec=0.5)
                self._publish(Twist(), REFLEX_STOP)
                return

            cmd = self._cmd_toward(best_dir, front)
            self._publish(cmd, VFH_STEER)
            return

        # ── P3: PREDICTIVE_BRAKE ──────────────────────────────────────────
        warn_thr  = self.warn_d + (self.hyst if in_cur_mode(PREDICTIVE_BRAKE) else 0.0)
        incoming_v = self._last_cmd.linear.x

        if front <= warn_thr and incoming_v > 0.0:
            # Si ya hay camino libre según VFH, mantener velocidad normal
            pod      = self._build_histogram(scan, ranges)
            smooth   = self._smooth_histogram(pod)
            best_dir = self._find_best_valley(smooth)

            if best_dir is not None:
                # Hay camino: solo frenar un poco, orientar suavemente
                t   = 1.0 - (front - self.emg_d) / max(self.warn_d - self.emg_d, 1e-6)
                t   = float(np.clip(t, 0.0, 0.5))   # máximo 50% reducción en zona warn
                cmd = Twist()
                cmd.linear.x  = incoming_v * (1.0 - t)
                cmd.angular.z = self._last_cmd.angular.z
                self._publish(cmd, PREDICTIVE_BRAKE)
            else:
                # No hay camino visible: frenar más agresivo
                t   = 1.0 - (front - self.emg_d) / max(self.warn_d - self.emg_d, 1e-6)
                t   = float(np.clip(t, 0.0, 1.0))
                cmd = Twist()
                cmd.linear.x  = incoming_v * (1.0 - t)
                cmd.angular.z = self._last_cmd.angular.z
                self._publish(cmd, PREDICTIVE_BRAKE)
            return

        # ── P4: PASS_THROUGH ──────────────────────────────────────────────
        if self._mode in (VFH_STEER, REFLEX_STOP):
            # Saliendo de evasión o paro de emergencia → replanificar
            self._maybe_replan(now)
            self._pub_pause.publish(self._bool(False))
            self.get_logger().info('[VFH+] Camino despejado — retomando navegación')

        self._publish(self._last_cmd, PASS_THROUGH)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _maybe_replan(self, now: float):
        if self._replanning:
            return
        if (now - self._last_replan_ts) < self._replan_cd:
            return
        msg   = Pose2D()
        msg.x = self._robot_x
        msg.y = self._robot_y
        self._pub_replan.publish(msg)
        self._replanning     = True
        self._last_replan_ts = now
        self.get_logger().warn(
            f'[VFH+] Disparando replanificación desde '
            f'({self._robot_x:.2f}, {self._robot_y:.2f})')

    def _bool(self, val: bool) -> Bool:
        m = Bool()
        m.data = val
        return m

    def _publish(self, cmd: Twist, mode: str):
        if mode != self._mode:
            if mode not in (PASS_THROUGH, PREDICTIVE_BRAKE):
                self.get_logger().warn(
                    f'[VFH+] {self._mode} → {mode}',
                    throttle_duration_sec=0.4)
            if self._mode in (VFH_STEER, REFLEX_STOP) and mode == PASS_THROUGH:
                self._pub_pause.publish(self._bool(False))
            self._mode = mode

        self._pub_cmd.publish(cmd)
        s      = String()
        s.data = mode
        self._pub_status.publish(s)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = VFHPlus()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._pub_cmd.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
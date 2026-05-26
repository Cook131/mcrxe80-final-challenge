#!/usr/bin/env python3
"""
aruco_localizer.py — Corrección de drift por landmark anchoring
================================================================

Problema que resuelve
---------------------
El robot navega por una pista con ArUcos en posiciones DESCONOCIDAS.
La odometría acumula drift con el tiempo. Cada vez que el robot ve un
ArUco, podemos corregir ese drift.

Método: Landmark Anchoring
--------------------------
Primera vez que se ve un marcador (ID nuevo):
  → Calcular su posición global usando la pose EKF actual + tvec de cámara
  → Guardar esa posición como "ancla" (landmark)

Siguientes veces que se ve el mismo marcador:
  → El robot debería estar a la misma distancia/ángulo del ancla
  → Calcular dónde DEBERÍA estar el robot dado el ancla y el tvec actual
  → Publicar esa pose como corrección para el EKF

Esto corrige el drift porque:
  - Si el robot drifteó a la derecha, al ver el mismo ArUco desde
    una posición estimada incorrecta, la corrección lo empuja de vuelta.

Flujo de datos
--------------
  aruco_detector.py  →  /aruco/id        (Int32)
                         /aruco/distance  (Float32)  distancia plano XZ [m]
                         /aruco/angle     (Float32)  bearing horizontal [°]
                         /aruco/waypoint  (PoseStamped) rvec+tvec completo

  puzzlebotOdometry  →  /odom  (Odometry)  pose EKF actual

  aruco_localizer    →  /aruco/pose   (PoseWithCovarianceStamped) → EKF
                         /aruco/debug  (String)

Integración con EKF
--------------------
  puzzlebotOdometry.py ya tiene el parche para suscribirse a /aruco/pose
  como tercer source (ARUCO_ACTIVE / ARUCO_PRIORITY).

Parámetros ROS
--------------
  camera_to_base_x   0.05   [m]   offset longitudinal cámara→base_link
  camera_to_base_y   0.00   [m]   offset lateral
  camera_to_base_z   0.10   [m]   altura cámara sobre el suelo
  anchor_min_dist    0.20   [m]   distancia mínima para anclar/corregir
  anchor_max_dist    3.50   [m]   distancia máxima para anclar/corregir
  anchor_reobserve   0.30   [m]   mínimo desplazamiento del robot para
                                  re-observar el mismo marcador (evita
                                  spamear correcciones estando quieto)
  r_base_pos         0.03   [m²]  varianza base de posición
  r_base_yaw         0.04   [rad²] varianza base de orientación
  distance_noise_k   0.025  factor de ruido proporcional a d²
  publish_rate       10.0   [Hz]
"""

import math
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import (
    PoseStamped, PoseWithCovarianceStamped, Quaternion
)
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Int32, String


# ─────────────────────────────────────────────────────────────────────────────
# Helpers matemáticos
# ─────────────────────────────────────────────────────────────────────────────

def normalize_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def euler_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw * 0.5)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    return q


def yaw_from_quaternion(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def pose_stamped_to_Rt(ps: PoseStamped):
    """PoseStamped (quaternión + posición) → R (3×3), t (3,)."""
    p  = ps.pose
    tx, ty, tz = p.position.x, p.position.y, p.position.z
    qx, qy, qz, qw = (p.orientation.x, p.orientation.y,
                       p.orientation.z, p.orientation.w)
    R = np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx**2+qy**2)],
    ], dtype=np.float64)
    return R, np.array([tx, ty, tz], dtype=np.float64)


def camera_tvec_to_robot_frame(tvec: np.ndarray, robot_yaw: float,
                                cam_x: float, cam_y: float) -> tuple[float, float]:
    """
    Convierte tvec (frame óptico cámara: X=derecha, Y=abajo, Z=frente)
    a desplazamiento en frame global (X=este, Y=norte).

    Pasos:
      1. Frame óptico → frame robot (X=adelante, Y=izquierda)
           dx_robot =  tvec[2]   (Z óptico = adelante del robot)
           dy_robot = -tvec[0]   (X óptico = derecha → izquierda negado)
      2. Frame robot → frame global (rotar por robot_yaw)
           dx_global = dx_robot·cos(yaw) - dy_robot·sin(yaw)
           dy_global = dx_robot·sin(yaw) + dy_robot·cos(yaw)
      3. Añadir offset de cámara a base_link
    """
    dx_robot = float(tvec[2])    # profundidad al marcador
    dy_robot = -float(tvec[0])   # desplazamiento lateral (izquierda positivo)

    cos_y = math.cos(robot_yaw)
    sin_y = math.sin(robot_yaw)

    # Posición del marcador en frame global desde la cámara
    dx_g = dx_robot * cos_y - dy_robot * sin_y
    dy_g = dx_robot * sin_y + dy_robot * cos_y

    # Offset cámara→base_link en frame global
    off_x = cam_x * cos_y - cam_y * sin_y
    off_y = cam_x * sin_y + cam_y * cos_y

    return dx_g - off_x, dy_g - off_y


# ─────────────────────────────────────────────────────────────────────────────
# Estructura de un landmark anclado
# ─────────────────────────────────────────────────────────────────────────────

class Landmark:
    """Posición global estimada de un marcador ArUco."""

    def __init__(self, pub_id: int, gx: float, gy: float,
                 robot_x: float, robot_y: float):
        self.pub_id    = pub_id
        self.gx        = gx        # posición global X del marcador [m]
        self.gy        = gy        # posición global Y del marcador [m]
        self.n_obs     = 1         # número de observaciones
        self.last_robot_x = robot_x  # última posición del robot al verlo
        self.last_robot_y = robot_y

    def update_position(self, gx: float, gy: float,
                        robot_x: float, robot_y: float):
        """
        Actualiza la posición del ancla con un promedio móvil suavizado.
        Cuantas más observaciones, menos peso a la nueva (el ancla se
        vuelve más estable con el tiempo).
        """
        alpha = 1.0 / (self.n_obs + 1)   # peso decreciente
        self.gx = (1 - alpha) * self.gx + alpha * gx
        self.gy = (1 - alpha) * self.gy + alpha * gy
        self.n_obs += 1
        self.last_robot_x = robot_x
        self.last_robot_y = robot_y


# ─────────────────────────────────────────────────────────────────────────────
# Nodo principal
# ─────────────────────────────────────────────────────────────────────────────

class ArucoLocalizerNode(Node):

    def __init__(self):
        super().__init__('aruco_localizer')

        # ── Parámetros ────────────────────────────────────────────────────
        self.declare_parameter('camera_to_base_x',  0.05)
        self.declare_parameter('camera_to_base_y',  0.00)
        self.declare_parameter('camera_to_base_z',  0.10)
        self.declare_parameter('anchor_min_dist',   0.20)
        self.declare_parameter('anchor_max_dist',   3.50)
        self.declare_parameter('anchor_reobserve',  0.30)
        self.declare_parameter('r_base_pos',        0.03)
        self.declare_parameter('r_base_yaw',        0.04)
        self.declare_parameter('distance_noise_k',  0.025)
        self.declare_parameter('publish_rate',     10.0)

        self._cam_x      = self.get_parameter('camera_to_base_x').value
        self._cam_y      = self.get_parameter('camera_to_base_y').value
        self._cam_z      = self.get_parameter('camera_to_base_z').value
        self._d_min      = self.get_parameter('anchor_min_dist').value
        self._d_max      = self.get_parameter('anchor_max_dist').value
        self._d_reobs    = self.get_parameter('anchor_reobserve').value
        self._r_pos      = self.get_parameter('r_base_pos').value
        self._r_yaw      = self.get_parameter('r_base_yaw').value
        self._k_dist     = self.get_parameter('distance_noise_k').value
        rate             = self.get_parameter('publish_rate').value

        # ── Estado ────────────────────────────────────────────────────────
        self._lock = threading.Lock()

        # Landmarks descubiertos: pub_id → Landmark
        self._landmarks: dict[int, Landmark] = {}

        # Última pose del EKF
        self._robot_x:   float = 0.0
        self._robot_y:   float = 0.0
        self._robot_yaw: float = 0.0
        self._odom_ready: bool = False

        # Última detección pendiente de procesar
        self._pending_id:   int            = -1
        self._pending_R:    np.ndarray | None = None
        self._pending_t:    np.ndarray | None = None
        self._pending_dist: float          = 0.0
        self._pending_ts:   float          = -1.0

        # Pose a publicar (None = no hay corrección nueva)
        self._correction: tuple[float, float, float, float] | None = None
        # (rx, ry, ryaw, distance)

        # ── Suscriptores ──────────────────────────────────────────────────
        self.create_subscription(
            Odometry, '/odom', self._cb_odom, 10)
        self.create_subscription(
            Int32, '/aruco/id', self._cb_id, 10)
        self.create_subscription(
            Float32, '/aruco/distance', self._cb_dist, 10)
        self.create_subscription(
            PoseStamped, '/aruco/waypoint', self._cb_waypoint, 10)

        # ── Publicadores ──────────────────────────────────────────────────
        self._pub_pose  = self.create_publisher(
            PoseWithCovarianceStamped, '/aruco/pose',  10)
        self._pub_debug = self.create_publisher(
            String, '/aruco/debug', 10)

        self.create_timer(1.0 / rate, self._process_and_publish)

        self.get_logger().info(
            'ArUco Localizer (landmark anchoring) listo\n'
            '  Primera detección de un ID → ancla su posición global\n'
            '  Re-detecciones            → corrección de drift al EKF\n'
            '  Publica → /aruco/pose  (PoseWithCovarianceStamped)'
        )

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _cb_odom(self, msg: Odometry):
        with self._lock:
            self._robot_x   = msg.pose.pose.position.x
            self._robot_y   = msg.pose.pose.position.y
            self._robot_yaw = yaw_from_quaternion(msg.pose.pose.orientation)
            self._odom_ready = True

    def _cb_id(self, msg: Int32):
        with self._lock:
            self._pending_id = msg.data

    def _cb_dist(self, msg: Float32):
        with self._lock:
            self._pending_dist = msg.data

    def _cb_waypoint(self, msg: PoseStamped):
        """Recibe rvec+tvec del marcador desde aruco_detector."""
        R, t = pose_stamped_to_Rt(msg)
        dist = float(np.linalg.norm(t))
        if not (self._d_min <= dist <= self._d_max):
            return
        with self._lock:
            self._pending_R  = R
            self._pending_t  = t.copy()
            self._pending_ts = time.monotonic()

    # ── Lógica principal ──────────────────────────────────────────────────

    def _process_and_publish(self):
        with self._lock:
            if not self._odom_ready:
                return
            if self._pending_id <= 0:
                return
            if self._pending_t is None:
                return
            if (time.monotonic() - self._pending_ts) > 0.3:
                return   # detección vieja

            pub_id   = self._pending_id
            t_vec    = self._pending_t.copy()
            dist     = float(np.linalg.norm(t_vec))
            rx_ekf   = self._robot_x
            ry_ekf   = self._robot_y
            yaw_ekf  = self._robot_yaw

        # Posición global del marcador según la pose EKF actual
        # (robot_x + desplazamiento al marcador en frame global)
        dx_g, dy_g = camera_tvec_to_robot_frame(
            t_vec, yaw_ekf, self._cam_x, self._cam_y
        )
        marker_gx = rx_ekf + dx_g
        marker_gy = ry_ekf + dy_g

        with self._lock:
            if pub_id not in self._landmarks:
                # ── PRIMERA VEZ: anclar este marcador ─────────────────────
                lm = Landmark(pub_id, marker_gx, marker_gy, rx_ekf, ry_ekf)
                self._landmarks[pub_id] = lm
                self.get_logger().info(
                    f'[ANCHOR] ID={pub_id} anclado en '
                    f'({marker_gx:.3f}, {marker_gy:.3f}) '
                    f'desde robot ({rx_ekf:.3f}, {ry_ekf:.3f}) '
                    f'd={dist:.3f}m'
                )
                self._correction = None   # primera vez no corregimos
                return

            lm = self._landmarks[pub_id]

            # ── RE-OBSERVACIÓN: verificar que el robot se movió ───────────
            moved = math.hypot(rx_ekf - lm.last_robot_x,
                               ry_ekf - lm.last_robot_y)
            if moved < self._d_reobs:
                return   # robot casi quieto, no spamear

            # ── CALCULAR CORRECCIÓN ───────────────────────────────────────
            # El marcador está en (lm.gx, lm.gy) en el mundo.
            # Dado el tvec actual (marcador relativo a cámara), el robot
            # DEBERÍA estar en:
            #   robot_corrected = landmark_pos - desplazamiento_al_marcador
            rx_corr = lm.gx - dx_g
            ry_corr = lm.gy - dy_g

            # Orientación: el eje Z de la cámara apunta al marcador.
            # bearing_global = dirección robot→marcador en el mapa
            bearing = math.atan2(dy_g, dx_g)
            # El robot mira en la dirección del marcador
            # (corregido por el ángulo lateral del tvec)
            lateral_angle = math.atan2(-float(t_vec[0]), float(t_vec[2]))
            ryaw_corr = normalize_angle(bearing - lateral_angle)

            # Actualizar ancla suavemente (promedio móvil)
            lm.update_position(marker_gx, marker_gy, rx_ekf, ry_ekf)

            drift = math.hypot(rx_corr - rx_ekf, ry_corr - ry_ekf)

            self.get_logger().info(
                f'[CORRECT] ID={pub_id} obs#{lm.n_obs} | '
                f'drift={drift:.3f}m | '
                f'EKF=({rx_ekf:.3f},{ry_ekf:.3f}) → '
                f'corr=({rx_corr:.3f},{ry_corr:.3f})'
            )

            self._correction = (rx_corr, ry_corr, ryaw_corr, dist)

        # ── Publicar corrección ───────────────────────────────────────────
        self._publish_correction()

    def _publish_correction(self):
        with self._lock:
            if self._correction is None:
                return
            rx, ry, ryaw, dist = self._correction
            self._correction = None

        # Covarianza adaptativa: más ruido a más distancia
        noise = self._k_dist * dist ** 2
        r_pos = self._r_pos + noise
        r_yaw = self._r_yaw + noise

        msg = PoseWithCovarianceStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'   # frame de odometría (no mapa global)

        msg.pose.pose.position.x  = rx
        msg.pose.pose.position.y  = ry
        msg.pose.pose.position.z  = 0.0
        msg.pose.pose.orientation = euler_to_quaternion(ryaw)

        cov = [0.0] * 36
        cov[0]  = r_pos    # σ²_xx
        cov[7]  = r_pos    # σ²_yy
        cov[14] = 1e-6
        cov[21] = 1e-6
        cov[28] = 1e-6
        cov[35] = r_yaw    # σ²_yaw
        msg.pose.covariance = cov

        self._pub_pose.publish(msg)

        with self._lock:
            n_lm = len(self._landmarks)
        debug = (
            f'landmarks={n_lm} | '
            f'corr=({rx:.3f},{ry:.3f}) θ={math.degrees(ryaw):.1f}° | '
            f'σ_pos={math.sqrt(r_pos):.3f}m d={dist:.3f}m'
        )
        self._pub_debug.publish(String(data=debug))


# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = ArucoLocalizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
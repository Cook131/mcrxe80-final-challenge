#!/usr/bin/env python3
"""
aruco_localizer.py — Corrección de drift por landmark anchoring
================================================================

Problema que resuelve
---------------------
El robot navega por una pista con ArUcos en posiciones CONOCIDAS (YAML)
o DESCONOCIDAS (anclado dinámico de fallback).
La odometría acumula drift con el tiempo. Cada vez que el robot ve un
ArUco, podemos corregir ese drift.

Método: Landmark Anchoring con posiciones predefinidas
------------------------------------------------------
Al arrancar:
  → Se carga aruco_landmarks.yaml con las posiciones globales conocidas
  → Esos landmarks se marcan como "fijos" (no se actualizan con promedio móvil)

Primera vez que se ve un marcador ID NO listado en el YAML:
  → Comportamiento de fallback: calcular su posición global usando la
    pose EKF actual + tvec de cámara y guardarlo como ancla dinámica

Siguientes veces que se ve el mismo marcador (fijo o dinámico):
  → El robot debería estar a la misma distancia/ángulo del ancla
  → Calcular dónde DEBERÍA estar el robot dado el ancla y el tvec actual
  → Publicar esa pose como corrección para el EKF

Flujo de datos
--------------
  aruco_detector.py  →  /aruco/id        (Int32)
                         /aruco/distance  (Float32)  distancia plano XZ [m]
                         /aruco/angle     (Float32)  bearing horizontal [°]
                         /aruco/waypoint  (PoseStamped) rvec+tvec completo

  puzzlebotOdometry  →  /odom  (Odometry)  pose EKF actual

  aruco_localizer    →  /aruco/pose   (PoseWithCovarianceStamped) → EKF
                         /aruco/debug  (String)

Parámetros ROS
--------------
  landmarks_file     ''     ruta al YAML con posiciones predefinidas.
                            Si está vacío, todos los landmarks son dinámicos.
  camera_to_base_x   0.05   [m]   offset longitudinal cámara→base_link
  camera_to_base_y   0.00   [m]   offset lateral
  camera_to_base_z   0.10   [m]   altura cámara sobre el suelo
  anchor_min_dist    0.20   [m]   distancia mínima para anclar/corregir
  anchor_max_dist    3.50   [m]   distancia máxima para anclar/corregir
  anchor_reobserve   0.30   [m]   mínimo desplazamiento del robot para
                                  re-observar el mismo marcador
  r_base_pos         0.03   [m²]  varianza base de posición
  r_base_yaw         0.04   [rad²] varianza base de orientación
  distance_noise_k   0.025  factor de ruido proporcional a d²
  publish_rate       10.0   [Hz]

Formato del YAML
----------------
  landmarks:
    1:
      x: 1.50
      y: 0.00
      yaw: 0.00   # opcional: orientación del marcador en el mapa [rad]
                  # si se omite, el yaw del robot no se corrige con ese landmark
    2:
      x: 3.00
      y: 1.50
      yaw: 1.5708
"""

import math
import os
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

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
      3. Restar offset de cámara a base_link (ya rotado a frame global)
         Nota: se resta porque cam_x/cam_y son la posición de la CÁMARA
         respecto al base_link. El tvec mide desde la cámara, no desde
         el base_link, así que hay que restar ese offset para obtener el
         desplazamiento al marcador medido desde el base_link.
    """
    dx_robot = float(tvec[2])
    dy_robot = -float(tvec[0])

    cos_y = math.cos(robot_yaw)
    sin_y = math.sin(robot_yaw)

    dx_g = dx_robot * cos_y - dy_robot * sin_y
    dy_g = dx_robot * sin_y + dy_robot * cos_y

    # Offset cámara→base_link en frame global
    off_x = cam_x * cos_y - cam_y * sin_y
    off_y = cam_x * sin_y + cam_y * cos_y

    return dx_g - off_x, dy_g - off_y


def yaw_from_marker_rotation(R_marker_cam: np.ndarray,
                              robot_yaw: float,
                              marker_yaw_map: float) -> float:
    """
    Estima el yaw del robot a partir de la rotación del marcador.

    La rotación R_marker_cam (3×3) expresa la orientación del marcador
    en frame cámara, publicada por aruco_detector como PoseStamped.

    El frame óptico de OpenCV tiene:
      X = derecha, Y = abajo, Z = frente (lejos del lente)

    El eje Z del marcador apunta hacia la cámara (normal al plano del marcador
    apuntando hacia fuera). En el frame óptico, ese eje es la 3ª columna de
    R_marker_cam con signo negado (la cámara ve el frente del marcador).

    Pasos:
      1. Extraer el eje Z del marcador en frame óptico: nz = -R_marker_cam[:,2]
         (signo negativo porque solvePnP define el eje Z del objeto apuntando
         hacia la cámara, pero queremos la normal hacia afuera del marcador)
      2. Proyectar al plano horizontal: ignorar la componente Y (vertical)
      3. Convertir de frame óptico a frame global:
           nz_x_robot =  nz[2]   (Z óptico → adelante robot)
           nz_y_robot = -nz[0]   (X óptico → izquierda robot, negado)
         Rotar por robot_yaw para obtener la normal en frame global.
      4. El marcador tiene una orientación conocida en el mapa (marker_yaw_map).
         La normal del marcador en el mapa apunta en la dirección marker_yaw_map.
         La diferencia entre esa dirección esperada y la medida da el error de yaw.
      5. yaw_robot = robot_yaw + (marker_yaw_map - marker_yaw_measured)

    Returns
    -------
    yaw_corr : float
        Yaw corregido del robot en radianes.
    """
    # Normal del marcador en frame óptico (apunta hacia la cámara)
    # R_marker_cam[:,2] es el eje Z del marcador en frame cámara.
    # Como el marcador "mira" a la cámara, su normal hacia fuera es -Z_cam.
    nz_cam = -R_marker_cam[:, 2]   # [nx, ny, nz] en frame óptico

    # Proyectar al plano horizontal del robot:
    # Frame óptico → frame robot:  X_robot = Z_cam,  Y_robot = -X_cam
    nz_robot_x =  nz_cam[2]   # componente adelante
    nz_robot_y = -nz_cam[0]   # componente izquierda

    # Rotar al frame global
    cos_y = math.cos(robot_yaw)
    sin_y = math.sin(robot_yaw)
    nz_global_x = nz_robot_x * cos_y - nz_robot_y * sin_y
    nz_global_y = nz_robot_x * sin_y + nz_robot_y * cos_y

    # Dirección medida de la normal del marcador en frame global
    marker_yaw_measured = math.atan2(nz_global_y, nz_global_x)

    # Corrección: el yaw del robot debe rotar marker_yaw_measured
    # hasta coincidir con marker_yaw_map
    yaw_error = normalize_angle(marker_yaw_map - marker_yaw_measured)
    yaw_corr  = normalize_angle(robot_yaw + yaw_error)

    return yaw_corr


# ─────────────────────────────────────────────────────────────────────────────
# Estructura de un landmark
# ─────────────────────────────────────────────────────────────────────────────

class Landmark:
    """Posición global de un marcador ArUco.

    Atributos
    ---------
    fixed : bool
        True  → posición cargada desde YAML; nunca se actualiza con
                promedio móvil (se confía 100 % en el mapa).
        False → posición estimada dinámicamente; se refina con cada
                observación.
    yaw_map : float | None
        Orientación del marcador en el mapa [rad].
        None si no está definida (landmarks dinámicos o YAML sin campo yaw).
        Cuando está definida, se usa para corregir también el yaw del robot.
    """

    def __init__(self, pub_id: int, gx: float, gy: float,
                 robot_x: float, robot_y: float,
                 fixed: bool = False,
                 yaw_map: float | None = None):
        self.pub_id    = pub_id
        self.gx        = gx
        self.gy        = gy
        self.n_obs     = 1
        self.fixed     = fixed
        self.yaw_map   = yaw_map
        self.last_robot_x = robot_x
        self.last_robot_y = robot_y

    def update_position(self, gx: float, gy: float,
                        robot_x: float, robot_y: float):
        """Actualiza la posición con promedio móvil (solo landmarks dinámicos)."""
        if self.fixed:
            # Landmark fijo: no modificar posición, solo contabilizar observación
            self.n_obs += 1
            self.last_robot_x = robot_x
            self.last_robot_y = robot_y
            return

        alpha = 1.0 / (self.n_obs + 1)
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
        self.declare_parameter('landmarks_file',    '')
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

        self._landmarks_file = self.get_parameter('landmarks_file').value
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

        # Landmarks: pub_id → Landmark
        self._landmarks: dict[int, Landmark] = {}

        # Última pose del EKF
        self._robot_x:   float = 0.0
        self._robot_y:   float = 0.0
        self._robot_yaw: float = 0.0
        self._odom_ready: bool = False

        # Última detección pendiente de procesar
        self._pending_id:   int              = -1
        self._pending_R:    np.ndarray | None = None   # rotación 3×3 del marcador
        self._pending_t:    np.ndarray | None = None   # traslación [tx,ty,tz]
        self._pending_dist: float            = 0.0
        self._pending_ts:   float            = -1.0

        # Pose a publicar (None = no hay corrección nueva)
        self._correction: tuple[float, float, float, float] | None = None

        # ── Cargar landmarks desde YAML ───────────────────────────────────
        self._load_landmarks_from_yaml()

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

        n_fixed = sum(1 for lm in self._landmarks.values() if lm.fixed)
        n_with_yaw = sum(
            1 for lm in self._landmarks.values()
            if lm.fixed and lm.yaw_map is not None
        )
        self.get_logger().info(
            f'ArUco Localizer (landmark anchoring) listo\n'
            f'  Landmarks fijos (YAML): {n_fixed} '
            f'({n_with_yaw} con yaw definido → corrección completa x,y,θ)\n'
            f'  IDs no listados usarán anclado dinámico de fallback '
            f'(solo corrección x,y)\n'
            f'  Publica → /aruco/pose  (PoseWithCovarianceStamped)'
        )

    # ── Carga de YAML ─────────────────────────────────────────────────────

    def _load_landmarks_from_yaml(self):
        """
        Carga posiciones predefinidas desde el archivo YAML.

        Espera la estructura:
            landmarks:
              <id>:
                x: <float>
                y: <float>
                yaw: <float>   # opcional, en radianes

        Los landmarks cargados se marcan como fixed=True.
        Si se proporciona 'yaw', se usará para corregir también la orientación
        del robot. Sin 'yaw', solo se corrige la posición (x, y).
        """
        path = self._landmarks_file

        if not path:
            self.get_logger().info(
                'Parámetro landmarks_file vacío → '
                'todos los landmarks serán dinámicos (solo corrección x,y).'
            )
            return

        if not _YAML_AVAILABLE:
            self.get_logger().error(
                'PyYAML no está instalado. '
                'Instálalo con: pip install pyyaml\n'
                'Continuando sin landmarks predefinidos.'
            )
            return

        if not os.path.isfile(path):
            self.get_logger().error(
                f'Archivo de landmarks no encontrado: {path}\n'
                f'Continuando sin landmarks predefinidos.'
            )
            return

        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            self.get_logger().error(
                f'Error al parsear YAML ({path}): {e}\n'
                f'Continuando sin landmarks predefinidos.'
            )
            return

        if not isinstance(data, dict) or 'landmarks' not in data:
            self.get_logger().error(
                f'El YAML no contiene la clave "landmarks". '
                f'Revisa el formato en {path}.'
            )
            return

        entries = data['landmarks']
        if not isinstance(entries, dict):
            self.get_logger().error(
                '"landmarks" debe ser un diccionario id → {x, y[, yaw]}.'
            )
            return

        loaded = 0
        for raw_id, coords in entries.items():
            try:
                pub_id = int(raw_id)
                gx = float(coords['x'])
                gy = float(coords['y'])
                yaw_map = float(coords['yaw']) if 'yaw' in coords else None
            except (TypeError, KeyError, ValueError) as e:
                self.get_logger().warn(
                    f'Entrada inválida para ID={raw_id}: {e} — omitida.'
                )
                continue

            lm = Landmark(
                pub_id=pub_id,
                gx=gx,
                gy=gy,
                robot_x=float('inf'),
                robot_y=float('inf'),
                fixed=True,
                yaw_map=yaw_map,
            )
            self._landmarks[pub_id] = lm
            loaded += 1

            yaw_str = f"  yaw={math.degrees(yaw_map):.1f}°" if yaw_map is not None else "  (sin yaw)"
            self.get_logger().info(
                f'  [YAML] ID={pub_id} → ({gx:.3f}, {gy:.3f}){yaw_str} [FIJO]'
            )

        self.get_logger().info(
            f'Cargados {loaded} landmarks predefinidos desde {path}'
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
        """Recibe R (rotación 3×3) y t (traslación) del marcador desde aruco_detector."""
        R, t = pose_stamped_to_Rt(msg)
        dist = float(np.linalg.norm(t))
        if not (self._d_min <= dist <= self._d_max):
            return
        with self._lock:
            self._pending_R  = R.copy()     # se usa para corrección de yaw
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
            R_marker = self._pending_R.copy() if self._pending_R is not None else None
            dist     = float(np.linalg.norm(t_vec))
            rx_ekf   = self._robot_x
            ry_ekf   = self._robot_y
            yaw_ekf  = self._robot_yaw

        # ── Posición global del marcador según pose EKF actual ────────────
        dx_g, dy_g = camera_tvec_to_robot_frame(
            t_vec, yaw_ekf, self._cam_x, self._cam_y
        )
        marker_gx = rx_ekf + dx_g
        marker_gy = ry_ekf + dy_g

        with self._lock:
            if pub_id not in self._landmarks:
                # ── PRIMERA VEZ y NO está en YAML: anclar dinámicamente ───
                # Los landmarks dinámicos no tienen yaw_map, por lo que solo
                # corregirán posición x,y (no orientación).
                lm = Landmark(pub_id, marker_gx, marker_gy,
                              rx_ekf, ry_ekf, fixed=False, yaw_map=None)
                self._landmarks[pub_id] = lm
                self.get_logger().info(
                    f'[ANCHOR-DYN] ID={pub_id} anclado dinámicamente en '
                    f'({marker_gx:.3f}, {marker_gy:.3f}) '
                    f'desde robot ({rx_ekf:.3f}, {ry_ekf:.3f}) '
                    f'd={dist:.3f}m  (solo corrección x,y)'
                )
                self._correction = None   # primera vez no corregimos
                return

            lm = self._landmarks[pub_id]

            # ── RE-OBSERVACIÓN: verificar que el robot se movió ───────────
            moved = math.hypot(rx_ekf - lm.last_robot_x,
                               ry_ekf - lm.last_robot_y)
            if moved < self._d_reobs:
                return   # robot casi quieto, no spamear

            # ── CALCULAR CORRECCIÓN DE POSICIÓN ───────────────────────────
            # El marcador está en (lm.gx, lm.gy) en el mapa.
            # Si eso es verdad, el robot DEBE estar en:
            rx_corr = lm.gx - dx_g
            ry_corr = lm.gy - dy_g

            # ── CALCULAR CORRECCIÓN DE YAW ────────────────────────────────
            # Solo posible si:
            #   a) tenemos la rotación del marcador (R_marker), y
            #   b) el landmark tiene yaw_map definido en el YAML.
            #
            # Si alguna condición falla, se mantiene el yaw actual del EKF
            # (no se degrada la corrección de posición).
            if R_marker is not None and lm.yaw_map is not None:
                ryaw_corr = yaw_from_marker_rotation(
                    R_marker, yaw_ekf, lm.yaw_map
                )
                yaw_corrected = True
            else:
                ryaw_corr = yaw_ekf   # sin corrección de yaw
                yaw_corrected = False

            # ── Actualizar ancla (solo landmarks dinámicos) ───────────────
            lm.update_position(marker_gx, marker_gy, rx_ekf, ry_ekf)

            drift_pos = math.hypot(rx_corr - rx_ekf, ry_corr - ry_ekf)
            drift_yaw = abs(normalize_angle(ryaw_corr - yaw_ekf))
            tag = 'FIXED' if lm.fixed else 'DYN'
            yaw_tag = f"θ={math.degrees(ryaw_corr):.1f}°" if yaw_corrected \
                      else "θ=EKF(sin corrección)"

            self.get_logger().info(
                f'[CORRECT-{tag}] ID={pub_id} obs#{lm.n_obs} | '
                f'drift_xy={drift_pos:.3f}m drift_yaw={math.degrees(drift_yaw):.1f}° | '
                f'EKF=({rx_ekf:.3f},{ry_ekf:.3f}) → '
                f'corr=({rx_corr:.3f},{ry_corr:.3f}) {yaw_tag}'
            )

            self._correction = (rx_corr, ry_corr, ryaw_corr, dist, yaw_corrected)

        # ── Publicar corrección ───────────────────────────────────────────
        self._publish_correction()

    def _publish_correction(self):
        with self._lock:
            if self._correction is None:
                return
            rx, ry, ryaw, dist, yaw_corrected = self._correction
            self._correction = None

        # Covarianza adaptativa: más ruido a más distancia
        noise = self._k_dist * dist ** 2
        r_pos = self._r_pos + noise

        # Si el yaw NO fue corregido (sin yaw_map), le asignamos
        # una varianza muy alta para que el EKF ignore esa componente
        # y confíe en su propia estimación de orientación.
        r_yaw = (self._r_yaw + noise) if yaw_corrected else 1e6

        msg = PoseWithCovarianceStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'

        msg.pose.pose.position.x  = rx
        msg.pose.pose.position.y  = ry
        msg.pose.pose.position.z  = 0.0
        msg.pose.pose.orientation = euler_to_quaternion(ryaw)

        cov = [0.0] * 36
        cov[0]  = r_pos    # σ²_xx
        cov[7]  = r_pos    # σ²_yy
        cov[14] = 1e-6     # σ²_zz  (z fijo en 0)
        cov[21] = 1e-6     # σ²_roll
        cov[28] = 1e-6     # σ²_pitch
        cov[35] = r_yaw    # σ²_yaw  (grande si no hay corrección de yaw)
        msg.pose.covariance = cov

        self._pub_pose.publish(msg)

        with self._lock:
            n_fixed  = sum(1 for lm in self._landmarks.values() if lm.fixed)
            n_dyn    = len(self._landmarks) - n_fixed
        debug = (
            f'lm_fixed={n_fixed} lm_dyn={n_dyn} | '
            f'corr=({rx:.3f},{ry:.3f}) θ={math.degrees(ryaw):.1f}° '
            f'[yaw_ok={yaw_corrected}] | '
            f'σ_pos={math.sqrt(r_pos):.3f}m '
            f'σ_yaw={math.sqrt(min(r_yaw, 1e3)):.3f}rad '
            f'd={dist:.3f}m'
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
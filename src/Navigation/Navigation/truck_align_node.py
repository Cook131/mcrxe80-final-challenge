#!/usr/bin/env python3
"""
truck_align_node.py — Alineación visual con camión  [v2 — integración A*/GoToGoal]
====================================================================================
Cambios respecto a v1
---------------------
  * El logo del camión está en el piso/frente bajo: la cámara lo pierde de vista
    cuando el robot se acerca. Esta versión maneja explícitamente ese caso.

  * Nuevo estado NAV_APPROACH: al confirmar alineación visual (logo centrado),
    el nodo estima la distancia al logo por el tamaño del bbox en píxeles y publica
    un goal relativo a /astar/goal para que GoToGoal + A* lleven al robot hasta
    el punto de depósito, sin necesidad de ver el logo durante el trayecto.

  * Approach ciego garantizado: LOGO_LOST_APPROACH es la fase donde el logo ya
    desapareció del frame (robot muy cerca / logo debajo del FOV). El nodo sigue
    el goal hasta GOAL_REACHED de A* y luego declara ALIGNED.

  * Suscripción a /odom para convertir offset visual → coordenadas mundo.
  * Suscripción a /astar/status para saber cuándo A* terminó.

FSM interna:
  IDLE → SCAN → ALIGN → NAV_APPROACH → LOGO_LOST_APPROACH → DONE → IDLE

Topics
------
  SUB  /truck_align/cmd      (String)   "align:<client_name>"
       /yolo/detecciones     (String)   JSON [{class, conf, bbox, bbox_cx}]
       /mission/mode         (String)   "stop" / "estop" aborta
       /odom                 (Odometry) pose del robot en mundo
       /astar/status         (String)   GOAL_REACHED / EXECUTING / ...

  PUB  /truck_align/result   (String)   "ALIGNED" | "FAILED" | "TIMEOUT"
       /truck_align/status   (String)   heartbeat estado FSM
       /cmd_vel              (Twist)    velocidades durante SCAN / ALIGN
       /astar/goal           (Pose2D)   goal absoluto para A* durante approach

Parámetros
----------
  frame_width_px      int    320      ancho del frame de cámara
  logo_real_width_m   float  0.35     ancho físico conocido del logo [m]
                                       → usado para estimar distancia por bbox
  focal_length_px     float  186.0    focal horizontal en píxeles (fx de calib)
                                       → dist_est = (logo_w_m * fx) / bbox_w_px
  approach_stop_dist  float  0.40     distancia final al logo tras approach [m]
                                       (goal = pos_logo - approach_stop_dist)
  yolo_kp             float  0.005    ganancia proporcional px→rad/s
  max_w               float  0.35     velocidad angular máxima rad/s
  err_ok_px           int    22       umbral de alineación en píxeles
  confirm_ticks       int    3        frames consecutivos para confirmar
  min_conf            float  0.60     confianza mínima YOLO
  scan_speed          float  0.22     rad/s durante búsqueda
  scan_timeout_s      float  18.0     tiempo total de scan antes de FAILED
  align_timeout_s     float  10.0     tiempo máximo en fase ALIGN
  nav_approach_timeout_s float 30.0   timeout A* approach
  fsm_rate_hz         float  20.0     frecuencia del timer interno
"""

from __future__ import annotations

import json
import math
import time
from enum import Enum, auto
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Twist, Pose2D
from nav_msgs.msg import Odometry
from std_msgs.msg import String


# ══════════════════════════════════════════════════════════════════════════════
# FSM INTERNA
# ══════════════════════════════════════════════════════════════════════════════

class AlignState(Enum):
    IDLE                = auto()  # esperando /truck_align/cmd
    SCAN                = auto()  # girando para encontrar el logo
    ALIGN               = auto()  # centrando el logo en frame con P-ctrl angular
    NAV_APPROACH        = auto()  # A* llevando al robot hasta el logo (blind ok)
    LOGO_LOST_APPROACH  = auto()  # logo fuera de FOV (robot muy cerca) — esperando GOAL_REACHED
    DONE                = auto()  # publicar resultado y volver a IDLE


# ══════════════════════════════════════════════════════════════════════════════
# NODO
# ══════════════════════════════════════════════════════════════════════════════

class TruckAlignNode(Node):

    def __init__(self):
        super().__init__('truck_align_node')

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('frame_width_px',          320)
        self.declare_parameter('logo_real_width_m',       0.35)   # ancho físico del logo
        self.declare_parameter('focal_length_px',         186.0)  # fx de calibración
        self.declare_parameter('approach_stop_dist',      0.40)   # distancia objetivo final
        self.declare_parameter('yolo_kp',                 0.005)
        self.declare_parameter('max_w',                   0.35)
        self.declare_parameter('err_ok_px',               22)
        self.declare_parameter('confirm_ticks',           3)
        self.declare_parameter('min_conf',                0.60)
        self.declare_parameter('scan_speed',              0.22)
        self.declare_parameter('scan_timeout_s',          18.0)
        self.declare_parameter('align_timeout_s',         10.0)
        self.declare_parameter('nav_approach_timeout_s',  30.0)
        self.declare_parameter('fsm_rate_hz',             20.0)

        self._frame_cx    = self.get_parameter('frame_width_px').value // 2
        self._kp          = self.get_parameter('yolo_kp').value
        self._max_w       = self.get_parameter('max_w').value
        self._err_ok      = self.get_parameter('err_ok_px').value
        self._confirm_ticks = self.get_parameter('confirm_ticks').value
        self._min_conf    = self.get_parameter('min_conf').value
        self._scan_speed  = self.get_parameter('scan_speed').value
        self._scan_timeout  = self.get_parameter('scan_timeout_s').value
        self._align_timeout = self.get_parameter('align_timeout_s').value
        self._nav_timeout   = self.get_parameter('nav_approach_timeout_s').value
        fsm_rate            = self.get_parameter('fsm_rate_hz').value

        # ── Estado interno ────────────────────────────────────────────────────
        self._state:           AlignState = AlignState.IDLE
        self._state_entry_t:   float      = time.monotonic()
        self._just_entered:    bool       = True

        self._target_class:    str   = ''
        self._detections:      list  = []
        self._mission_mode:    str   = ''

        self._ok_ticks:        int   = 0
        self._result:          str   = ''

        # Pose del robot (de /odom)
        self._robot_x:         float = 0.0
        self._robot_y:         float = 0.0
        self._robot_th:        float = 0.0

        # Estado A*
        self._astar_status:    str   = ''

        # ── QOS ───────────────────────────────────────────────────────────────
        best_effort = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=5)

        # ── Suscriptores ──────────────────────────────────────────────────────
        self.create_subscription(String,   '/truck_align/cmd',  self._cb_cmd,    10)
        self.create_subscription(String,   '/yolo/detecciones', self._cb_yolo,   best_effort)
        self.create_subscription(String,   '/mission/mode',     self._cb_mode,   10)
        self.create_subscription(Odometry, '/odom',             self._cb_odom,   10)
        self.create_subscription(String,   '/astar/status',     self._cb_astar,  10)

        # ── Publicadores ──────────────────────────────────────────────────────
        self._pub_result = self.create_publisher(String, '/truck_align/result', 10)
        self._pub_status = self.create_publisher(String, '/truck_align/status', 10)
        self._pub_cmd    = self.create_publisher(Twist,  '/cmd_vel',            10)
        self._pub_goal   = self.create_publisher(Pose2D, '/astar/goal',         10)

        # ── Timer ─────────────────────────────────────────────────────────────
        self.create_timer(1.0 / fsm_rate, self._tick)

        self.get_logger().info(
            f'TruckAlignNode v2 listo | '
            f'frame_cx={self._frame_cx}px | err_ok={self._err_ok}px | '
            f'confirm={self._confirm_ticks} ticks | '
            f'logo_w={self.get_parameter("logo_real_width_m").value}m | '
            f'approach_stop={self.get_parameter("approach_stop_dist").value}m'
        )

    # ══════════════════════════════════════════════════════════════════════════
    # CALLBACKS
    # ══════════════════════════════════════════════════════════════════════════

    def _cb_cmd(self, msg: String):
        raw = msg.data.strip()
        if not raw.startswith('align:'):
            self.get_logger().warning(f'[TruckAlign] Comando no reconocido: "{raw}"')
            return

        client = raw[len('align:'):].strip().lower()
        if not client:
            self.get_logger().warning('[TruckAlign] Nombre de cliente vacío')
            return

        if self._state != AlignState.IDLE:
            self.get_logger().warning(
                f'[TruckAlign] Recibido "{raw}" pero estado={self._state.name} — ignorando')
            return

        self.get_logger().info(f'[TruckAlign] Iniciando alineación para "{client}"')
        self._target_class = client
        self._ok_ticks     = 0
        self._result       = ''
        self._transition(AlignState.SCAN)

    def _cb_yolo(self, msg: String):
        try:
            raw = json.loads(msg.data)
            for d in raw:
                if 'bbox_cx' not in d and 'bbox' in d:
                    x1, _, x2, _ = d['bbox']
                    d['bbox_cx'] = (x1 + x2) // 2
            self._detections = raw
        except Exception:
            self._detections = []

    def _cb_mode(self, msg: String):
        self._mission_mode = msg.data.strip().lower()

    def _cb_odom(self, msg: Odometry):
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self._robot_th = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def _cb_astar(self, msg: String):
        self._astar_status = msg.data.strip()

    # ══════════════════════════════════════════════════════════════════════════
    # FSM TICK
    # ══════════════════════════════════════════════════════════════════════════

    def _tick(self):
        self._pub_status.publish(String(data=self._state.name))

        # Abort por E_STOP o stop manual
        if self._mission_mode in ('estop', 'stop') and self._state != AlignState.IDLE:
            self.get_logger().warning(
                f'[TruckAlign] Abort por mode="{self._mission_mode}"')
            self._stop()
            self._publish_result('FAILED')
            self._transition(AlignState.IDLE)
            return

        if self._state == AlignState.IDLE:
            pass

        elif self._state == AlignState.SCAN:
            self._do_scan()

        elif self._state == AlignState.ALIGN:
            self._do_align()

        elif self._state == AlignState.NAV_APPROACH:
            self._do_nav_approach()

        elif self._state == AlignState.LOGO_LOST_APPROACH:
            self._do_logo_lost_approach()

        elif self._state == AlignState.DONE:
            self._do_done()

        self._just_entered = False

    # ══════════════════════════════════════════════════════════════════════════
    # FASES
    # ══════════════════════════════════════════════════════════════════════════

    def _do_scan(self):
        """
        Gira buscando el logo target.
        Dirección: izquierda primera mitad, derecha segunda.
        Si detecta → ALIGN. Si timeout → FAILED.
        """
        t = self._time_in_state()

        if t > self._scan_timeout:
            self.get_logger().warning(
                f'[SCAN] Timeout {self._scan_timeout}s sin encontrar "{self._target_class}"')
            self._stop()
            self._publish_result('TIMEOUT')
            self._transition(AlignState.IDLE)
            return

        det = self._find_target()
        if det is not None:
            self.get_logger().info(
                f'[SCAN] "{self._target_class}" encontrado (conf={det["conf"]:.2f}) → ALIGN')
            self._stop()
            self._ok_ticks = 0
            self._transition(AlignState.ALIGN)
            return

        direction = 1.0 if t < self._scan_timeout / 2 else -1.0
        cmd = Twist()
        cmd.angular.z = direction * self._scan_speed
        self._pub_cmd.publish(cmd)

    def _do_align(self):
        """
        Centra el logo con P-controller angular puro.
        Cuando confirma alineación → estima distancia y lanza NAV_APPROACH.
        Si el logo desaparece antes de confirmar → vuelve a SCAN.
        """
        t = self._time_in_state()

        if t > self._align_timeout:
            self.get_logger().warning(f'[ALIGN] Timeout {self._align_timeout}s')
            self._stop()
            self._publish_result('TIMEOUT')
            self._transition(AlignState.IDLE)
            return

        det = self._find_target()

        if det is None:
            self.get_logger().warning(f'[ALIGN] "{self._target_class}" perdido → SCAN')
            self._ok_ticks = 0
            self._transition(AlignState.SCAN)
            return

        bbox_cx = det.get('bbox_cx', self._frame_cx)
        err_x   = bbox_cx - self._frame_cx

        w = max(-self._max_w, min(self._max_w, -self._kp * err_x))
        cmd = Twist()
        cmd.angular.z = w
        self._pub_cmd.publish(cmd)

        if abs(err_x) < self._err_ok:
            self._ok_ticks += 1
            self.get_logger().debug(
                f'[ALIGN] err_x={err_x}px | ok_ticks={self._ok_ticks}/{self._confirm_ticks}')
        else:
            self._ok_ticks = 0

        if self._ok_ticks >= self._confirm_ticks:
            self.get_logger().info(
                f'[ALIGN] Alineado ✓ | err_x={err_x}px | conf={det["conf"]:.2f} '
                f'→ calculando distancia y lanzando NAV_APPROACH')
            self._stop()
            # Lanzar approach navigado
            if self._publish_logo_approach_goal(det):
                self._transition(AlignState.NAV_APPROACH)
            else:
                # No se pudo estimar distancia (bbox demasiado pequeño) → FAILED
                self.get_logger().error('[ALIGN] No se pudo estimar distancia al logo → FAILED')
                self._publish_result('FAILED')
                self._transition(AlignState.IDLE)

    def _do_nav_approach(self):
        """
        Espera que A* lleve el robot hasta el goal publicado.
        Si el logo desaparece del frame durante este trayecto, es NORMAL
        (lo perdemos porque está en el piso / ángulo bajo) → pasamos a
        LOGO_LOST_APPROACH que solo espera GOAL_REACHED.
        """
        if self._time_in_state() > self._nav_timeout:
            self.get_logger().error(
                f'[NAV_APPROACH] Timeout {self._nav_timeout}s → FAILED')
            self._publish_result('TIMEOUT')
            self._transition(AlignState.IDLE)
            return

        det = self._find_target()

        if det is None:
            # Logo perdido — esperado al acercarse al piso
            self.get_logger().info(
                '[NAV_APPROACH] Logo fuera de FOV (approach ciego normal) → LOGO_LOST_APPROACH')
            self._transition(AlignState.LOGO_LOST_APPROACH)
            return

        # Logo todavía visible: verificar si se está descentrando mucho
        # (el robot se movió lateralmente). Si sí, corregir publicando nuevo goal.
        bbox_cx = det.get('bbox_cx', self._frame_cx)
        err_x   = bbox_cx - self._frame_cx
        if abs(err_x) > self._err_ok * 3:
            self.get_logger().info(
                f'[NAV_APPROACH] Desvío lateral err_x={err_x}px — recalculando goal')
            self._publish_logo_approach_goal(det)

        if self._astar_status == 'GOAL_REACHED':
            self.get_logger().info('[NAV_APPROACH] GOAL_REACHED → DONE')
            self._result = 'ALIGNED'
            self._transition(AlignState.DONE)

    def _do_logo_lost_approach(self):
        """
        Logo fuera de FOV por proximidad. Solo esperamos GOAL_REACHED de A*.
        No se intenta recuperar el logo — ya perdemos de vista el logo cuando
        el robot se acerca al piso, esto es el comportamiento esperado.
        """
        if self._time_in_state() > self._nav_timeout:
            self.get_logger().error(
                f'[LOGO_LOST_APPROACH] Timeout {self._nav_timeout}s → FAILED')
            self._publish_result('TIMEOUT')
            self._transition(AlignState.IDLE)
            return

        if self._astar_status == 'GOAL_REACHED':
            self.get_logger().info('[LOGO_LOST_APPROACH] GOAL_REACHED → DONE (approach ciego exitoso)')
            self._result = 'ALIGNED'
            self._transition(AlignState.DONE)
        else:
            self.get_logger().debug(
                f'[LOGO_LOST_APPROACH] Esperando A*... status={self._astar_status}')

    def _do_done(self):
        if self._just_entered:
            self._publish_result(self._result)
        if self._time_in_state() > 0.1:
            self._target_class = ''
            self._result       = ''
            self._transition(AlignState.IDLE)

    # ══════════════════════════════════════════════════════════════════════════
    # ESTIMACIÓN DE DISTANCIA Y GOAL A*
    # ══════════════════════════════════════════════════════════════════════════

    def _estimate_logo_distance(self, det: dict) -> Optional[float]:
        """
        Estima la distancia al logo usando la fórmula del modelo pin-hole:
            dist = (logo_w_real * focal_px) / bbox_width_px

        Asume que el logo es aproximadamente frontal (no muy girado lateralmente).
        Retorna None si el bbox es inválido o demasiado pequeño.
        """
        bbox = det.get('bbox')
        if not bbox or len(bbox) < 4:
            return None

        x1, _, x2, _ = bbox
        bbox_w = abs(x2 - x1)
        if bbox_w < 5:  # píxeles mínimos para una estimación fiable
            return None

        logo_real_w = self.get_parameter('logo_real_width_m').value
        focal_px    = self.get_parameter('focal_length_px').value
        dist = (logo_real_w * focal_px) / bbox_w
        return dist

    def _publish_logo_approach_goal(self, det: dict) -> bool:
        """
        Calcula y publica el goal en /astar/goal para que A* lleve el robot
        hasta approach_stop_dist delante del logo.

        El robot está mirando al logo: el bearing en mundo es self._robot_th
        más el error angular normalizado del centro del bbox.

        Retorna True si el goal se publicó, False si no se pudo estimar distancia.
        """
        dist = self._estimate_logo_distance(det)
        if dist is None:
            return False

        # Ángulo lateral del logo respecto al frente del robot
        bbox_cx = det.get('bbox_cx', self._frame_cx)
        err_x   = bbox_cx - self._frame_cx
        # Conversión px → rad: aprox err_x / focal_px (ángulo pequeño)
        focal_px   = self.get_parameter('focal_length_px').value
        angle_rad  = math.atan2(err_x, focal_px)   # + = logo a la derecha
        bearing    = self._robot_th + angle_rad     # dirección al logo en mundo

        # Posición estimada del logo en mundo
        logo_wx = self._robot_x + dist * math.cos(bearing)
        logo_wy = self._robot_y + dist * math.sin(bearing)

        # Goal: detenerse a approach_stop_dist delante del logo
        stop_dist = self.get_parameter('approach_stop_dist').value
        goal_x = logo_wx - stop_dist * math.cos(bearing)
        goal_y = logo_wy - stop_dist * math.sin(bearing)

        goal_msg = Pose2D()
        goal_msg.x = goal_x
        goal_msg.y = goal_y
        self._pub_goal.publish(goal_msg)

        self.get_logger().info(
            f'[TruckAlign] Goal A* publicado: ({goal_x:.3f}, {goal_y:.3f}) | '
            f'logo estimado: ({logo_wx:.3f}, {logo_wy:.3f}) | '
            f'dist={dist:.2f}m | bearing={math.degrees(bearing):.1f}°'
        )
        return True

    # ══════════════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _find_target(self) -> Optional[dict]:
        best = None
        for d in self._detections:
            if (d.get('class', '').lower() == self._target_class
                    and d.get('conf', 0) >= self._min_conf):
                if best is None or d['conf'] > best['conf']:
                    best = d
        return best

    def _publish_result(self, result: str):
        self.get_logger().info(f'[TruckAlign] → result="{result}"')
        self._pub_result.publish(String(data=result))

    def _stop(self):
        self._pub_cmd.publish(Twist())

    def _transition(self, new_state: AlignState):
        if new_state == self._state:
            return
        self.get_logger().info(
            f'[TruckAlign FSM] {self._state.name} → {new_state.name}')
        self._state         = new_state
        self._state_entry_t = time.monotonic()
        self._just_entered  = True

    def _time_in_state(self) -> float:
        return time.monotonic() - self._state_entry_t


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = TruckAlignNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

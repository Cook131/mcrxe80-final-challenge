#!/usr/bin/env python3
"""
qr_align_node.py — Iolair QR Align + Collect  [v3 — goal-based, sin cmd_vel fino]
====================================================================================
Resumen de cambios respecto a v2
---------------------------------
  v2 tenía un tramo "fino" PD que publicaba cmd_vel directamente para ALIGN y
  ADVANCE. Ese tramo fue eliminado por completo. Ahora TODA la navegación —
  incluyendo la alineación lateral y el acercamiento final a 35 cm — se hace
  mediante goals relativos publicados en /astar/goal, que A* convierte en
  waypoints y GoToGoal ejecuta con su PID.

  Esto logra:
    * Coherencia: un único controlador de movimiento en todo el stack.
    * No hay conflicto con VFH+: el bypass sigue activo (ya existía), pero
      el nodo ya no "pelea" contra GoToGoal por cmd_vel.
    * Alineación progresiva: cada tick de ALIGNING recalcula un goal que
      corrige lateral + ángulo en coordenadas mundo; la convergencia la
      hace GoToGoal sin oscilaciones bruscas.
    * Tramo final (APPROACH_FINAL, ~30 cm) sigue usando un goal relativo
      directo, sin mapa: se publica en /goal (waypoint GoToGoal) con
      bypass de A*.

Máquina de estados
------------------
  IDLE
    → SEARCH_QR   al recibir /collect/trigger (siempre)

  SEARCH_QR
    → ALIGNING      cuando el QR es visible
    → RECOVER_SCAN  si timeout sin QR  (safeguard)

  ALIGNING
    → APPROACH_FINAL  cuando centrado + dist ≤ align_stop_dist
    → ALIGNING        cada tick: recalcula y publica goal de alineación
    → RECOVER_SCAN    si QR perdido  (safeguard)

  RECOVER_SCAN  ← safeguard: barrido ±30° buscando el QR
    Fases:
      1. LEFT_SWEEP : gira +scan_range_deg a la izquierda
      2. RIGHT_SWEEP: gira −2×scan_range_deg a la derecha (cubre ambos lados)
      3. CENTER     : vuelve al centro (+scan_range_deg)
    → estado_origen  si recupera el QR en cualquier fase
    → ABORT          si el barrido completo termina sin QR

  APPROACH_FINAL
    → HOLD        cuando A* publica GOAL_REACHED  (llegó a 35 cm)
    → ABORT       timeout

  HOLD
    → BACK_AWAY   lift subido + pallet confirmado + /collect/done publicado

  BACK_AWAY
    → DELIVERY    retroceso completado

  DELIVERY
    → IDLE        control cedido a FSM de misión

Coordinación con VFH+
---------------------
  /align/active = True  →  VFH+ ignora todo (collect_bypass ON).
  Se activa en SEARCH_QR y se desactiva al salir de BACK_AWAY.
  Durante DELIVERY el bypass ya está OFF y el VFH+ vuelve a funcionar.

Topics
------
  SUB:  /collect/trigger     (std_msgs/String)   "rack" | "conveyor" | "abort"
  PUB:  /collect/done        (std_msgs/String)   "SUCCESS" | "ABORT"
  SUB:  /aruco/qr            (std_msgs/String)
  SUB:  /aruco/qr/distance   (std_msgs/Float32)  metros en plano XZ
  SUB:  /aruco/qr/angle      (std_msgs/Float32)  grados, + = derecha
  SUB:  /odom                (nav_msgs/Odometry)
  SUB:  /astar/status        (std_msgs/String)   GOAL_REACHED | EXECUTING | ...
  PUB:  /astar/goal          (geometry_msgs/Pose2D)  goal mundo → A* → GoToGoal
  PUB:  /goal                (geometry_msgs/Pose2D)  waypoint directo → GoToGoal
  PUB:  /cmd_vel             (geometry_msgs/Twist)   SOLO para BACK_AWAY
  PUB:  /lift_auto           (std_msgs/String)   n1 | n2 | hold | down
  SUB:  /lift_done           (std_msgs/String)   AT_N1 | AT_N2 | HOLD | DOWN
  PUB:  /align/active        (std_msgs/Bool)     VFH+ bypass
  PUB:  /collect/qr_payload  (std_msgs/String)

Parámetros ROS2
---------------
  align_stop_dist       float  0.35   Distancia objetivo de alineación [m]
  approach_final_dist   float  0.05   Distancia objetivo de encaje final [m]
  align_lateral_tol     float  0.03   Tolerancia lateral para considerar centrado [m]
  angle_tol_deg         float  4.0    Tolerancia angular (alineación fina) [°]
  goal_replan_dist      float  0.06   Re-publicar goal si el lateral cambió > este valor [m]
  goal_replan_angle     float  3.0    Re-publicar goal si el ángulo cambió > este valor [°]
  back_away_speed       float  0.10   Velocidad de retroceso [m/s]
  back_away_time        float  1.8    Duración del retroceso [s]
  lift_timeout          float  8.0    Timeout /lift_done [s]
  align_timeout         float  20.0   Timeout en ALIGNING [s]
  approach_timeout      float  15.0   Timeout en APPROACH_FINAL [s]
  search_timeout        float  10.0   Timeout en SEARCH_QR [s]
  qr_timeout            float  2.5    Segundos sin QR antes de pausar [s]
  cam_offset_deg        float  0.0    Offset angular cámara→base_link [°]
  fsm_rate_hz           float  20.0   Frecuencia del tick [Hz]
  scan_range_deg        float  30.0   Semi-amplitud del barrido de recuperación [°]
  scan_speed_dps        float  20.0   Velocidad angular del barrido [°/s]
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos  import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist, Pose2D
from nav_msgs.msg      import Odometry
from std_msgs.msg      import Bool, Float32, String


# ══════════════════════════════════════════════════════════════════════════════
# Estados internos
# ══════════════════════════════════════════════════════════════════════════════

class _S:
    IDLE           = 'IDLE'
    SEARCH_QR      = 'SEARCH_QR'
    ALIGNING       = 'ALIGNING'
    RECOVER_SCAN   = 'RECOVER_SCAN'   # safeguard: barrido ±30° al perder el QR
    APPROACH_FINAL = 'APPROACH_FINAL'
    HOLD           = 'HOLD'
    BACK_AWAY      = 'BACK_AWAY'
    DELIVERY       = 'DELIVERY'
    ABORT          = 'ABORT'


# VFH+ bypass ON: estados donde la evasión debe estar inhibida.
_BYPASS_STATES = {_S.SEARCH_QR, _S.ALIGNING, _S.APPROACH_FINAL, _S.HOLD}

# Mapa zona → comando lift + estado de confirmación esperado en /lift_done
_ZONE_LIFT = {
    'rack':     ('n1', 'AT_N1'),
    'conveyor': ('n2', 'AT_N2'),
}


# ══════════════════════════════════════════════════════════════════════════════
class QRAlignNode(Node):

    def __init__(self):
        super().__init__('qr_align_node')

        # ── Parámetros ────────────────────────────────────────────────────
        self.declare_parameter('align_stop_dist',     0.35)
        self.declare_parameter('approach_final_dist', 0.05)
        self.declare_parameter('align_lateral_tol',   0.03)
        self.declare_parameter('angle_tol_deg',       4.0)
        self.declare_parameter('goal_replan_dist',    0.06)
        self.declare_parameter('goal_replan_angle',   3.0)
        self.declare_parameter('back_away_speed',     0.10)
        self.declare_parameter('back_away_time',      1.8)
        self.declare_parameter('lift_timeout',        8.0)
        self.declare_parameter('align_timeout',       20.0)
        self.declare_parameter('approach_timeout',    15.0)
        self.declare_parameter('search_timeout',      10.0)
        self.declare_parameter('qr_timeout',          2.5)
        self.declare_parameter('cam_offset_deg',      0.0)
        self.declare_parameter('fsm_rate_hz',         20.0)
        self.declare_parameter('scan_range_deg',      30.0)
        self.declare_parameter('scan_speed_dps',      20.0)
        self.declare_parameter('scan_max_attempts',   3)
        self.declare_parameter('cam_fwd_m',   0.15)
        self.declare_parameter('cam_left_m',  0.07)

        self._p = lambda n: self.get_parameter(n).value

        # ── Estado FSM ────────────────────────────────────────────────────
        self._state       = _S.IDLE
        self._state_entry = time.monotonic()

        self._zone        = ''
        self._lift_cmd    = ''
        self._lift_expect = ''

        # ── Datos QR ──────────────────────────────────────────────────────
        self._qr_payload  = ''
        self._qr_angle    = 0.0    # grados, + = derecha
        self._qr_dist     = 999.0  # metros
        self._qr_stamp    = 0.0

        # ── Pose odométrica ───────────────────────────────────────────────
        self._rx   = 0.0
        self._ry   = 0.0
        self._rth  = 0.0           # yaw en radianes

        # ── A*/GoToGoal ───────────────────────────────────────────────────
        self._astar_status = ''
        # Último goal publicado (para detectar si hay que re-publicar)
        self._last_goal_x = None   # coordenada mundo del último goal publicado
        self._last_goal_y = None

        # ── RECOVER_SCAN — barrido de recuperación ────────────────────────
        # Estado al que se vuelve si el barrido encuentra el QR.
        self._scan_return_state = _S.SEARCH_QR
        # Fase del barrido: 'LEFT' | 'RIGHT' | 'CENTER'
        self._scan_phase        = 'LEFT'
        # Yaw odométrico al inicio de la fase actual (referencia de giro).
        self._scan_phase_start_yaw = 0.0
        self._scan_attempts        = 0

        # ── Lift ──────────────────────────────────────────────────────────
        self._lift_done_label = ''

        # ── QOS BEST_EFFORT para tópicos de sensor ────────────────────────
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Suscriptores ──────────────────────────────────────────────────
        self.create_subscription(String,   '/collect/trigger',   self._cb_trigger,      10)
        self.create_subscription(String,   '/aruco/qr',          self._cb_qr,           qos_be)
        self.create_subscription(Float32,  '/aruco/qr/distance', self._cb_qr_dist,      qos_be)
        self.create_subscription(Float32,  '/aruco/qr/angle',    self._cb_qr_angle,     qos_be)
        self.create_subscription(String,   '/lift_done',         self._cb_lift_done,    10)
        self.create_subscription(Odometry, '/odom',              self._cb_odom,         10)
        self.create_subscription(String,   '/astar/status',      self._cb_astar_status, 10)

        # ── Publicadores ──────────────────────────────────────────────────
        # /cmd_vel: usado SOLO durante BACK_AWAY (retroceso en línea recta)
        self._pub_cmd     = self.create_publisher(Twist,  '/cmd_vel',            10)
        self._pub_lift    = self.create_publisher(String, '/lift_auto',          10)
        self._pub_done    = self.create_publisher(String, '/collect/done',       10)
        self._pub_payload = self.create_publisher(String, '/collect/qr_payload', 10)
        # /astar/goal: goal mundo completo → A* planifica y ejecuta con GoToGoal
        self._pub_astar   = self.create_publisher(Pose2D, '/astar/goal',         10)
        # /goal: waypoint directo a GoToGoal (bypass A* para tramo final)
        self._pub_wp      = self.create_publisher(Pose2D, '/goal',               10)
        self._pub_active  = self.create_publisher(Bool,   '/align/active',       10)

        # ── Timer FSM ─────────────────────────────────────────────────────
        rate = float(self._p('fsm_rate_hz'))
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            'qr_align_node v3 listo (control 100% goal-based, sin PD cmd_vel)\n'
            f'  align_stop_dist={self._p("align_stop_dist")}m  '
            f'approach_final_dist={self._p("approach_final_dist")}m  '
            f'angle_tol={self._p("angle_tol_deg")}°'
        )

    # ══════════════════════════════════════════════════════════════════════
    # CALLBACKS
    # ══════════════════════════════════════════════════════════════════════

    def _cb_trigger(self, msg: String):
        cmd = msg.data.strip().lower()

        if cmd == 'abort':
            if self._state != _S.IDLE:
                self.get_logger().warn('[Collect] ABORT recibido por FSM')
                self._transition(_S.ABORT)
            return

        if cmd not in _ZONE_LIFT:
            self.get_logger().warn(f'[Collect] Zona desconocida: "{cmd}" — ignorado')
            return

        if self._state != _S.IDLE:
            self.get_logger().warn(
                f'[Collect] Trigger ignorado — estado actual: {self._state}')
            return

        self._zone       = cmd
        self._lift_cmd, self._lift_expect = _ZONE_LIFT[cmd]
        self.get_logger().info(
            f'[Collect] Trigger zona="{cmd}" → lift_cmd={self._lift_cmd}')

        self._set_vfh_bypass(True)
        self._transition(_S.SEARCH_QR)

    def _cb_qr(self, msg: String):
        payload = msg.data.strip()
        if payload:
            if payload != self._qr_payload:
                self.get_logger().info(f'[QR] Payload: {payload}')
                self._qr_payload = payload
                self._pub_payload.publish(String(data=payload))
            self._qr_stamp = time.monotonic()

    def _cb_qr_dist(self, msg: Float32):
        self._qr_dist  = float(msg.data)
        self._qr_stamp = time.monotonic()

    def _cb_qr_angle(self, msg: Float32):
        # El offset lateral ya se corrige en aruco_detector._angle_distance.
        # cam_offset_deg queda en 0.0; se preserva el parámetro por compatibilidad.
        self._qr_angle = float(msg.data) + float(self._p('cam_offset_deg'))
      
    def _cb_lift_done(self, msg: String):
        label = msg.data.strip()
        if label:
            self.get_logger().info(f'[Lift] /lift_done: {label}')
            self._lift_done_label = label

    def _cb_odom(self, msg: Odometry):
        self._rx  = msg.pose.pose.position.x
        self._ry  = msg.pose.pose.position.y
        q         = msg.pose.pose.orientation
        self._rth = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def _cb_astar_status(self, msg: String):
        prev = self._astar_status
        self._astar_status = msg.data.strip()
        if self._astar_status != prev:
            self.get_logger().debug(f'[A*] status: {self._astar_status}')

    # ══════════════════════════════════════════════════════════════════════
    # FSM TICK PRINCIPAL
    # ══════════════════════════════════════════════════════════════════════

    def _tick(self):
        s = self._state

        if s == _S.IDLE:
            return

        # ── SEARCH_QR ─────────────────────────────────────────────────────
        elif s == _S.SEARCH_QR:
            # Esperar hasta que el QR sea visible.
            # El movimiento de búsqueda lo gestiona la misión externa (FSM).
            if self._qr_visible():
                self.get_logger().info(
                    f'[SEARCH_QR] QR detectado dist={self._qr_dist:.2f}m '
                    f'angle={self._qr_angle:+.1f}° → ALIGNING')
                self._transition(_S.ALIGNING)
                return

            if self._time_in_state() > self._p('search_timeout'):
                self.get_logger().warn(
                    '[SEARCH_QR] Timeout sin QR → RECOVER_SCAN')
                self._start_recover_scan(return_to=_S.SEARCH_QR)
                return

        # ── ALIGNING ──────────────────────────────────────────────────────
        elif s == _S.ALIGNING:
            if not self._qr_visible():
                # Breve gracia de ruido de detección antes de reaccionar
                if (time.monotonic() - self._qr_stamp) > self._p('qr_timeout'):
                    self.get_logger().warn(
                        '[ALIGNING] QR perdido → RECOVER_SCAN')
                    self._start_recover_scan(return_to=_S.ALIGNING)
                return  # No publicar goals con datos viejos

            if self._time_in_state() > self._p('align_timeout'):
                self.get_logger().warn('[ALIGNING] Timeout → ABORT')
                self._transition(_S.ABORT)
                return

            # Error lateral y angular medidos desde base_link, no desde la cámara.
            # Se reutiliza la misma geometría de _compute_align_goal.
            CAM_FWD  = float(self._p('cam_fwd_m'))
            CAM_LEFT = float(self._p('cam_left_m'))
            cam_x = self._rx + CAM_FWD  * math.cos(self._rth) - CAM_LEFT * math.sin(self._rth)
            cam_y = self._ry + CAM_FWD  * math.sin(self._rth) + CAM_LEFT * math.cos(self._rth)
            bearing_cam   = self._rth + math.radians(self._qr_angle)
            qr_x = cam_x + self._qr_dist * math.cos(bearing_cam)
            qr_y = cam_y + self._qr_dist * math.sin(bearing_cam)
            bearing_robot = math.atan2(qr_y - self._ry, qr_x - self._rx)
            angle_err_robot = math.degrees(self._angle_diff(bearing_robot, self._rth))
            lateral_err = self._qr_dist * math.sin(math.radians(angle_err_robot))
            aligned = (
                abs(angle_err_robot) < self._p('angle_tol_deg')
                and abs(lateral_err)  < self._p('align_lateral_tol')
            )
            close_enough = self._qr_dist <= self._p('align_stop_dist')

            if aligned and close_enough:
                self.get_logger().info(
                    f'[ALIGNING] ✔ Centrado a {self._qr_dist:.3f}m '
                    f'(lateral={lateral_err*100:.1f}cm, '
                    f'angle={self._qr_angle:+.1f}°) → APPROACH_FINAL')
                self._publish_approach_final_goal()
                self._transition(_S.APPROACH_FINAL)
                return

            # Publicar goal de alineación (solo si cambió lo suficiente)
            self._publish_align_goal_if_needed()

        # ── RECOVER_SCAN ──────────────────────────────────────────────────
        elif s == _S.RECOVER_SCAN:
            self._tick_recover_scan()

        # ── APPROACH_FINAL ────────────────────────────────────────────────
        elif s == _S.APPROACH_FINAL:
            # Esperamos que A*/GoToGoal confirme llegada al waypoint final.
            # El goal ya fue publicado en la transición desde ALIGNING.
            if self._astar_status == 'GOAL_REACHED':
                self.get_logger().info('[APPROACH_FINAL] GOAL_REACHED → HOLD')
                self._transition(_S.HOLD)
                return

            if self._time_in_state() > self._p('approach_timeout'):
                self.get_logger().warn('[APPROACH_FINAL] Timeout → ABORT')
                self._transition(_S.ABORT)

        # ── HOLD ──────────────────────────────────────────────────────────
        elif s == _S.HOLD:
            # Entrada al estado: enviar comando de lift
            if self._time_in_state() < 0.1:
                self.get_logger().info(f'[HOLD] Subiendo lift: {self._lift_cmd}')
                self._lift_done_label = ''
                self._pub_lift.publish(String(data=self._lift_cmd))
                return

            # Esperar confirmación del lift
            if self._lift_done_label == self._lift_expect:
                # Lift en posición → elevar a HOLD para agarrar el pallet
                self.get_logger().info(
                    f'[HOLD] Lift en {self._lift_done_label} → elevando a hold')
                self._pub_lift.publish(String(data='hold'))
                self._lift_done_label = ''   # reset para esperar AT_HOLD
                # Avanzar directamente a BACK_AWAY luego de un breve settle
                # (el hold confirma la recogida; no esperamos otro /lift_done
                #  para mantener la secuencia simple y evitar falso bloqueo)
                self.get_logger().info('[HOLD] Pallet recogido → /collect/done SUCCESS')
                self._pub_done.publish(String(data='SUCCESS'))
                self._transition(_S.BACK_AWAY)
                return

            if self._time_in_state() > self._p('lift_timeout'):
                self.get_logger().error(
                    f'[HOLD] Timeout esperando {self._lift_expect} → ABORT')
                self._transition(_S.ABORT)

        # ── BACK_AWAY ─────────────────────────────────────────────────────
        elif s == _S.BACK_AWAY:
            # Único uso de cmd_vel en todo el nodo: retroceso en línea recta.
            # No usamos A* porque el espacio detrás del robot ya fue atravesado
            # y queremos una maniobra determinista de baja velocidad.
            elapsed = self._time_in_state()
            if elapsed < self._p('back_away_time'):
                cmd = Twist()
                cmd.linear.x = -abs(self._p('back_away_speed'))
                self._pub_cmd.publish(cmd)
            else:
                self._stop()
                self.get_logger().info('[BACK_AWAY] Retroceso completado → DELIVERY')
                # Re-habilitar VFH+ antes de ceder el control
                self._set_vfh_bypass(False)
                self._transition(_S.DELIVERY)

        # ── DELIVERY ──────────────────────────────────────────────────────
        elif s == _S.DELIVERY:
            # Este estado es un "paso de mano" a la FSM de misión.
            # qr_align_node se limpia a sí mismo y vuelve a IDLE.
            # La FSM de misión recibe la señal via /collect/done (ya enviado)
            # y toma el control para navegar al punto de entrega.
            self.get_logger().info(
                f'[DELIVERY] Control cedido a FSM de misión — '
                f'payload="{self._qr_payload}" zona="{self._zone}"')
            self._reset()
            self._transition(_S.IDLE)

        # ── ABORT ─────────────────────────────────────────────────────────
        elif s == _S.ABORT:
            self._stop()
            if self._lift_cmd:
                self._pub_lift.publish(String(data='down'))
            self.get_logger().warn('[Collect] ❌ ABORT')
            self._set_vfh_bypass(False)
            self._pub_done.publish(String(data='ABORT'))
            self._reset()
            self._transition(_S.IDLE)

    # ══════════════════════════════════════════════════════════════════════
    # RECOVER_SCAN — barrido de recuperación de QR
    # ══════════════════════════════════════════════════════════════════════

    def _start_recover_scan(self, return_to: str):
        """
        Inicia el barrido de recuperación.

        Parámetro
        ---------
        return_to : estado al que volver si el QR es encontrado.
                    (SEARCH_QR o ALIGNING)

        El barrido se hace en sitio (giro puro, sin traslación) en tres
        fases:
          LEFT  → gira +scan_range_deg  (izquierda, antihorario)
          RIGHT → gira −2×scan_range_deg (derecha, cruzando el centro)
          CENTER → gira +scan_range_deg  (vuelve al heading original)

        En cada tick se compara el yaw odométrico actual con el yaw al
        inicio de la fase para saber cuándo se completó el arco. La
        velocidad angular se publica en cmd_vel directamente — es el
        único uso de cmd_vel junto con BACK_AWAY, y es legítimo porque
        el robot está parado buscando visibilidad, no navegando.
        """
        self._scan_return_state    = return_to
        self._scan_phase           = 'LEFT'
        self._scan_phase_start_yaw = self._rth
        self.get_logger().warn(
            f'[RECOVER_SCAN] Iniciando barrido ±{self._p("scan_range_deg")}°  '
            f'(volveré a {return_to} si encuentro el QR)')
        self._transition(_S.RECOVER_SCAN)

    def _tick_recover_scan(self):
        """
        Ejecuta un tick del barrido de recuperación.

        Lógica por fase
        ---------------
        LEFT   → gira antihorario hasta alcanzar +scan_range_deg desde el
                 heading de entrada.
        RIGHT  → gira horario hasta alcanzar −scan_range_deg desde el
                 heading de entrada (recorre 2× el rango, cruzando el centro).
        CENTER → gira antihorario hasta volver al heading de entrada
                 (recorre +scan_range_deg).

        En cualquier fase: si el QR se vuelve visible, se para el robot
        y se hace la transición al estado de retorno configurado.

        Si las tres fases se completan sin detectar el QR → ABORT.

        Convención de ángulos
        ---------------------
        El yaw de odometría crece en sentido antihorario (positivo = izquierda).
        angular.z positivo = giro antihorario = hacia la izquierda.
        """
        # ── ¿Recuperamos el QR? ───────────────────────────────────────────
        if self._qr_visible():
            self._stop()
            self.get_logger().info(
                f'[RECOVER_SCAN] QR recuperado en fase {self._scan_phase} '
                f'(dist={self._qr_dist:.2f}m, angle={self._qr_angle:+.1f}°) '
                f'→ {self._scan_return_state}')
            self._transition(self._scan_return_state)
            return

        scan_range = self._p('scan_range_deg')
        scan_speed = math.radians(self._p('scan_speed_dps'))  # rad/s

        # Ángulo girado desde el inicio de la fase actual (con signo)
        delta_yaw_deg = math.degrees(
            self._angle_diff(self._rth, self._scan_phase_start_yaw)
        )

        if self._scan_phase == 'LEFT':
            # Objetivo: girar +scan_range_deg (antihorario)
            if delta_yaw_deg < scan_range:
                self._pub_cmd.publish(self._spin_cmd(+scan_speed))
            else:
                # Fase LEFT completada → pasar a RIGHT
                self._stop()
                self.get_logger().info(
                    f'[RECOVER_SCAN] LEFT completado ({delta_yaw_deg:+.1f}°) → RIGHT')
                self._scan_phase           = 'RIGHT'
                self._scan_phase_start_yaw = self._rth

        elif self._scan_phase == 'RIGHT':
            # Objetivo: girar −2×scan_range_deg (horario) desde el punto más izquierdo
            if delta_yaw_deg > -2.0 * scan_range:
                self._pub_cmd.publish(self._spin_cmd(-scan_speed))
            else:
                # Fase RIGHT completada → volver al centro
                self._stop()
                self.get_logger().info(
                    f'[RECOVER_SCAN] RIGHT completado ({delta_yaw_deg:+.1f}°) → CENTER')
                self._scan_phase           = 'CENTER'
                self._scan_phase_start_yaw = self._rth

        elif self._scan_phase == 'CENTER':
            # Objetivo: girar +scan_range_deg (antihorario, volver al heading original)
            if delta_yaw_deg < scan_range:
                self._pub_cmd.publish(self._spin_cmd(+scan_speed))
            else:
                # Barrido completo sin QR
                self._stop()
                self._scan_attempts = getattr(self, '_scan_attempts', 0) + 1
                max_attempts = 3
                self.get_logger().warn(
                    '[RECOVER_SCAN] Barrido completo sin QR '
                    f'(intento {self._scan_attempts}/{max_attempts})')
                if self._scan_attempts >= max_attempts:
                    self.get_logger().error('[RECOVER_SCAN] Sin QR tras todos los intentos → ABORT')
                    self._scan_attempts = 0
                    self._transition(_S.ABORT)
                else:
                    # Reiniciar el barrido desde el heading actual
                    self._scan_phase           = 'LEFT'
                    self._scan_phase_start_yaw = self._rth

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Diferencia angular con signo en [-π, π]: a − b."""
        d = a - b
        while d >  math.pi: d -= 2.0 * math.pi
        while d < -math.pi: d += 2.0 * math.pi
        return d

    @staticmethod
    def _spin_cmd(angular_z: float) -> Twist:
        """Twist de giro puro en sitio (linear.x = 0)."""
        cmd = Twist()
        cmd.angular.z = angular_z
        return cmd

    # ══════════════════════════════════════════════════════════════════════
    # PUBLICACIÓN DE GOALS  (núcleo del refactor)
    # ══════════════════════════════════════════════════════════════════════

    def _publish_align_goal_if_needed(self):
        goal = self._compute_align_goal(self._p('align_stop_dist'))

        # Hysteresis en coordenadas mundo, no en ángulo de cámara.
        # Así detecta cambios aunque vengan de rotación del robot o de nueva lectura.
        if self._last_goal_x is not None:
            dx = abs(goal.x - self._last_goal_x)
            dy = abs(goal.y - self._last_goal_y)
            if dx < self._p('goal_replan_dist') and dy < self._p('goal_replan_dist'):
                return

        self._pub_astar.publish(goal)
        self._last_goal_x = goal.x
        self._last_goal_y = goal.y

        self.get_logger().info(
            f'[ALIGNING] Goal → ({goal.x:.3f}, {goal.y:.3f}) θ={math.degrees(goal.theta):.1f}°  '
            f'[QR dist={self._qr_dist:.2f}m ang={self._qr_angle:+.1f}°]'
        )

    def _publish_approach_final_goal(self):
        """
        Publica el goal de acercamiento final (~5 cm del QR) directamente en
        /goal (waypoint a GoToGoal, bypass de A*).

        Por qué bypass de A* en este tramo:
          - El robot ya está a ≤35 cm del QR; el mapa de ocupación puede
            marcar ese espacio como libre o con resolución insuficiente.
          - Queremos una maniobra corta y precisa, no replanificación.
          - GoToGoal (PID puro) es suficiente para este último paso.
        """
        goal = self._compute_align_goal(self._p('approach_final_dist'))
        self._pub_wp.publish(goal)
        self.get_logger().info(
            f'[APPROACH_FINAL] Goal directo → ({goal.x:.3f}, {goal.y:.3f}) '
            f'θ={goal.theta:.2f}rad'
        )

    def _compute_align_goal(self, stop_dist: float) -> Pose2D:
        angle_rad = math.radians(self._qr_angle)

        CAM_FWD  = float(self._p('cam_fwd_m'))
        CAM_LEFT = float(self._p('cam_left_m'))

        # Posición de la cámara en mundo
        cam_x = self._rx + CAM_FWD  * math.cos(self._rth) - CAM_LEFT * math.sin(self._rth)
        cam_y = self._ry + CAM_FWD  * math.sin(self._rth) + CAM_LEFT * math.cos(self._rth)

        # Bearing cámara→QR (en mundo)
        bearing_cam = self._rth + angle_rad

        # Posición estimada del QR en mundo (medida desde la cámara)
        qr_x = cam_x + self._qr_dist * math.cos(bearing_cam)
        qr_y = cam_y + self._qr_dist * math.sin(bearing_cam)

        # Bearing base_link→QR (el robot debe quedar mirando esto, no el bearing de cámara)
        bearing_robot = math.atan2(qr_y - self._ry, qr_x - self._rx)

        # Goal: base_link a stop_dist metros del QR, a lo largo del eje base_link→QR
        gx = qr_x - stop_dist * math.cos(bearing_robot)
        gy = qr_y - stop_dist * math.sin(bearing_robot)

        goal = Pose2D()
        goal.x     = gx
        goal.y     = gy
        goal.theta = bearing_robot   # orientación correcta: base_link mira al QR
        return goal

    # ══════════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _qr_visible(self) -> bool:
        return (
            self._qr_payload != ''
            and (time.monotonic() - self._qr_stamp) < self._p('qr_timeout')
        )

    def _stop(self):
        """Para el robot publicando un Twist cero."""
        self._pub_cmd.publish(Twist())

    def _time_in_state(self) -> float:
        return time.monotonic() - self._state_entry

    def _set_vfh_bypass(self, active: bool):
        self._pub_active.publish(Bool(data=active))
        state_str = 'ON  (evasión inhibida)' if active else 'OFF (evasión normal)'
        self.get_logger().info(f'[VFH+] /align/active → {state_str}')

    def _transition(self, new_state: str):
        if new_state == self._state:
            return
        self.get_logger().info(f'[FSM] {self._state} → {new_state}')
        self._state       = new_state
        self._state_entry = time.monotonic()
        # Reset del último goal para forzar re-publicación al entrar a ALIGNING
        self._last_goal_x = None
        self._last_goal_y = None

    def _reset(self):
        self._zone            = ''
        self._lift_cmd        = ''
        self._lift_expect     = ''
        self._lift_done_label = ''
        self._qr_payload      = ''
        self._qr_angle        = 0.0
        self._qr_dist         = 999.0
        self._astar_status    = ''
        self._last_goal_x = None
        self._last_goal_y = None
        self._scan_return_state    = _S.SEARCH_QR
        self._scan_phase           = 'LEFT'
        self._scan_phase_start_yaw = 0.0
        self._scan_attempts = 0


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = QRAlignNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

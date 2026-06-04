#!/usr/bin/env python3
"""
mission_manager.py — Iolair FSM Central
========================================
Máquina de estados principal del sistema autónomo Iolair.

Arquitectura de topics
----------------------
  SUB:  /mission/mode        (std_msgs/String)   — comando desde HMI
        /astar/status        (std_msgs/String)   — PLANNING / MOVING / GOAL_REACHED / NO_PATH
        /aruco/qr            (std_msgs/String)   — payload QR detectado
        /aruco/id            (std_msgs/Int32)    — ID ArUco detectado
        /aruco/label         (std_msgs/String)   — label ArUco
        /aruco/distance      (std_msgs/Float32)  — distancia al marker
        /yolo/detecciones    (std_msgs/String)   — JSON detecciones YOLO
        /odom                (nav_msgs/Odometry) — pose actual
        /reflex_status       (std_msgs/String)   — estado Bug/IBA

  PUB:  /astar/goal          (geometry_msgs/Pose2D) — goal al planificador
        /cmd_vel             (geometry_msgs/Twist)   — stop de emergencia
        /mission/mode        (std_msgs/String)       — reflejo de modo activo al HMI
        /mission/state       (std_msgs/String)       — estado FSM actual
        /servo_command       (std_msgs/String)       — LIFT_UP / LIFT_DOWN / STOP
        /ekf/active_source   (std_msgs/String)       — fuente EKF activa (info)

Pipeline de navegación (ya existente)
--------------------------------------
  /astar/goal → astar_planner → /goal
              → go_to_goal    → /cmd_raw
              → bug_IBA       → /cmd_vel
              → controller    → /VelocitySetL/R

Estados FSM
-----------
  INIT          → IDLE
  IDLE          → NAVSELECT (al recibir /mission/mode)
  NAVSELECT     → TELEOP        (mode == "teleop")
                → VOICE_CMD     (mode == "voice")
                → AUTONAV_INIT  (mode == "auto")
                → MAPPING       (mode == "mapping")   [pre-misión]
  TELEOP        → IDLE
  VOICE_CMD     → IDLE
  AUTONAV_INIT  → ASTAR_EXPLORE
  ASTAR_EXPLORE → QR_ALIGN      (QR detectado)        [TODO: implementar]
                → IDLE          (stop manual desde HMI)
  QR_ALIGN      → COLLECT       (alineado)             [TODO: implementar]
  COLLECT       → GO2GOAL       (pallet recogido)      [TODO: implementar]
  GO2GOAL       → TRUCK_ALIGN   (arrived)
  TRUCK_ALIGN   → DROP_PALLET   (logo confirmado)      [TODO: implementar parcial]
  DROP_PALLET   → ASTAR_EXPLORE (siguiente objetivo)
"""

import json
import math
import time

import rclpy
from rclpy.node       import Node
from rclpy.qos        import QoSProfile, ReliabilityPolicy

from std_msgs.msg         import String, Bool
from std_msgs.msg         import Int32
from std_msgs.msg         import Float32
from geometry_msgs.msg    import Twist, Pose2D
from nav_msgs.msg         import Odometry


# ══════════════════════════════════════════════════════════════════════════════
# STATE CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

class State:
    INIT          = 'INIT'
    IDLE          = 'IDLE'
    NAVSELECT     = 'NAVSELECT'
    MAPPING       = 'MAPPING'
    TELEOP        = 'TELEOP'
    VOICE_CMD     = 'VOICE_CMD'
    AUTONAV_INIT  = 'AUTONAV_INIT'
    ASTAR_EXPLORE = 'ASTAR_EXPLORE'
    QR_ALIGN      = 'QR_ALIGN'       # TODO: implementar
    COLLECT       = 'COLLECT'        # TODO: implementar
    GO2GOAL       = 'GO2GOAL'
    TRUCK_ALIGN   = 'TRUCK_ALIGN'    # TODO: implementar (YOLO)
    DROP_PALLET   = 'DROP_PALLET'    # TODO: implementar
    MISSION_DONE  = 'MISSION_DONE'


# ══════════════════════════════════════════════════════════════════════════════
# LIFT POSES  (comandos para TangServoDriver)
# ══════════════════════════════════════════════════════════════════════════════

class LiftCmd:
    DEFAULT   = 'STOP'       # posición home/neutral
    LOW       = 'LIFT_DOWN'  # nivel suelo
    PICK      = 'LIFT_UP'    # levantar pallet
    TRANSPORT = 'LIFT_UP'    # portar (mismo nivel que PICK por ahora)
    DROP      = 'LIFT_DOWN'  # depositar


# ══════════════════════════════════════════════════════════════════════════════
# MISSION MANAGER NODE
# ══════════════════════════════════════════════════════════════════════════════

class MissionManagerNode(Node):

    def __init__(self):
        super().__init__('mission_manager')

        # ── Parámetros ────────────────────────────────────────────────────
        self.declare_parameter('fsm_rate_hz',       10.0)
        self.declare_parameter('goal_timeout_s',    90.0)   # timeout por waypoint
        self.declare_parameter('qr_detect_dist',     1.5)   # m — distancia máxima válida
        self.declare_parameter('truck_zone_id',      20)    # ArUco ID del camión (External WP)
        self.declare_parameter('exploration_file',   '')    # si vacío usa waypoints hardcoded

        self._fsm_rate     = self.get_parameter('fsm_rate_hz').value
        self._goal_timeout = self.get_parameter('goal_timeout_s').value
        self._qr_dist_max  = self.get_parameter('qr_detect_dist').value
        self._truck_id     = self.get_parameter('truck_zone_id').value
        self._expl_file    = self.get_parameter('exploration_file').value

        # ── Estado FSM ────────────────────────────────────────────────────
        self._state          = State.INIT
        self._prev_state     = None
        self._state_entry_t  = time.monotonic()

        # ── Datos de sensor ───────────────────────────────────────────────
        self._mission_mode   = ''          # último /mission/mode recibido
        self._astar_status   = 'IDLE'
        self._qr_payload     = ''          # último QR decodificado
        self._aruco_id       = -1
        self._aruco_label    = ''
        self._aruco_dist     = 999.0
        self._yolo_detections= []          # lista de dicts {class, conf}
        self._reflex_status  = 'PASS'
        self._robot_x        = 0.0
        self._robot_y        = 0.0
        self._robot_theta    = 0.0

        # ── Estado de misión ──────────────────────────────────────────────
        self._exploration_waypoints = self._default_exploration_waypoints()
        self._expl_idx       = 0           # índice waypoint exploración
        self._goal_sent_t    = -1.0
        self._pallet_client  = ''          # cliente del QR leído
        self._truck_goal     = None        # Pose2D del camión destino
        self._pallets_done   = 0

        # ── QOS ───────────────────────────────────────────────────────────
        best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)

        # ── Suscriptores ──────────────────────────────────────────────────
        self.create_subscription(String,   '/mission/mode',     self._cb_mode,    10)
        self.create_subscription(String,   '/astar/status',     self._cb_astar,   10)
        self.create_subscription(String,   '/aruco/qr',         self._cb_qr,      10)
        self.create_subscription(Int32,    '/aruco/id',         self._cb_aruco_id,10)
        self.create_subscription(String,   '/aruco/label',      self._cb_aruco_lbl,10)
        self.create_subscription(Float32,  '/aruco/distance',   self._cb_aruco_d, 10)
        self.create_subscription(String,   '/yolo/detecciones', self._cb_yolo,    10)
        self.create_subscription(Odometry, '/odom',             self._cb_odom,    best_effort)
        self.create_subscription(String,   '/reflex_status',    self._cb_reflex,  10)

        # ── Publicadores ──────────────────────────────────────────────────
        self._pub_goal    = self.create_publisher(Pose2D,  '/astar/goal',       10)
        self._pub_cmd     = self.create_publisher(Twist,   '/cmd_vel',          10)
        self._pub_mode    = self.create_publisher(String,  '/mission/mode',     10)
        self._pub_state   = self.create_publisher(String,  '/mission/state',    10)
        self._pub_servo   = self.create_publisher(String,  '/servo_command',    10)
        self._pub_ekf_src = self.create_publisher(String,  '/ekf/active_source',10)

        # ── Timer principal FSM ───────────────────────────────────────────
        dt = 1.0 / self._fsm_rate
        self.create_timer(dt, self._fsm_tick)

        self.get_logger().info('MissionManager iniciado — estado: INIT')

    # ══════════════════════════════════════════════════════════════════════
    # CALLBACKS DE SENSORES
    # ══════════════════════════════════════════════════════════════════════

    def _cb_mode(self, msg: String):
        """Recibe comando de modo desde HMI (/mission/mode)."""
        new_mode = msg.data.strip().lower()
        if new_mode != self._mission_mode:
            self.get_logger().info(f'[HMI] /mission/mode: {new_mode}')
        self._mission_mode = new_mode

    def _cb_astar(self, msg: String):
        self._astar_status = msg.data

    def _cb_qr(self, msg: String):
        payload = msg.data.strip()
        if payload and payload != self._qr_payload:
            self.get_logger().info(f'[Percepción] QR detectado: {payload}')
            self._qr_payload = payload

    def _cb_aruco_id(self, msg: Int32):
        self._aruco_id = msg.data

    def _cb_aruco_lbl(self, msg: String):
        self._aruco_label = msg.data

    def _cb_aruco_d(self, msg: Float32):
        self._aruco_dist = msg.data

    def _cb_yolo(self, msg: String):
        try:
            self._yolo_detections = json.loads(msg.data)
        except Exception:
            self._yolo_detections = []

    def _cb_odom(self, msg: Odometry):
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self._robot_theta = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def _cb_reflex(self, msg: String):
        self._reflex_status = msg.data

    # ══════════════════════════════════════════════════════════════════════
    # FSM TICK PRINCIPAL
    # ══════════════════════════════════════════════════════════════════════

    def _fsm_tick(self):
        """Se ejecuta a _fsm_rate Hz. Evalúa transiciones y ejecuta acciones."""

        # Publicar estado actual al dashboard
        self._pub_state.publish(String(data=self._state))

        # ── INIT ──────────────────────────────────────────────────────────
        if self._state == State.INIT:
            self._log_state_entry()
            self._servo(LiftCmd.DEFAULT)
            self._transition(State.IDLE)

        # ── IDLE ──────────────────────────────────────────────────────────
        elif self._state == State.IDLE:
            self._log_state_entry()
            # Esperar comando del HMI
            if self._mission_mode in ('teleop', 'voice', 'auto', 'mapping'):
                self._transition(State.NAVSELECT)

        # ── NAVSELECT ─────────────────────────────────────────────────────
        elif self._state == State.NAVSELECT:
            self._log_state_entry()
            mode = self._mission_mode

            if mode == 'teleop':
                self.get_logger().info('[FSM] → TELEOP')
                self._transition(State.TELEOP)

            elif mode == 'voice':
                self.get_logger().info('[FSM] → VOICE_CMD')
                self._transition(State.VOICE_CMD)

            elif mode == 'auto':
                self.get_logger().info('[FSM] → AUTONAV_INIT')
                self._transition(State.AUTONAV_INIT)

            elif mode == 'mapping':
                self.get_logger().info('[FSM] → MAPPING')
                self._transition(State.MAPPING)

        # ── MAPPING ───────────────────────────────────────────────────────
        elif self._state == State.MAPPING:
            self._log_state_entry()
            # En modo mapping el robot es teleoperado mientras slam_node
            # construye el mapa. El manager solo monitorea el comando de stop.
            if self._mission_mode == 'stop' or self._mission_mode == '':
                self._stop_robot()
                self._transition(State.IDLE)

        # ── TELEOP ────────────────────────────────────────────────────────
        elif self._state == State.TELEOP:
            self._log_state_entry()
            # El control proviene del HMI via /cmd_vel directo.
            # Salir cuando el modo cambia.
            if self._mission_mode != 'teleop':
                self._stop_robot()
                self._transition(State.IDLE)

        # ── VOICE_CMD ─────────────────────────────────────────────────────
        elif self._state == State.VOICE_CMD:
            self._log_state_entry()
            # voice_command_node publica en /voice_command, el robot responde.
            # El manager solo gestiona la salida del modo.
            if self._mission_mode != 'voice':
                self._stop_robot()
                self._transition(State.IDLE)

        # ── AUTONAV_INIT ──────────────────────────────────────────────────
        elif self._state == State.AUTONAV_INIT:
            self._log_state_entry()
            # Inicializar variables de exploración
            self._expl_idx    = 0
            self._pallet_client = ''
            self._qr_payload    = ''
            self._pallets_done  = 0
            self.get_logger().info(
                f'[AutoNav] Iniciando misión con {len(self._exploration_waypoints)} waypoints')
            self._transition(State.ASTAR_EXPLORE)

        # ── ASTAR_EXPLORE ─────────────────────────────────────────────────
        elif self._state == State.ASTAR_EXPLORE:
            self._log_state_entry()

            # Stop manual desde HMI
            if self._mission_mode == 'stop' or self._mission_mode == 'teleop':
                self.get_logger().warn('[AutoNav] Stop manual → IDLE')
                self._stop_robot()
                self._transition(State.IDLE)
                return

            # ── TODO: detección de QR ──────────────────────────────────
            # Cuando QR_ALIGN esté implementado, descomentar:
            #
            # if (self._qr_payload
            #         and self._aruco_dist < self._qr_dist_max
            #         and self._aruco_dist > 0.1):
            #     self.get_logger().info(
            #         f'[AutoNav] QR detectado: {self._qr_payload} '
            #         f'(dist={self._aruco_dist:.2f} m)')
            #     self._pallet_client = self._qr_payload
            #     self._stop_robot()
            #     self._transition(State.QR_ALIGN)
            #     return
            # ──────────────────────────────────────────────────────────────

            # Navegación por waypoints de exploración
            self._run_exploration()

        # ── QR_ALIGN ──────────────────────────────────────────────────────
        elif self._state == State.QR_ALIGN:
            # TODO: Implementar alineación visual con ArUco
            # Centrar el marker en el frame de cámara usando /aruco/angle
            # y /aruco/distance para maniobrar hasta distancia de recolección.
            #
            # Condición de transición:
            #   abs(aruco_angle) < 0.05 rad  AND  aruco_dist < 0.30 m
            #
            # Pseudocódigo:
            #   err_angle = aruco_angle
            #   err_dist  = aruco_dist - 0.25  (deseado)
            #   cmd.angular.z = -kp_angle * err_angle
            #   cmd.linear.x  =  kp_dist  * err_dist (si abs(err_angle) < thr)
            #   if aligned: transition(COLLECT)
            #
            self.get_logger().warn('[QR_ALIGN] Estado no implementado — TODO')
            # Por ahora transicionar directo a COLLECT (bypass visual alignment)
            # TODO: QUITAR esta línea cuando QR_ALIGN esté implementado
            self._transition(State.COLLECT)

        # ── COLLECT ───────────────────────────────────────────────────────
        elif self._state == State.COLLECT:
            # TODO: Implementar secuencia de recolección
            #
            # 1. Clasificar zona desde /aruco/label:
            #    - "rack"      → LIFT pose RACK
            #    - "conveyor"  → LIFT pose CONVEYOR
            #
            # 2. Secuencia:
            #    a) Bajar lift a pose LOW
            #    b) Avanzar hasta contacto (~0.05 m)
            #    c) Subir lift a pose PICK
            #    d) Retroceder para despejar zona
            #    e) Transicionar a GO2GOAL
            #
            # Pseudocódigo:
            #   zone = classify_zone(aruco_label)
            #   servo(LOW)    ; wait(1.5)
            #   advance(0.05) ; wait(1.0)
            #   servo(PICK)   ; wait(2.0)
            #   retreat(0.3)  ; wait(1.5)
            #   find_truck_goal()
            #   transition(GO2GOAL)
            #
            self.get_logger().warn('[COLLECT] Estado no implementado — TODO')
            # Determinar goal del camión (por ahora desde ArUco IDs 20-22 = External WPs)
            truck_goal = self._find_truck_goal()
            if truck_goal:
                self._truck_goal = truck_goal
                self._servo(LiftCmd.TRANSPORT)
                self._transition(State.GO2GOAL)
            else:
                # Sin goal de camión, volver a explorar
                self.get_logger().warn('[COLLECT] No se encontró goal de camión → explorar')
                self._transition(State.ASTAR_EXPLORE)

        # ── GO2GOAL ───────────────────────────────────────────────────────
        elif self._state == State.GO2GOAL:
            self._log_state_entry()

            # Stop manual
            if self._mission_mode == 'stop':
                self._stop_robot()
                self._servo(LiftCmd.DEFAULT)
                self._transition(State.IDLE)
                return

            if self._truck_goal is None:
                self.get_logger().error('[GO2GOAL] No hay goal de camión — abortando')
                self._transition(State.IDLE)
                return

            # Enviar goal al A* la primera vez que entramos al estado
            if self._prev_state != State.GO2GOAL:
                self._send_goal(self._truck_goal.x, self._truck_goal.y)

            # Monitorear llegada
            if self._astar_status == 'GOAL_REACHED':
                self.get_logger().info('[GO2GOAL] Llegado a zona de camión')
                self._stop_robot()
                self._transition(State.TRUCK_ALIGN)

            # Timeout
            elif (self._goal_sent_t > 0 and
                  time.monotonic() - self._goal_sent_t > self._goal_timeout):
                self.get_logger().warn('[GO2GOAL] Timeout — reintentando')
                self._send_goal(self._truck_goal.x, self._truck_goal.y)

        # ── TRUCK_ALIGN ───────────────────────────────────────────────────
        elif self._state == State.TRUCK_ALIGN:
            # TODO: Implementar alineación con YOLO
            #
            # 1. Activar yolo_vision node (ya corre desde manchester.launch)
            # 2. Suscribirse a /yolo/detecciones
            # 3. Buscar clase == self._pallet_client (Cliente A/B/C)
            # 4. Calcular error de centrado: err_x = det.bbox.cx - frame_w/2
            # 5. Girar hasta err_x < 20 px
            # 6. Transicionar a DROP_PALLET
            #
            # Pseudocódigo:
            #   for det in yolo_detections:
            #       if det['class'] matches pallet_client:
            #           err = det['bbox_cx'] - 160  (frame 320px)
            #           cmd.angular.z = -kp * err / 160
            #           if abs(err) < 20: transition(DROP_PALLET)
            #
            self.get_logger().warn('[TRUCK_ALIGN] Estado YOLO no implementado — TODO')

            # Por ahora: si hay detecciones de YOLO asumir alineado
            if self._yolo_detections:
                cls_found = [d.get('class','') for d in self._yolo_detections]
                self.get_logger().info(f'[TRUCK_ALIGN] YOLO detectó: {cls_found}')
                self._stop_robot()
                self._transition(State.DROP_PALLET)
            elif self._time_in_state() > 15.0:
                # Timeout YOLO — depositar igual (degraded mode)
                self.get_logger().warn('[TRUCK_ALIGN] Timeout YOLO — depositando sin alineación')
                self._transition(State.DROP_PALLET)

        # ── DROP_PALLET ───────────────────────────────────────────────────
        elif self._state == State.DROP_PALLET:
            # TODO: Refinar secuencia de depósito
            #
            # Secuencia básica implementada:
            # 1. Bajar lift
            # 2. Esperar
            # 3. Retroceder
            # 4. Subir lift a DEFAULT
            # 5. Loop a ASTAR_EXPLORE
            #
            self._log_state_entry()
            elapsed = self._time_in_state()

            if elapsed < 0.5:
                self._servo(LiftCmd.DROP)        # Bajar

            elif elapsed < 2.5:
                pass                             # Esperar depósito

            elif elapsed < 4.0:
                cmd = Twist()
                cmd.linear.x = -0.10             # Retroceder
                self._pub_cmd.publish(cmd)

            elif elapsed < 5.5:
                self._stop_robot()
                self._servo(LiftCmd.DEFAULT)     # Lift a default

            else:
                self._pallets_done += 1
                self.get_logger().info(
                    f'[DROP_PALLET] Pallet depositado #{self._pallets_done}')
                self._pallet_client = ''
                self._qr_payload    = ''
                self._truck_goal    = None
                self._transition(State.ASTAR_EXPLORE)

        # ── MISSION_DONE ──────────────────────────────────────────────────
        elif self._state == State.MISSION_DONE:
            self._log_state_entry()
            self._stop_robot()
            self._servo(LiftCmd.DEFAULT)
            self.get_logger().info(
                f'✅ [MISIÓN COMPLETA] Pallets procesados: {self._pallets_done}')
            # Esperar siguiente comando HMI
            if self._mission_mode in ('teleop', 'voice', 'auto'):
                self._transition(State.IDLE)

    # ══════════════════════════════════════════════════════════════════════
    # EXPLORACIÓN A*
    # ══════════════════════════════════════════════════════════════════════

    def _run_exploration(self):
        """Gestiona el ciclo de exploración por waypoints."""

        # ¿Misión completada? (recorrimos todos los waypoints)
        if self._expl_idx >= len(self._exploration_waypoints):
            self.get_logger().info('[Exploración] Todos los waypoints visitados')
            self._expl_idx = 0          # loop continuo
            # TODO: si se detectaron todos los pallets → MISSION_DONE

        wp = self._exploration_waypoints[self._expl_idx]
        wx, wy = wp['x'], wp['y']

        # Enviar goal si es la primera vez en este waypoint
        #   (o si acabamos de entrar al estado)
        if self._prev_state != State.ASTAR_EXPLORE and self._expl_idx == 0:
            self._send_goal(wx, wy)
            return

        # Monitorear progreso
        if self._astar_status == 'GOAL_REACHED':
            self.get_logger().info(
                f'[Exploración] WP {self._expl_idx+1}/{len(self._exploration_waypoints)} '
                f'({wx:.2f},{wy:.2f}) alcanzado')
            self._expl_idx += 1
            if self._expl_idx < len(self._exploration_waypoints):
                nwp = self._exploration_waypoints[self._expl_idx]
                self._send_goal(nwp['x'], nwp['y'])

        elif self._astar_status == 'NO_PATH':
            self.get_logger().warn(
                f'[Exploración] NO_PATH en WP {self._expl_idx} — saltando')
            self._expl_idx += 1
            if self._expl_idx < len(self._exploration_waypoints):
                nwp = self._exploration_waypoints[self._expl_idx]
                self._send_goal(nwp['x'], nwp['y'])

        elif (self._goal_sent_t > 0 and
              time.monotonic() - self._goal_sent_t > self._goal_timeout):
            self.get_logger().warn(
                f'[Exploración] Timeout en WP {self._expl_idx} — reintentando')
            self._send_goal(wx, wy)

    # ══════════════════════════════════════════════════════════════════════
    # UTILIDADES
    # ══════════════════════════════════════════════════════════════════════

    def _send_goal(self, x: float, y: float):
        """Publica un goal al astar_planner."""
        msg       = Pose2D()
        msg.x     = float(x)
        msg.y     = float(y)
        msg.theta = 0.0
        self._pub_goal.publish(msg)
        self._goal_sent_t  = time.monotonic()
        self._astar_status = 'PLANNING'
        self.get_logger().info(f'[Goal] → ({x:.2f}, {y:.2f})')

    def _stop_robot(self):
        """Publica Twist cero para detener el robot."""
        self._pub_cmd.publish(Twist())

    def _servo(self, cmd: str):
        """Publica comando al TangServoDriver."""
        self._pub_servo.publish(String(data=cmd))
        self.get_logger().info(f'[Lift] → {cmd}')

    def _transition(self, new_state: str):
        """Cambia de estado y registra la transición."""
        if new_state != self._state:
            self.get_logger().info(
                f'[FSM] {self._state} → {new_state}')
            self._prev_state    = self._state
            self._state         = new_state
            self._state_entry_t = time.monotonic()

    def _log_state_entry(self):
        """Loggea una sola vez al entrar a un estado."""
        if self._prev_state != self._state:
            pass  # ya loggeado en _transition

    def _time_in_state(self) -> float:
        """Segundos transcurridos en el estado actual."""
        return time.monotonic() - self._state_entry_t

    def _find_truck_goal(self) -> Pose2D | None:
        """
        Determina el goal del camión destino según el cliente del pallet.

        La correspondencia cliente→ArUco ID viene de aruco_landmarks.yaml:
          External WPs: IDs 0-2 (marcadores de camiones)
          El QR del pallet contiene el nombre del cliente.

        TODO: leer coordenadas reales desde landmarks_yaml cuando estén calibradas.
        Por ahora usa valores aproximados del arena.
        """
        # Mapping cliente → coordenadas goal del camión
        # Ajustar con coordenadas reales del arena una vez medidas
        client_goals = {
            'Cliente A': (2.06,  1.06),   # ArUco landmark 10
            'Cliente B': (2.06, -1.47),   # ArUco landmark 9
            'Cliente C': (-1.78, 1.84),   # ArUco landmark 2
        }

        # Si tenemos cliente identificado del QR, usarlo
        if self._pallet_client in client_goals:
            x, y = client_goals[self._pallet_client]
            goal = Pose2D()
            goal.x = x
            goal.y = y
            return goal

        # Si no hay cliente identificado, ir al landmark ArUco más cercano
        # entre los External WPs (IDs 20-22)
        self.get_logger().warn(
            f'[find_truck_goal] Cliente "{self._pallet_client}" no reconocido '
            f'— usando goal por defecto')
        goal = Pose2D()
        goal.x = 2.06
        goal.y = 0.0
        return goal

    def _default_exploration_waypoints(self) -> list:
        """
        Waypoints de exploración del arena.
        Cargados desde exploration_waypoints.yaml si está disponible,
        o usando los valores calibrados del arena real.
        """
        # Si se especificó un archivo externo, cargarlo
        if self._expl_file:
            try:
                import yaml
                with open(self._expl_file, 'r') as f:
                    data = yaml.safe_load(f)
                wps = [{'x': float(wp['x']), 'y': float(wp['y'])}
                       for wp in data.get('waypoints', [])]
                if wps:
                    self.get_logger().info(
                        f'[Waypoints] Cargados {len(wps)} desde {self._expl_file}')
                    return wps
            except Exception as e:
                self.get_logger().error(f'[Waypoints] Error cargando YAML: {e}')

        # Waypoints calibrados del arena real (de exploration_waypoints.yaml)
        return [
            {'x': -1.046, 'y': -0.051},  # WP 1
            {'x': -2.043, 'y':  0.004},  # WP 2
            {'x': -1.976, 'y':  1.109},  # WP 3
            {'x': -0.617, 'y':  1.089},  # WP 4
            {'x':  1.054, 'y':  1.009},  # WP 5
            {'x':  1.174, 'y': -0.050},  # WP 6
            {'x':  1.099, 'y': -1.356},  # WP 7
            {'x': -0.045, 'y': -1.330},  # WP 8
            {'x':  0.030, 'y':  0.500},  # WP 9 (centro)
            {'x': -0.947, 'y': -1.286},  # WP 10
            {'x': -1.956, 'y': -1.324},  # WP 11
            {'x': -1.961, 'y': -0.047},  # WP 12
        ]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = MissionManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

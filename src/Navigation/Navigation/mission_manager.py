#!/usr/bin/env python3
"""
mission_manager.py — Iolair HFSM Central  (refactor v2.1)
===========================================================
Máquina de estados jerárquica para el robot Iolair.

v2.1 — cambios respecto a v2.0
---------------------------------
  • LOC_RECOVERY eliminado: los nodos MCL/EKF/aruco_localizer manejan
    su propia robustez. 3 NO_PATH consecutivos escalan a NAV_RECOVERY.
  • _is_state_entry() reemplazado por flag booleano _just_entered — evita
    dependencia frágil en threshold de tiempo.
  • Bug corregido en COLLECT_APPROACH: elapsed_since_done mal calculado.
  • _reset_lift_flags() centraliza el reset de _lift_cmd_sent /
    _lift_done_received / _lift_done_label antes de cada secuencia.
  • TRUCK_ALIGN delegado a truck_align_node (nodo independiente) vía
    /truck_align/cmd  y  /truck_align/result.

Arquitectura de topics
-----------------------
  SUB:  /mission/mode          (String)   auto/stop/estop/teleop/resume/reset
        /emergency_stop        (Bool)     ★ Topic dedicado HMI — True activa E_STOP
                                          inmediatamente en el hilo del callback,
                                          sin esperar el siguiente tick del FSM.
                                          False es ignorado (el reset es por /mission/mode).
        /astar/status          (String)   PLANNING/MOVING/GOAL_REACHED/NO_PATH
        /aruco/id              (Int32)
        /aruco/label           (String)
        /aruco/angle           (Float32)  rad, + = derecha
        /aruco/distance        (Float32)  m
        /aruco/qr              (String)
        /aruco/qr/distance     (Float32)  m
        /lift_done             (String)   AT_N1/AT_N2/HOLD/DOWN
        /lift_state            (String)
        /odom                  (Odometry)
        /reflex_status         (String)   PASS/WARN/EMERGENCY
        /truck_align/result    (String)   ALIGNED/FAILED/TIMEOUT
        /truck_align/status    (String)   IDLE/SCAN/ALIGN/DONE — heartbeat

  PUB:  /astar/goal            (Pose2D)
        /astar/cancel          (String)   "cancel" → descarta path activo en E_STOP
        /cmd_vel               (Twist)    cero en cada tick durante E_STOP
        /mission/state         (String)
        /mission/context       (String)   JSON
        /lift_auto             (String)   n1/n2/hold/down → spi_servo_node
        /recovery/active       (String)
        /truck_align/cmd       (String)   "align:<client>" → truck_align_node
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import Pose2D, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Bool, Float32, Int32, String


# ══════════════════════════════════════════════════════════════════════════════
# ENUMERACIONES
# ══════════════════════════════════════════════════════════════════════════════

class State(Enum):
    INIT                    = auto()
    IDLE                    = auto()
    NAVSELECT               = auto()
    MAPPING                 = auto()
    TELEOP                  = auto()
    VOICE_CMD               = auto()
    # ── Autonomous mission ────────────────────────────────────────────────
    AUTONAV_INIT            = auto()
    ASTAR_EXPLORE           = auto()
    QR_ALIGN                = auto()
    # ── Collect ───────────────────────────────────────────────────────────
    COLLECT_APPROACH        = auto()
    COLLECT_INSERT_ACQUIRE  = auto()
    # ── Delivery ──────────────────────────────────────────────────────────
    GO2GOAL                 = auto()
    TRUCK_ALIGN             = auto()   # delegado a truck_align_node
    DROP_PALLET             = auto()
    # ── End ───────────────────────────────────────────────────────────────
    MISSION_DONE            = auto()
    # ── Recovery (transversal) ────────────────────────────────────────────
    QR_RECOVERY             = auto()
    MANIP_RECOVERY          = auto()
    NAV_RECOVERY            = auto()
    YOLO_RECOVERY           = auto()
    E_STOP                  = auto()


RECOVERY_STATES = frozenset({
    State.QR_RECOVERY,
    State.MANIP_RECOVERY,
    State.NAV_RECOVERY,
    State.YOLO_RECOVERY,
    State.E_STOP,
})


class LiftCmd:
    N1   = 'n1'    # Conveyor height  (FPGA GO_N1 → AT_N1)
    N2   = 'n2'    # Rack height      (FPGA GO_N2 → AT_N2)
    HOLD = 'hold'  # Lift pallet      (FPGA GO_HOLD → HOLD)
    DOWN = 'down'  # Lower to ground  (FPGA GO_DOWN → IDLE/DOWN)


# ══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MissionContext:
    pallets_done:    int   = 0
    expl_idx:        int   = 0
    pallet_client:   str   = ''
    zone_type:       str   = ''     # 'conveyor' | 'rack'
    truck_goal_x:    float = 0.0
    truck_goal_y:    float = 0.0
    has_truck_goal:  bool  = False
    target_aruco_id: int   = -1
    qr_payload:      str   = ''
    pallet_acquired: bool  = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> 'MissionContext':
        return cls(**json.loads(s))


@dataclass
class SensorSnapshot:
    mission_mode:    str   = ''
    astar_status:    str   = 'IDLE'
    aruco_id:        int   = -1
    aruco_label:     str   = ''
    aruco_angle:     float = 0.0
    aruco_dist:      float = 999.0
    qr_payload:      str   = ''
    qr_dist:            float = 999.0
    yolo_detections:    list  = field(default_factory=list)
    lift_done:          str   = ''
    lift_state:         str   = ''
    reflex_status:      str   = 'PASS'
    truck_align_result: str   = ''
    truck_align_status: str   = 'IDLE'   # IDLE/SCAN/ALIGN/DONE — heartbeat del nodo
    robot_x:            float = 0.0
    robot_y:            float = 0.0
    robot_theta:        float = 0.0
    map_ready:          bool  = False


# ══════════════════════════════════════════════════════════════════════════════
# RECOVERY MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class RecoveryManager:
    """
    Contadores de recovery + guard anti-bucle.

    Reglas:
      • Cada recovery state tiene su propio counter.
      • Mismo recovery 3 veces seguidas en historial → escalar a IDLE.
      • Counter > MAX_RETRIES → escalar a IDLE.
      • reset(state) al completar exitosamente un estado protegido.
    """

    MAX_RETRIES = 3

    def __init__(self, logger):
        self._log = logger
        self._counts: dict[State, int] = {}
        self._history: deque[State] = deque(maxlen=6)

    def enter(self, recovery_state: State) -> bool:
        """
        Registra entrada a recovery. Devuelve True para proceder,
        False para escalar a IDLE.
        """
        self._counts[recovery_state] = self._counts.get(recovery_state, 0) + 1
        self._history.append(recovery_state)
        count = self._counts[recovery_state]

        self._log.warning(
            f'[RecoveryMgr] {recovery_state.name} intento {count}/{self.MAX_RETRIES}')

        # Guard: mismo recovery 3× seguidas
        if len(self._history) >= 3:
            last3 = list(self._history)[-3:]
            if all(s == recovery_state for s in last3):
                self._log.error(
                    f'[RecoveryMgr] {recovery_state.name} 3× seguidas → IDLE')
                return False

        if count > self.MAX_RETRIES:
            self._log.error(
                f'[RecoveryMgr] {recovery_state.name} superó {self.MAX_RETRIES} → IDLE')
            return False

        return True

    def reset(self, state: State):
        self._counts.pop(state, None)

    def reset_all(self):
        self._counts.clear()
        self._history.clear()


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

# QR_ALIGN P-controller
KP_ANGLE        = 1.2
KP_DIST         = 0.15
MAX_W_ALIGN     = 0.50
MAX_V_ALIGN     = 0.08
ALIGN_ANGLE_OK  = 0.06    # rad
ALIGN_DIST_OK   = 0.06    # m
ALIGN_TARGET_D  = 0.25    # m

# Lift timeouts
LIFT_MOVE_TIMEOUT = 6.0   # s

# Navegación
NAV_GOAL_TIMEOUT  = 90.0  # s
NAV_MAX_RETRIES   = 2
QR_DIST_MAX       = 1.5   # m
QR_DIST_MIN       = 0.15  # m

# Persistencia
CHECKPOINT_FILE   = '/tmp/iolair_checkpoint.json'

# ── Geometría / misión ────────────────────────────────────────────────────────

CLIENT_TRUCK_GOALS: dict[str, tuple[float, float]] = {
    'nalmart': (2.06,  1.06),
    'nemezon': (2.06, -1.47),
    'nepsi':   (1.08,  1.84),
}

ZONE_LIFT_CMD: dict[str, str] = {
    'conveyor': LiftCmd.N1,
    'rack':     LiftCmd.N2,
}

LANDMARK_POSITIONS: dict[int, tuple[float, float]] = {
    0: (-1.78, -1.84), 1: (-2.81,  0.18), 2: (-1.78,  1.84),
    3: ( 1.08,  1.84), 4: ( 1.08, -1.84), 5: (-0.41,  0.54),
    6: (-0.41, -0.64), 7: (-1.61,  0.54), 8: (-1.61, -0.64),
    9: ( 2.06, -1.47), 10:(2.06,   1.06),
}


# ══════════════════════════════════════════════════════════════════════════════
# NODO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class MissionManagerNode(Node):

    def __init__(self):
        super().__init__('mission_manager')

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('fsm_rate_hz',     10.0)
        self.declare_parameter('exploration_file', '')
        self.declare_parameter('total_pallets',    2)
        self.declare_parameter('load_checkpoint',  False)

        self._fsm_rate      = self.get_parameter('fsm_rate_hz').value
        self._expl_file     = self.get_parameter('exploration_file').value
        self._total_pallets = self.get_parameter('total_pallets').value
        self._load_ckpt     = self.get_parameter('load_checkpoint').value

        # ── Estado FSM ────────────────────────────────────────────────────────
        self._state:       State           = State.INIT
        self._prev_state:  Optional[State] = None
        self._state_entry_t: float         = time.monotonic()
        # Flag booleano limpio para "primer tick en este estado"
        self._just_entered: bool           = True

        self._ctx      = MissionContext()
        self._sensors  = SensorSnapshot()
        self._recovery = RecoveryManager(self.get_logger())

        self._exploration_waypoints = self._load_exploration_waypoints()

        # ── Variables de control inter-estado ─────────────────────────────────
        self._goal_sent_t:    float = -1.0
        self._goal_retries:   int   = 0

        # QR_ALIGN hysteresis
        self._align_ok_ticks: int   = 0

        # Watchdog pausable para TRUCK_ALIGN:
        # cuenta solo los segundos en que truck_align_status == 'IDLE'
        # (nodo no arrancó o murió). Se pausa si el nodo está SCAN/ALIGN.
        self._truck_watchdog_idle_s: float = 0.0   # acumulado en IDLE
        self._truck_watchdog_last_t: float = 0.0   # último tick procesado

        # Lift (flags compartidos entre COLLECT y DROP, siempre reseteados
        # por _reset_lift_flags() al entrar a cada estado que los usa)
        self._lift_cmd_sent:       bool  = False
        self._lift_done_received:  bool  = False
        self._lift_done_label:     str   = ''
        self._lift_sent_t:         float = 0.0

        # E_STOP — flag de activación directa desde callback (sin pasar por tick)
        self._estop_active: bool = False

        # Recovery origin
        self._recovery_origin: State = State.IDLE

        if self._load_ckpt:
            self._try_load_checkpoint()

        # ── QOS ───────────────────────────────────────────────────────────────
        best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        transient = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)

        # ── Suscriptores ──────────────────────────────────────────────────────
        self.create_subscription(String,       '/mission/mode',       self._cb_mode,        10)
        self.create_subscription(String,       '/astar/status',       self._cb_astar,       10)
        self.create_subscription(Int32,        '/aruco/id',           self._cb_aruco_id,    10)
        self.create_subscription(String,       '/aruco/label',        self._cb_aruco_lbl,   10)
        self.create_subscription(Float32,      '/aruco/angle',        self._cb_aruco_ang,   10)
        self.create_subscription(Float32,      '/aruco/distance',     self._cb_aruco_d,     10)
        self.create_subscription(String,       '/aruco/qr',           self._cb_qr,          10)
        self.create_subscription(Float32,      '/aruco/qr/distance',  self._cb_qr_dist,     10)
        self.create_subscription(String,       '/lift_done',          self._cb_lift_done,   10)
        self.create_subscription(String,       '/lift_state',         self._cb_lift_state,  10)
        self.create_subscription(Odometry,     '/odom',               self._cb_odom,        best_effort)
        self.create_subscription(String,       '/reflex_status',      self._cb_reflex,      10)
        self.create_subscription(OccupancyGrid,'/map',                self._cb_map,         transient)
        self.create_subscription(OccupancyGrid,'/slam_map',           self._cb_map,         10)
        self.create_subscription(String,       '/truck_align/result', self._cb_truck_result,  10)
        self.create_subscription(String,       '/truck_align/status', self._cb_truck_status,  10)
        self.create_subscription(Bool,         '/emergency_stop',     self._cb_emergency_stop,10)

        # ── Publicadores ──────────────────────────────────────────────────────
        self._pub_goal        = self.create_publisher(Pose2D,  '/astar/goal',       10)
        self._pub_cmd         = self.create_publisher(Twist,   '/cmd_vel',          10)
        self._pub_state       = self.create_publisher(String,  '/mission/state',    10)
        self._pub_context     = self.create_publisher(String,  '/mission/context',  10)
        self._pub_lift_auto   = self.create_publisher(String,  '/lift_auto',        10)
        self._pub_recovery    = self.create_publisher(String,  '/recovery/active',  10)
        self._pub_truck_cmd   = self.create_publisher(String,  '/truck_align/cmd',  10)
        self._pub_astar_cancel= self.create_publisher(String,  '/astar/cancel',     10)

        # ── Timer FSM ─────────────────────────────────────────────────────────
        self.create_timer(1.0 / self._fsm_rate, self._fsm_tick)

        self.get_logger().info(
            f'MissionManager v2.1 | rate={self._fsm_rate}Hz | '
            f'pallets={self._total_pallets} | WPs={len(self._exploration_waypoints)}')

    # ══════════════════════════════════════════════════════════════════════════
    # CALLBACKS
    # ══════════════════════════════════════════════════════════════════════════

    def _cb_mode(self, msg: String):
        v = msg.data.strip().lower()
        if v != self._sensors.mission_mode:
            self.get_logger().info(f'[HMI] mode → "{v}"')
        self._sensors.mission_mode = v

    def _cb_astar(self, msg: String):
        self._sensors.astar_status = msg.data

    def _cb_aruco_id(self, msg: Int32):
        self._sensors.aruco_id = msg.data

    def _cb_aruco_lbl(self, msg: String):
        self._sensors.aruco_label = msg.data

    def _cb_aruco_ang(self, msg: Float32):
        self._sensors.aruco_angle = msg.data

    def _cb_aruco_d(self, msg: Float32):
        self._sensors.aruco_dist = msg.data

    def _cb_qr(self, msg: String):
        payload = msg.data.strip()
        if payload and payload != self._sensors.qr_payload:
            self.get_logger().info(f'[Percepción] QR: "{payload}"')
        self._sensors.qr_payload = payload

    def _cb_qr_dist(self, msg: Float32):
        self._sensors.qr_dist = msg.data

    def _cb_yolo(self, msg: String):
        try:
            raw = json.loads(msg.data)
            # Normalizar bbox_cx si no viene del nodo YOLO
            for d in raw:
                if 'bbox_cx' not in d and 'bbox' in d:
                    x1, _, x2, _ = d['bbox']
                    d['bbox_cx'] = (x1 + x2) // 2
            self._sensors.yolo_detections = raw
        except Exception:
            self._sensors.yolo_detections = []

    def _cb_lift_done(self, msg: String):
        label = msg.data.strip()
        if label:
            self.get_logger().info(f'[Lift] done: "{label}"')
            self._lift_done_received = True
            self._lift_done_label    = label
            self._sensors.lift_done  = label

    def _cb_lift_state(self, msg: String):
        self._sensors.lift_state = msg.data

    def _cb_odom(self, msg: Odometry):
        self._sensors.robot_x = msg.pose.pose.position.x
        self._sensors.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self._sensors.robot_theta = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _cb_reflex(self, msg: String):
        self._sensors.reflex_status = msg.data
        if msg.data == 'EMERGENCY' and self._state != State.E_STOP:
            self.get_logger().error('[BugIBA] EMERGENCY → E_STOP inmediato')
            self._pub_cmd.publish(Twist())
            self._pub_lift_auto.publish(String(data=LiftCmd.DOWN))
            self._estop_active = True
            self._enter_recovery(State.E_STOP)

    def _cb_map(self, _):
        if not self._sensors.map_ready:
            self._sensors.map_ready = True
            self.get_logger().info('[Map] disponible')

    def _cb_truck_result(self, msg: String):
        result = msg.data.strip()
        if result:
            self.get_logger().info(f'[TruckAlign] result: "{result}"')
        self._sensors.truck_align_result = result

    def _cb_truck_status(self, msg: String):
        self._sensors.truck_align_status = msg.data.strip()

    def _cb_emergency_stop(self, msg: Bool):
        """
        Callback dedicado para /emergency_stop (std_msgs/Bool).
        
        True  -> E_STOP inmediato en el hilo del executor, sin esperar tick.
                 Publica cmd_vel=0, lift=down y cancela el path activo.
        False -> ignorado. El reset se hace con /mission/mode="reset".
        """
        if not msg.data:
            return
        if self._state == State.E_STOP:
            return
        self.get_logger().error('[/emergency_stop] Senial recibida -> E_STOP inmediato')
        # Freno total, cancelar path y bajar lift antes del proximo tick
        self._pub_cmd.publish(Twist())
        self._pub_lift_auto.publish(String(data=LiftCmd.DOWN))
        self._pub_astar_cancel.publish(String(data='cancel'))
        self._estop_active = True
        self._enter_recovery(State.E_STOP)

    # ══════════════════════════════════════════════════════════════════════════
    # FSM TICK
    # ══════════════════════════════════════════════════════════════════════════

    def _fsm_tick(self):
        # ── E_STOP tiene prioridad absoluta ───────────────────────────────
        # Dos fuentes: /emergency_stop callback (ya procesado, _estop_active=True)
        # o /mission/mode == 'estop' como fallback desde HMI.
        if self._state == State.E_STOP:
            # En E_STOP: publicar Twist cero en CADA tick para anular cualquier
            # /cmd_raw que GoToGoal o BugIBA puedan seguir publicando.
            self._pub_cmd.publish(Twist())
            # Blanquear /astar/goal con coords imposibles para que el planner
            # no procese ningún goal residual.
            _blank = Pose2D()
            _blank.x, _blank.y = 1e6, 1e6
            self._pub_goal.publish(_blank)
            # Publicar estado y delegar al handler
            self._pub_state.publish(String(data=self._state.name))
            self._pub_recovery.publish(String(data=self._state.name))
            self._state_estop()
            self._just_entered = False
            return

        if (not self._estop_active
                and self._sensors.mission_mode == 'estop'
                and self._state != State.E_STOP):
            self.get_logger().error('[HMI] mission_mode=estop -> E_STOP')
            self._pub_cmd.publish(Twist())
            self._pub_lift_auto.publish(String(data=LiftCmd.DOWN))
            self._pub_astar_cancel.publish(String(data='cancel'))
            self._estop_active = True
            self._enter_recovery(State.E_STOP)
            return

        self._pub_state.publish(String(data=self._state.name))
        if self._state in RECOVERY_STATES:
            self._pub_recovery.publish(String(data=self._state.name))

        handlers = {
            State.INIT:                     self._state_init,
            State.IDLE:                     self._state_idle,
            State.NAVSELECT:                self._state_navselect,
            State.MAPPING:                  self._state_mapping,
            State.TELEOP:                   self._state_teleop,
            State.VOICE_CMD:                self._state_voice_cmd,
            State.AUTONAV_INIT:             self._state_autonav_init,
            State.ASTAR_EXPLORE:            self._state_astar_explore,
            State.QR_ALIGN:                 self._state_qr_align,
            State.COLLECT_APPROACH:         self._state_collect_approach,
            State.COLLECT_INSERT_ACQUIRE:   self._state_collect_insert_acquire,
            State.GO2GOAL:                  self._state_go2goal,
            State.TRUCK_ALIGN:              self._state_truck_align,
            State.DROP_PALLET:              self._state_drop_pallet,
            State.MISSION_DONE:             self._state_mission_done,
            State.QR_RECOVERY:              self._recovery_qr,
            State.MANIP_RECOVERY:           self._recovery_manip,
            State.NAV_RECOVERY:             self._recovery_nav,
            State.YOLO_RECOVERY:            self._recovery_yolo,
            # State.E_STOP se despacha directamente al inicio de _fsm_tick
            # antes de llegar a este dict — no incluir aquí para evitar doble llamada.
        }
        handler = handlers.get(self._state)
        if handler:
            handler()
            # Apagar _just_entered tras el primer tick
            self._just_entered = False
        else:
            self.get_logger().error(f'[FSM] sin handler para {self._state}')

    # ══════════════════════════════════════════════════════════════════════════
    # ESTADOS DEL SISTEMA
    # ══════════════════════════════════════════════════════════════════════════

    def _state_init(self):
        self._lift_auto(LiftCmd.DOWN)
        self._transition(State.IDLE)

    def _state_idle(self):
        mode = self._sensors.mission_mode
        if mode in ('teleop', 'voice', 'auto', 'mapping'):
            self._transition(State.NAVSELECT)

    def _state_navselect(self):
        mapping = {
            'teleop':  State.TELEOP,
            'voice':   State.VOICE_CMD,
            'auto':    State.AUTONAV_INIT,
            'mapping': State.MAPPING,
        }
        dest = mapping.get(self._sensors.mission_mode)
        if dest:
            self._transition(dest)

    def _state_mapping(self):
        if self._sensors.mission_mode in ('stop', ''):
            self._stop_robot()
            self._transition(State.IDLE)

    def _state_teleop(self):
        if self._sensors.mission_mode != 'teleop':
            self._stop_robot()
            self._transition(State.IDLE)

    def _state_voice_cmd(self):
        if self._sensors.mission_mode != 'voice':
            self._stop_robot()
            self._transition(State.IDLE)

    # ══════════════════════════════════════════════════════════════════════════
    # AUTONOMOUS_MISSION
    # ══════════════════════════════════════════════════════════════════════════

    def _state_autonav_init(self):
        if not self._sensors.map_ready:
            if self._just_entered:
                self.get_logger().warning('[AutoNav] Esperando mapa...')
            return

        self._ctx            = MissionContext()
        self._recovery.reset_all()
        self._goal_retries   = 0
        self._align_ok_ticks = 0
        self.get_logger().info(
            f'[AutoNav] Misión iniciada | WPs={len(self._exploration_waypoints)} '
            f'| target={self._total_pallets} pallets')
        self._transition(State.ASTAR_EXPLORE)

    # ─────────────────────────────────────────────────────────────────────────

    def _state_astar_explore(self):
        s = self._sensors

        if s.mission_mode in ('stop', 'teleop'):
            self._stop_robot()
            self._transition(State.IDLE)
            return

        # ── Detección QR válida ───────────────────────────────────────────
        if (s.qr_payload
                and QR_DIST_MIN < s.qr_dist < QR_DIST_MAX):
            self.get_logger().info(
                f'[AutoNav] QR "{s.qr_payload}" dist={s.qr_dist:.2f}m → QR_ALIGN')
            self._ctx.qr_payload      = s.qr_payload
            self._ctx.pallet_client   = s.qr_payload
            self._ctx.target_aruco_id = s.aruco_id
            self._ctx.zone_type       = 'rack' if 'rack' in s.aruco_label.lower() else 'conveyor'
            self._stop_robot()
            self._transition(State.QR_ALIGN)
            return

        # ── Primer tick: enviar goal ───────────────────────────────────────
        if self._just_entered:
            self._goal_retries = 0
            self._send_goal_wp(self._ctx.expl_idx)
            return

        # ── Monitorear progreso ────────────────────────────────────────────
        if s.astar_status == 'GOAL_REACHED':
            self._recovery.reset(State.NAV_RECOVERY)
            self._goal_retries   = 0
            self._ctx.expl_idx   = (self._ctx.expl_idx + 1) % len(self._exploration_waypoints)
            self.get_logger().info(
                f'[Exploración] WP {self._ctx.expl_idx}/{len(self._exploration_waypoints)} alcanzado')
            self._send_goal_wp(self._ctx.expl_idx)

        elif s.astar_status == 'NO_PATH':
            self._goal_retries += 1
            self.get_logger().warning(
                f'[Exploración] NO_PATH WP {self._ctx.expl_idx} '
                f'(intento {self._goal_retries}/{NAV_MAX_RETRIES + 1})')
            if self._goal_retries > NAV_MAX_RETRIES:
                self.get_logger().error('[Exploración] NO_PATH repetido → NAV_RECOVERY')
                self._enter_recovery(State.NAV_RECOVERY)
                return
            # Saltar al siguiente WP
            self._ctx.expl_idx = (self._ctx.expl_idx + 1) % len(self._exploration_waypoints)
            self._send_goal_wp(self._ctx.expl_idx)

        elif self._goal_timeout():
            self._goal_retries += 1
            self.get_logger().warning(
                f'[Exploración] Timeout WP {self._ctx.expl_idx} '
                f'(intento {self._goal_retries})')
            if self._goal_retries > NAV_MAX_RETRIES:
                self.get_logger().error('[Exploración] Timeouts repetidos → NAV_RECOVERY')
                self._enter_recovery(State.NAV_RECOVERY)
                return
            self._send_goal_wp(self._ctx.expl_idx)

    # ─────────────────────────────────────────────────────────────────────────

    def _state_qr_align(self):
        s = self._sensors
        t = self._time_in_state()

        if s.mission_mode == 'stop':
            self._stop_robot()
            self._transition(State.IDLE)
            return

        # Pérdida de marcador objetivo
        if s.aruco_id != self._ctx.target_aruco_id and t > 1.5:
            self.get_logger().warning(
                f'[QR_ALIGN] ArUco {self._ctx.target_aruco_id} perdido → QR_RECOVERY')
            self._stop_robot()
            self._enter_recovery(State.QR_RECOVERY)
            return

        if t > 30.0:
            self.get_logger().warning('[QR_ALIGN] Timeout → QR_RECOVERY')
            self._stop_robot()
            self._enter_recovery(State.QR_RECOVERY)
            return

        # P-controller
        err_angle = s.aruco_angle
        err_dist  = s.aruco_dist - ALIGN_TARGET_D

        cmd = Twist()
        cmd.angular.z = max(-MAX_W_ALIGN,
                            min(MAX_W_ALIGN, -KP_ANGLE * err_angle))
        if abs(err_angle) < 0.15:
            cmd.linear.x = max(-MAX_V_ALIGN,
                               min(MAX_V_ALIGN,  KP_DIST  * err_dist))
        self._pub_cmd.publish(cmd)

        # Hysteresis: 5 ticks consecutivos OK
        if abs(err_angle) < ALIGN_ANGLE_OK and abs(err_dist) < ALIGN_DIST_OK:
            self._align_ok_ticks += 1
        else:
            self._align_ok_ticks = 0

        if self._align_ok_ticks >= 5:
            self.get_logger().info('[QR_ALIGN] Alineado → COLLECT_APPROACH')
            self._stop_robot()
            self._align_ok_ticks = 0
            self._recovery.reset(State.QR_RECOVERY)
            self._transition(State.COLLECT_APPROACH)

    # ══════════════════════════════════════════════════════════════════════════
    # COLLECT
    # ══════════════════════════════════════════════════════════════════════════

    def _state_collect_approach(self):
        """
        t=0   → retroceder 0.3 m (1.8 s a 0.13 m/s)
        t=1.8 → enviar lift N1/N2 (una sola vez)
        lift_done → avanzar a 0.06 m/s manteniendo alineación ArUco
        aruco_dist < 0.10 m → INSERT_ACQUIRE
        """
        s = self._sensors
        t = self._time_in_state()

        # ── Phase 0: retroceder ───────────────────────────────────────────
        if t < 1.8:
            cmd = Twist()
            cmd.linear.x = -0.13
            if s.aruco_id == self._ctx.target_aruco_id:
                cmd.angular.z = max(-0.3, min(0.3, -0.3 * s.aruco_angle))
            self._pub_cmd.publish(cmd)
            return

        # ── Phase 1: ajustar altura tool (una sola vez) ───────────────────
        if not self._lift_cmd_sent:
            self._stop_robot()
            lift_cmd = ZONE_LIFT_CMD.get(self._ctx.zone_type, LiftCmd.N1)
            self.get_logger().info(
                f'[APPROACH] zona={self._ctx.zone_type} → lift {lift_cmd}')
            self._reset_lift_flags()
            self._lift_auto(lift_cmd)
            self._lift_cmd_sent = True
            self._lift_sent_t   = time.monotonic()
            return

        # ── Phase 2: esperar lift_done ────────────────────────────────────
        if not self._lift_done_received:
            elapsed_lift = time.monotonic() - self._lift_sent_t
            if elapsed_lift > LIFT_MOVE_TIMEOUT:
                self.get_logger().error('[APPROACH] Timeout lift → MANIP_RECOVERY')
                self._enter_recovery(State.MANIP_RECOVERY)
            return

        # ── Phase 3: avanzar hacia pallet ────────────────────────────────
        if s.aruco_dist > 0.10:
            cmd = Twist()
            cmd.linear.x = 0.06
            if s.aruco_id == self._ctx.target_aruco_id:
                cmd.angular.z = max(-0.4, min(0.4, -0.5 * s.aruco_angle))
            self._pub_cmd.publish(cmd)
        else:
            self._stop_robot()
            self.get_logger().info('[APPROACH] Fork en posición → INSERT_ACQUIRE')
            self._transition(State.COLLECT_INSERT_ACQUIRE)

    def _state_collect_insert_acquire(self):
        """
        Entrada → lift HOLD
        lift_done==HOLD → retroceder hasta dist > 0.6 m
        dist ok → pallet_acquired = True → GO2GOAL
        """
        s = self._sensors

        # ── Phase 0: enviar HOLD (una sola vez) ──────────────────────────
        if not self._lift_cmd_sent:
            self.get_logger().info('[INSERT] Levantando pallet → HOLD')
            self._reset_lift_flags()
            self._lift_auto(LiftCmd.HOLD)
            self._lift_cmd_sent = True
            self._lift_sent_t   = time.monotonic()
            return

        # ── Phase 1: esperar HOLD ─────────────────────────────────────────
        if not self._lift_done_received:
            if time.monotonic() - self._lift_sent_t > LIFT_MOVE_TIMEOUT:
                self.get_logger().error('[INSERT] Timeout lift HOLD → MANIP_RECOVERY')
                self._enter_recovery(State.MANIP_RECOVERY)
            return

        if self._lift_done_label != 'HOLD':
            self.get_logger().error(
                f'[INSERT] lift_done="{self._lift_done_label}" ≠ HOLD → MANIP_RECOVERY')
            self._enter_recovery(State.MANIP_RECOVERY)
            return

        # ── Phase 2: retroceder con pallet ───────────────────────────────
        if s.aruco_dist < 0.6:
            cmd = Twist()
            cmd.linear.x = -0.08
            self._pub_cmd.publish(cmd)
            return

        # ── Phase 3: adquisición confirmada ──────────────────────────────
        self._stop_robot()
        self._ctx.pallet_acquired = True

        truck_goal = self._find_truck_goal()
        if truck_goal is None:
            self.get_logger().error('[INSERT] Sin truck_goal → NAV_RECOVERY')
            self._enter_recovery(State.NAV_RECOVERY)
            return

        self._ctx.truck_goal_x   = truck_goal[0]
        self._ctx.truck_goal_y   = truck_goal[1]
        self._ctx.has_truck_goal = True
        self.get_logger().info(
            f'[INSERT] Pallet adquirido ✓ camión=({truck_goal[0]:.2f},{truck_goal[1]:.2f})')
        self._recovery.reset(State.MANIP_RECOVERY)
        self._save_checkpoint()
        self._transition(State.GO2GOAL)

    # ══════════════════════════════════════════════════════════════════════════
    # GO2GOAL
    # ══════════════════════════════════════════════════════════════════════════

    def _state_go2goal(self):
        s = self._sensors

        if s.mission_mode == 'stop':
            self._stop_robot()
            self._transition(State.IDLE)
            return

        if not self._ctx.has_truck_goal:
            self.get_logger().error('[GO2GOAL] Sin truck_goal')
            self._transition(State.IDLE)
            return

        if self._just_entered:
            self._goal_retries = 0
            self._send_goal(self._ctx.truck_goal_x, self._ctx.truck_goal_y)
            return

        if s.astar_status == 'GOAL_REACHED':
            self.get_logger().info('[GO2GOAL] Llegado → TRUCK_ALIGN')
            self._stop_robot()
            self._recovery.reset(State.NAV_RECOVERY)
            self._transition(State.TRUCK_ALIGN)

        elif self._goal_timeout():
            self._goal_retries += 1
            if self._goal_retries > NAV_MAX_RETRIES:
                self.get_logger().warning('[GO2GOAL] Timeout repetido → NAV_RECOVERY')
                self._enter_recovery(State.NAV_RECOVERY)
                return
            self.get_logger().warning(
                f'[GO2GOAL] Timeout (intento {self._goal_retries})')
            self._send_goal(self._ctx.truck_goal_x, self._ctx.truck_goal_y)

    # ══════════════════════════════════════════════════════════════════════════
    # TRUCK_ALIGN  — delegado a truck_align_node
    # ══════════════════════════════════════════════════════════════════════════

    def _state_truck_align(self):
        """
        Envía el comando al truck_align_node y espera su resultado.

        Protocolo:
          PUB /truck_align/cmd    → "align:<client_name>"
          SUB /truck_align/result → "ALIGNED" | "FAILED" | "TIMEOUT"
          SUB /truck_align/status → "IDLE" | "SCAN" | "ALIGN" | "DONE"

        Watchdog pausable:
          Solo cuenta el tiempo en que truck_align_status == 'IDLE'
          (nodo no arrancó, murió, o nunca recibió el comando).
          Si el nodo está SCAN o ALIGN el timer se pausa — está trabajando.
          Umbral: 8 s acumulados en IDLE → YOLO_RECOVERY.
        """
        s = self._sensors

        # ── Primer tick: enviar comando y armar watchdog ──────────────────
        if self._just_entered:
            cmd_msg = f'align:{self._ctx.pallet_client}'
            self.get_logger().info(f'[TRUCK_ALIGN] → truck_align_node: "{cmd_msg}"')
            self._pub_truck_cmd.publish(String(data=cmd_msg))
            self._sensors.truck_align_result = ''
            self._truck_watchdog_idle_s = 0.0
            self._truck_watchdog_last_t = time.monotonic()
            return

        # ── Acumular tiempo idle (watchdog pausable) ──────────────────────
        now  = time.monotonic()
        dt   = now - self._truck_watchdog_last_t
        self._truck_watchdog_last_t = now

        node_active = s.truck_align_status in ('SCAN', 'ALIGN', 'DONE')
        if not node_active:
            self._truck_watchdog_idle_s += dt

        if self._truck_watchdog_idle_s > 8.0:
            self.get_logger().error(
                f'[TRUCK_ALIGN] Watchdog: nodo en IDLE por '
                f'{self._truck_watchdog_idle_s:.1f}s → YOLO_RECOVERY')
            self._enter_recovery(State.YOLO_RECOVERY)
            return

        # ── Procesar resultado ────────────────────────────────────────────
        result = s.truck_align_result

        if result == 'ALIGNED':
            self.get_logger().info('[TRUCK_ALIGN] Alineado ✓ → DROP_PALLET')
            self._sensors.truck_align_result = ''
            self._recovery.reset(State.YOLO_RECOVERY)
            self._transition(State.DROP_PALLET)

        elif result == 'FAILED':
            self.get_logger().warning('[TRUCK_ALIGN] FAILED → YOLO_RECOVERY')
            self._sensors.truck_align_result = ''
            self._enter_recovery(State.YOLO_RECOVERY)

        elif result == 'TIMEOUT':
            self.get_logger().warning('[TRUCK_ALIGN] TIMEOUT → YOLO_RECOVERY')
            self._sensors.truck_align_result = ''
            self._enter_recovery(State.YOLO_RECOVERY)

    # ══════════════════════════════════════════════════════════════════════════
    # DROP_PALLET
    # ══════════════════════════════════════════════════════════════════════════

    def _state_drop_pallet(self):
        t = self._time_in_state()

        # Phase 0: avanzar ligeramente al interior (0.8 s)
        if t < 0.8:
            cmd = Twist()
            cmd.linear.x = 0.07
            self._pub_cmd.publish(cmd)
            return

        self._stop_robot()

        # Phase 1: bajar fork (una sola vez)
        if not self._lift_cmd_sent:
            self.get_logger().info('[DROP] Bajando fork → DOWN')
            self._reset_lift_flags()
            self._lift_auto(LiftCmd.DOWN)
            self._lift_cmd_sent = True
            self._lift_sent_t   = time.monotonic()
            return

        # Phase 2: esperar DOWN
        if not self._lift_done_received:
            if time.monotonic() - self._lift_sent_t > LIFT_MOVE_TIMEOUT:
                self.get_logger().error('[DROP] Timeout lift DOWN → MANIP_RECOVERY')
                self._enter_recovery(State.MANIP_RECOVERY)
            return

        # Phase 3: retroceder (2.5 s después del lift_done)
        elapsed_since_lift_done = time.monotonic() - self._lift_sent_t
        if elapsed_since_lift_done < 2.5:
            cmd = Twist()
            cmd.linear.x = -0.10
            self._pub_cmd.publish(cmd)
            return

        # Phase 4: completar entrega
        self._stop_robot()
        self._ctx.pallet_acquired = False
        self._ctx.pallet_client   = ''
        self._ctx.qr_payload      = ''
        self._ctx.has_truck_goal  = False
        self._lift_cmd_sent       = False
        self._ctx.pallets_done   += 1

        self.get_logger().info(
            f'[DROP] Pallet #{self._ctx.pallets_done} entregado ✓')
        self._save_checkpoint()
        self._recovery.reset(State.MANIP_RECOVERY)

        if self._ctx.pallets_done >= self._total_pallets:
            self._transition(State.MISSION_DONE)
        else:
            self.get_logger().info(
                f'[DROP] Retomando exploración desde WP {self._ctx.expl_idx}')
            self._transition(State.ASTAR_EXPLORE)

    # ─────────────────────────────────────────────────────────────────────────

    def _state_mission_done(self):
        self._stop_robot()
        if self._just_entered:
            self._lift_auto(LiftCmd.DOWN)
            self.get_logger().info(
                f'✅ MISIÓN COMPLETA | pallets={self._ctx.pallets_done}')
        if self._sensors.mission_mode in ('auto', 'teleop', 'voice'):
            self._transition(State.IDLE)

    def _state_estop(self):
        """
        Paro de emergencia.

        Activacion:
          • /emergency_stop Bool(True)  — callback directo, sin delay de tick
          • /mission/mode == "estop"    — fallback HMI via topic de modo
          • /reflex_status == EMERGENCY — colision inminente desde bug_IBA

        Acciones en cada tick mientras E_STOP esta activo:
          • /cmd_vel = Twist() cero     — anula GoToGoal y cualquier nodo de velocidad
          • /astar/goal = (1e6, 1e6)    — hace que el planner devuelva NO_PATH
          Al entrar (_just_entered):
          • /lift_auto = "down"          — baja fork a piso incondicionalmente
          • /astar/cancel = "cancel"     — descarta path activo en el planner

        Salida (via /mission/mode):
          • "resume" -> regresar al estado de origen (limpia goal residual primero)
          • "reset"  -> MissionContext nuevo -> IDLE
        """
        if self._just_entered:
            self.get_logger().error('[E_STOP] *** PARO DE EMERGENCIA ACTIVO ***')
            # Bajar lift a piso siempre — sin importar si hay pallet a bordo
            self._pub_lift_auto.publish(String(data=LiftCmd.DOWN))
            # Cancelar path activo en el planner
            self._pub_astar_cancel.publish(String(data='cancel'))
            # cmd_vel=0 y astar_goal blanqueado se publican en cada tick (ver _fsm_tick)

        mode = self._sensors.mission_mode
        if mode == 'resume':
            self.get_logger().info('[E_STOP] Resume -> regresando a origen')
            # Limpiar goal residual antes de devolver el control al planner
            self._pub_astar_cancel.publish(String(data='cancel'))
            self._estop_active = False
            self._transition(self._recovery_origin)
        elif mode == 'reset':
            self.get_logger().info('[E_STOP] Reset -> IDLE')
            self._pub_astar_cancel.publish(String(data='cancel'))
            self._estop_active = False
            self._ctx = MissionContext()
            self._transition(State.IDLE)

    # ══════════════════════════════════════════════════════════════════════════
    # RECOVERY STATES
    # ══════════════════════════════════════════════════════════════════════════

    def _enter_recovery(self, recovery_state: State):
        self._recovery_origin = self._state
        if not self._recovery.enter(recovery_state):
            self.get_logger().error(
                f'[Recovery] {recovery_state.name} → IDLE incondicional')
            self._stop_robot()
            self._lift_auto(LiftCmd.DOWN)
            self._ctx = MissionContext()
            self._transition(State.IDLE)
            return
        self._transition(recovery_state)

    # ── QR_RECOVERY ───────────────────────────────────────────────────────────

    def _recovery_qr(self):
        """
        t < 1.5 s: retroceder 0.2 m
        t < 9.5 s: scan ±30° buscando target_aruco_id
        Si redetecta → QR_ALIGN
        Si no → saltar pallet, continuar exploración
        """
        t = self._time_in_state()
        s = self._sensors

        if t < 1.5:
            cmd = Twist()
            cmd.linear.x = -0.13
            self._pub_cmd.publish(cmd)
            return

        self._stop_robot()

        if s.aruco_id == self._ctx.target_aruco_id:
            self.get_logger().info('[QR_RECOVERY] Marcador redetectado → QR_ALIGN')
            self._recovery.reset(State.QR_RECOVERY)
            self._align_ok_ticks = 0
            self._transition(State.QR_ALIGN)
            return

        if t < 9.5:
            direction = 1.0 if t < 5.5 else -1.0
            cmd = Twist()
            cmd.angular.z = direction * 0.25
            self._pub_cmd.publish(cmd)

            if s.aruco_id == self._ctx.target_aruco_id:
                self._stop_robot()
                self.get_logger().info('[QR_RECOVERY] Marcador encontrado en scan → QR_ALIGN')
                self._recovery.reset(State.QR_RECOVERY)
                self._align_ok_ticks = 0
                self._transition(State.QR_ALIGN)
            return

        # No encontrado — saltar pallet
        self._stop_robot()
        self.get_logger().warning(
            f'[QR_RECOVERY] Marcador {self._ctx.target_aruco_id} no encontrado — '
            f'saltando, continuando exploración')
        self._ctx.qr_payload      = ''
        self._ctx.pallet_client   = ''
        self._ctx.target_aruco_id = -1
        self._ctx.expl_idx        = (self._ctx.expl_idx + 1) % len(self._exploration_waypoints)
        self._recovery.reset(State.QR_RECOVERY)
        self._transition(State.ASTAR_EXPLORE)

    # ── MANIP_RECOVERY ────────────────────────────────────────────────────────

    def _recovery_manip(self):
        """
        t < 4 s: retroceder a posición segura
        Bajar fork a DOWN (esperar lift_done)
        Reintentar COLLECT_APPROACH
        """
        t = self._time_in_state()

        if t < 4.0:
            cmd = Twist()
            cmd.linear.x = -0.10
            self._pub_cmd.publish(cmd)
            return

        self._stop_robot()

        if not self._lift_cmd_sent:
            self.get_logger().info('[MANIP_RECOVERY] Bajando fork a posición segura')
            self._reset_lift_flags()
            self._lift_auto(LiftCmd.DOWN)
            self._lift_cmd_sent = True
            self._lift_sent_t   = time.monotonic()
            return

        if not self._lift_done_received:
            if time.monotonic() - self._lift_sent_t > LIFT_MOVE_TIMEOUT + 2.0:
                self.get_logger().error('[MANIP_RECOVERY] Lift no responde → IDLE')
                self._lift_cmd_sent = False
                self._transition(State.IDLE)
            return

        # Fork seguro → reintentar
        self.get_logger().info('[MANIP_RECOVERY] Fork seguro → COLLECT_APPROACH')
        self._ctx.pallet_acquired = False
        self._transition(State.COLLECT_APPROACH)

    # ── NAV_RECOVERY ──────────────────────────────────────────────────────────

    def _recovery_nav(self):
        """
        Rotate-escape: girar 90° (~3 s) + avanzar 0.3 m (~3 s)
        Re-enviar goal original al estado de origen
        """
        t = self._time_in_state()

        if self._just_entered:
            self.get_logger().info('[NAV_RECOVERY] Rotate-escape')

        if t < 3.0:
            cmd = Twist()
            cmd.angular.z = 0.52   # ~90° en 3 s
            self._pub_cmd.publish(cmd)
            return

        if t < 6.0:
            cmd = Twist()
            cmd.linear.x = 0.10
            self._pub_cmd.publish(cmd)
            return

        self._stop_robot()
        if t < 6.5:
            return

        self.get_logger().info(
            f'[NAV_RECOVERY] Re-enviando goal → {self._recovery_origin.name}')
        self._goal_retries = 0
        self._recovery.reset(State.NAV_RECOVERY)

        if self._recovery_origin == State.GO2GOAL and self._ctx.has_truck_goal:
            self._send_goal(self._ctx.truck_goal_x, self._ctx.truck_goal_y)
        elif self._recovery_origin in (State.ASTAR_EXPLORE,):
            self._send_goal_wp(self._ctx.expl_idx)

        self._transition(self._recovery_origin)

    # ── YOLO_RECOVERY ─────────────────────────────────────────────────────────

    def _recovery_yolo(self):
        """
        Scan lateral ±90° buscando clase target
        Si detecta (conf > 0.5) → publicar nuevo cmd a truck_align_node → TRUCK_ALIGN
        Si no → depositar en modo degradado → DROP_PALLET
        """
        t          = self._time_in_state()
        target_cls = self._ctx.pallet_client.lower()

        if t < 18.0:
            direction = 1.0 if t < 9.0 else -1.0
            cmd = Twist()
            cmd.angular.z = direction * 0.20
            self._pub_cmd.publish(cmd)

            for d in self._sensors.yolo_detections:
                if (d.get('class', '').lower() == target_cls
                        and d.get('conf', 0) >= 0.50):
                    self._stop_robot()
                    self.get_logger().info('[YOLO_RECOVERY] Logo encontrado → TRUCK_ALIGN')
                    self._sensors.truck_align_result = ''
                    self._pub_truck_cmd.publish(
                        String(data=f'align:{self._ctx.pallet_client}'))
                    self._recovery.reset(State.YOLO_RECOVERY)
                    self._transition(State.TRUCK_ALIGN)
                    return
            return

        self._stop_robot()
        self.get_logger().warning('[YOLO_RECOVERY] Logo no encontrado — depositando sin alineación')
        self._recovery.reset(State.YOLO_RECOVERY)
        self._transition(State.DROP_PALLET)

    # ══════════════════════════════════════════════════════════════════════════
    # HELPERS DE NAVEGACIÓN
    # ══════════════════════════════════════════════════════════════════════════

    def _send_goal(self, x: float, y: float):
        if not self._sensors.map_ready:
            self.get_logger().warning(f'[Goal] Mapa no listo, goal ({x:.2f},{y:.2f}) ignorado')
            return
        msg = Pose2D()
        msg.x, msg.y, msg.theta = float(x), float(y), 0.0
        self._pub_goal.publish(msg)
        self._goal_sent_t          = time.monotonic()
        self._sensors.astar_status = 'PLANNING'
        self.get_logger().info(f'[Goal] → ({x:.2f},{y:.2f})')

    def _send_goal_wp(self, idx: int):
        wps = self._exploration_waypoints
        if not wps:
            return
        wp = wps[idx % len(wps)]
        self.get_logger().info(
            f'[Exploración] WP {idx % len(wps) + 1}/{len(wps)} '
            f'→ ({wp["x"]:.2f},{wp["y"]:.2f})')
        self._send_goal(wp['x'], wp['y'])

    def _goal_timeout(self) -> bool:
        return (self._goal_sent_t > 0
                and time.monotonic() - self._goal_sent_t > NAV_GOAL_TIMEOUT)

    def _find_truck_goal(self) -> Optional[tuple[float, float]]:
        key = self._ctx.pallet_client.strip().lower()
        if key in CLIENT_TRUCK_GOALS:
            return CLIENT_TRUCK_GOALS[key]
        self.get_logger().warning(f'[find_truck_goal] "{key}" desconocido — default')
        return (2.06, 0.0)

    def _nearest_landmark(self) -> Optional[int]:
        rx, ry = self._sensors.robot_x, self._sensors.robot_y
        best_id, best_d = None, float('inf')
        for lid, (lx, ly) in LANDMARK_POSITIONS.items():
            d = math.hypot(lx - rx, ly - ry)
            if d < best_d:
                best_d, best_id = d, lid
        return best_id

    # ══════════════════════════════════════════════════════════════════════════
    # HELPERS GENERALES
    # ══════════════════════════════════════════════════════════════════════════

    def _stop_robot(self):
        self._pub_cmd.publish(Twist())

    def _lift_auto(self, cmd: str):
        self.get_logger().info(f'[Lift] → {cmd}')
        self._pub_lift_auto.publish(String(data=cmd))

    def _reset_lift_flags(self):
        """Limpia flags de lift. Llamado automáticamente en _transition."""
        self._lift_cmd_sent      = False
        self._lift_done_received = False
        self._lift_done_label    = ''
        self._lift_sent_t        = 0.0

    def _transition(self, new_state: State):
        if new_state == self._state:
            return
        self.get_logger().info(f'[FSM] {self._state.name} → {new_state.name}')
        self._prev_state     = self._state
        self._state          = new_state
        self._state_entry_t  = time.monotonic()
        self._just_entered   = True
        self._reset_lift_flags()          # nunca heredar flags de lift entre estados
        self._truck_watchdog_idle_s = 0.0 # nunca heredar watchdog entre activaciones
        self._truck_watchdog_last_t = 0.0
        self._pub_context.publish(String(data=self._ctx.to_json()))
        self._save_checkpoint()

    def _time_in_state(self) -> float:
        return time.monotonic() - self._state_entry_t

    # ══════════════════════════════════════════════════════════════════════════
    # PERSISTENCIA
    # ══════════════════════════════════════════════════════════════════════════

    def _save_checkpoint(self):
        try:
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump({
                    'state':   self._state.name,
                    'context': asdict(self._ctx),
                    'ts':      time.time(),
                }, f, indent=2)
        except Exception as e:
            self.get_logger().warning(f'[Checkpoint] Error: {e}')

    def _try_load_checkpoint(self):
        try:
            with open(CHECKPOINT_FILE) as f:
                data = json.load(f)
            self._ctx = MissionContext(**data.get('context', {}))
            self.get_logger().info(
                f'[Checkpoint] Cargado | pallets={self._ctx.pallets_done} '
                f'| expl_idx={self._ctx.expl_idx}')
        except FileNotFoundError:
            pass
        except Exception as e:
            self.get_logger().warning(f'[Checkpoint] Error cargando: {e}')

    # ══════════════════════════════════════════════════════════════════════════
    # WAYPOINTS
    # ══════════════════════════════════════════════════════════════════════════

    def _load_exploration_waypoints(self) -> list[dict]:
        if self._expl_file:
            try:
                import yaml
                with open(self._expl_file) as f:
                    data = yaml.safe_load(f)
                wps = [{'x': float(wp['x']), 'y': float(wp['y'])}
                       for wp in data.get('waypoints', [])]
                if wps:
                    self.get_logger().info(
                        f'[Waypoints] {len(wps)} cargados desde {self._expl_file}')
                    return wps
            except Exception as e:
                self.get_logger().error(f'[Waypoints] Error: {e}')

        return [
            {'x': -1.04, 'y':  0.00},
            {'x': -2.02, 'y':  0.00},
            {'x': -1.98, 'y':  1.11},
            {'x': -0.62, 'y':  1.09},
            {'x':  1.05, 'y':  1.01},
            {'x':  1.17, 'y': -0.04},
            {'x':  1.09, 'y': -1.35},
            {'x': -0.04, 'y': -1.33},
            {'x':  0.03, 'y':  0.49},
            {'x': -0.94, 'y': -1.28},
            {'x': -1.95, 'y': -1.32},
            {'x': -1.96, 'y': -0.04},
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

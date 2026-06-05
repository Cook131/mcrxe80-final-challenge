#!/usr/bin/env python3
"""
qr_collect_node.py — Iolair QR Align + Collect
================================================
Nodo autónomo de alineación con QR y recolección de pallet.

Integración con FSM (mission_manager.py)
-----------------------------------------
  SUB:  /collect/trigger     (std_msgs/String)
          "rack"     → nivel de recolección N1  (racks bajos)
          "conveyor" → nivel de recolección N2  (conveyors altos)
          "abort"    → cancela operación activa

  PUB:  /collect/done        (std_msgs/String)
          "SUCCESS"  → pallet recogido, listo para GO2GOAL
          "ABORT"    → operación cancelada o timeout

  SUB:  /aruco/qr            (std_msgs/String)   — payload QR
  SUB:  /aruco/distance      (std_msgs/Float32)  — distancia plano XZ [m]  (cualquier marcador)
  SUB:  /aruco/angle         (std_msgs/Float32)  — ángulo horizontal [°], + = derecha
  SUB:  /aruco/id            (std_msgs/Int32)    — ID del marcador prioritario (para saber si hay marcador)

  PUB:  /cmd_vel             (geometry_msgs/Twist)
  PUB:  /lift_auto           (std_msgs/String)   — n1 | n2 | hold | down
  SUB:  /lift_done           (std_msgs/String)   — AT_N1 | AT_N2 | HOLD | DOWN

  PUB:  /collect/qr_payload  (std_msgs/String)   — reenvía contenido QR a FSM

Máquina de estados interna
---------------------------
  IDLE
    → ALIGN        al recibir /collect/trigger con zona válida y QR visible

  ALIGN
    → ADVANCE      cuando |angle| < angle_tol  AND  dist > approach_dist
    → IDLE         si se pierde el QR (timeout) o llega "abort"

  ADVANCE
    → LIFT_CMD     al llegar a dist <= approach_dist  AND  |angle| < angle_tol
    → ALIGN        si el ángulo se desvía más de angle_tol * 2 durante avance

  LIFT_CMD
    → WAIT_LIFT    después de publicar comando SPI
    → ABORT        timeout esperando /lift_done

  WAIT_LIFT
    → EXTRACT      al recibir /lift_done con nivel correcto (AT_N1 / AT_N2)
    → ABORT        timeout

  EXTRACT
    → REVERSE      después de avanzar micro-distancia para encajar el pallet

  REVERSE
    → DONE         robot alejado, pallet asegurado

  DONE / ABORT
    → IDLE         publica /collect/done y resetea estado

Parámetros ROS2 configurables
------------------------------
  kp_angle        float   0.018    P angular [rad/s / °]
  kd_angle        float   0.004    D angular [rad/s / (°/s)]
  kp_dist         float   0.40     P lineal  [m/s / m]
  kd_dist         float   0.08     D lineal  [m/s / (m/s)]
  angle_tol_deg   float   4.0      Tolerancia angular alineación [°]
  approach_dist   float   0.28     Distancia objetivo de parada [m]
  dist_tol        float   0.03     Tolerancia de distancia [m]
  cam_offset_deg  float   0.0      Offset angular cámara→base_link [°]
                                   (se suma al ángulo recibido)
  max_angular     float   0.45     Velocidad angular máx [rad/s]
  max_linear      float   0.18     Velocidad lineal máx [m/s]
  extract_speed   float   0.08     Velocidad de encaje [m/s]
  extract_time    float   0.6      Duración avance de encaje [s]
  reverse_speed   float   0.10     Velocidad de retroceso [m/s]
  reverse_time    float   1.8      Duración retroceso [s]
  qr_timeout      float   2.5      Segundos sin QR antes de pausar [s]
  lift_timeout    float   8.0      Timeout esperando /lift_done [s]
  fsm_rate_hz     float   20.0     Frecuencia del tick interno [Hz]
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos  import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from std_msgs.msg      import Float32, Int32, String


# ══════════════════════════════════════════════════════════════════════════════
# Estados internos
# ══════════════════════════════════════════════════════════════════════════════

class _S:
    IDLE      = 'IDLE'
    ALIGN     = 'ALIGN'
    ADVANCE   = 'ADVANCE'
    LIFT_CMD  = 'LIFT_CMD'
    WAIT_LIFT = 'WAIT_LIFT'
    EXTRACT   = 'EXTRACT'
    REVERSE   = 'REVERSE'
    DONE      = 'DONE'
    ABORT     = 'ABORT'


# Mapa zona → comando lift y estado esperado en /lift_done
_ZONE_LIFT = {
    'rack':     ('n1', 'AT_N1'),
    'conveyor': ('n2', 'AT_N2'),
}


# ══════════════════════════════════════════════════════════════════════════════
class QRCollectNode(Node):

    def __init__(self):
        super().__init__('qr_collect_node')

        # ── Parámetros ────────────────────────────────────────────────────
        self.declare_parameter('kp_angle',       0.018)
        self.declare_parameter('kd_angle',       0.004)
        self.declare_parameter('kp_dist',        0.40)
        self.declare_parameter('kd_dist',        0.08)
        self.declare_parameter('angle_tol_deg',  4.0)
        self.declare_parameter('approach_dist',  0.28)
        self.declare_parameter('dist_tol',       0.03)
        self.declare_parameter('cam_offset_deg', 0.0)
        self.declare_parameter('max_angular',    0.45)
        self.declare_parameter('max_linear',     0.18)
        self.declare_parameter('extract_speed',  0.08)
        self.declare_parameter('extract_time',   0.6)
        self.declare_parameter('reverse_speed',  0.10)
        self.declare_parameter('reverse_time',   1.8)
        self.declare_parameter('qr_timeout',     2.5)
        self.declare_parameter('lift_timeout',   8.0)
        self.declare_parameter('fsm_rate_hz',    20.0)

        self._p = lambda n: self.get_parameter(n).value   # shortcut

        # ── Estado FSM interno ────────────────────────────────────────────
        self._state        = _S.IDLE
        self._prev_state   = None
        self._state_entry  = time.monotonic()

        # zona activa ('rack' | 'conveyor')
        self._zone         = ''
        # lift cmd y estado esperado para la zona activa
        self._lift_cmd     = ''
        self._lift_expect  = ''

        # ── Datos de percepción ───────────────────────────────────────────
        self._qr_payload   = ''          # string del QR (puede estar vacío si es ArUco)
        self._qr_angle     = 0.0         # grados  (viene de /aruco/angle)
        self._qr_dist      = 999.0       # metros  (viene de /aruco/distance)
        self._qr_stamp     = 0.0         # monotonic del último dato válido
        self._marker_id    = -1          # ID del marcador prioritario (-1 = ninguno)

        # ── Control PD — derivada ─────────────────────────────────────────
        self._prev_angle_err = 0.0
        self._prev_dist_err  = 0.0
        self._prev_ctrl_t    = time.monotonic()

        # ── Estado del lift ───────────────────────────────────────────────
        self._lift_done_label = ''

        # ── QOS sensor (BEST_EFFORT para /aruco/*) ────────────────────────
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Suscriptores ──────────────────────────────────────────────────
        self.create_subscription(
            String,  '/collect/trigger',    self._cb_trigger,   10)
        self.create_subscription(
            String,  '/aruco/qr',           self._cb_qr,        qos_be)
        self.create_subscription(
            Float32, '/aruco/distance',     self._cb_qr_dist,   qos_be)
        self.create_subscription(
            Float32, '/aruco/angle',        self._cb_qr_angle,  qos_be)
        self.create_subscription(
            Int32,   '/aruco/id',           self._cb_aruco_id,  qos_be)
        self.create_subscription(
            String,  '/lift_done',          self._cb_lift_done, 10)

        # ── Publicadores ──────────────────────────────────────────────────
        self._pub_cmd     = self.create_publisher(Twist,  '/cmd_vel',           10)
        self._pub_lift    = self.create_publisher(String, '/lift_auto',         10)
        self._pub_done    = self.create_publisher(String, '/collect/done',      10)
        self._pub_payload = self.create_publisher(String, '/collect/qr_payload',10)

        # ── Timer FSM ─────────────────────────────────────────────────────
        rate = float(self._p('fsm_rate_hz'))
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            'qr_collect_node listo\n'
            '  SUB: /collect/trigger  (rack | conveyor | abort)\n'
            '  PUB: /collect/done     (SUCCESS | ABORT)\n'
            '  PUB: /collect/qr_payload\n'
            f'  Tolerancias: angle={self._p("angle_tol_deg")}°  '
            f'dist={self._p("approach_dist")}±{self._p("dist_tol")}m'
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
        self._transition(_S.ALIGN)

    def _cb_qr(self, msg: String):
        payload = msg.data.strip()
        if payload:
            if payload != self._qr_payload:
                self.get_logger().info(f'[Percepción] QR: {payload}')
                self._qr_payload = payload
                # Reenviar payload a FSM para que determine zona de depósito
                self._pub_payload.publish(String(data=payload))
            self._qr_stamp = time.monotonic()

    def _cb_aruco_id(self, msg: Int32):
        """Trackea el ID del marcador prioritario para saber si hay marcador visible."""
        self._marker_id = int(msg.data)
        # Si hay marcador válido, refrescar timestamp de percepción
        unknown_id = -1  # coincide con el default de aruco_detector
        if self._marker_id != unknown_id:
            self._qr_stamp = time.monotonic()

    def _cb_qr_dist(self, msg: Float32):
        self._qr_dist  = float(msg.data)
        # stamp se actualiza desde _cb_aruco_id para mayor coherencia;
        # pero también lo hacemos aquí para el caso en que no haya /aruco/id
        self._qr_stamp = time.monotonic()

    def _cb_qr_angle(self, msg: Float32):
        # Aplicar offset angular cámara→base_link
        raw = float(msg.data)
        self._qr_angle = raw + float(self._p('cam_offset_deg'))
        self._qr_stamp = time.monotonic()

    def _cb_lift_done(self, msg: String):
        label = msg.data.strip()
        if label:
            self.get_logger().info(f'[Lift] /lift_done: {label}')
            self._lift_done_label = label

    # ══════════════════════════════════════════════════════════════════════
    # FSM TICK
    # ══════════════════════════════════════════════════════════════════════

    def _tick(self):
        s = self._state

        # ── IDLE ──────────────────────────────────────────────────────────
        if s == _S.IDLE:
            return   # espera /collect/trigger

        # ── ALIGN ─────────────────────────────────────────────────────────
        elif s == _S.ALIGN:
            if not self._qr_visible():
                # QR perdido — detener y esperar; si se excede timeout → ABORT
                self._stop()
                if self._time_in_state() > self._p('qr_timeout') * 3:
                    self.get_logger().warn('[ALIGN] QR perdido — ABORT')
                    self._transition(_S.ABORT)
                return

            angle_err = self._qr_angle                     # °, objetivo 0
            dist_err  = self._qr_dist - self._p('approach_dist')  # m

            cmd = self._pd_control(angle_err, dist_err)
            self._pub_cmd.publish(cmd)

            aligned = (
                abs(angle_err) < self._p('angle_tol_deg')
                and abs(dist_err)  > self._p('dist_tol')   # todavía lejos
            )
            arrived = (
                abs(angle_err) < self._p('angle_tol_deg')
                and abs(dist_err)  <= self._p('dist_tol')
            )

            if arrived:
                self._stop()
                self.get_logger().info(
                    f'[ALIGN] Alineado y en distancia '
                    f'angle={angle_err:+.1f}° dist={self._qr_dist:.3f}m → LIFT_CMD')
                self._transition(_S.LIFT_CMD)
            elif aligned:
                # Ángulo OK pero todavía lejos: pasar a ADVANCE
                self._stop()
                self.get_logger().info(
                    f'[ALIGN] Alineado, avanzando '
                    f'angle={angle_err:+.1f}° dist={self._qr_dist:.3f}m → ADVANCE')
                self._transition(_S.ADVANCE)

        # ── ADVANCE ───────────────────────────────────────────────────────
        elif s == _S.ADVANCE:
            if not self._qr_visible():
                self._stop()
                if self._time_in_state() > self._p('qr_timeout'):
                    self.get_logger().warn('[ADVANCE] QR perdido → ALIGN')
                    self._transition(_S.ALIGN)
                return

            angle_err = self._qr_angle
            dist_err  = self._qr_dist - self._p('approach_dist')

            # Si el ángulo se desvió durante el avance, volver a alinear
            if abs(angle_err) > self._p('angle_tol_deg') * 2.0:
                self._stop()
                self.get_logger().info(
                    f'[ADVANCE] Desvío angular {angle_err:+.1f}° → ALIGN')
                self._transition(_S.ALIGN)
                return

            # Avanzar con corrección angular suave
            cmd = self._pd_control(angle_err, dist_err, angular_scale=0.5)
            self._pub_cmd.publish(cmd)

            if abs(dist_err) <= self._p('dist_tol'):
                self._stop()
                self.get_logger().info(
                    f'[ADVANCE] Distancia alcanzada {self._qr_dist:.3f}m → LIFT_CMD')
                self._transition(_S.LIFT_CMD)

        # ── LIFT_CMD ──────────────────────────────────────────────────────
        elif s == _S.LIFT_CMD:
            if self._time_in_state() < 0.1:
                # Solo publicar una vez al entrar
                self.get_logger().info(
                    f'[Lift] Enviando comando: {self._lift_cmd}')
                self._pub_lift.publish(String(data=self._lift_cmd))
                self._lift_done_label = ''   # limpiar flag anterior
            self._transition(_S.WAIT_LIFT)

        # ── WAIT_LIFT ─────────────────────────────────────────────────────
        elif s == _S.WAIT_LIFT:
            if self._lift_done_label == self._lift_expect:
                self.get_logger().info(
                    f'[Lift] Confirmado: {self._lift_done_label} → EXTRACT')
                self._transition(_S.EXTRACT)

            elif self._time_in_state() > self._p('lift_timeout'):
                self.get_logger().error(
                    f'[Lift] Timeout esperando {self._lift_expect} → ABORT')
                self._transition(_S.ABORT)

        # ── EXTRACT ───────────────────────────────────────────────────────
        elif s == _S.EXTRACT:
            # Avance corto para meter el pallet en el fork
            elapsed = self._time_in_state()
            if elapsed < self._p('extract_time'):
                cmd = Twist()
                cmd.linear.x = self._p('extract_speed')
                self._pub_cmd.publish(cmd)
            else:
                self._stop()
                self.get_logger().info('[EXTRACT] Encaje completado → REVERSE')
                # Subir a HOLD para asegurar el pallet
                self._pub_lift.publish(String(data='hold'))
                self._transition(_S.REVERSE)

        # ── REVERSE ───────────────────────────────────────────────────────
        elif s == _S.REVERSE:
            elapsed = self._time_in_state()
            if elapsed < self._p('reverse_time'):
                cmd = Twist()
                cmd.linear.x = -self._p('reverse_speed')
                self._pub_cmd.publish(cmd)
            else:
                self._stop()
                self.get_logger().info('[REVERSE] Pallet asegurado → DONE')
                self._transition(_S.DONE)

        # ── DONE ──────────────────────────────────────────────────────────
        elif s == _S.DONE:
            self._stop()
            self.get_logger().info(
                f'[Collect] ✅ SUCCESS — payload="{self._qr_payload}" '
                f'zona="{self._zone}"')
            self._pub_done.publish(String(data='SUCCESS'))
            self._reset()
            self._transition(_S.IDLE)

        # ── ABORT ─────────────────────────────────────────────────────────
        elif s == _S.ABORT:
            self._stop()
            # Bajar lift a posición segura si estaba activo
            if self._lift_cmd:
                self._pub_lift.publish(String(data='down'))
            self.get_logger().warn('[Collect] ❌ ABORT')
            self._pub_done.publish(String(data='ABORT'))
            self._reset()
            self._transition(_S.IDLE)

    # ══════════════════════════════════════════════════════════════════════
    # CONTROL PD
    # ══════════════════════════════════════════════════════════════════════

    def _pd_control(
        self,
        angle_err: float,
        dist_err: float,
        angular_scale: float = 1.0,
    ) -> Twist:
        """
        Control PD desacoplado.
          angle_err : error angular en grados (+ = derecha)
          dist_err  : error de distancia en metros (+ = lejos)
          angular_scale: factor de atenuación para la corrección angular
                         durante el avance (evitar zigzag)
        """
        now = time.monotonic()
        dt  = now - self._prev_ctrl_t
        dt  = max(dt, 1e-3)

        # Derivadas
        d_angle = (angle_err - self._prev_angle_err) / dt
        d_dist  = (dist_err  - self._prev_dist_err)  / dt

        # Ganancias
        kp_a = self._p('kp_angle')
        kd_a = self._p('kd_angle')
        kp_d = self._p('kp_dist')
        kd_d = self._p('kd_dist')

        # Salidas — nota: ángulo + = derecha → angular.z negativo (giro antihorario)
        angular_z = -(kp_a * angle_err + kd_a * d_angle) * angular_scale
        linear_x  =   kp_d * dist_err  + kd_d * d_dist

        # Saturar
        max_w = self._p('max_angular')
        max_v = self._p('max_linear')
        angular_z = max(-max_w, min(max_w, angular_z))
        linear_x  = max(-max_v, min(max_v, linear_x))

        # No retroceder durante ALIGN/ADVANCE (solo avanzar o detenerse)
        linear_x = max(0.0, linear_x)

        # Actualizar derivada
        self._prev_angle_err = angle_err
        self._prev_dist_err  = dist_err
        self._prev_ctrl_t    = now

        cmd = Twist()
        cmd.angular.z = angular_z
        cmd.linear.x  = linear_x
        return cmd

    # ══════════════════════════════════════════════════════════════════════
    # UTILIDADES
    # ══════════════════════════════════════════════════════════════════════

    def _qr_visible(self) -> bool:
        """
        True si hay un marcador visible (ArUco O QR) con datos frescos.
        No requiere QR payload — cualquier marcador con ángulo/distancia válidos sirve.
        """
        unknown_id = -1
        fresh = (time.monotonic() - self._qr_stamp) < self._p('qr_timeout')
        has_marker = (self._marker_id != unknown_id)
        return fresh and has_marker

    def _stop(self):
        self._pub_cmd.publish(Twist())

    def _time_in_state(self) -> float:
        return time.monotonic() - self._state_entry

    def _transition(self, new_state: str):
        if new_state != self._state:
            self.get_logger().info(f'[FSM] {self._state} → {new_state}')
            self._prev_state  = self._state
            self._state       = new_state
            self._state_entry = time.monotonic()
            # Resetear derivadas al cambiar de estado
            self._prev_angle_err = 0.0
            self._prev_dist_err  = 0.0
            self._prev_ctrl_t    = time.monotonic()

    def _reset(self):
        """Limpia variables de sesión al volver a IDLE."""
        self._zone            = ''
        self._lift_cmd        = ''
        self._lift_expect     = ''
        self._lift_done_label = ''
        self._qr_payload      = ''
        self._qr_angle        = 0.0
        self._qr_dist         = 999.0
        self._marker_id       = -1


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = QRCollectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
puzzlebotController.py — Closed-loop wheel velocity controller
==============================================================

Arquitectura
------------
  /cmd_vel (Twist)
      │
      ▼ inverse kinematics
  [target_wr, target_wl]  ← velocidades angulares deseadas [rad/s]
      │
      ▼  PID (una por rueda, corre a control_rate Hz)
  [VelocitySetR, VelocitySetL]  → firmware PWM driver
      ▲
  [VelocityEncR, VelocityEncL]  ← velocidades reales de encoders

Tuning guide
------------
  Kp = 0.5   → subir hasta que las ruedas alcancen el set-point rápido sin overshoot
  Ki = 0.002 → subir despacio para eliminar error en estado estable a velocidad baja
  Kd = 0.0001→ subir sólo si ves oscilación

Override en tiempo de ejecución:
  ros2 run <pkg> controller --ros-args -p Kp:=1.0 -p Ki:=0.1 -p Kd:=0.01

Suscribe a:
    /cmd_vel       (geometry_msgs/Twist)  — comando de velocidad
    /VelocityEncR  (std_msgs/Float32)     — rueda derecha medida [rad/s]
    /VelocityEncL  (std_msgs/Float32)     — rueda izquierda medida [rad/s]

Publica en:
    /VelocitySetR  (std_msgs/Float32)     — set-point rueda derecha [rad/s]
    /VelocitySetL  (std_msgs/Float32)     — set-point rueda izquierda [rad/s]
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


# ── PID ───────────────────────────────────────────────────────────────────────

class PID:
    """
    PID discreto con anti-windup y saturación de salida.

    Parámetros
    ----------
    Kp, Ki, Kd : ganancias
    dt         : paso de tiempo [s] — debe coincidir con la tasa del loop
    out_min    : límite inferior de salida (también limita integral windup)
    out_max    : límite superior de salida
    """

    def __init__(self, Kp: float, Ki: float, Kd: float,
                 dt: float, out_min: float, out_max: float):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.out_min = out_min
        self.out_max = out_max

        self._integral   = 0.0
        self._prev_error = 0.0

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0

    def update(self, error: float) -> float:
        # Proporcional
        p = self.Kp * error

        # Integral con anti-windup por saturación
        self._integral += error * self.dt
        i_raw = self.Ki * self._integral
        i_clamped = max(self.out_min, min(self.out_max, i_raw))
        # Re-calcular el acumulador para que refleje el término clampeado
        if self.Ki != 0.0:
            self._integral = i_clamped / self.Ki

        # Derivativo (sobre el error, no la medición — aceptable a esta escala)
        d = self.Kd * (error - self._prev_error) / self.dt
        self._prev_error = error

        output = p + i_clamped + d
        return max(self.out_min, min(self.out_max, output))

    def update_gains(self, Kp: float, Ki: float, Kd: float):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.reset()


# ── Controller Node ───────────────────────────────────────────────────────────

class PuzzlebotController(Node):

    # Constantes físicas (deben coincidir con odometría y modelo)
    WHEEL_RADIUS = 0.05   # metros
    WHEEL_BASE   = 0.19   # metros

    # Velocidad angular máxima que puede entregar el motor [rad/s]
    # ~200 rpm → 200/60 * 2π ≈ 21 rad/s
    MAX_WHEEL_VEL = 21.0

    def __init__(self):
        super().__init__('puzzlebot_main_controller')

        # ── Parámetros ROS 2 ──────────────────────────────────────────────
        self.declare_parameter('Kp',           0.5)
        self.declare_parameter('Ki',           0.002)
        self.declare_parameter('Kd',           0.0001)
        self.declare_parameter('control_rate', 50.0)   # Hz

        Kp   = self.get_parameter('Kp').value
        Ki   = self.get_parameter('Ki').value
        Kd   = self.get_parameter('Kd').value
        rate = self.get_parameter('control_rate').value
        dt   = 1.0 / rate

        self._pid_r = PID(Kp, Ki, Kd, dt,
                          -self.MAX_WHEEL_VEL, self.MAX_WHEEL_VEL)
        self._pid_l = PID(Kp, Ki, Kd, dt,
                          -self.MAX_WHEEL_VEL, self.MAX_WHEEL_VEL)

        # Velocidades deseadas (set-point) y medidas
        self._target_r = 0.0
        self._target_l = 0.0
        self._meas_r   = 0.0
        self._meas_l   = 0.0

        # ── QoS: coincidir con los publishers BEST_EFFORT del firmware ────
        enc_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(Twist,   '/cmd_vel',      self._cb_cmd_vel, 10)
        self.create_subscription(Float32, '/VelocityEncR', self._cb_enc_r,   enc_qos)
        self.create_subscription(Float32, '/VelocityEncL', self._cb_enc_l,   enc_qos)

        # ── Publishers ────────────────────────────────────────────────────
        self._pub_r = self.create_publisher(Float32, '/VelocitySetR', 10)
        self._pub_l = self.create_publisher(Float32, '/VelocitySetL', 10)

        # ── Timer del loop de control ─────────────────────────────────────
        self.create_timer(dt, self._control_loop)

        self.get_logger().info(
            f'PuzzlebotController listo — '
            f'Kp={Kp}, Ki={Ki}, Kd={Kd}, rate={rate} Hz'
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cb_cmd_vel(self, msg: Twist):
        """Convierte Twist → velocidades angulares de rueda deseadas [rad/s]."""
        v = msg.linear.x
        w = msg.angular.z
        r = self.WHEEL_RADIUS
        L = self.WHEEL_BASE

        # Cinemática inversa diferencial
        self._target_r = (2.0 * v + w * L) / (2.0 * r)
        self._target_l = (2.0 * v - w * L) / (2.0 * r)

        # Saturar a límites del motor
        self._target_r = max(-self.MAX_WHEEL_VEL,
                             min(self.MAX_WHEEL_VEL, self._target_r))
        self._target_l = max(-self.MAX_WHEEL_VEL,
                             min(self.MAX_WHEEL_VEL, self._target_l))

        # Reset integral cuando el target cambia de signo (evita windup en reversa)
        # FIX: comparar target vs. error anterior (no target * prev_error,
        # que daba falsos positivos cuando prev_error era ruido cerca de cero)
        if self._target_r * self._pid_r._prev_error < -1.0:
            self._pid_r.reset()
        if self._target_l * self._pid_l._prev_error < -1.0:
            self._pid_l.reset()

    def _cb_enc_r(self, msg: Float32):
        self._meas_r = msg.data

    def _cb_enc_l(self, msg: Float32):
        self._meas_l = msg.data

    # ── Loop de control ───────────────────────────────────────────────────────

    def _control_loop(self):
        """
        Corre a control_rate Hz.
        Calcula la salida PID por rueda y publica el set-point.
        """
        # Parada limpia cuando el target es cero
        if self._target_r == 0.0 and self._target_l == 0.0:
            self._pid_r.reset()
            self._pid_l.reset()
            self._publish(0.0, 0.0)
            return

        error_r = self._target_r - self._meas_r
        error_l = self._target_l - self._meas_l

        set_r = float(self._pid_r.update(error_r))
        set_l = float(self._pid_l.update(error_l))

        self._publish(set_r, set_l)

    # ── Helper de publicación ─────────────────────────────────────────────────

    def _publish(self, vel_r: float, vel_l: float):
        msg_r = Float32()
        msg_l = Float32()
        msg_r.data = vel_r
        msg_l.data = vel_l
        self._pub_r.publish(msg_r)
        self._pub_l.publish(msg_l)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = PuzzlebotController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Parada de seguridad al cerrar
        node._publish(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
puzzlebotTeleop.py — Teleoperación por teclado para el Puzzlebot
=================================================================

Teclas de movimiento:
  W / S   : avance / retroceso
  A / D   : giro izquierda / derecha
  Espacio : parada de emergencia (hard stop)

Teclas de lift:
  1 : N1   (posición baja)
  2 : N2   (posición alta)
  3 : HOLD (sostener)
  4 : DOWN (bajar)

  Q / Ctrl+C : salir
"""

import sys
import select
import termios
import tty
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String


# Mapa de tecla → comando lift
LIFT_KEYS = {
    '1': 'n1',
    '2': 'n2',
    '3': 'hold',
    '4': 'down',
}


class PuzzlebotTeleop(Node):

    def __init__(self):
        super().__init__('puzzlebot_teleop')

        # ── Publishers ────────────────────────────────────────────────────
        self._pub_cmd  = self.create_publisher(Twist,  '/cmd_vel',   10)
        self._pub_lift = self.create_publisher(String, '/lift_auto', 10)

        # ── Parámetros ROS 2 ──────────────────────────────────────────────
        self.declare_parameter('max_lin',   0.2)
        self.declare_parameter('max_ang',   0.5)
        self.declare_parameter('accel_lin', 0.05)
        self.declare_parameter('accel_ang', 0.05)

        self.max_lin   = self.get_parameter('max_lin').value
        self.max_ang   = self.get_parameter('max_ang').value
        self.accel_lin = self.get_parameter('accel_lin').value
        self.accel_ang = self.get_parameter('accel_ang').value

        # ── Estado de velocidad ───────────────────────────────────────────
        self.target_lin  = 0.0
        self.target_ang  = 0.0
        self.current_lin = 0.0
        self.current_ang = 0.0
        self._vel_lock   = threading.Lock()

        # ── Estado del lift (solo para logging) ───────────────────────────
        self._lift_state = 'down'

        # ── Timer de publicación a 50 Hz ──────────────────────────────────
        self.create_timer(0.02, self._publish_velocity)

        self.get_logger().info(
            "\n┌─────────────────────────────┐\n"
            "│     Puzzlebot  Teleop       │\n"
            "├──────────────┬──────────────┤\n"
            "│  W / S       │ Avance / Ret │\n"
            "│  A / D       │ Giro izq/der │\n"
            "│  Espacio     │ Hard stop    │\n"
            "├──────────────┼──────────────┤\n"
            "│  1           │ Lift → N1    │\n"
            "│  2           │ Lift → N2    │\n"
            "│  3           │ Lift → HOLD  │\n"
            "│  4           │ Lift → DOWN  │\n"
            "├──────────────┴──────────────┤\n"
            "│  Q / Ctrl+C  →  Salir       │\n"
            "└─────────────────────────────┘"
        )

    # ── Procesamiento de teclas ───────────────────────────────────────────
    def _process_key(self, key: str):
        # — Lift (sin lock, sólo publica una vez) —
        if key in LIFT_KEYS:
            cmd = LIFT_KEYS[key]
            msg = String()
            msg.data = cmd
            self._pub_lift.publish(msg)
            self._lift_state = cmd
            self.get_logger().info(f'Lift → {cmd.upper()}')
            return

        # — Movimiento —
        key_lower = key.lower()
        with self._vel_lock:
            if key_lower == 'w':
                self.target_lin =  self.max_lin
                self.target_ang =  0.0
            elif key_lower == 's':
                self.target_lin = -self.max_lin
                self.target_ang =  0.0
            elif key_lower == 'a':
                self.target_lin =  0.0
                self.target_ang =  self.max_ang
            elif key_lower == 'd':
                self.target_lin =  0.0
                self.target_ang = -self.max_ang
            elif key == ' ':
                self.target_lin  = 0.0
                self.target_ang  = 0.0
                self.current_lin = 0.0
                self.current_ang = 0.0

    # ── Timer: rampa + publicación ────────────────────────────────────────
    def _publish_velocity(self):
        with self._vel_lock:
            tl = self.target_lin
            ta = self.target_ang

        if tl > self.current_lin:
            self.current_lin = min(tl, self.current_lin + self.accel_lin)
        elif tl < self.current_lin:
            self.current_lin = max(tl, self.current_lin - self.accel_lin)

        if ta > self.current_ang:
            self.current_ang = min(ta, self.current_ang + self.accel_ang)
        elif ta < self.current_ang:
            self.current_ang = max(ta, self.current_ang - self.accel_ang)

        msg = Twist()
        msg.linear.x  = self.current_lin
        msg.angular.z = self.current_ang
        self._pub_cmd.publish(msg)

    # ── Bucle de teclado (hilo principal) ─────────────────────────────────
    def key_loop(self):
        settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            while rclpy.ok():
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if rlist:
                    key = sys.stdin.read(1)
                    if key == '\x03' or key.lower() == 'q':
                        break
                    self._process_key(key)
                else:
                    # Timeout → ninguna tecla → decelerar a cero
                    with self._vel_lock:
                        self.target_lin = 0.0
                        self.target_ang = 0.0
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

    def _stop_robot(self):
        self._pub_cmd.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = PuzzlebotTeleop()

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.key_loop()
    except KeyboardInterrupt:
        pass
    finally:
        print("\rSaliendo del teleop...         ")
        node._stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
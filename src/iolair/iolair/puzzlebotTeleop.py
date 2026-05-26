#!/usr/bin/env python3
"""
puzzlebotTeleop.py — Teleoperación por teclado para el Puzzlebot
=================================================================

Teclas:
  W / S   : avance / retroceso
  A / D   : giro izquierda / derecha
  Espacio : parada de emergencia (hard stop)
  Q / Ctrl+C : salir

Características:
  - Suavizado de velocidad (rampa de aceleración) a 50 Hz
  - Parámetros configurables vía --ros-args
  - Bucle de teclado en hilo principal; callbacks ROS en hilo secundario
  - Timeout de tecla (0.1 s) → coast to stop automático
  - Restauración garantizada del terminal en cualquier caso de salida
"""

import sys
import select
import termios
import tty
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class PuzzlebotTeleop(Node):

    def __init__(self):
        super().__init__('puzzlebot_teleop')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

        # ── Parámetros ROS 2 ──────────────────────────────────────────────
        self.declare_parameter('max_lin',   0.2)
        self.declare_parameter('max_ang',   0.5)
        self.declare_parameter('accel_lin', 0.05)
        self.declare_parameter('accel_ang', 0.05)

        self.max_lin   = self.get_parameter('max_lin').value
        self.max_ang   = self.get_parameter('max_ang').value
        self.accel_lin = self.get_parameter('accel_lin').value
        self.accel_ang = self.get_parameter('accel_ang').value

        # Velocidades objetivo (lo que el usuario pide)
        self.target_lin = 0.0
        self.target_ang = 0.0

        # Velocidades actuales (lo que el robot realmente hace)
        self.current_lin = 0.0
        self.current_ang = 0.0

        # FIX: lock para acceso seguro a target_* desde el hilo de teclado
        # mientras publish_velocity corre en el hilo del executor ROS
        self._vel_lock = threading.Lock()

        # Timer de publicación a 50 Hz
        self.timer = self.create_timer(0.02, self.publish_velocity)

        self.get_logger().info(
            "\nTeleop Active:\n"
            "---------------------------\n"
            "  W / S : Linear Move\n"
            "  A / D : Angular Turn\n"
            "  Space : Emergency Stop\n"
            "  'q' or CTRL+C to quit.\n"
            "---------------------------"
        )

    def process_key(self, key: str):
        """Mapea teclas a velocidades objetivo."""
        key = key.lower()
        with self._vel_lock:
            if key == 'w':
                self.target_lin =  self.max_lin
                self.target_ang =  0.0
            elif key == 's':
                self.target_lin = -self.max_lin
                self.target_ang =  0.0
            elif key == 'a':
                self.target_lin =  0.0
                self.target_ang =  self.max_ang
            elif key == 'd':
                self.target_lin =  0.0
                self.target_ang = -self.max_ang
            elif key == ' ':
                # Hard stop: reset también la velocidad actual
                self.target_lin  = 0.0
                self.target_ang  = 0.0
                self.current_lin = 0.0
                self.current_ang = 0.0

    def publish_velocity(self):
        """Interpola la velocidad actual hacia el objetivo y publica."""
        with self._vel_lock:
            tl = self.target_lin
            ta = self.target_ang

        # Ramp lineal
        if tl > self.current_lin:
            self.current_lin = min(tl, self.current_lin + self.accel_lin)
        elif tl < self.current_lin:
            self.current_lin = max(tl, self.current_lin - self.accel_lin)

        # Ramp angular
        if ta > self.current_ang:
            self.current_ang = min(ta, self.current_ang + self.accel_ang)
        elif ta < self.current_ang:
            self.current_ang = max(ta, self.current_ang - self.accel_ang)

        msg = Twist()
        msg.linear.x  = self.current_lin
        msg.angular.z = self.current_ang
        self.publisher_.publish(msg)

    def key_loop(self):
        """
        Bucle bloqueante de lectura de teclado (corre en el hilo principal).
        Un timeout de 0.1 s actúa como detector de "tecla soltada".
        """
        settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            while rclpy.ok():
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if rlist:
                    key = sys.stdin.read(1)
                    if key == '\x03' or key.lower() == 'q':
                        break
                    self.process_key(key)
                else:
                    # Timeout → ninguna tecla presionada → decelerar a cero
                    with self._vel_lock:
                        self.target_lin = 0.0
                        self.target_ang = 0.0
        finally:
            # Garantizar restauración del terminal siempre
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

    def stop_robot(self):
        """Publica velocidad cero como parada de seguridad final."""
        msg = Twist()
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PuzzlebotTeleop()

    # Callbacks ROS en hilo secundario (desbloquea el hilo principal para el teclado)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.key_loop()
    except KeyboardInterrupt:
        pass
    finally:
        print("\rExiting teleop node...         ")
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
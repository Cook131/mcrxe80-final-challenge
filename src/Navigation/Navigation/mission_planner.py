#!/usr/bin/env python3
import time
import yaml
import sys
import tty
import termios
import threading
import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D
from std_msgs.msg import String


class MissionPlannerNode(Node):

    def __init__(self):
        super().__init__('mission_planner')

        # ── Parámetros ────────────────────────────────────────────────────
        self.declare_parameter('loop_mission', False)
        self.declare_parameter('start_delay', 2.0)
        self.declare_parameter('goal_timeout', 60.0)
        self.declare_parameter('waypoints_file', 'src/iolair/configs/waypoints.yaml')

        self._loop        = self.get_parameter('loop_mission').value
        self._start_delay = self.get_parameter('start_delay').value
        self._timeout     = self.get_parameter('goal_timeout').value
        self._file_path   = self.get_parameter('waypoints_file').value

        # ── Cargar Waypoints ──────────────────────────────────────────────
        self._waypoints = self._load_waypoints_from_yaml()

        # ── Estado ────────────────────────────────────────────────────────
        self._current_idx    = 0
        self._astar_status   = 'IDLE'
        self._goal_sent_time : float = -1.0
        self._mission_done   = False
        self._started        = False
        self._waiting_for_key = False   # <-- pausa activa en este waypoint

        # Thread-safe: Enter desbloquea la pausa
        self._advance_flag  = threading.Event()
        self._resend_flag   = threading.Event()
        self._shutdown_flag = threading.Event()

        # ── Suscriptores / Publicadores ───────────────────────────────────
        self._pub_goal = self.create_publisher(Pose2D, '/astar/goal', 10)
        self.create_subscription(String, '/astar/status', self._cb_status, 10)

        # ── Hilo lector de teclado ────────────────────────────────────────
        self._kb_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self._kb_thread.start()

        # ── Timer principal ───────────────────────────────────────────────
        self.create_timer(0.2, self._mission_loop)

        self.get_logger().info(
            f'Mission Planner iniciado con {len(self._waypoints)} waypoints.\n'
            '  Enter → siguiente WP  |  r → reenviar actual  |  q → salir'
        )

    # ── Carga YAML ────────────────────────────────────────────────────────
    def _load_waypoints_from_yaml(self):
        if not os.path.exists(self._file_path):
            self.get_logger().error(f'No se encuentra el archivo: {self._file_path}')
            return []
        with open(self._file_path, 'r') as f:
            data = yaml.safe_load(f)
        waypoints = []
        for wp in data.get('waypoints', []):
            waypoints.append((float(wp['x']), float(wp['y']), f"WP ID: {wp.get('id', 'N/A')}"))
        return waypoints

    # ── Lector de teclado (hilo separado) ─────────────────────────────────
    def _keyboard_listener(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while not self._shutdown_flag.is_set():
                ch = sys.stdin.read(1)
                if ch in ('\r', '\n', 'n', 'N'):
                    self._advance_flag.set()
                elif ch in ('r', 'R'):
                    self._resend_flag.set()
                elif ch in ('q', 'Q', '\x03'):
                    self._shutdown_flag.set()
                    self.get_logger().info('Cerrando por teclado...')
                    rclpy.shutdown()
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    # ── Callback status A* ────────────────────────────────────────────────
    def _cb_status(self, msg: String):
        self._astar_status = msg.data

    # ── Loop principal ────────────────────────────────────────────────────
    def _mission_loop(self):
        if self._mission_done or not self._waypoints:
            return

        # Delay inicial
        if not self._started:
            if not hasattr(self, '_start_ts'):
                self._start_ts = time.monotonic()
                return
            if time.monotonic() - self._start_ts < self._start_delay:
                return
            self._started = True
            self._send_current_goal()
            return

        if self._current_idx >= len(self._waypoints):
            if self._loop:
                self._current_idx = 0
                self._send_current_goal()
            else:
                if not self._mission_done:
                    self.get_logger().info('¡Misión completada!')
                self._mission_done = True
            return

        # ── Reenvío manual (disponible siempre) ───────────────────────────
        if self._resend_flag.is_set():
            self._resend_flag.clear()
            self.get_logger().info('[r] Reenviando waypoint actual...')
            self._waiting_for_key = False   # cancela pausa si la hubiera
            self._send_current_goal()
            return

        # ── Pausa esperando Enter ──────────────────────────────────────────
        if self._waiting_for_key:
            if self._advance_flag.is_set():
                self._advance_flag.clear()
                self._waiting_for_key = False
                self._current_idx += 1
                if self._current_idx < len(self._waypoints):
                    self._send_current_goal()
                else:
                    self.get_logger().info('¡Misión completada!')
                    self._mission_done = True
            return  # sigue esperando si aún no se presionó Enter

        # ── Lógica automática por status ───────────────────────────────────
        if self._astar_status == 'GOAL_REACHED':
            self._astar_status = 'WAITING'   # evita re-trigger
            self._waiting_for_key = True
            self.get_logger().info(
                f'[{self._current_idx + 1}/{len(self._waypoints)}] '
                f'Goal alcanzado. Presiona Enter para continuar al siguiente WP...'
            )

        elif self._astar_status == 'NO_PATH':
            time.sleep(2.0)
            self._send_current_goal()

        elif (self._goal_sent_time > 0 and
              time.monotonic() - self._goal_sent_time > self._timeout):
            self._send_current_goal()

    # ── Publicar goal ──────────────────────────────────────────────────────
    def _send_current_goal(self):
        if self._current_idx >= len(self._waypoints):
            return
        x, y, desc = self._waypoints[self._current_idx]
        msg = Pose2D()
        msg.x = x
        msg.y = y
        self._pub_goal.publish(msg)
        self._goal_sent_time = time.monotonic()
        self._astar_status   = 'PLANNING'
        self.get_logger().info(
            f'[{self._current_idx + 1}/{len(self._waypoints)}] '
            f'Enviando {desc}: ({x:.2f}, {y:.2f})'
        )


def main(args=None):
    rclpy.init(args=args)
    node = MissionPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
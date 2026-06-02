#!/usr/bin/env python3
"""
mission_planner.py — Secuenciador de waypoints para el A* del Puzzlebot
========================================================================

Publica una lista de posiciones (waypoints) al tópico /astar/goal
(geometry_msgs/Pose2D) de forma secuencial: espera a que el A* reporte
GOAL_REACHED antes de mandar el siguiente.

Flujo
-----
  mission_planner → /astar/goal  (Pose2D)
  /astar/status   → mission_planner  (String: IDLE | PLANNING | EXECUTING |
                                               GOAL_REACHED | NO_PATH)

Parámetros ROS
--------------
  loop_mission   False  si True, repite la secuencia indefinidamente
  start_delay    2.0    segundos de espera antes de mandar el primer goal
                        (para que el A* y el mapa terminen de inicializarse)
  goal_timeout   60.0   segundos máximos por waypoint antes de reintentar

Waypoints
---------
  Se definen en la lista WAYPOINTS al inicio del archivo (coordenadas en
  metros, frame map). Edítalos a tu gusto.

  Las coordenadas vienen de tu mapa SLAM:
    - El origen (0,0) es la posición donde arrancó el SLAM.
    - X crece hacia la derecha, Y crece hacia arriba.
"""

import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D
from std_msgs.msg import String


# ─────────────────────────────────────────────────────────────────────────────
# WAYPOINTS — edita esta lista con las posiciones que quieres visitar
# Formato: (x [m], y [m], descripción)
# ─────────────────────────────────────────────────────────────────────────────

WAYPOINTS = [
    ( 0.50,  0.00, 'Punto 1'),
    ( 0.50,  0.50, 'Punto 2'),
    ( 0.00,  0.50, 'Punto 3'),
    (-0.50,  0.50, 'Punto 4'),
    (-0.50,  0.00, 'Punto 5'),
    (-0.50, -0.50, 'Punto 6'),
    ( 0.00, -0.50, 'Punto 7'),
    ( 0.50, -0.50, 'Punto 8'),
    ( 0.00,  0.00, 'Origen — regreso'),
]


# ─────────────────────────────────────────────────────────────────────────────

class MissionPlannerNode(Node):

    def __init__(self):
        super().__init__('mission_planner')

        # ── Parámetros ────────────────────────────────────────────────────
        self.declare_parameter('loop_mission',  False)
        self.declare_parameter('start_delay',   2.0)
        self.declare_parameter('goal_timeout',  60.0)

        self._loop        = self.get_parameter('loop_mission').value
        self._start_delay = self.get_parameter('start_delay').value
        self._timeout     = self.get_parameter('goal_timeout').value

        # ── Estado ────────────────────────────────────────────────────────
        self._waypoints   = list(WAYPOINTS)
        self._current_idx = 0
        self._astar_status = 'IDLE'
        self._goal_sent_time: float = -1.0
        self._mission_done  = False
        self._started       = False          # bandera de start_delay

        # ── Suscriptores / Publicadores ───────────────────────────────────
        self._pub_goal = self.create_publisher(Pose2D, '/astar/goal', 10)
        self.create_subscription(String, '/astar/status', self._cb_status, 10)

        # ── Timer principal ───────────────────────────────────────────────
        self.create_timer(0.2, self._mission_loop)

        self.get_logger().info(
            f'Mission Planner iniciado\n'
            f'  Waypoints   : {len(self._waypoints)}\n'
            f'  Loop        : {self._loop}\n'
            f'  Start delay : {self._start_delay} s\n'
        )
        for i, (x, y, desc) in enumerate(self._waypoints):
            self.get_logger().info(f'  [{i+1}/{len(self._waypoints)}] '
                                   f'({x:.2f}, {y:.2f})  ← {desc}')

    # ── Callback de estado del A* ─────────────────────────────────────────

    def _cb_status(self, msg: String):
        self._astar_status = msg.data

    # ── Loop de misión ────────────────────────────────────────────────────

    def _mission_loop(self):
        if self._mission_done:
            return

        # Espera inicial para que el A* y el mapa inicialicen
        if not self._started:
            if not hasattr(self, '_start_ts'):
                self._start_ts = time.monotonic()
                self.get_logger().info(
                    f'Esperando {self._start_delay:.1f}s para que el A* inicialice...')
                return
            if time.monotonic() - self._start_ts < self._start_delay:
                return
            self._started = True
            self.get_logger().info('¡Iniciando misión!')
            self._send_current_goal()
            return

        # Misión completada
        if self._current_idx >= len(self._waypoints):
            if self._loop:
                self.get_logger().info('🔁 Misión completada — reiniciando secuencia.')
                self._current_idx = 0
                self._send_current_goal()
            else:
                self.get_logger().info('✅ Misión completada. Todos los waypoints visitados.')
                self._mission_done = True
            return

        # El A* llegó al goal → mandar el siguiente
        if self._astar_status == 'GOAL_REACHED':
            self._current_idx += 1
            if self._current_idx < len(self._waypoints):
                self._send_current_goal()
            # Si ya terminamos, el siguiente ciclo lo detecta arriba

        # El A* no encontró ruta → reintentar el mismo waypoint
        elif self._astar_status == 'NO_PATH':
            self.get_logger().warn(
                f'[{self._current_idx+1}/{len(self._waypoints)}] '
                f'No se encontró ruta. Reintentando en 2 s...')
            time.sleep(2.0)
            self._send_current_goal()

        # Timeout: el A* tardó demasiado → reintentar
        elif (self._goal_sent_time > 0 and
              time.monotonic() - self._goal_sent_time > self._timeout):
            self.get_logger().warn(
                f'[{self._current_idx+1}/{len(self._waypoints)}] '
                f'Timeout de {self._timeout:.0f}s alcanzado. Reintentando...')
            self._send_current_goal()

    # ── Envío de goal ─────────────────────────────────────────────────────

    def _send_current_goal(self):
        if self._current_idx >= len(self._waypoints):
            return

        x, y, desc = self._waypoints[self._current_idx]

        msg = Pose2D()
        msg.x = x
        msg.y = y
        self._pub_goal.publish(msg)
        self._goal_sent_time = time.monotonic()
        self._astar_status   = 'PLANNING'   # reset local para no re-triggerear

        self.get_logger().info(
            f'[{self._current_idx+1}/{len(self._waypoints)}] '
            f'Goal enviado → ({x:.2f}, {y:.2f})  ← {desc}'
        )


# ─────────────────────────────────────────────────────────────────────────────

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
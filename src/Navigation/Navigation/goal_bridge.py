#!/usr/bin/env python3
"""
rviz_goal_bridge.py
===================
Convierte el click del botón "Publish Point" de RViz
  /clicked_point  (geometry_msgs/PointStamped)
hacia el tópico que espera el A* planner:
  /astar/goal     (geometry_msgs/Pose2D)

Uso:
  1. En RViz activa el tool "Publish Point" (Ctrl+click en la barra de tools,
     o tecla rápida P si está configurada)
  2. Haz click en cualquier punto del mapa
  3. Este nodo lo reenvía automáticamente al A*

Agregar al setup.py de iolair:
  'rviz_goal_bridge = iolair.rviz_goal_bridge:main',
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Pose2D


class RvizGoalBridge(Node):

    def __init__(self):
        super().__init__('rviz_goal_bridge')

        self.sub = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self._cb,
            10
        )
        self.pub = self.create_publisher(Pose2D, '/astar/goal', 10)

        self.get_logger().info(
            '[RvizGoalBridge] Listo. '
            'Usa el botón "Publish Point" en RViz para mandar goals al A*.'
        )

    def _cb(self, msg: PointStamped):
        goal = Pose2D()
        goal.x = msg.point.x
        goal.y = msg.point.y
        goal.theta = 0.0

        self.get_logger().info(
            f'[RvizGoalBridge] Goal recibido → x={goal.x:.3f}  y={goal.y:.3f}'
        )
        self.pub.publish(goal)


def main(args=None):
    rclpy.init(args=args)
    node = RvizGoalBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
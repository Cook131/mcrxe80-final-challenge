#!/usr/bin/env python3
"""
================================================================================
Proyecto:    voice_hmm_ros (Paquete ROS 2)
Módulo:      voice_command_handler_node.py
Descripción: Nodo de ROS 2 que recibe comandos de voz clasificados desde el nodo
             speech_recognition_node y los traduce en velocidades para mover el
             robot diferencial (PuzzleBot).

             Suscribe:
               /voice_command        (std_msgs/String) — palabra reconocida
               /voice_command_valid  (std_msgs/Bool)   — flag de confianza

             Publica:
               /cmd_vel              (geometry_msgs/Twist) — velocidad del robot
                 ↳ Opción B: se publica DIRECTO a /cmd_vel, sin pasar por
                   bug_IBA. El robot ejecuta los comandos sin interferencia.
                   Usar en simulación o entornos sin obstáculos.

             Comandos soportados:
               "adelante"  → avanzar hacia el frente
               "atras"     → retroceder
               "izquierda" → girar a la izquierda (en sitio)
               "derecha"   → girar a la derecha (en sitio)
               "gira"      → spin rápido en sitio
               "detente"   → detener el robot
               "stop"      → detener el robot (alias)
               "<unk>"     → desconocido, detiene el robot por seguridad

             Lifter (pendiente de integración):
               Los comandos "arriba" y "abajo" están definidos como
               constantes y documentados, pero todo el código relacionado
               está comentado. Para activarlos:
                 1. Descomentar CMD_LIFTER_ARRIBA / CMD_LIFTER_ABAJO.
                 2. Descomentar el publisher self._pub_lifter en __init__.
                 3. Descomentar los elif en _build_twist().
                 4. Ajustar el nombre del tópico y el tipo de mensaje.

Parámetros ROS 2 configurables (--ros-args -p <param>:=<value>):
    linear_speed     Velocidad lineal en m/s           (default: 0.22)
    angular_speed    Velocidad angular normal en rad/s  (default: 0.40)
    spin_speed       Velocidad de spin rápido en rad/s  (default: 0.60)
    cmd_duration     Duración del movimiento en seg      (default: 1.50)

Uso en ROS 2:
    $ colcon build --packages-select voice_hmm_ros
    $ source install/setup.bash
    $ ros2 run voice_hmm_ros voice_command_handler_node

Registro en setup.py (console_scripts):
    'voice_command_handler_node = voice_hmm_ros.voice_command_handler_node:main'

Autor:  Equipo voice_hmm_ros
Fecha:  2025
================================================================================
"""

from __future__ import annotations

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist

# ── Constantes: palabras clave reconocidas ────────────────────────────────────
# Definidas como frozenset para admitir aliases y búsqueda O(1).

CMD_ADELANTE:  frozenset[str] = frozenset({"adelante", "avanza", "avanzar"})
CMD_ATRAS:     frozenset[str] = frozenset({"atras", "retrocede", "retroceder"})
CMD_IZQUIERDA: frozenset[str] = frozenset({"izquierda", "izquierdo"})
CMD_DERECHA:   frozenset[str] = frozenset({"derecha", "derecho"})
CMD_GIRA:      frozenset[str] = frozenset({"gira", "girar", "spin"})
CMD_STOP:      frozenset[str] = frozenset({"detente", "stop", "alto", "parar"})
CMD_UNKNOWN:   str             = "<unk>"

# Comandos del lifter — descomentar cuando el hardware esté disponible
# CMD_LIFTER_ARRIBA: frozenset[str] = frozenset({"arriba", "sube"})
# CMD_LIFTER_ABAJO:  frozenset[str] = frozenset({"abajo",  "baja"})


class VoiceCommandHandlerNode(Node):
    """
    Traduce comandos de voz validados en mensajes Twist para el PuzzleBot.

    Opción B: publica directo en /cmd_vel, sin pasar por bug_IBA.

    Política de seguridad:
      - Si /voice_command_valid llega como False → robot se detiene.
      - Si el comando es "<unk>" o no está mapeado → robot se detiene.
      - Tras cmd_duration segundos sin nuevo comando → robot se detiene.

    Nota sobre el temporizador:
      Se usa time.monotonic() + timer loop a 10 Hz en lugar de
      create_timer() para el auto-stop. Esto evita la race condition
      del timer que no es truly one-shot en rclpy.
    """

    def __init__(self) -> None:
        super().__init__("voice_command_handler_node")

        # ── Parámetros configurables ──────────────────────────────────────
        self.declare_parameter("linear_speed",  0.22)   # m/s
        self.declare_parameter("angular_speed", 0.40)   # rad/s — giro normal
        self.declare_parameter("spin_speed",    0.60)   # rad/s — giro rápido
        self.declare_parameter("cmd_duration",  1.50)   # segundos

        self._v_lin  = self.get_parameter("linear_speed").value
        self._w_ang  = self.get_parameter("angular_speed").value
        self._w_spin = self.get_parameter("spin_speed").value
        self._dur    = self.get_parameter("cmd_duration").value

        # ── Estado interno ────────────────────────────────────────────────
        self._last_valid:   bool  = False
        self._action_end_t: float = 0.0
        self._active_twist: Twist = Twist()

        # ── Suscriptores ──────────────────────────────────────────────────
        self.create_subscription(Bool,   "voice_command_valid", self._cb_valid,   10)
        self.create_subscription(String, "voice_command",       self._cb_command, 10)

        # ── Publicador — directo a /cmd_vel (Opción B) ────────────────────
        self._pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)

        # ── Lifter (pendiente de hardware) ────────────────────────────────
        # Descomentar junto con los elif en _build_twist():
        # from std_msgs.msg import Int8
        # self._pub_lifter = self.create_publisher(Int8, "/lifter_cmd", 10)

        # ── Timer de control a 10 Hz ──────────────────────────────────────
        # Publica el Twist activo mientras la acción no haya expirado.
        # También maneja el auto-stop cuando expira cmd_duration.
        self.create_timer(0.10, self._timer_loop)

        self.get_logger().info(
            "VoiceCommandHandlerNode iniciado → /cmd_vel (Opción B, sin bug_IBA) | "
            f"v={self._v_lin} m/s | w={self._w_ang} rad/s | "
            f"spin={self._w_spin} rad/s | dur={self._dur} s"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _cb_valid(self, msg: Bool) -> None:
        """Actualiza el flag de validez del último paquete de reconocimiento."""
        self._last_valid = bool(msg.data)

    def _cb_command(self, msg: String) -> None:
        """
        Procesa el comando de voz recién llegado.
        Si es válido, activa el Twist correspondiente durante cmd_duration segundos.
        """
        command = msg.data.strip().lower()
        self.get_logger().info(f'Comando: "{command}" | válido={self._last_valid}')

        # Seguridad: descartar si el HMM no confía en el comando
        if not self._last_valid or command == CMD_UNKNOWN:
            self._stop_robot()
            self.get_logger().warning(
                f'Comando ignorado ("{command}"). Robot detenido.'
            )
            return

        # Construir Twist y activar la acción temporizada
        twist = self._build_twist(command)

        if twist is None:
            self.get_logger().warning(
                f'Sin mapeo para "{command}". Robot detenido por seguridad.'
            )
            self._stop_robot()
            return

        # Activar acción: el timer_loop la publica a 10 Hz hasta que expire
        self._active_twist = twist
        self._action_end_t = time.monotonic() + self._dur
        self.get_logger().info(
            f'Ejecutando "{command}" → '
            f'v={twist.linear.x:.2f} m/s, '
            f'w={twist.angular.z:.2f} rad/s '
            f'por {self._dur:.1f} s'
        )

    # ── Timer de publicación (10 Hz) ──────────────────────────────────────

    def _timer_loop(self) -> None:
        """
        Publica el Twist activo mientras la acción no haya expirado.
        Cuando expira, publica Twist=0 una sola vez y limpia el estado.
        """
        now = time.monotonic()

        if now < self._action_end_t:
            self._pub_cmd.publish(self._active_twist)
        else:
            # Solo detiene una vez (cuando aún hay movimiento activo)
            active = (
                self._active_twist.linear.x  != 0.0 or
                self._active_twist.angular.z != 0.0
            )
            if active:
                self._stop_robot()
                if self._action_end_t != 0.0:
                    self.get_logger().info("Auto-stop: duración completada.")

    # ── Construcción del Twist ─────────────────────────────────────────────

    def _build_twist(self, command: str) -> Twist | None:
        """
        Mapea una palabra clave a un mensaje Twist.

        Returns:
            Twist configurado, o None si el comando no está mapeado.
        """
        twist = Twist()

        if command in CMD_ADELANTE:
            twist.linear.x = self._v_lin

        elif command in CMD_ATRAS:
            twist.linear.x = -self._v_lin

        elif command in CMD_IZQUIERDA:
            twist.angular.z = self._w_ang

        elif command in CMD_DERECHA:
            twist.angular.z = -self._w_ang

        elif command in CMD_GIRA:
            twist.angular.z = self._w_spin

        elif command in CMD_STOP:
            pass  # Twist en cero = detener

        # ── Lifter (pendiente de integración) ─────────────────────────────
        # Descomentar cuando el hardware esté listo y ajustar tipo de mensaje.
        #
        # elif command in CMD_LIFTER_ARRIBA:
        #     from std_msgs.msg import Int8
        #     msg_lift = Int8()
        #     msg_lift.data = 1          # 1 = subir
        #     self._pub_lifter.publish(msg_lift)
        #     self.get_logger().info("Lifter: subiendo")
        #     return twist               # Sin movimiento de base
        #
        # elif command in CMD_LIFTER_ABAJO:
        #     from std_msgs.msg import Int8
        #     msg_lift = Int8()
        #     msg_lift.data = -1         # -1 = bajar
        #     self._pub_lifter.publish(msg_lift)
        #     self.get_logger().info("Lifter: bajando")
        #     return twist               # Sin movimiento de base
        # ──────────────────────────────────────────────────────────────────

        else:
            return None  # Comando no mapeado

        return twist

    # ── Helper ────────────────────────────────────────────────────────────

    def _stop_robot(self) -> None:
        """Publica Twist=0 y resetea el estado de la acción activa."""
        self._active_twist = Twist()
        self._action_end_t = 0.0
        self._pub_cmd.publish(self._active_twist)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None) -> None:
    rclpy.init(args=args)
    node = VoiceCommandHandlerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Garantizar que el robot quede detenido al cerrar el nodo
        node._stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
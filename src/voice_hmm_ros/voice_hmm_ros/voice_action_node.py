#!/usr/bin/env python3
"""
================================================================================
Proyecto:    voice_hmm_ros
Módulo:      voice_action_node.py

Descripción:
    Nodo ROS 2 que traduce comandos de voz reconocidos en acciones del robot.

    Entrada:
        /voice_command   (std_msgs/String)
            Palabra reconocida por el nodo HMM:
            avanza, atras, izquierda, derecha, gira, detente,
            arriba, abajo, toma, suelta o "<unk>".

        /lift_done       (std_msgs/String)
            Confirmación del nodo spi_servo_node.py:
            AT_N1, AT_N2, HOLD, DOWN, MANUAL_DONE, etc.

    Salidas:
        /cmd_vel         (geometry_msgs/Twist)
            Movimiento de la base móvil.

        /lift_auto       (std_msgs/String)
            Comandos automáticos para el nodo spi_servo_node.py:
            n1, n2, hold, down.

        /lift_trigger    (std_msgs/Int8)
            Stop manual del lifter usando 0.

    Mapeo de comandos:
        Base:
            avanza      -> avanzar
            atras       -> retroceder
            izquierda   -> girar izquierda
            derecha     -> girar derecha
            gira        -> giro 360 grados
            detente     -> paro general

        Lifter:
            arriba      -> n2
            abajo       -> down / IDLE
            toma        -> n1 y después hold (encadenado vía /lift_done)
            suelta      -> down / IDLE

    Diseño:
        - Este nodo NO controla SPI directamente.
        - spi_servo_node.py sigue siendo responsable del FPGA, MISO,
          validación de estados y publicación de /lift_done.
        - Este nodo solo traduce voz a intención de movimiento/lifter.

================================================================================
"""

from __future__ import annotations

import math
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Int8


# ============================================================
# Vocabulario entrenado  (frozensets para O(1) lookup)
# ============================================================

CMD_ADELANTE  = frozenset({"avanza"})
CMD_ATRAS     = frozenset({"atras"})
CMD_IZQUIERDA = frozenset({"izquierda"})
CMD_DERECHA   = frozenset({"derecha"})
CMD_GIRA      = frozenset({"gira"})
CMD_STOP      = frozenset({"detente"})

CMD_LIFT_N2   = frozenset({"arriba"})
CMD_LIFT_DOWN = frozenset({"abajo", "suelta"})
CMD_LIFT_TAKE = frozenset({"toma"})

CMD_UNKNOWN   = "<unk>"

# Timeout de seguridad para la secuencia n1 → hold (segundos).
# Si /lift_done no llega en este tiempo, la acción pendiente se cancela.
LIFT_SEQUENCE_TIMEOUT_S: float = 8.0


class VoiceActionNode(Node):
    """
    Convierte comandos de voz en acciones de base y lifter.

    Base:
        Publica Twist en /cmd_vel.

    Lifter:
        Publica String en /lift_auto para el nodo spi_servo_node.py.
        Publica Int8 en /lift_trigger solo para stop manual seguro (valor 0).
        Escucha /lift_done para encadenar toma: n1 -> hold.
    """

    def __init__(self) -> None:
        super().__init__("voice_action_node")

        # ============================================================
        # Parámetros configurables
        # ============================================================

        self.declare_parameter("linear_speed",  0.22)
        self.declare_parameter("angular_speed", 0.40)
        self.declare_parameter("spin_speed",    0.60)
        self.declare_parameter("cmd_duration",  1.50)

        self._v_lin       = float(self.get_parameter("linear_speed").value)
        self._w_ang       = float(self.get_parameter("angular_speed").value)
        self._w_spin      = float(self.get_parameter("spin_speed").value)
        self._cmd_duration = float(self.get_parameter("cmd_duration").value)

        # --- Validación de parámetros ---
        errors: list[str] = []
        if self._v_lin < 0.0:
            errors.append("linear_speed debe ser >= 0.0")
        if self._w_ang < 0.0:
            errors.append("angular_speed debe ser >= 0.0")
        if self._w_spin <= 0.0:
            errors.append("spin_speed debe ser > 0.0 para calcular giro 360°")
        if errors:
            raise ValueError("; ".join(errors))

        if self._cmd_duration <= 0.0:
            self.get_logger().warning(
                "cmd_duration <= 0.0; usando 1.5 s por seguridad."
            )
            self._cmd_duration = 1.5

        # Duración de giro 360°: θ = ω·t  →  t = 2π / ω
        self._spin_360_duration: float = (2.0 * math.pi) / self._w_spin

        # ============================================================
        # Estado interno — base móvil
        # ============================================================

        self._active_twist: Twist = Twist()
        self._active_until: float = 0.0   # tiempo absoluto (monotonic)

        # ============================================================
        # Estado interno — lifter
        # ============================================================

        # "HOLD_AFTER_N1"  → esperando AT_N1 de /lift_done para mandar hold.
        # None             → sin secuencia pendiente.
        self._pending_lift_action: Optional[str] = None

        # Marca de tiempo para detectar timeout en la secuencia toma.
        self._lift_sequence_start: float = 0.0

        # ============================================================
        # ROS I/O
        # ============================================================

        # FIX: el nodo HMM publica en /voice_command (sin barra inicial)
        # para permanecer consistente con la convención del paquete.
        # Ambas formas ("voice_command" y "/voice_command") se resuelven igual
        # en ROS 2 si el nodo no tiene namespace, pero se usa el nombre relativo
        # por consistencia con el resto del package.
        self.create_subscription(
            String,
            "voice_command",
            self._cb_voice_command,
            10,
        )

        self.create_subscription(
            String,
            "/lift_done",
            self._cb_lift_done,
            10,
        )

        self._pub_cmd_vel = self.create_publisher(Twist,  "/cmd_vel",       10)
        self._pub_lift_auto    = self.create_publisher(String, "/lift_auto",    10)
        self._pub_lift_trigger = self.create_publisher(Int8,   "/lift_trigger", 10)

        # Timer a 20 Hz: mantiene /cmd_vel activo y aplica auto-stop de base.
        # También vigila el timeout de la secuencia de lift.
        self.create_timer(0.05, self._timer_loop)

        self.get_logger().info(
            "VoiceActionNode listo.\n"
            "  Entrada: /voice_command\n"
            "  Lifter feedback: /lift_done\n"
            "  Salidas: /cmd_vel, /lift_auto, /lift_trigger\n"
            f"  Base: v={self._v_lin:.2f} m/s | "
            f"w_turn={self._w_ang:.2f} rad/s | "
            f"w_spin={self._w_spin:.2f} rad/s | "
            f"dur_base={self._cmd_duration:.2f} s | "
            f"dur_360={self._spin_360_duration:.2f} s"
        )

    # ============================================================
    # Callback principal de voz
    # ============================================================

    def _cb_voice_command(self, msg: String) -> None:
        command: str = msg.data.strip().lower()

        self.get_logger().info(f'Comando de voz recibido: "{command}"')

        # Comando vacío o desconocido → paro seguro general
        if not command or command == CMD_UNKNOWN:
            self._stop_all("Comando desconocido o vacío: paro seguro.")
            return

        # Stop general (detente)
        if command in CMD_STOP:
            self._stop_all(f'Paro general por comando "{command}".')
            return

        # Lifter (arriba / abajo / toma / suelta)
        if self._handle_lift_command(command):
            return

        # Base (avanza / atras / izquierda / derecha / gira)
        twist, duration = self._build_base_action(command)
        if twist is None:
            self._stop_all(f'Comando sin mapeo: "{command}". Paro seguro.')
            return

        self._active_twist = twist
        self._active_until = time.monotonic() + duration

        self.get_logger().info(
            f'"{command}" → /cmd_vel  '
            f"linear.x={twist.linear.x:.2f}  "
            f"angular.z={twist.angular.z:.2f}  "
            f"dur={duration:.2f} s"
        )

    # ============================================================
    # Callback de feedback del lifter
    # ============================================================

    def _cb_lift_done(self, msg: String) -> None:
        label: str = msg.data.strip().upper()

        self.get_logger().info(f'/lift_done recibido: "{label}"')

        if self._pending_lift_action != "HOLD_AFTER_N1":
            return

        if label == "AT_N1":
            self.get_logger().info(
                'Secuencia "toma": AT_N1 confirmado → enviando "hold".'
            )
            self._send_lift_auto("hold", "toma/hold")
            self._pending_lift_action = None

        elif label in {"DOWN", "AT_N2", "HOLD", "MANUAL_DONE"}:
            # Estado inesperado: cancelar para no dejar una secuencia colgada.
            self.get_logger().warning(
                f'Secuencia "toma" esperaba AT_N1 pero recibió "{label}". '
                "Cancelando acción pendiente."
            )
            self._pending_lift_action = None

    # ============================================================
    # Timer principal (20 Hz)
    # ============================================================

    def _timer_loop(self) -> None:
        now = time.monotonic()

        # --- Auto-stop de base ---
        if self._active_until > 0.0:
            if now < self._active_until:
                self._pub_cmd_vel.publish(self._active_twist)
            else:
                self._stop_base("Auto-stop: duración de base completada.")

        # --- Timeout de secuencia de lift ---
        # FIX: el original nunca vigilaba que la secuencia "toma" pudiera
        # quedarse pendiente indefinidamente si /lift_done nunca llegaba
        # (nodo spi apagado, cable SPI defectuoso, etc.).
        if (
            self._pending_lift_action == "HOLD_AFTER_N1"
            and self._lift_sequence_start > 0.0
            and (now - self._lift_sequence_start) > LIFT_SEQUENCE_TIMEOUT_S
        ):
            self.get_logger().warning(
                f'Secuencia "toma": timeout ({LIFT_SEQUENCE_TIMEOUT_S:.1f} s) '
                "sin confirmación AT_N1. Cancelando."
            )
            self._pending_lift_action = None
            self._lift_sequence_start = 0.0

    # ============================================================
    # Acciones de base
    # ============================================================

    def _build_base_action(self, command: str) -> tuple[Optional[Twist], float]:
        """
        Construye el Twist y la duración para los comandos de base.

        Para "gira" calcula la duración de una vuelta completa:
            t = 2π / spin_speed
        """
        twist    = Twist()
        duration = self._cmd_duration

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
            duration = self._spin_360_duration

        else:
            return None, 0.0

        return twist, duration

    # ============================================================
    # Acciones de lifter
    # ============================================================

    def _handle_lift_command(self, command: str) -> bool:
        """
        Ejecuta comandos del lifter.

        Mapeo:
            arriba → n2
            abajo  → down
            suelta → down
            toma   → n1, luego hold cuando /lift_done indique AT_N1
        """
        if command in CMD_LIFT_N2:
            self._pending_lift_action = None
            self._send_lift_auto("n2", command)
            return True

        if command in CMD_LIFT_DOWN:
            self._pending_lift_action = None
            self._send_lift_auto("down", command)
            return True

        if command in CMD_LIFT_TAKE:
            # FIX: registrar timestamp para poder detectar timeout.
            self._pending_lift_action = "HOLD_AFTER_N1"
            self._lift_sequence_start = time.monotonic()
            self._send_lift_auto("n1", command)
            return True

        return False

    def _send_lift_auto(self, lift_cmd: str, source_word: str) -> None:
        """Publica un comando en /lift_auto para spi_servo_node."""
        msg = String()
        msg.data = lift_cmd
        self._pub_lift_auto.publish(msg)
        self.get_logger().info(f'"{source_word}" → /lift_auto: "{lift_cmd}"')

    def _stop_lifter(self) -> None:
        """
        Para el lifter de forma manual usando /lift_trigger = 0.

        spi_servo_node interpreta Int8(0) como CMD_STOP.

        FIX: también cancela cualquier secuencia pendiente y resetea
        el timestamp de secuencia.
        """
        self._pending_lift_action = None
        self._lift_sequence_start = 0.0

        msg = Int8()
        msg.data = 0
        self._pub_lift_trigger.publish(msg)
        self.get_logger().info("Lifter detenido vía /lift_trigger: 0")

    # ============================================================
    # Stops seguros
    # ============================================================

    def _stop_base(self, reason: Optional[str] = None) -> None:
        """Detiene la base y cancela el temporizador de movimiento."""
        self._active_twist = Twist()
        self._active_until = 0.0
        self._pub_cmd_vel.publish(self._active_twist)
        if reason:
            self.get_logger().info(reason)

    def _stop_all(self, reason: Optional[str] = None) -> None:
        """Paro de emergencia: detiene base y lifter."""
        self._stop_base()
        self._stop_lifter()
        if reason:
            self.get_logger().info(reason)


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = VoiceActionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_all("Cierre seguro: base y lifter detenidos.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
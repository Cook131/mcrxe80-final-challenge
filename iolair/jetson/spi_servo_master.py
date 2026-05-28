#!/usr/bin/env python3
"""
spi_servo_node.py
Jetson Nano → Tang Nano 20K (SPI slave) → 2× MG90S 360°

Protocolo (3 bytes por transferencia):
  Byte 0 : 0xAB  ← header / identificador de comando
  Byte 1 : servo1 (0-255 | 0=reversa max, 127=stop, 255=adelante max)
  Byte 2 : servo2 (misma escala)

Topics suscritos:
  /servo_cmd  [geometry_msgs/Twist]
      linear.x  → servo1  (-1.0 .. +1.0  →  -100 .. +100)
      angular.z → servo2  (misma escala)

  /servo_raw  [std_msgs/Int16MultiArray]
      data[0] → servo1  (-100 .. +100, entero directo)
      data[1] → servo2

Topic publicado:
  /servo_state [std_msgs/Int16MultiArray]
      data[0], data[1] = valores actuales enviados al FPGA

Parámetros ROS2 declarables (ros2 run ... --ros-args -p spi_speed:=1000000):
  spi_device   (str)   → "/dev/spidev0.0"
  spi_speed    (int)   → 500000
  spi_mode     (int)   → 0
  publish_rate (float) → 10.0  (Hz del publisher de estado)

Compilar / instalar: colcon build --packages-select <tu_paquete>
                     o directamente:  python3 spi_servo_node.py
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from geometry_msgs.msg import Twist
from std_msgs.msg import Int16MultiArray

try:
    import spidev
    SPIDEV_AVAILABLE = True
except ImportError:
    SPIDEV_AVAILABLE = False


# ── Constantes de protocolo ────────────────────────────────────────────────────
CMD_SERVO   = 0xAB
SERVO_STOP  = 127


# ── Helpers ────────────────────────────────────────────────────────────────────

def speed_to_byte(speed: float) -> int:
    """
    Convierte velocidad normalizada (-100..+100) al byte que espera el FPGA.
      -100  →    0   (reversa máxima, PWM ~1 ms)
         0  →  127   (stop,           PWM ~1.5 ms)
      +100  →  255   (adelante max,   PWM ~2 ms)
    Acepta float; aplica clamp interno.
    """
    speed = max(-100.0, min(100.0, float(speed)))
    return int((speed + 100.0) * 255.0 / 200.0)


def twist_to_speeds(msg: Twist):
    """
    Mapea un Twist a (s1, s2) en el rango -100..+100.
    linear.x  controla servo 1.
    angular.z controla servo 2.
    El rango Twist estándar es -1.0..+1.0, se escala a -100..+100.
    """
    s1 = msg.linear.x  * 100.0
    s2 = msg.angular.z * 100.0
    return s1, s2


# ── Nodo principal ─────────────────────────────────────────────────────────────

class SpiServoNode(Node):

    def __init__(self):
        super().__init__('spi_servo_node')

        # ── Parámetros ─────────────────────────────────────────────────────────
        self.declare_parameter('spi_device',   '/dev/spidev0.0')
        self.declare_parameter('spi_speed',    500_000)
        self.declare_parameter('spi_mode',     0)
        self.declare_parameter('publish_rate', 10.0)

        spi_device   = self.get_parameter('spi_device').value
        spi_speed    = self.get_parameter('spi_speed').value
        spi_mode     = self.get_parameter('spi_mode').value
        publish_rate = self.get_parameter('publish_rate').value

        # ── Estado interno ─────────────────────────────────────────────────────
        self._s1: float = 0.0   # velocidad actual servo 1  (-100..+100)
        self._s2: float = 0.0   # velocidad actual servo 2

        # ── SPI ────────────────────────────────────────────────────────────────
        self._spi = None
        if SPIDEV_AVAILABLE:
            try:
                bus, device = self._parse_spidev(spi_device)
                self._spi = spidev.SpiDev()
                self._spi.open(bus, device)
                self._spi.max_speed_hz = spi_speed
                self._spi.mode         = spi_mode
                self._spi.bits_per_word = 8
                self.get_logger().info(
                    f'SPI abierto: {spi_device} | '
                    f'mode={spi_mode} | speed={spi_speed} Hz'
                )
            except Exception as e:
                self.get_logger().error(f'No se pudo abrir SPI: {e}')
                self._spi = None
        else:
            self.get_logger().warn(
                'spidev no está instalado — modo DRY-RUN '
                '(los comandos se loguean pero no se envían)'
            )

        # Detener servos al arrancar por seguridad
        self._send_servos(0.0, 0.0)

        # ── Subscribers ────────────────────────────────────────────────────────
        self._sub_twist = self.create_subscription(
            Twist,
            '/servo_cmd',
            self._cb_twist,
            10
        )
        self._sub_raw = self.create_subscription(
            Int16MultiArray,
            '/servo_raw',
            self._cb_raw,
            10
        )

        # ── Publisher de estado ────────────────────────────────────────────────
        self._pub_state = self.create_publisher(Int16MultiArray, '/servo_state', 10)
        self._timer = self.create_timer(
            1.0 / publish_rate,
            self._publish_state
        )

        self.get_logger().info('spi_servo_node activo ✓')

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _cb_twist(self, msg: Twist):
        """Recibe geometry_msgs/Twist y actualiza los servos."""
        s1, s2 = twist_to_speeds(msg)
        self._send_servos(s1, s2)

    def _cb_raw(self, msg: Int16MultiArray):
        """
        Recibe Int16MultiArray con [s1, s2] directamente en -100..+100.
        Útil para control manual o teleop personalizado.
        """
        if len(msg.data) < 2:
            self.get_logger().warn(
                '/servo_raw requiere al menos 2 elementos en data[]'
            )
            return
        self._send_servos(float(msg.data[0]), float(msg.data[1]))

    # ── Lógica SPI ─────────────────────────────────────────────────────────────

    def _send_servos(self, s1: float, s2: float):
        """
        Construye el paquete [0xAB, b1, b2] y lo transfiere por SPI.
        Actualiza estado interno independientemente de si el envío fue exitoso.
        """
        self._s1 = max(-100.0, min(100.0, s1))
        self._s2 = max(-100.0, min(100.0, s2))

        b1 = speed_to_byte(self._s1)
        b2 = speed_to_byte(self._s2)
        packet = [CMD_SERVO, b1, b2]

        if self._spi is not None:
            try:
                self._spi.xfer2(packet)
            except Exception as e:
                self.get_logger().error(f'SPI transfer falló: {e}')
                return
        else:
            # Dry-run: solo loguear
            self.get_logger().debug(
                f'[DRY-RUN] TX → [0x{CMD_SERVO:02X}] '
                f'[0x{b1:02X} ({self._s1:+.0f})] '
                f'[0x{b2:02X} ({self._s2:+.0f})]'
            )

        self.get_logger().debug(
            f'Servos → S1:{self._s1:+.0f}  S2:{self._s2:+.0f}  '
            f'TX:[0x{CMD_SERVO:02X}, 0x{b1:02X}, 0x{b2:02X}]'
        )

    def _publish_state(self):
        """Publica el estado actual de velocidad en /servo_state."""
        msg = Int16MultiArray()
        msg.data = [int(self._s1), int(self._s2)]
        self._pub_state.publish(msg)

    # ── Utilidades ─────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_spidev(device: str):
        """
        Extrae (bus, device) de un path tipo '/dev/spidev0.0'.
        Retorna (0, 0) como fallback seguro.
        """
        try:
            parts = device.replace('/dev/spidev', '').split('.')
            return int(parts[0]), int(parts[1])
        except Exception:
            return 0, 0

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def destroy_node(self):
        """Detiene los servos y cierra el bus SPI antes de destruir el nodo."""
        self.get_logger().info('Deteniendo servos y cerrando SPI...')
        self._send_servos(0.0, 0.0)
        if self._spi is not None:
            try:
                self._spi.close()
            except Exception:
                pass
        super().destroy_node()


# ── Entrypoint ─────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = SpiServoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
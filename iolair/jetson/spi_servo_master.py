#!/usr/bin/env python3
"""
spi_servo_node.py
Jetson Nano → Tang Nano 20K (SPI slave) → 2× Servo 360°

Protocolo (3 bytes por transferencia, MSB first):
  Byte 0 : 0xAB  ← header
  Byte 1 : servo1  (0x00=reversa máx · 0x7F=stop · 0xFF=adelante máx)
  Byte 2 : servo2  (misma escala)

Topics suscritos:
  /servo_cmd  [geometry_msgs/Twist]
      linear.x  → servo1  (-1.0 .. +1.0)
      angular.z → servo2  (-1.0 .. +1.0)

  /servo_raw  [std_msgs/Int16MultiArray]
      data[0] → servo1  (-100 .. +100)
      data[1] → servo2

Topic publicado:
  /servo_state [std_msgs/Int16MultiArray]
      data[0], data[1] = valores actuales (-100..+100)

Parámetros ROS2:
  spi_device   (str)   → "/dev/spidev0.0"
  spi_speed    (int)   → 500000
  spi_mode     (int)   → 0
  publish_rate (float) → 10.0

Uso rápido de prueba (sin ROS2):
  python3 spi_servo_node.py   ← usa __main__ con test de barrido
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Int16MultiArray

try:
    import spidev
    SPIDEV_AVAILABLE = True
except ImportError:
    SPIDEV_AVAILABLE = False


# ── Constantes de protocolo ────────────────────────────────────────────────────
CMD_HEADER  = 0xAB
SERVO_STOP  = 127


# ── Helpers ────────────────────────────────────────────────────────────────────

def speed_to_byte(speed: float) -> int:
    """
    Convierte velocidad normalizada (-100..+100) al byte del FPGA.
      -100  →   0   (reversa máxima, PWM 1.0 ms)
         0  → 127   (stop,           PWM 1.5 ms)
      +100  → 255   (adelante máx,   PWM 2.0 ms)
    """
    speed = max(-100.0, min(100.0, float(speed)))
    return int((speed + 100.0) * 255.0 / 200.0)


def byte_to_speed(b: int) -> float:
    """Inversa de speed_to_byte, para el topic de estado."""
    return round((b / 255.0) * 200.0 - 100.0, 1)


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
        self._s1: float = 0.0
        self._s2: float = 0.0

        # ── SPI ────────────────────────────────────────────────────────────────
        self._spi = None
        if SPIDEV_AVAILABLE:
            try:
                bus, device = self._parse_spidev(spi_device)
                spi = spidev.SpiDev()
                spi.open(bus, device)
                spi.max_speed_hz  = spi_speed
                spi.mode          = spi_mode       # CPOL=0 CPHA=0
                spi.bits_per_word = 8
                spi.lsbfirst      = False           # MSB first — igual que el FPGA
                spi.cshigh        = False           # CS activo bajo — igual que cs_n del FPGA
                # no_cs = False (default): el kernel maneja CS automáticamente
                self._spi = spi
                self.get_logger().info(
                    f'SPI abierto: {spi_device} | mode={spi_mode} | '
                    f'{spi_speed // 1000} kHz | MSB-first | CS active-low'
                )
            except Exception as e:
                self.get_logger().error(
                    f'No se pudo abrir SPI ({spi_device}): {e}\n'
                    f'  → Verifica: ls /dev/spi*  y  sudo chmod 666 {spi_device}'
                )
        else:
            self.get_logger().warn(
                'spidev no disponible — modo DRY-RUN '
                '(comandos logueados, no enviados)'
            )

        # ── Detener servos al arrancar ─────────────────────────────────────────
        self._send_servos(0.0, 0.0)

        # ── Subscribers ────────────────────────────────────────────────────────
        self.create_subscription(Twist,            '/servo_cmd', self._cb_twist, 10)
        self.create_subscription(Int16MultiArray,  '/servo_raw', self._cb_raw,   10)

        # ── Publisher de estado ────────────────────────────────────────────────
        self._pub_state = self.create_publisher(Int16MultiArray, '/servo_state', 10)
        self.create_timer(1.0 / publish_rate, self._publish_state)

        self.get_logger().info('spi_servo_node activo ✓')

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _cb_twist(self, msg: Twist):
        self._send_servos(msg.linear.x * 100.0, msg.angular.z * 100.0)

    def _cb_raw(self, msg: Int16MultiArray):
        if len(msg.data) < 2:
            self.get_logger().warn('/servo_raw necesita data[0] y data[1]')
            return
        self._send_servos(float(msg.data[0]), float(msg.data[1]))

    # ── Lógica SPI ─────────────────────────────────────────────────────────────

    def _send_servos(self, s1: float, s2: float):
        """
        Envía [0xAB, b1, b2] por SPI.

        xfer2() mantiene CS bajo durante los 3 bytes (un solo ciclo CS).
        Si el kernel de Jetson hace toggle de CS entre bytes, el FPGA
        igual procesa correctamente gracias al shift-register deslizante.
        """
        self._s1 = max(-100.0, min(100.0, s1))
        self._s2 = max(-100.0, min(100.0, s2))

        b1 = speed_to_byte(self._s1)
        b2 = speed_to_byte(self._s2)
        packet = [CMD_HEADER, b1, b2]

        if self._spi is not None:
            try:
                self._spi.xfer2(packet)
                self.get_logger().info(
                    f'SPI TX → [0x{CMD_HEADER:02X} 0x{b1:02X} 0x{b2:02X}]  '
                    f'S1:{self._s1:+.0f}%  S2:{self._s2:+.0f}%'
                )
            except Exception as e:
                self.get_logger().error(f'SPI transfer falló: {e}')
        else:
            self.get_logger().info(
                f'[DRY-RUN] TX → [0x{CMD_HEADER:02X} 0x{b1:02X} 0x{b2:02X}]  '
                f'S1:{self._s1:+.0f}%  S2:{self._s2:+.0f}%'
            )

    def _publish_state(self):
        msg = Int16MultiArray()
        msg.data = [int(self._s1), int(self._s2)]
        self._pub_state.publish(msg)

    # ── Utilidades ─────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_spidev(device: str):
        try:
            parts = device.replace('/dev/spidev', '').split('.')
            return int(parts[0]), int(parts[1])
        except Exception:
            return 0, 0

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def destroy_node(self):
        self.get_logger().info('Deteniendo servos y cerrando SPI...')
        self._send_servos(0.0, 0.0)
        if self._spi is not None:
            try:
                self._spi.close()
            except Exception:
                pass
        super().destroy_node()


# ── Entrypoint ROS2 ────────────────────────────────────────────────────────────

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


# ── Test rápido sin ROS2 ───────────────────────────────────────────────────────
# Ejecuta:  python3 spi_servo_node.py
# Hace un barrido S1: -100→+100, S2: fijo en stop
# Sirve para verificar que el FPGA responde correctamente.

if __name__ == '__main__':
    import time

    if not SPIDEV_AVAILABLE:
        print('[ERROR] spidev no instalado. Ejecuta: pip install spidev')
        exit(1)

    print('=== Test SPI sin ROS2 ===')
    print('Verifica con osciloscopio en pin 15 (cs_debug) de Tang Nano.')
    print('Los servos deberían barrer de reversa a adelante.\n')

    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.max_speed_hz  = 500_000
    spi.mode          = 0
    spi.bits_per_word = 8
    spi.lsbfirst      = False
    spi.cshigh        = False

    try:
        # Enviar STOP primero
        spi.xfer2([CMD_HEADER, SERVO_STOP, SERVO_STOP])
        print(f'STOP → [0xAB 0x{SERVO_STOP:02X} 0x{SERVO_STOP:02X}]')
        time.sleep(1.0)

        # Barrido S1: reversa → adelante, S2 fijo en stop
        print('Barrido servo1...')
        for speed in range(-100, 101, 10):
            b = speed_to_byte(float(speed))
            spi.xfer2([CMD_HEADER, b, SERVO_STOP])
            print(f'  S1:{speed:+4d}% → byte 0x{b:02X}')
            time.sleep(0.3)

        # Volver a STOP
        spi.xfer2([CMD_HEADER, SERVO_STOP, SERVO_STOP])
        print('\nSTOP enviado. Test finalizado.')

    finally:
        spi.xfer2([CMD_HEADER, SERVO_STOP, SERVO_STOP])
        spi.close()

#!/usr/bin/env python3
"""
spi_servo_node.py
Jetson Nano → Tang Nano 20K (SPI slave) → 1× Servo 360° (lift)

Protocolo SPI:  [0xAB] [servo1_byte] [0x7F]

─── Interfaz principal ────────────────────────────────────────
  /lift_trigger  [std_msgs/Int8]
       1  → SUBIENDO
      -1  → BAJANDO
       0  → STOPPED
  (también acepta /lift_cmd String "up"/"down"/"stop" por compatibilidad)

─── Publicado ─────────────────────────────────────────────────
  /lift_state   [std_msgs/String]   "STOPPED"|"SUBIENDO"|"BAJANDO"
  /servo_state  [std_msgs/Int16MultiArray]  [velocidad_actual, 0]

─── Parámetros ROS2 ───────────────────────────────────────────
  speed_up      (float) → 100.0   % al subir
  speed_down    (float) → 100.0   % al bajar
  neutral_byte  (int)   → 127     byte real de stop del servo
                                  (calibrar con __main__ --cal)
  invert        (bool)  → False   invierte dirección
  spi_device    (str)   → "/dev/spidev0.0"
  spi_speed     (int)   → 500000
  publish_rate  (float) → 10.0
"""

import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray, String, Int8

try:
    import spidev
    SPIDEV_AVAILABLE = True
except ImportError:
    SPIDEV_AVAILABLE = False

CMD_HEADER = 0xAB

class LiftState:
    STOPPED  = 'STOPPED'
    SUBIENDO = 'SUBIENDO'
    BAJANDO  = 'BAJANDO'


def speed_to_byte(speed_pct: float, neutral: int = 127) -> int:
    """
    Convierte velocidad % al byte PWM usando el neutral real del servo.

    neutral  → byte de stop real (calibrado)
    +100%    → 255  (adelante máx)
    -100%    → 0    (reversa máx)

    La escala se parte en dos mitades desde 'neutral':
      [neutral … 255]  para velocidades positivas
      [0 … neutral]    para velocidades negativas
    Así el servo respeta su zona muerta real.
    """
    speed_pct = max(-100.0, min(100.0, float(speed_pct)))
    if speed_pct >= 0:
        return neutral + int((255 - neutral) * speed_pct / 100.0)
    else:
        return neutral + int(neutral * speed_pct / 100.0)   # speed_pct negativo


# ── Nodo ───────────────────────────────────────────────────────────────────────

class SpiServoNode(Node):

    def __init__(self):
        super().__init__('spi_servo_node')

        # ── Parámetros ─────────────────────────────────────────────────────────
        self.declare_parameter('spi_device',    '/dev/spidev0.0')
        self.declare_parameter('spi_speed',     500_000)
        self.declare_parameter('spi_mode',      0)
        self.declare_parameter('publish_rate',  10.0)
        self.declare_parameter('speed_up',      100.0)
        self.declare_parameter('speed_down',    100.0)
        self.declare_parameter('neutral_byte',  127)
        self.declare_parameter('invert',        False)

        spi_device   = self.get_parameter('spi_device').value
        spi_speed    = self.get_parameter('spi_speed').value
        spi_mode     = self.get_parameter('spi_mode').value
        publish_rate = self.get_parameter('publish_rate').value
        self._speed_up    = float(self.get_parameter('speed_up').value)
        self._speed_down  = float(self.get_parameter('speed_down').value)
        self._neutral     = int(self.get_parameter('neutral_byte').value)
        self._invert      = bool(self.get_parameter('invert').value)

        self._speed      : float = 0.0
        self._lift_state : str   = LiftState.STOPPED

        # ── SPI ────────────────────────────────────────────────────────────────
        self._spi = None
        if SPIDEV_AVAILABLE:
            try:
                bus, dev = self._parse_spidev(spi_device)
                spi = spidev.SpiDev()
                spi.open(bus, dev)
                spi.max_speed_hz  = spi_speed
                spi.mode          = spi_mode
                spi.bits_per_word = 8
                spi.lsbfirst      = False
                spi.cshigh        = False
                self._spi = spi
                self.get_logger().info(
                    f'SPI {spi_device} {spi_speed//1000}kHz | '
                    f'neutral={self._neutral} | '
                    f'up={self._speed_up:.0f}% down={self._speed_down:.0f}% | '
                    f'invert={self._invert}'
                )
            except Exception as e:
                self.get_logger().error(f'SPI error: {e}')
        else:
            self.get_logger().warn('spidev no disponible — DRY-RUN')

        self._send(0.0)

        # ── Topics ─────────────────────────────────────────────────────────────
        # trigger principal (Int8: 1=up, -1=down, 0=stop)
        self.create_subscription(Int8,   '/lift_trigger', self._cb_trigger, 10)
        # compatibilidad con String
        self.create_subscription(String, '/lift_cmd',     self._cb_lift_cmd, 10)

        self._pub_state       = self.create_publisher(String,          '/lift_state',  10)
        self._pub_servo_state = self.create_publisher(Int16MultiArray, '/servo_state', 10)
        self.create_timer(1.0 / publish_rate, self._publish_state)

        self.get_logger().info(
            'spi_servo_node activo ✓\n'
            '  Trigger:  ros2 topic pub --once /lift_trigger std_msgs/msg/Int8 "{data: 1}"\n'
            '            ros2 topic pub --once /lift_trigger std_msgs/msg/Int8 "{data: -1}"\n'
            '            ros2 topic pub --once /lift_trigger std_msgs/msg/Int8 "{data: 0}"'
        )

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _cb_trigger(self, msg: Int8):
        if   msg.data ==  1: self._transition(LiftState.SUBIENDO)
        elif msg.data == -1: self._transition(LiftState.BAJANDO)
        elif msg.data ==  0: self._transition(LiftState.STOPPED)
        else:
            self.get_logger().warn(f'/lift_trigger: valor {msg.data} inválido (usa 1/-1/0)')

    def _cb_lift_cmd(self, msg: String):
        cmd = msg.data.strip().lower()
        if   cmd == 'up':   self._transition(LiftState.SUBIENDO)
        elif cmd == 'down': self._transition(LiftState.BAJANDO)
        elif cmd == 'stop': self._transition(LiftState.STOPPED)
        else:
            self.get_logger().warn(f'/lift_cmd: "{cmd}" no reconocido')

    # ── SM ─────────────────────────────────────────────────────────────────────

    def _transition(self, new_state: str):
        if new_state == self._lift_state:
            return
        prev = self._lift_state
        self._lift_state = new_state

        if new_state == LiftState.SUBIENDO:
            pct = +self._speed_up   if not self._invert else -self._speed_up
        elif new_state == LiftState.BAJANDO:
            pct = -self._speed_down if not self._invert else +self._speed_down
        else:
            pct = 0.0

        self._send(pct)
        self.get_logger().info(
            f'Lift: {prev} → {new_state}  '
            f'({pct:+.0f}% → byte {speed_to_byte(pct, self._neutral):3d})'
        )

    # ── SPI ────────────────────────────────────────────────────────────────────

    def _send(self, speed_pct: float):
        self._speed = max(-100.0, min(100.0, speed_pct))
        b1 = speed_to_byte(self._speed, self._neutral)
        if self._spi is not None:
            try:
                self._spi.xfer2([CMD_HEADER, b1, self._neutral])
            except Exception as e:
                self.get_logger().error(f'SPI falló: {e}')
                return
        self.get_logger().info(
            f'TX → [0xAB 0x{b1:02X} 0x{self._neutral:02X}]  '
            f'{self._speed:+.0f}%  [{self._lift_state}]'
        )

    def _publish_state(self):
        s = String(); s.data = self._lift_state
        self._pub_state.publish(s)
        m = Int16MultiArray(); m.data = [int(self._speed), 0]
        self._pub_servo_state.publish(m)

    @staticmethod
    def _parse_spidev(device: str):
        try:
            parts = device.replace('/dev/spidev', '').split('.')
            return int(parts[0]), int(parts[1])
        except Exception:
            return 0, 0

    def destroy_node(self):
        self._send(0.0)
        if self._spi:
            try: self._spi.close()
            except Exception: pass
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


# ── Test / calibración sin ROS2 ────────────────────────────────────────────────
#
#  MODO NORMAL:
#    python3 spi_servo_node.py [speed_up%] [speed_down%] [neutral_byte]
#    python3 spi_servo_node.py 90 60 127
#
#  MODO CALIBRACIÓN (encuentra el neutral real de tu servo):
#    python3 spi_servo_node.py --cal
#    Envía bytes 120→135 uno por uno; el servo debe PARAR en alguno.
#    Ese byte es tu neutral_byte real.

if __name__ == '__main__':
    import time

    if not SPIDEV_AVAILABLE:
        print('[ERROR] pip install spidev'); exit(1)

    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.max_speed_hz = 500_000
    spi.mode = 0; spi.bits_per_word = 8
    spi.lsbfirst = False; spi.cshigh = False

    def tx_raw(b):
        spi.xfer2([CMD_HEADER, b, 127])

    try:
        # ── Modo calibración ───────────────────────────────────────────────────
        if '--cal' in sys.argv:
            print('=== Calibración de neutral_byte ===')
            print('El servo debería moverse lento ahora.')
            print('Observa en qué byte se DETIENE completamente.\n')
            tx_raw(110); time.sleep(1.0)   # punto de partida lento
            for b in range(110, 150):
                tx_raw(b)
                print(f'  byte={b:3d}  (Ctrl+C si ya paró)')
                time.sleep(0.4)
            tx_raw(127)
            print('\nAl byte donde paró, pasa --neutral_byte:=<ese_byte> al nodo ROS2.')

        # ── Modo test normal ───────────────────────────────────────────────────
        else:
            speed_up   = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
            speed_down = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
            neutral    = int(sys.argv[3])   if len(sys.argv) > 3 else 127

            UP_B   = speed_to_byte(+speed_up,   neutral)
            DOWN_B = speed_to_byte(-speed_down,  neutral)

            print(f'=== Test lift ===  neutral={neutral}  up={speed_up:.0f}%(→{UP_B})  down={speed_down:.0f}%(→{DOWN_B})')

            print('[STOP]');     tx_raw(neutral); time.sleep(1.0)
            print('[SUBIENDO]'); tx_raw(UP_B);    time.sleep(3.0)
            print('[STOP]');     tx_raw(neutral); time.sleep(1.0)
            print('[BAJANDO]');  tx_raw(DOWN_B);  time.sleep(3.0)
            print('[STOP]');     tx_raw(neutral)
            print('Test finalizado.')

    except KeyboardInterrupt:
        print(f'\nAbortado en byte actual.')
    finally:
        tx_raw(127)
        spi.close()
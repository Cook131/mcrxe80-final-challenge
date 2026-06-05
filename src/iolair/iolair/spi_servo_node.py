#!/usr/bin/env python3
"""
spi_servo_node.py
Jetson Nano → Tang Nano 20K (SPI slave) → 1× Servo 360° (lift)

La SM y sus timers viven en el FPGA.
El FPGA devuelve su sm_state real en MISO en cada transacción SPI.
El 'done' se detecta cuando sm_state cambia a un estado estable.

─── sm_state devuelto en MISO ─────────────────────────────────
  0=IDLE   1=MAN_UP   2=MAN_DOWN
  3=TO_N1  4=AT_N1    5=TO_N2   6=AT_N2
  7=LIFTING  8=HOLD   9=LOWERING

─── /lift_done publica cuando FPGA transiciona a estado estable ─
  "AT_N1" | "AT_N2" | "HOLD" | "DOWN" | "MANUAL_DONE"

─── Protocolo SPI ─────────────────────────────────────────────
  TX [0xAC][cmd][0x00]  RX [sm_state][0x00][0x00]
  CMD_PING 0xFF = solo leer estado, sin cambiar nada

─── Test sin ROS2 ─────────────────────────────────────────────
  python3 spi_servo_node.py --test stop|n1|n2|hold|down|cycle|manual|status
"""

import sys
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int8, Int16MultiArray

try:
    import spidev
    SPIDEV_AVAILABLE = True
except ImportError:
    SPIDEV_AVAILABLE = False

# ── Protocolo ──────────────────────────────────────────────────────────────────
HDR_CMD    = 0xAC
HDR_DIRECT = 0xAB

CMD_PING     = 0xFF   # solo leer estado, sin cambiar nada
CMD_STOP     = 0x00
CMD_MAN_UP   = 0x20
CMD_MAN_DOWN = 0x21
CMD_GO_N1    = 0x10
CMD_GO_N2    = 0x11
CMD_GO_HOLD  = 0x12
CMD_GO_DOWN  = 0x13

CMD_NAMES = {
    CMD_PING: 'PING', CMD_STOP: 'STOP',
    CMD_MAN_UP: 'MAN_UP', CMD_MAN_DOWN: 'MAN_DOWN',
    CMD_GO_N1: 'GO_N1', CMD_GO_N2: 'GO_N2',
    CMD_GO_HOLD: 'GO_HOLD', CMD_GO_DOWN: 'GO_DOWN',
}

# sm_state devuelto por MISO → nombre legible
FPGA_STATE = {
    0: 'IDLE',     1: 'MAN_UP',   2: 'MAN_DOWN',
    3: 'TO_N1',    4: 'AT_N1',    5: 'TO_N2',
    6: 'AT_N2',    7: 'LIFTING',  8: 'HOLD',
    9: 'LOWERING',
}

# Estados estables del FPGA (servo parado)
FPGA_STABLE = {0, 4, 6, 8}   # IDLE, AT_N1, AT_N2, HOLD

# Qué publicar en /lift_done al llegar a cada estado estable
DONE_LABEL = {
    0: 'DOWN',   4: 'AT_N1',
    6: 'AT_N2',  8: 'HOLD',
}


class SpiServoNode(Node):

    def __init__(self):
        super().__init__('spi_servo_node')

        self.declare_parameter('spi_device',    '/dev/spidev0.0')
        self.declare_parameter('spi_speed',     500_000)
        self.declare_parameter('spi_mode',      0)
        self.declare_parameter('publish_rate',  10.0)
        self.declare_parameter('poll_rate',     20.0)   # Hz para leer MISO

        spi_device   = self.get_parameter('spi_device').value
        spi_speed    = self.get_parameter('spi_speed').value
        spi_mode     = self.get_parameter('spi_mode').value
        publish_rate = self.get_parameter('publish_rate').value
        poll_rate    = self.get_parameter('poll_rate').value

        self._fpga_state  = 0          # último sm_state leído del FPGA
        self._prev_fpga   = 0          # para detectar transiciones
        self._spi         = None

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
                self.get_logger().info(f'SPI {spi_device} {spi_speed//1000}kHz | MISO activo')
            except Exception as e:
                self.get_logger().error(f'SPI error: {e}')
        else:
            self.get_logger().warn('spidev no disponible — DRY-RUN')

        self.send_cmd(CMD_STOP, repeats=5)

        # ── Subscribers ────────────────────────────────────────────────────────
        self.create_subscription(Int8,   '/lift_trigger', self._cb_trigger, 10)
        self.create_subscription(String, '/lift_auto',    self._cb_auto,    10)

        # ── Publishers ─────────────────────────────────────────────────────────
        self._pub_state = self.create_publisher(String,          '/lift_state',  10)
        self._pub_pos   = self.create_publisher(String,          '/lift_pos',    10)
        self._pub_done  = self.create_publisher(String,          '/lift_done',   10)
        self._pub_servo = self.create_publisher(Int16MultiArray, '/servo_state', 10)

        # Timer de publicación de estado
        self.create_timer(1.0 / publish_rate, self._publish_state)

        # Timer de polling MISO — lee sm_state real del FPGA
        self.create_timer(1.0 / poll_rate, self._poll_fpga)

        self.get_logger().info(
            'spi_servo_node activo ✓  (SM + timers en FPGA, done via MISO)\n'
            '  ros2 topic echo /lift_done    ← para coordinar otros nodos\n'
            '  ros2 topic echo /lift_state   ← estado FPGA en tiempo real'
        )

    # ── SPI ────────────────────────────────────────────────────────────────────

    def send_cmd(self, cmd: int, repeats: int = 3) -> int:
        """
        Envía [0xAC, cmd, 0x00] N veces.
        Devuelve el sm_state leído en el último envío (byte 0 de MISO).
        """
        pkt = [HDR_CMD, cmd, 0x00]
        state = 0
        for i in range(repeats):
            if self._spi is not None:
                resp = self._spi.xfer2(pkt)
                state = resp[0] & 0x0F   # sm_state en nibble bajo
            if i < repeats - 1:
                time.sleep(0.0005)
        self.get_logger().info(
            f'SPI CMD 0x{cmd:02X} ({CMD_NAMES.get(cmd,"?")}) ×{repeats}'
            f'  FPGA={FPGA_STATE.get(state, "?")}({state})'
        )
        return state

    def _read_state(self) -> int:
        """Lee el sm_state actual del FPGA sin cambiar nada (CMD_PING)."""
        if self._spi is None:
            return self._fpga_state
        resp = self._spi.xfer2([HDR_CMD, CMD_PING, 0x00])
        return resp[0] & 0x0F

    # ── Polling MISO ───────────────────────────────────────────────────────────

    def _poll_fpga(self):
        """
        Lee sm_state del FPGA a poll_rate Hz.
        Si transiciona de moving→stable, publica /lift_done.
        """
        new_state = self._read_state()

        if new_state != self._fpga_state:
            prev = self._fpga_state
            self._fpga_state = new_state

            self.get_logger().info(
                f'FPGA: {FPGA_STATE.get(prev,"?")}({prev}) → '
                f'{FPGA_STATE.get(new_state,"?")}({new_state})'
            )

            # Transición a estado ESTABLE → publicar done
            if new_state in FPGA_STABLE:
                label = DONE_LABEL.get(new_state, 'DONE')
                msg = String()
                msg.data = label
                self._pub_done.publish(msg)
                self.get_logger().info(f'/lift_done: "{label}"')

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _cb_trigger(self, msg: Int8):
        if msg.data == 1:
            self.send_cmd(CMD_MAN_UP, 3)
        elif msg.data == -1:
            self.send_cmd(CMD_MAN_DOWN, 3)
        elif msg.data == 0:
            state = self.send_cmd(CMD_STOP, 5)
            # Al parar manual, done inmediato
            if state in FPGA_STABLE:
                label = DONE_LABEL.get(state, 'MANUAL_DONE')
            else:
                label = 'MANUAL_DONE'
            m = String(); m.data = label
            self._pub_done.publish(m)
        else:
            self.get_logger().warn(f'lift_trigger: {msg.data} inválido')

    def _cb_auto(self, msg: String):
        cmd_map = {
            'n1':   CMD_GO_N1,
            'n2':   CMD_GO_N2,
            'hold': CMD_GO_HOLD,
            'down': CMD_GO_DOWN,
        }
        required_state = {
            'n1': 0, 'n2': 0,          # necesita IDLE
            'hold': None,               # AT_N1 o AT_N2
            'down': 8,                  # necesita HOLD
        }

        key = msg.data.strip().lower()
        if key not in cmd_map:
            self.get_logger().warn(f'lift_auto: "{key}" no reconocido')
            return

        # Validación contra estado FPGA real (vía MISO)
        current = self._read_state()
        req = required_state[key]

        if key == 'hold' and current not in (4, 6):  # AT_N1 o AT_N2
            self.get_logger().warn(
                f'Auto "hold" ignorado — FPGA en {FPGA_STATE.get(current,"?")}({current})'
                f', necesita AT_N1(4) o AT_N2(6)'
            )
            return
        elif req is not None and current != req:
            self.get_logger().warn(
                f'Auto "{key}" ignorado — FPGA en {FPGA_STATE.get(current,"?")}({current})'
                f', necesita {FPGA_STATE.get(req,"?")}({req})'
            )
            return

        self.send_cmd(cmd_map[key], 3)

    # ── Publicación continua ────────────────────────────────────────────────────

    def _publish_state(self):
        state_name = FPGA_STATE.get(self._fpga_state, 'UNKNOWN')
        pos_map = {0: 'DOWN', 4: 'N1', 6: 'N2', 8: 'HOLD'}
        pos = pos_map.get(self._fpga_state, 'UNKNOWN')

        s = String(); s.data = state_name
        self._pub_state.publish(s)
        p = String(); p.data = pos
        self._pub_pos.publish(p)
        m = Int16MultiArray(); m.data = [self._fpga_state, 0]
        self._pub_servo.publish(m)

    @staticmethod
    def _parse_spidev(device: str):
        try:
            parts = device.replace('/dev/spidev', '').split('.')
            return int(parts[0]), int(parts[1])
        except Exception:
            return 0, 0

    def destroy_node(self):
        self.send_cmd(CMD_STOP, 5)
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


# ── Test sin ROS2 ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    if '--test' not in sys.argv:
        print(__doc__)
        sys.exit(0)

    if not SPIDEV_AVAILABLE:
        print('[ERROR] pip install spidev'); sys.exit(1)

    idx  = sys.argv.index('--test')
    test = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else ''

    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.max_speed_hz  = 500_000
    spi.mode          = 0
    spi.bits_per_word = 8
    spi.lsbfirst      = False
    spi.cshigh        = False

    def tx(cmd, repeats=3):
        pkt = [HDR_CMD, cmd, 0x00]
        state = 0
        for i in range(repeats):
            resp = spi.xfer2(pkt)
            state = resp[0] & 0x0F
            if i < repeats - 1:
                time.sleep(0.0005)
        name = CMD_NAMES.get(cmd, f'0x{cmd:02X}')
        fname = FPGA_STATE.get(state, '?')
        print(f'  TX [{name}] ×{repeats}  MISO→ {fname}({state})')
        return state

    def read_state():
        resp = spi.xfer2([HDR_CMD, CMD_PING, 0x00])
        s = resp[0] & 0x0F
        return s, FPGA_STATE.get(s, '?')

    def wait_until(target_states, timeout=5.0, label=''):
        """Espera (polling MISO) hasta que FPGA llegue a un estado esperado."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            s, name = read_state()
            elapsed = time.time() - t0
            print(f'  {label}  {elapsed:.1f}s  FPGA={name}({s})', end='\r')
            if s in target_states:
                print(f'  {label}  {elapsed:.1f}s  FPGA={name}({s}) ✓')
                return s
            time.sleep(0.05)
        print(f'\n  TIMEOUT después de {timeout}s')
        return None

    print(f'\n=== Test: {test} ===\n')

    try:
        if test == 'status':
            s, name = read_state()
            print(f'FPGA sm_state = {name} ({s})')

        elif test == 'stop':
            tx(CMD_STOP, 5)

        elif test == 'n1':
            print('GO_N1 → esperando AT_N1 (MISO polling)...')
            tx(CMD_GO_N1)
            wait_until({4}, timeout=3.0, label='[TO_N1]')

        elif test == 'n2':
            print('GO_N2 → esperando AT_N2 (MISO polling)...')
            tx(CMD_GO_N2)
            wait_until({6}, timeout=4.0, label='[TO_N2]')

        elif test == 'hold':
            s, name = read_state()
            print(f'Estado actual: {name}({s})')
            if s not in (4, 6):
                print('Necesitas estar en AT_N1(4) o AT_N2(6). Abort.')
            else:
                tx(CMD_GO_HOLD)
                wait_until({8}, timeout=3.0, label='[LIFTING]')

        elif test == 'down':
            s, name = read_state()
            print(f'Estado actual: {name}({s})')
            if s != 8:
                print('Necesitas estar en HOLD(8). Abort.')
            else:
                tx(CMD_GO_DOWN)
                wait_until({0}, timeout=4.0, label='[LOWERING]')

        elif test == 'cycle':
            print('Ciclo: IDLE → N1 → HOLD → DOWN  (polling MISO)\n')
            tx(CMD_STOP, 5); time.sleep(0.3)
            print('[1/4] GO_N1')
            tx(CMD_GO_N1)
            wait_until({4}, timeout=3.0, label='[TO_N1]')
            print('[2/4] GO_HOLD')
            tx(CMD_GO_HOLD)
            wait_until({8}, timeout=3.0, label='[LIFTING]')
            print('[3/4] Simulando traslado...')
            wait_until({8}, timeout=2.0, label='[HOLD]')  # espera en hold
            print('[4/4] GO_DOWN')
            tx(CMD_GO_DOWN)
            wait_until({0}, timeout=4.0, label='[LOWERING]')
            tx(CMD_STOP, 5)
            print('\nCiclo completado ✓')

        elif test == 'manual':
            import threading
            print(f'Estado inicial: {read_state()[1]}')
            print('  1=MAN_UP  -1=MAN_DOWN  0=STOP  s=status  q=salir\n')
            tx(CMD_STOP, 5)
            state = {'cmd': CMD_STOP}
            stop_ev = threading.Event()

            def sender():
                while not stop_ev.is_set():
                    spi.xfer2([HDR_CMD, state['cmd'], 0x00])
                    time.sleep(0.1)

            threading.Thread(target=sender, daemon=True).start()

            while True:
                raw = input('> ').strip()
                if raw == 'q':
                    break
                elif raw == 's':
                    s, n = read_state()
                    print(f'  FPGA: {n}({s})')
                elif raw == '1':
                    state['cmd'] = CMD_MAN_UP;   print('  MAN_UP')
                elif raw == '-1':
                    state['cmd'] = CMD_MAN_DOWN; print('  MAN_DOWN')
                elif raw == '0':
                    state['cmd'] = CMD_STOP;     print('  STOP')
                else:
                    print('  Usa: 1 / -1 / 0 / s / q')

            stop_ev.set()
            tx(CMD_STOP, 5)

        else:
            print(f'Opciones: status | stop | n1 | n2 | hold | down | cycle | manual')

    except KeyboardInterrupt:
        print('\nAbortado.')
    finally:
        for _ in range(5):
            spi.xfer2([HDR_CMD, CMD_STOP, 0x00])
        spi.close()

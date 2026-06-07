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

─── Transiciones válidas (/lift_auto) ─────────────────────────
  IDLE(0)          → n1, n2
  AT_N1(4)         → hold              ← NO puede bajar directo
  AT_N2(6)         → hold              ← NO puede bajar directo
  HOLD(8)          → down
  cualquier estado → stop (emergencia)

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

# ── Tabla de transiciones válidas ──────────────────────────────────────────────
# cmd_key → conjunto de estados FPGA desde los que se permite ejecutar
# None en el set = sin restricción de origen (no aplica aquí, pero por claridad)
#
#   n1, n2  → solo desde IDLE (0)
#   hold    → solo desde AT_N1 (4) o AT_N2 (6)  — DEBE pasar por HOLD antes de bajar
#   down    → solo desde HOLD (8)
VALID_FROM = {
    'n1':   {0},      # IDLE
    'n2':   {0},      # IDLE
    'hold': {4, 6},   # AT_N1 o AT_N2
    'down': {8},      # HOLD
}

# Cuántos ciclos consecutivos del mismo estado se requieren para
# considerarlo "firme" y evitar jitter en transiciones
STABLE_CONFIRM_COUNT = 3      # × poll_period = 150 ms @ 20 Hz
KEEPALIVE_PERIOD     = 0.05   # segundos entre keepalives durante pausa


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

        self._fpga_state      = 0     # último sm_state crudo leído del FPGA
        self._stable_count    = 0     # contador de confirmaciones consecutivas
        self._confirmed_state = None  # None hasta que el primer poll confirme el estado
        self._spi             = None

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

        self.send_cmd(CMD_STOP, repeats=5)

        # ── Inicialización bloqueante del estado confirmado ─────────────────────
        # Lee STABLE_CONFIRM_COUNT+1 veces para tener _confirmed_state válido
        # antes de aceptar cualquier comando. Evita rechazar "hold" si el
        # FPGA ya está en AT_N1/AT_N2 cuando el nodo arranca.
        if self._spi is not None:
            raw = 0
            for _ in range(STABLE_CONFIRM_COUNT + 1):
                raw = self._read_state()
                time.sleep(0.05)
            self._fpga_state      = raw
            self._confirmed_state = raw
            self._stable_count    = STABLE_CONFIRM_COUNT
            self.get_logger().info(
                f'Estado FPGA inicial confirmado: '
                f'{FPGA_STATE.get(raw, "?")}({raw})'
            )

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
            '  ros2 topic echo /lift_state   ← estado FPGA en tiempo real\n'
            '  Transiciones: IDLE→n1/n2 | AT_N1/AT_N2→hold | HOLD→down'
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
            return self._fpga_state if self._fpga_state is not None else 0
        resp = self._spi.xfer2([HDR_CMD, CMD_PING, 0x00])
        return resp[0] & 0x0F

    # ── Polling MISO ───────────────────────────────────────────────────────────

    def _poll_fpga(self):
        """
        Lee sm_state del FPGA a poll_rate Hz.
        Aplica confirmación por mayoría (STABLE_CONFIRM_COUNT lecturas
        consecutivas iguales) antes de considerar el estado como firme,
        eliminando el jitter en transiciones como TO_N1→AT_N1.
        Publica /lift_done solo al confirmar un estado estable.
        """
        raw = self._read_state()

        # Acumular confirmaciones del mismo valor crudo
        if raw == self._fpga_state:
            self._stable_count = min(self._stable_count + 1, STABLE_CONFIRM_COUNT)
        else:
            # Nuevo valor — reiniciar contador
            self._fpga_state   = raw
            self._stable_count = 1
            return   # esperar más muestras antes de actuar

        # Solo actuar cuando el estado lleva N lecturas consecutivas iguales
        if self._stable_count < STABLE_CONFIRM_COUNT:
            return

        # Estado firme — detectar cambio respecto al último confirmado
        if raw == self._confirmed_state:
            return   # sin cambio real

        prev = self._confirmed_state
        self._confirmed_state = raw

        self.get_logger().info(
            f'FPGA (confirmado): {FPGA_STATE.get(prev,"?")}({prev}) → '
            f'{FPGA_STATE.get(raw,"?")}({raw})'
        )

        # Transición a estado ESTABLE → publicar done
        if raw in FPGA_STABLE:
            label = DONE_LABEL.get(raw, 'DONE')
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

        key = msg.data.strip().lower()
        if key not in cmd_map:
            self.get_logger().warn(f'lift_auto: "{key}" no reconocido')
            return

        # Guard: estado aún no confirmado (ventana de arranque del nodo)
        if self._confirmed_state is None:
            self.get_logger().warn(
                f'Auto "{key}" ignorado — FPGA aún no confirmado '
                f'(esperando {STABLE_CONFIRM_COUNT} polls)'
            )
            return

        current = self._confirmed_state
        allowed = VALID_FROM[key]

        if current not in allowed:
            # Mensaje de error descriptivo según el caso
            if key == 'hold':
                reason = 'necesita AT_N1(4) o AT_N2(6) — debe pasar por HOLD antes de bajar'
            elif key == 'down':
                reason = 'necesita HOLD(8) — no se puede bajar directo desde AT_N1/AT_N2'
            elif key in ('n1', 'n2'):
                reason = 'necesita IDLE(0)'
            else:
                reason = f'estados válidos: {allowed}'

            self.get_logger().warn(
                f'Auto "{key}" ignorado — FPGA en '
                f'{FPGA_STATE.get(current,"?")}({current}), {reason}'
            )
            return

        self.send_cmd(cmd_map[key], 3)

    # ── Publicación continua ────────────────────────────────────────────────────

    def _publish_state(self):
        # Usar 0 como fallback mientras _confirmed_state es None (arranque)
        state = self._confirmed_state if self._confirmed_state is not None else 0
        state_name = FPGA_STATE.get(state, 'UNKNOWN')
        pos_map = {0: 'DOWN', 4: 'N1', 6: 'N2', 8: 'HOLD'}
        pos = pos_map.get(state, 'UNKNOWN')

        s = String(); s.data = state_name
        self._pub_state.publish(s)
        p = String(); p.data = pos
        self._pub_pos.publish(p)
        m = Int16MultiArray(); m.data = [state, 0]
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

    def tx(cmd, repeats=3, post_delay=0.08):
        """
        Envía el comando N veces y espera post_delay segundos.
        El delay es crítico: el FPGA necesita tiempo para procesar
        el comando y transicionar su SM antes del primer poll.
        """
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
        if post_delay > 0:
            time.sleep(post_delay)   # dar tiempo al FPGA para actualizar su SM
        return state

    def read_state_raw():
        """Lee sm_state crudo una sola vez."""
        resp = spi.xfer2([HDR_CMD, CMD_PING, 0x00])
        s = resp[0] & 0x0F
        return s, FPGA_STATE.get(s, '?')

    def read_state_confirmed(confirm_count=3, period=0.02):
        """
        Lee hasta obtener `confirm_count` lecturas consecutivas iguales.
        Evita falsos positivos por jitter en transiciones del FPGA.
        """
        last = None
        streak = 0
        while True:
            s, name = read_state_raw()
            if s == last:
                streak += 1
                if streak >= confirm_count:
                    return s, name
            else:
                last   = s
                streak = 1
            time.sleep(period)

    def wait_until(target_states, timeout=5.0, label='',
                   confirm_count=STABLE_CONFIRM_COUNT,
                   reinforce_cmd=None, reinforce_every=0.5):
        """
        Espera (polling MISO con confirmación anti-jitter) hasta que
        el FPGA llegue de forma estable a uno de los estados esperados.
        Si reinforce_cmd se especifica, lo reenvía cada reinforce_every
        segundos para asegurar que el FPGA no pierda el comando.
        """
        t0 = time.time()
        last   = None
        streak = 0
        last_reinforce = t0

        while time.time() - t0 < timeout:
            # Reforzar comando si se especificó
            now = time.time()
            if reinforce_cmd is not None and (now - last_reinforce) >= reinforce_every:
                spi.xfer2([HDR_CMD, reinforce_cmd, 0x00])
                last_reinforce = now

            resp   = spi.xfer2([HDR_CMD, CMD_PING, 0x00])
            s      = resp[0] & 0x0F
            name   = FPGA_STATE.get(s, '?')
            elapsed = time.time() - t0

            if s == last:
                streak += 1
            else:
                last   = s
                streak = 1

            confirmed = streak >= confirm_count
            marker = ' ✓' if (s in target_states and confirmed) else ''
            print(f'  {label}  {elapsed:.1f}s  FPGA={name}({s})'
                  f'  streak={streak}{marker}', end='\r')

            if s in target_states and confirmed:
                print()   # nueva línea
                return s

            time.sleep(0.02)

        print(f'\n  TIMEOUT después de {timeout}s')
        return None

    def keepalive_pause(cmd, duration, label='pausa'):
        """
        Mantiene el estado del FPGA enviando el mismo comando cada
        KEEPALIVE_PERIOD segundos durante `duration` segundos.
        Evita que un timer interno del FPGA caiga a IDLE durante la pausa.
        """
        print(f'  {label} {duration:.1f}s con keepalive CMD 0x{cmd:02X}...')
        t0 = time.time()
        while time.time() - t0 < duration:
            spi.xfer2([HDR_CMD, cmd, 0x00])
            time.sleep(KEEPALIVE_PERIOD)
        # Verificar que seguimos en el estado correcto tras la pausa
        s, name = read_state_confirmed()
        print(f'  Tras pausa: FPGA={name}({s})')
        return s

    print(f'\n=== Test: {test} ===\n')

    try:
        if test == 'status':
            s, name = read_state_confirmed()
            print(f'FPGA sm_state (confirmado) = {name} ({s})')

        elif test == 'stop':
            tx(CMD_STOP, 5)

        elif test == 'n1':
            s, name = read_state_confirmed()
            print(f'Estado actual (confirmado): {name}({s})')
            if s != 0:
                print('Necesitas estar en IDLE(0) para ir a N1. Abort.')
            else:
                print('GO_N1 → esperando AT_N1 (MISO polling con confirmación)...')
                tx(CMD_GO_N1)
                wait_until({4}, timeout=3.0, label='[TO_N1]')

        elif test == 'n2':
            s, name = read_state_confirmed()
            print(f'Estado actual (confirmado): {name}({s})')
            if s != 0:
                print('Necesitas estar en IDLE(0) para ir a N2. Abort.')
            else:
                print('GO_N2 → esperando AT_N2 (MISO polling con confirmación)...')
                tx(CMD_GO_N2)
                wait_until({6}, timeout=4.0, label='[TO_N2]')

        elif test == 'hold':
            s, name = read_state_confirmed()
            print(f'Estado actual (confirmado): {name}({s})')
            if s not in (4, 6):
                print('Necesitas estar en AT_N1(4) o AT_N2(6). '
                      'No se puede bajar directo — debe pasar por HOLD. Abort.')
            else:
                tx(CMD_GO_HOLD)
                wait_until({8}, timeout=3.0, label='[LIFTING]')

        elif test == 'down':
            s, name = read_state_confirmed()
            print(f'Estado actual (confirmado): {name}({s})')
            if s != 8:
                print('Necesitas estar en HOLD(8). '
                      'Desde AT_N1/AT_N2 primero debes ir a hold. Abort.')
            else:
                tx(CMD_GO_DOWN)
                wait_until({0}, timeout=4.0, label='[LOWERING]')

        elif test == 'cycle':
            pause = float(sys.argv[idx + 2]) if len(sys.argv) > idx + 2 else 1.0
            print(f'Ciclo: N1 → HOLD → DOWN → N2 → HOLD → DOWN'
                  f'  (pausa={pause:.1f}s, keepalive activo)\n')

            abort = False   # bandera para salida limpia sin cerrar spi en medio

            # Asegurar punto de partida limpio
            tx(CMD_STOP, 5, post_delay=0.3)
            s, name = read_state_confirmed()
            print(f'Estado inicial confirmado: {name}({s})\n')
            if s != 0:
                print('Se esperaba IDLE(0) para iniciar ciclo. Abort.')
                abort = True

            # ── Sub-ciclo N1 ──────────────────────────────────────────────────
            if not abort:
                print('[1/6] GO_N1 → esperando AT_N1...')
                tx(CMD_GO_N1)   # post_delay=0.08 por defecto
                reached = wait_until({4}, timeout=5.0, label='[TO_N1]',
                                     reinforce_cmd=CMD_GO_N1)
                if reached is None:
                    print('No se alcanzó AT_N1. Abort.')
                    abort = True

            if not abort:
                # Pausa en N1 con keepalive para evitar caída a IDLE
                keepalive_pause(CMD_GO_N1, pause, label='[AT_N1]')

            if not abort:
                print('[2/6] HOLD desde N1 → esperando HOLD...')
                tx(CMD_GO_HOLD)   # post_delay=0.08 da tiempo al FPGA
                reached = wait_until({8}, timeout=8.0, label='[LIFTING N1]',
                                     reinforce_cmd=CMD_GO_HOLD)
                if reached is None:
                    print('No se alcanzó HOLD. Abort.')
                    abort = True

            if not abort:
                print('[3/6] DOWN → esperando IDLE...')
                tx(CMD_GO_DOWN)
                reached = wait_until({0}, timeout=10.0, label='[LOWERING]',
                                     reinforce_cmd=CMD_GO_DOWN)
                if reached is None:
                    print('No se alcanzó IDLE. Abort.')
                    abort = True
                else:
                    time.sleep(0.3)

            # ── Sub-ciclo N2 ──────────────────────────────────────────────────
            if not abort:
                print('[4/6] GO_N2 → esperando AT_N2...')
                tx(CMD_GO_N2)
                reached = wait_until({6}, timeout=6.0, label='[TO_N2]',
                                     reinforce_cmd=CMD_GO_N2)
                if reached is None:
                    print('No se alcanzó AT_N2. Abort.')
                    abort = True

            if not abort:
                keepalive_pause(CMD_GO_N2, pause, label='[AT_N2]')

            if not abort:
                print('[5/6] HOLD desde N2 → esperando HOLD...')
                tx(CMD_GO_HOLD)
                reached = wait_until({8}, timeout=8.0, label='[LIFTING N2]',
                                     reinforce_cmd=CMD_GO_HOLD)
                if reached is None:
                    print('No se alcanzó HOLD. Abort.')
                    abort = True

            if not abort:
                keepalive_pause(CMD_GO_HOLD, pause, label='[HOLD]')

            if not abort:
                print('[6/6] DOWN → esperando IDLE...')
                tx(CMD_GO_DOWN)
                reached = wait_until({0}, timeout=10.0, label='[LOWERING]',
                                     reinforce_cmd=CMD_GO_DOWN)
                if reached is None:
                    print('No se alcanzó IDLE final. Abort.')
                    abort = True

            tx(CMD_STOP, 5, post_delay=0)
            if abort:
                print('\nCiclo abortado ✗')
            else:
                print('\nCiclo completado ✓')

        elif test == 'manual':
            import threading
            s, name = read_state_confirmed()
            print(f'Estado inicial (confirmado): {name}({s})')
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
                raw_input = input('> ').strip()
                if raw_input == 'q':
                    break
                elif raw_input == 's':
                    s, n = read_state_confirmed()
                    print(f'  FPGA (confirmado): {n}({s})')
                elif raw_input == '1':
                    state['cmd'] = CMD_MAN_UP;   print('  MAN_UP')
                elif raw_input == '-1':
                    state['cmd'] = CMD_MAN_DOWN; print('  MAN_DOWN')
                elif raw_input == '0':
                    state['cmd'] = CMD_STOP;     print('  STOP')
                else:
                    print('  Usa: 1 / -1 / 0 / s / q')

            stop_ev.set()
            tx(CMD_STOP, 5)

        else:
            print('Opciones: status | stop | n1 | n2 | hold | down | cycle | manual')

    except KeyboardInterrupt:
        print('\nAbortado.')
    finally:
        for _ in range(5):
            spi.xfer2([HDR_CMD, CMD_STOP, 0x00])
        spi.close()

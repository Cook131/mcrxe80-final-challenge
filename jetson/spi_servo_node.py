#!/usr/bin/env python3
"""
spi_servo_node.py  v2
Jetson Nano → Tang Nano 20K (SPI slave) → 1× Servo 360° (lift)

La SM y sus timers viven en el FPGA.
El FPGA devuelve su sm_state real en MISO en cada transacción SPI.
El 'done' se detecta cuando sm_state cambia a un estado estable.

─── Fix P1: _read_state() con retry anti-glitch ──────────────
  La versión anterior aceptaba cualquier valor crudo de MISO
  inmediatamente. Si la línea tenía ruido durante la transición
  CS↓, el nibble bajo podía ser basura. Ahora _read_state()
  hace 2 lecturas consecutivas y solo acepta el valor si ambas
  coinciden; si no, reintenta hasta max_tries.
  Impacto: elimina falsos "AT_N1" o "HOLD" reportados por
  glitches en la línea MISO durante pruebas.

─── Fix P2: _poll_fpga suspende durante ventana de expiración ─
  La versión anterior enviaba CMD_PING a 20 Hz durante toda la
  operación. Esto creaba una race condition en el FPGA (Bug F1
  del Verilog): si el CMD_PING llegaba exactamente cuando
  dur_cnt==1, spi_new tenía prioridad sobre timer_done y la SM
  quedaba atascada en TO_N1/LIFTING/etc. para siempre.
  Con el Verilog v2 (timer_fired flag) esa race está resuelta
  en hardware. Este fix P2 es una capa defensiva adicional:
  cuando el nodo sabe que el FPGA está en una transición con
  timer conocido, reduce el poll rate al 10% del normal durante
  los últimos POLL_SILENCE_MS ms antes de la expiración esperada.
  Esto hace la probabilidad de coincidencia prácticamente cero
  incluso con Verilog v1 si fuera necesario un rollback.

─── Fix P3: send_cmd documenta que MISO = estado PREVIO ───────
  El FPGA carga miso_sr en cs_fall (antes de decodificar el cmd).
  send_cmd() ya no usa el valor de retorno para inferir el
  estado post-comando; solo lo registra en el log con la
  etiqueta "(pre-cmd)".

─── sm_state devuelto en MISO ─────────────────────────────────
  0=IDLE   1=MAN_UP   2=MAN_DOWN
  3=TO_N1  4=AT_N1    5=TO_N2   6=AT_N2
  7=LIFTING  8=HOLD   9=LOWERING

─── /lift_done publica cuando FPGA transiciona a estado estable ─
  "AT_N1" | "AT_N2" | "HOLD" | "DOWN" | "MANUAL_DONE"

─── Protocolo SPI ─────────────────────────────────────────────
  TX [0xAC][cmd][0x00]  RX [sm_state_prev][0x00][0x00]
  CMD_PING 0xFF = solo leer estado, sin cambiar nada
  MISO byte 0 = estado ANTES de procesar el comando actual

─── Transiciones válidas (/lift_auto) ─────────────────────────
  IDLE(0)          → n1, n2
  AT_N1(4)         → hold
  AT_N2(6)         → hold
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

CMD_PING     = 0xFF
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

FPGA_STATE = {
    0: 'IDLE',     1: 'MAN_UP',   2: 'MAN_DOWN',
    3: 'TO_N1',    4: 'AT_N1',    5: 'TO_N2',
    6: 'AT_N2',    7: 'LIFTING',  8: 'HOLD',
    9: 'LOWERING',
}

FPGA_STABLE = {0, 4, 6, 8}

DONE_LABEL = {
    0: 'DOWN',   4: 'AT_N1',
    6: 'AT_N2',  8: 'HOLD',
}

VALID_FROM = {
    'n1':   {0},
    'n2':   {0},
    'hold': {4, 6},
    'down': {8},
}

# Duración conocida de cada transición en segundos @ 27 MHz.
# Usado por Fix P2 para suspender el poll antes de la expiración.
_FPGA_CLK   = 27_000_000
_TRANSITION_DUR_S = {
    3: 12_400_000 / _FPGA_CLK,   # TO_N1
    5: 37_700_000 / _FPGA_CLK,   # TO_N2
    7: 20_000_000 / _FPGA_CLK,   # LIFTING (worst case N1; N2 es menor)
    9: 55_700_000 / _FPGA_CLK,   # LOWERING
    1: 56_700_000 / _FPGA_CLK,   # MAN_UP
    2: 56_700_000 / _FPGA_CLK,   # MAN_DOWN
}

STABLE_CONFIRM_COUNT = 3
KEEPALIVE_PERIOD     = 0.05

# Fix P2: en los últimos N ms de una transición conocida, bajar
# la frecuencia de poll para reducir la probabilidad de
# coincidir con la expiración del timer del FPGA.
# Con Verilog v2 (timer_fired flag) esto ya no es necesario,
# pero se conserva como defensa en profundidad.
POLL_SILENCE_MS      = 120   # ms antes del fin del timer
POLL_SILENCE_PERIOD  = 0.5   # segundos entre polls en modo silencioso


class SpiServoNode(Node):

    def __init__(self):
        super().__init__('spi_servo_node')

        self.declare_parameter('spi_device',    '/dev/spidev0.0')
        self.declare_parameter('spi_speed',     500_000)
        self.declare_parameter('spi_mode',      0)
        self.declare_parameter('publish_rate',  10.0)
        self.declare_parameter('poll_rate',     20.0)

        spi_device   = self.get_parameter('spi_device').value
        spi_speed    = self.get_parameter('spi_speed').value
        spi_mode     = self.get_parameter('spi_mode').value
        publish_rate = self.get_parameter('publish_rate').value
        poll_rate    = self.get_parameter('poll_rate').value

        self._fpga_state      = 0
        self._stable_count    = 0
        self._confirmed_state = None
        self._spi             = None

        # Fix P2: timestamp de inicio de la última transición con timer
        self._transition_start = None
        self._transition_dur   = None

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
                    f'SPI {spi_device} {spi_speed//1000}kHz | MISO activo')
            except Exception as e:
                self.get_logger().error(f'SPI error: {e}')

        self.send_cmd(CMD_STOP, repeats=5)

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

        self.create_subscription(Int8,   '/lift_trigger', self._cb_trigger, 10)
        self.create_subscription(String, '/lift_auto',    self._cb_auto,    10)

        self._pub_state = self.create_publisher(String,          '/lift_state',  10)
        self._pub_pos   = self.create_publisher(String,          '/lift_pos',    10)
        self._pub_done  = self.create_publisher(String,          '/lift_done',   10)
        self._pub_servo = self.create_publisher(Int16MultiArray, '/servo_state', 10)

        self.create_timer(1.0 / publish_rate, self._publish_state)
        self.create_timer(1.0 / poll_rate,    self._poll_fpga)

        self.get_logger().info(
            'spi_servo_node v2 activo ✓\n'
            '  fixes: anti-glitch MISO, poll silence, timer_fired (FPGA v2)\n'
            '  ros2 topic echo /lift_done    ← coordinar otros nodos\n'
            '  ros2 topic echo /lift_state   ← estado FPGA en tiempo real\n'
            '  Transiciones: IDLE→n1/n2 | AT_N1/AT_N2→hold | HOLD→down'
        )

    # ── SPI ────────────────────────────────────────────────────────────────────

    def send_cmd(self, cmd: int, repeats: int = 3) -> int:
        """
        Envía [0xAC, cmd, 0x00] N veces.
        Fix P3: el valor devuelto es sm_state ANTES del comando (pre-cmd),
        no el estado posterior. No usar para inferir el resultado del comando.
        """
        pkt = [HDR_CMD, cmd, 0x00]
        pre_state = 0
        for i in range(repeats):
            if self._spi is not None:
                resp = self._spi.xfer2(pkt)
                pre_state = resp[0] & 0x0F
            if i < repeats - 1:
                time.sleep(0.0005)
        self.get_logger().info(
            f'SPI CMD 0x{cmd:02X} ({CMD_NAMES.get(cmd,"?")}) ×{repeats}'
            f'  MISO(pre-cmd)={FPGA_STATE.get(pre_state, "?")}({pre_state})'
        )
        return pre_state

    def _read_state(self, max_tries: int = 3) -> int:
        """
        Fix P1: lee sm_state con validación anti-glitch.
        Hace hasta max_tries lecturas; acepta el valor solo si dos
        lecturas consecutivas coinciden. Esto filtra glitches en
        MISO que podían reportar estados inválidos durante transiciones.
        """
        if self._spi is None:
            return self._fpga_state if self._fpga_state is not None else 0

        prev = None
        for _ in range(max_tries):
            resp = self._spi.xfer2([HDR_CMD, CMD_PING, 0x00])
            val  = resp[0] & 0x0F
            if val == prev:
                return val
            prev = val
            time.sleep(0.0002)

        # Si no hubo coincidencia en max_tries, devolver la última lectura
        # (mejor que un valor stale de _fpga_state)
        return prev if prev is not None else 0

    # ── Fix P2: gestión de ventana de silencio ─────────────────────────────────

    def _record_transition_start(self, to_state: int):
        """Registra el inicio de una transición con timer conocido."""
        dur = _TRANSITION_DUR_S.get(to_state)
        if dur is not None:
            self._transition_start = time.monotonic()
            self._transition_dur   = dur

    def _in_poll_silence(self) -> bool:
        """
        Fix P2: devuelve True si estamos en los últimos POLL_SILENCE_MS
        antes de la expiración esperada del timer del FPGA.
        En esa ventana el caller debe reducir la frecuencia de poll.
        """
        if self._transition_start is None or self._transition_dur is None:
            return False
        elapsed  = time.monotonic() - self._transition_start
        time_left = self._transition_dur - elapsed
        return 0 < time_left < (POLL_SILENCE_MS / 1000.0)

    def _clear_transition(self):
        self._transition_start = None
        self._transition_dur   = None

    # ── Polling MISO ───────────────────────────────────────────────────────────

    def _poll_fpga(self):
        """
        Lee sm_state del FPGA a poll_rate Hz con confirmación por mayoría.
        Fix P2: si estamos en la ventana de silencio pre-expiración,
        saltamos este poll para no interferir con el timer del FPGA.
        """
        # Fix P2: saltar poll en ventana crítica
        if self._in_poll_silence():
            self.get_logger().debug('[poll] silencio pre-expiración')
            return

        raw = self._read_state()   # Fix P1: anti-glitch integrado aquí

        if raw == self._fpga_state:
            self._stable_count = min(self._stable_count + 1, STABLE_CONFIRM_COUNT)
        else:
            self._fpga_state   = raw
            self._stable_count = 1
            return

        if self._stable_count < STABLE_CONFIRM_COUNT:
            return

        if raw == self._confirmed_state:
            return

        prev = self._confirmed_state
        self._confirmed_state = raw

        self.get_logger().info(
            f'FPGA (confirmado): {FPGA_STATE.get(prev,"?")}({prev}) → '
            f'{FPGA_STATE.get(raw,"?")}({raw})'
        )

        # Si llegamos a un estado estable, limpiar tracking de transición
        if raw in FPGA_STABLE:
            self._clear_transition()
            label = DONE_LABEL.get(raw, 'DONE')
            msg   = String()
            msg.data = label
            self._pub_done.publish(msg)
            self.get_logger().info(f'/lift_done: "{label}"')

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _cb_trigger(self, msg: Int8):
        if msg.data == 1:
            self.send_cmd(CMD_MAN_UP, 3)
            self._record_transition_start(1)   # MAN_UP
        elif msg.data == -1:
            self.send_cmd(CMD_MAN_DOWN, 3)
            self._record_transition_start(2)   # MAN_DOWN
        elif msg.data == 0:
            self._clear_transition()
            self.send_cmd(CMD_STOP, 5)
            # Fix P3: no usar retorno de send_cmd para inferir estado
            # El polling confirmará el estado real
            m = String(); m.data = 'MANUAL_DONE'
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

        if self._confirmed_state is None:
            self.get_logger().warn(
                f'Auto "{key}" ignorado — FPGA aún no confirmado '
                f'(esperando {STABLE_CONFIRM_COUNT} polls)'
            )
            return

        current = self._confirmed_state
        allowed = VALID_FROM[key]

        if current not in allowed:
            if key == 'hold':
                reason = 'necesita AT_N1(4) o AT_N2(6)'
            elif key == 'down':
                reason = 'necesita HOLD(8)'
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

        # Fix P2: registrar inicio de transición para gestión de silencio
        # El estado destino es el estado transitorio (no estable)
        transitioning_to = {
            'n1':   3,   # TO_N1
            'n2':   5,   # TO_N2
            'hold': 7,   # LIFTING
            'down': 9,   # LOWERING
        }
        self._record_transition_start(transitioning_to[key])

    # ── Publicación continua ────────────────────────────────────────────────────

    def _publish_state(self):
        state      = self._confirmed_state if self._confirmed_state is not None else 0
        state_name = FPGA_STATE.get(state, 'UNKNOWN')
        pos_map    = {0: 'DOWN', 4: 'N1', 6: 'N2', 8: 'HOLD'}
        pos        = pos_map.get(state, 'UNKNOWN')

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
        pkt = [HDR_CMD, cmd, 0x00]
        pre = 0
        for i in range(repeats):
            resp = spi.xfer2(pkt)
            pre  = resp[0] & 0x0F
            if i < repeats - 1:
                time.sleep(0.0005)
        name  = CMD_NAMES.get(cmd, f'0x{cmd:02X}')
        fname = FPGA_STATE.get(pre, '?')
        # Fix P3: etiquetar como pre-cmd para no confundir
        print(f'  TX [{name}] ×{repeats}  MISO(pre-cmd)→ {fname}({pre})')
        if post_delay > 0:
            time.sleep(post_delay)
        return pre

    def read_state_raw():
        """Fix P1: 2 lecturas consecutivas deben coincidir."""
        prev = None
        for _ in range(4):
            resp = spi.xfer2([HDR_CMD, CMD_PING, 0x00])
            s    = resp[0] & 0x0F
            if s == prev:
                return s, FPGA_STATE.get(s, '?')
            prev = s
            time.sleep(0.0002)
        return prev, FPGA_STATE.get(prev, '?')

    def read_state_confirmed(confirm_count=3, period=0.02):
        last   = None
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
                   reinforce_cmd=None, reinforce_every=0.5,
                   silence_before_end_s=POLL_SILENCE_MS/1000.0,
                   known_duration_s=None):
        """
        Espera hasta que FPGA llegue a un estado en target_states.
        Fix P2: si known_duration_s se especifica, los últimos
        silence_before_end_s se sondean más lentamente para no
        interferir con la expiración del timer del FPGA.
        """
        t0 = time.time()
        last   = None
        streak = 0
        last_reinforce = t0

        while time.time() - t0 < timeout:
            now     = time.time()
            elapsed = now - t0

            # Fix P2: reducir poll en ventana pre-expiración
            if known_duration_s is not None:
                time_left = known_duration_s - elapsed
                if 0 < time_left < silence_before_end_s:
                    time.sleep(POLL_SILENCE_PERIOD)
                    continue

            if reinforce_cmd is not None and (now - last_reinforce) >= reinforce_every:
                spi.xfer2([HDR_CMD, reinforce_cmd, 0x00])
                last_reinforce = now

            s, name = read_state_raw()   # Fix P1 incluido en read_state_raw

            if s == last:
                streak += 1
            else:
                last   = s
                streak = 1

            confirmed = streak >= confirm_count
            marker    = ' ✓' if (s in target_states and confirmed) else ''
            print(f'  {label}  {elapsed:.1f}s  FPGA={name}({s})'
                  f'  streak={streak}{marker}', end='\r')

            if s in target_states and confirmed:
                print()
                return s

            time.sleep(0.02)

        print(f'\n  TIMEOUT después de {timeout}s')
        return None

    def keepalive_pause(cmd, duration, label='pausa'):
        print(f'  {label} {duration:.1f}s con keepalive CMD 0x{cmd:02X}...')
        t0 = time.time()
        while time.time() - t0 < duration:
            spi.xfer2([HDR_CMD, cmd, 0x00])
            time.sleep(KEEPALIVE_PERIOD)
        s, name = read_state_confirmed()
        print(f'  Tras pausa: FPGA={name}({s})')
        return s

    print(f'\n=== Test: {test} ===\n')
    # Duraciones conocidas para Fix P2 en modo test
    _DUR = _TRANSITION_DUR_S

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
                print('GO_N1 → esperando AT_N1...')
                tx(CMD_GO_N1)
                wait_until({4}, timeout=3.0, label='[TO_N1]',
                           known_duration_s=_DUR[3])

        elif test == 'n2':
            s, name = read_state_confirmed()
            print(f'Estado actual (confirmado): {name}({s})')
            if s != 0:
                print('Necesitas estar en IDLE(0) para ir a N2. Abort.')
            else:
                print('GO_N2 → esperando AT_N2...')
                tx(CMD_GO_N2)
                wait_until({6}, timeout=4.0, label='[TO_N2]',
                           known_duration_s=_DUR[5])

        elif test == 'hold':
            s, name = read_state_confirmed()
            print(f'Estado actual (confirmado): {name}({s})')
            if s not in (4, 6):
                print('Necesitas AT_N1(4) o AT_N2(6). Abort.')
            else:
                dur = _DUR[7]
                tx(CMD_GO_HOLD)
                wait_until({8}, timeout=3.0, label='[LIFTING]',
                           known_duration_s=dur)

        elif test == 'down':
            s, name = read_state_confirmed()
            print(f'Estado actual (confirmado): {name}({s})')
            if s != 8:
                print('Necesitas HOLD(8). Abort.')
            else:
                tx(CMD_GO_DOWN)
                wait_until({0}, timeout=4.0, label='[LOWERING]',
                           known_duration_s=_DUR[9])

        elif test == 'cycle':
            pause = float(sys.argv[idx + 2]) if len(sys.argv) > idx + 2 else 1.0
            print(f'Ciclo: N1 → HOLD → DOWN → N2 → HOLD → DOWN'
                  f'  (pausa={pause:.1f}s)\n')

            abort = False

            tx(CMD_STOP, 5, post_delay=0.3)
            s, name = read_state_confirmed()
            print(f'Estado inicial confirmado: {name}({s})\n')
            if s != 0:
                print('Se esperaba IDLE(0). Abort.')
                abort = True

            if not abort:
                print('[1/6] GO_N1 → esperando AT_N1...')
                tx(CMD_GO_N1)
                reached = wait_until({4}, timeout=5.0, label='[TO_N1]',
                                     reinforce_cmd=CMD_GO_N1,
                                     known_duration_s=_DUR[3])
                if reached is None:
                    abort = True

            if not abort:
                keepalive_pause(CMD_GO_N1, pause, '[AT_N1]')

            if not abort:
                print('[2/6] HOLD desde N1 → esperando HOLD...')
                tx(CMD_GO_HOLD)
                reached = wait_until({8}, timeout=8.0, label='[LIFTING N1]',
                                     reinforce_cmd=CMD_GO_HOLD,
                                     known_duration_s=_DUR[7])
                if reached is None:
                    abort = True

            if not abort:
                print('[3/6] DOWN → esperando IDLE...')
                tx(CMD_GO_DOWN)
                reached = wait_until({0}, timeout=10.0, label='[LOWERING]',
                                     reinforce_cmd=CMD_GO_DOWN,
                                     known_duration_s=_DUR[9])
                if reached is None:
                    abort = True
                else:
                    time.sleep(0.3)

            if not abort:
                print('[4/6] GO_N2 → esperando AT_N2...')
                tx(CMD_GO_N2)
                reached = wait_until({6}, timeout=6.0, label='[TO_N2]',
                                     reinforce_cmd=CMD_GO_N2,
                                     known_duration_s=_DUR[5])
                if reached is None:
                    abort = True

            if not abort:
                keepalive_pause(CMD_GO_N2, pause, '[AT_N2]')

            if not abort:
                print('[5/6] HOLD desde N2 → esperando HOLD...')
                tx(CMD_GO_HOLD)
                reached = wait_until({8}, timeout=8.0, label='[LIFTING N2]',
                                     reinforce_cmd=CMD_GO_HOLD,
                                     known_duration_s=_DUR[7])
                if reached is None:
                    abort = True

            if not abort:
                keepalive_pause(CMD_GO_HOLD, pause, '[HOLD]')

            if not abort:
                print('[6/6] DOWN → esperando IDLE...')
                tx(CMD_GO_DOWN)
                reached = wait_until({0}, timeout=10.0, label='[LOWERING]',
                                     reinforce_cmd=CMD_GO_DOWN,
                                     known_duration_s=_DUR[9])
                if reached is None:
                    abort = True

            tx(CMD_STOP, 5, post_delay=0)
            print('\nCiclo completado ✓' if not abort else '\nCiclo abortado ✗')

        elif test == 'manual':
            import threading
            s, name = read_state_confirmed()
            print(f'Estado inicial (confirmado): {name}({s})')
            print('  1=MAN_UP  -1=MAN_DOWN  0=STOP  s=status  q=salir\n')
            tx(CMD_STOP, 5)
            state   = {'cmd': CMD_STOP}
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
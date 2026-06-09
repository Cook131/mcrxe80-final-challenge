#!/usr/bin/env python3
"""
================================================================================
Proyecto:    voice_hmm_ros
Módulo:      voice_action_node.py

Descripción:
    Nodo ROS 2 integrado: graba audio del micrófono, reconoce el comando con
    HMM + VQ y ejecuta directamente la acción del robot.

    Cambios aplicados:
        1. QoS Profile (RELIABLE) añadido para compatibilidad con Micro-ROS.
        2. Publicación continua a 20Hz en /cmd_vel (Watchdog fix).
        3. Selección de dispositivo de audio (--device) para sounddevice.
        4. threading.Lock() para proteger _active_twist y _active_until
           contra race conditions entre el hilo de voz y el timer de ROS.
        5. Secuencia de hold automático integrada para "toma" (n1) y "arriba" (n2).
        6. Ejecución continua (auto-loop) por defecto con paths preconfigurados.

================================================================================
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np
import sounddevice as sd

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Int8

from .hmm_from_scratch import WordHMMRecognizer
from .run_hmm import (
    Config,
    vad_trim,
    pre_emphasis,
    frame_signal,
    apply_hamming,
    extract_mfcc_from_frames,
    quantize_sequence,
)


# ============================================================
# Vocabulario entrenado  (frozensets para O(1) lookup)
# ============================================================

CMD_ADELANTE  = frozenset({"avanza"})
CMD_ATRAS     = frozenset({"atras", "atrás"})
CMD_IZQUIERDA = frozenset({"izquierda"})
CMD_DERECHA   = frozenset({"derecha"})
CMD_GIRA      = frozenset({"gira"})
CMD_STOP      = frozenset({"detente"})

CMD_LIFT_N2   = frozenset({"arriba"})
CMD_LIFT_DOWN = frozenset({"abajo", "suelta"})
CMD_LIFT_TAKE = frozenset({"toma"})

CMD_UNKNOWN   = "<unk>"


# ============================================================
# Pipeline de inferencia HMM + VQ desde micrófono
# ============================================================

def extract_mfcc_from_signal(signal: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Replica el pipeline de inferencia de run_hmm.py,
    pero recibe directamente la señal grabada del micrófono.
    """
    sig = np.asarray(signal, dtype=np.float64).flatten()

    # 1. VAD primero
    sig = vad_trim(sig, cfg)

    # 2. Normalización por utterance
    if sig.size > 0:
        mx = float(np.max(np.abs(sig)))
        if mx > 0.0:
            sig = sig / mx

    # 3. Pre-énfasis
    sig = pre_emphasis(sig, cfg.pre_emph)

    # 4. Framing + ventana
    frames = frame_signal(sig, cfg.frame_len, cfg.hop_len)
    frames = apply_hamming(frames)

    # 5. MFCC
    mfcc = extract_mfcc_from_frames(frames, cfg)

    return mfcc


def predict_from_signal(
    signal: np.ndarray,
    recognizer: WordHMMRecognizer,
    codebook: np.ndarray,
    cfg: Config,
    threshold: float = 0.0,
) -> Tuple[str, bool, Dict[str, float]]:
    """
    Señal de audio → MFCC → símbolos VQ → HMM → palabra.

    threshold usa una confianza simple:
        gap = mejor_log_likelihood - segundo_mejor_log_likelihood

    Si threshold <= 0.0, no se filtra por confianza.
    """
    mfcc = extract_mfcc_from_signal(signal, cfg)

    if mfcc.shape[0] == 0:
        return CMD_UNKNOWN, False, {}

    obs = quantize_sequence(mfcc, codebook)

    if len(obs) == 0:
        return CMD_UNKNOWN, False, {}

    word, scores = recognizer.predict(obs)

    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    if len(sorted_scores) >= 2:
        best_score = sorted_scores[0][1]
        second_score = sorted_scores[1][1]
        gap = best_score - second_score
    else:
        gap = float("inf")

    valid = True

    if threshold > 0.0 and gap < threshold:
        word = CMD_UNKNOWN
        valid = False

    return word, valid, scores


class VoiceActionNode(Node):
    """
    Nodo integrado de voz + acción.
    """

    def __init__(
        self,
        model_dir: str,
        codebook_path: str,
        duration: float,
        samplerate: int,
        threshold: float,
        auto_loop: bool,
        device: Optional[int] = None,
    ) -> None:
        super().__init__("voice_action_node")

        self.device = device

        # ============================================================
        # Parámetros configurables de movimiento
        # ============================================================

        self.declare_parameter("linear_speed",  1.0)
        self.declare_parameter("angular_speed", 1.0)
        self.declare_parameter("spin_speed",    0.6)
        self.declare_parameter("cmd_duration",  3.0)
        self.declare_parameter("hold_delay",    1.20)

        self._v_lin        = float(self.get_parameter("linear_speed").value)
        self._w_ang        = float(self.get_parameter("angular_speed").value)
        self._w_spin       = float(self.get_parameter("spin_speed").value)
        self._cmd_duration = float(self.get_parameter("cmd_duration").value)
        self._hold_delay   = float(self.get_parameter("hold_delay").value)

        # ============================================================
        # Parámetros de grabación / reconocimiento
        # ============================================================

        self.duration   = float(duration)
        self.samplerate = int(samplerate)
        self.threshold  = float(threshold)
        self.auto_loop  = bool(auto_loop)

        errors: list[str] = []
        if self._v_lin < 0.0:
            errors.append("linear_speed debe ser >= 0.0")
        if self._w_ang < 0.0:
            errors.append("angular_speed debe ser >= 0.0")
        if self._w_spin <= 0.0:
            errors.append("spin_speed debe ser > 0.0 para calcular giro 360°")
        if self._hold_delay <= 0.0:
            errors.append("hold_delay debe ser > 0.0")
        if self.duration <= 0.0:
            errors.append("duration debe ser > 0.0")
        if self.samplerate <= 0:
            errors.append("samplerate debe ser > 0")
        if errors:
            raise ValueError("; ".join(errors))

        if self._cmd_duration <= 0.0:
            self.get_logger().warning(
                "cmd_duration <= 0.0; usando 2.0 s por seguridad."
            )
            self._cmd_duration = 2.0

        # Duración de giro 360°: θ = ω·t  →  t = 2π / ω
        self._spin_360_duration: float = (2.0 * math.pi) / self._w_spin

        # ============================================================
        # Cargar modelos y codebook
        # ============================================================

        self.recognizer = WordHMMRecognizer.load(model_dir)
        self.codebook   = np.load(codebook_path)

        if not self.recognizer.models:
            raise RuntimeError(f"No se cargaron modelos HMM desde: {model_dir}")

        self.cfg = Config(
            dataset_dir="",
            results_dir="",
            target_sr=self.samplerate,
            frame_len=int(round(0.025 * self.samplerate)),  # 25 ms
            hop_len=int(round(0.010 * self.samplerate)),    # 10 ms
            pre_emph=0.97,
            vad_threshold_ratio=0.05,
            min_segment_ms=200,
            n_fft=512,
            n_mels=26,
            n_mfcc=12,
            include_c0=False,
            codebook_size=256,
        )

        # ============================================================
        # Estado interno — base móvil
        # ============================================================

        self._state_lock   = threading.Lock()
        self._active_twist = Twist()
        self._active_until = 0.0

        # ============================================================
        # Estado interno — lifter
        # ============================================================

        self._pending_lift_action: Optional[str] = None
        self._lift_sequence_start: float = 0.0

        # ============================================================
        # ROS I/O
        # ============================================================

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._pub_cmd_vel      = self.create_publisher(Twist,  "/cmd_vel",       qos_profile)
        self._pub_lift_auto    = self.create_publisher(String, "/lift_auto",      10)
        self._pub_lift_trigger = self.create_publisher(Int8,   "/lift_trigger",   10)

        # Timer a 20 Hz
        self.create_timer(0.05, self._timer_loop)

        self._running = True
        self._voice_thread = threading.Thread(
            target=self._voice_input_loop,
            name="voice_input_loop",
            daemon=True,
        )
        self._voice_thread.start()

        self.get_logger().info(
            "VoiceActionNode integrado listo.\n"
            f"  Dispositivo de Audio: {self.device if self.device is not None else 'Predeterminado del SO'}\n"
            f"  Audio: duration={self.duration:.2f} s | sr={self.samplerate} Hz | "
            f"threshold={self.threshold:.2f}\n"
            f"  Base: v={self._v_lin:.2f} m/s | "
            f"w_turn={self._w_ang:.2f} rad/s | "
            f"w_spin={self._w_spin:.2f} rad/s | "
            f"dur_base={self._cmd_duration:.2f} s | "
            f"dur_360={self._spin_360_duration:.2f} s\n"
            f"  Lifter: toma/arriba -> espera {self._hold_delay:.2f} s -> hold\n"
            f"  Modo: {'Continuo (Auto-loop)' if self.auto_loop else 'Interactivo'}"
        )

    # ============================================================
    # Loop de entrada por micrófono
    # ============================================================

    def _voice_input_loop(self) -> None:
        while self._running and rclpy.ok():
            try:
                if not self.auto_loop:
                    input("\nPresiona ENTER para grabar el comando de voz...")

                self.record_predict_and_execute()

                if self.auto_loop:
                    time.sleep(0.25)

            except EOFError:
                self.get_logger().warning(
                    "No hay entrada interactiva disponible. "
                    "Usa una terminal interactiva o ejecuta en modo auto-loop."
                )
                break
            except KeyboardInterrupt:
                break
            except Exception as exc:
                self.get_logger().error(f"Error en loop de voz: {exc}")
                time.sleep(0.5)

    def record_predict_and_execute(self) -> None:
        self.get_logger().info(f"Grabando {self.duration:.1f} s. Habla ahora.")

        n_samples = int(self.duration * self.samplerate)

        recording = sd.rec(
            n_samples,
            samplerate=self.samplerate,
            channels=1,
            dtype="float64",
            device=self.device,
        )
        sd.wait()

        signal = recording.flatten()

        word, valid, scores = predict_from_signal(
            signal=signal,
            recognizer=self.recognizer,
            codebook=self.codebook,
            cfg=self.cfg,
            threshold=self.threshold,
        )

        if valid:
            self.get_logger().info(f'Comando reconocido: "{word}"')
        else:
            self.get_logger().warning('Comando poco confiable: "<unk>"')

        if scores:
            self.get_logger().info("Puntuaciones por modelo:")
            for label, score in sorted(
                scores.items(),
                key=lambda item: item[1],
                reverse=True,
            ):
                self.get_logger().info(f"  {label:12s}: {score:.4f}")

        self._process_voice_command(word)

    # ============================================================
    # Procesamiento principal del comando reconocido
    # ============================================================

    def _process_voice_command(self, command_raw: str) -> None:
        command: str = command_raw.strip().lower()

        self.get_logger().info(f'Procesando comando: "{command}"')

        if not command or command == CMD_UNKNOWN:
            self._stop_all("Comando desconocido o vacío: paro seguro.")
            return

        if command in CMD_STOP:
            self._stop_all(f'Paro general por comando "{command}".')
            return

        if self._handle_lift_command(command):
            return

        twist, duration = self._build_base_action(command)
        if twist is None:
            self._stop_all(f'Comando sin mapeo: "{command}". Paro seguro.')
            return

        with self._state_lock:
            self._active_twist = twist
            self._active_until = time.monotonic() + duration

        self.get_logger().info(
            f'"{command}" → /cmd_vel  '
            f"linear.x={twist.linear.x:.2f}  "
            f"angular.z={twist.angular.z:.2f}  "
            f"dur={duration:.2f} s"
        )

    # ============================================================
    # Timer principal (20 Hz)
    # ============================================================

    def _timer_loop(self) -> None:
        now = time.monotonic()

        with self._state_lock:
            if self._active_until > 0.0 and now >= self._active_until:
                self._active_twist = Twist()
                self._active_until = 0.0
                self.get_logger().info("Auto-stop: duración de base completada.")

            twist_to_publish = self._active_twist

        self._pub_cmd_vel.publish(twist_to_publish)

        # --- Secuencia temporizada de lift: n1 o n2 -> hold ---
        if (
            self._pending_lift_action == "SEND_HOLD_AFTER_DELAY"
            and self._lift_sequence_start > 0.0
            and (now - self._lift_sequence_start) >= self._hold_delay
        ):
            self.get_logger().info(
                f'Secuencia auto-hold: espera de {self._hold_delay:.2f} s completada '
                '→ enviando "hold".'
            )
            self._send_lift_auto("hold", "secuencia/hold")
            self._pending_lift_action = None
            self._lift_sequence_start = 0.0

    # ============================================================
    # Acciones de base
    # ============================================================

    def _build_base_action(self, command: str) -> tuple[Optional[Twist], float]:
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
        """
        if command in CMD_LIFT_N2:
            self._pending_lift_action = "SEND_HOLD_AFTER_DELAY"
            self._lift_sequence_start = time.monotonic()
            self._send_lift_auto("n2", command)
            self.get_logger().info(
                f'Secuencia "arriba": se mandó "n2". '
                f'Se mandará "hold" en {self._hold_delay:.2f} s.'
            )
            return True

        if command in CMD_LIFT_DOWN:
            self._pending_lift_action = None
            self._lift_sequence_start = 0.0
            self._send_lift_auto("down", command)
            return True

        if command in CMD_LIFT_TAKE:
            self._pending_lift_action = "SEND_HOLD_AFTER_DELAY"
            self._lift_sequence_start = time.monotonic()
            self._send_lift_auto("n1", command)
            self.get_logger().info(
                f'Secuencia "toma": se mandó "n1". '
                f'Se mandará "hold" en {self._hold_delay:.2f} s.'
            )
            return True

        return False

    def _send_lift_auto(self, lift_cmd: str, source_word: str) -> None:
        msg = String()
        msg.data = lift_cmd
        self._pub_lift_auto.publish(msg)
        self.get_logger().info(f'"{source_word}" → /lift_auto: "{lift_cmd}"')

    def _stop_lifter(self) -> None:
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
        with self._state_lock:
            self._active_twist = Twist()
            self._active_until = 0.0
        if reason:
            self.get_logger().info(reason)

    def _stop_all(self, reason: Optional[str] = None) -> None:
        self._stop_base()
        self._stop_lifter()
        if reason:
            self.get_logger().info(reason)

    def shutdown(self) -> None:
        self._running = False
        self._stop_all("Cierre seguro: base y lifter detenidos.")


# ============================================================
# Main
# ============================================================

def main(args: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    
    # Rutas por defecto preconfiguradas
    parser.add_argument(
        "--model-dir",
        type=str,
        default="src/voice_hmm_ros/voice_hmm_ros/resultados_hmm_bw_tol05/models",
        help="Ruta a la carpeta con los .npz de los HMM.",
    )
    parser.add_argument(
        "--codebook-path",
        type=str,
        default="src/voice_hmm_ros/voice_hmm_ros/resultados_hmm_bw_tol05/codebook.npy",
        help="Ruta al archivo codebook.npy.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Duración de grabación en segundos.",
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=16000,
        help="Frecuencia de muestreo.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=70.0,
        help="Umbral opcional de confianza basado en gap ent    re top-1 y top-2.",
    )
    # auto-loop activado por defecto. --interactive pide usar ENTER manualmente.
    parser.add_argument(
        "--interactive",
        action="store_false",
        help="Desactiva el auto-loop continuo y pide presionar ENTER para cada comando.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="ID del dispositivo de audio en sounddevice.",
    )

    parsed_args, ros_args = parser.parse_known_args(args=args)

    rclpy.init(args=ros_args)

    node = VoiceActionNode(
        model_dir=parsed_args.model_dir,
        codebook_path=parsed_args.codebook_path,
        duration=parsed_args.duration,
        samplerate=parsed_args.samplerate,
        threshold=parsed_args.threshold,
        auto_loop=not parsed_args.interactive,
        device=parsed_args.device,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
================================================================================
Proyecto:    voice_hmm_ros (Paquete ROS 2)
Módulo:      voice_command_node.py
Descripción: Nodo de ROS 2 que captura audio del micrófono en tiempo real, extrae
             características MFCC, las cuantiza mediante el codebook global y
             clasifica el comando de voz usando los modelos HMM entrenados.
             Publica los resultados en los tópicos correspondientes.

Uso en ROS 2:
    1. Compilar el espacio de trabajo (desde la raíz ~/ros2_ws):
       $ colcon build --packages-select voice_hmm_ros
       
    2. Sourcing del entorno:
       $ source install/setup.bash
       
    3. Ejecutar el nodo pasándole los modelos entrenados:
       $ ros2 run voice_hmm_ros voice_command_node --model-dir src/voice_hmm_ros/resultados_hmm/models --codebook-path src/voice_hmm_ros/resultados_hmm/codebook.npy

Argumentos Principales:
    --model-dir       Ruta a la carpeta que contiene los archivos .npz de los HMM.
    --codebook-path   Ruta al archivo global codebook.npy generado en el entrenamiento.
    --duration        Duración de la ventana de grabación en segundos (Por defecto: 2.0).
    --threshold       Umbral de confianza basado en el margen de log-likelihood (Por defecto: 0.0).

================================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import sounddevice as sd

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

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
    """
    mfcc = extract_mfcc_from_signal(signal, cfg)

    if mfcc.shape[0] == 0:
        return "<unk>", False, {}

    obs = quantize_sequence(mfcc, codebook)

    if len(obs) == 0:
        return "<unk>", False, {}

    word, scores = recognizer.predict(obs)

    # Confianza simple: diferencia entre las 2 mejores puntuaciones
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    if len(sorted_scores) >= 2:
        best_score = sorted_scores[0][1]
        second_score = sorted_scores[1][1]
        gap = best_score - second_score
    else:
        gap = float("inf")

    valid = True

    if threshold > 0.0 and gap < threshold:
        word = "<unk>"
        valid = False

    return word, valid, scores


class VoiceCommandNode(Node):
    def __init__(
        self,
        model_dir: str,
        codebook_path: str,
        duration: float,
        samplerate: int,
        threshold: float,
    ) -> None:
        super().__init__("voice_command_node")

        # Publishers ROS2
        self.command_pub = self.create_publisher(String, "voice_command", 10)
        self.valid_pub = self.create_publisher(Bool, "voice_command_valid", 10)

        # Parámetros de grabación
        self.duration = duration
        self.samplerate = samplerate
        self.threshold = threshold

        # Cargar modelos y codebook
        self.recognizer = WordHMMRecognizer.load(model_dir)
        self.codebook = np.load(codebook_path)

        if not self.recognizer.models:
            raise RuntimeError(
                f"No se cargaron modelos HMM desde: {model_dir}"
            )

        # Configuración idéntica a la del entrenamiento descrita en tu plantilla
        self.cfg = Config(
            dataset_dir="",
            results_dir="",
            target_sr=samplerate,
            frame_len=int(round(0.025 * samplerate)),  # 25 ms
            hop_len=int(round(0.010 * samplerate)),    # 10 ms
            pre_emph=0.97,
            vad_threshold_ratio=0.05,
            min_segment_ms=200,
            n_fft=512,
            n_mels=26,
            n_mfcc=12,
            include_c0=False,
            codebook_size=256,
        )

        self.get_logger().info("Nodo de reconocimiento de voz listo.")
        self.get_logger().info("Publicará comandos en /voice_command")

    def record_predict_and_publish(self) -> None:
        """
        Grabación única para demo:
        ENTER → graba → predice → publica en ROS2.
        """
        input("\nPresiona ENTER para grabar el comando de voz...")

        self.get_logger().info(
            f"Grabando {self.duration:.1f} s. Habla ahora."
        )

        n_samples = int(self.duration * self.samplerate)

        recording = sd.rec(
            n_samples,
            samplerate=self.samplerate,
            channels=1,
            dtype="float64",
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

        # Mensaje de palabra
        msg_word = String()
        msg_word.data = word
        self.command_pub.publish(msg_word)

        # Mensaje de validez
        msg_valid = Bool()
        msg_valid.data = valid
        self.valid_pub.publish(msg_valid)

        if valid:
            self.get_logger().info(
                f'Comando reconocido y publicado: "{word}"'
            )
        else:
            self.get_logger().warning(
                'Comando poco confiable. Se publicó "<unk>".'
            )

        if scores:
            self.get_logger().info("Puntuaciones por modelo:")
            for label, score in sorted(
                scores.items(),
                key=lambda item: item[1],
                reverse=True,
            ):
                self.get_logger().info(f"  {label:12s}: {score:.4f}")


def main(args=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Ruta a la carpeta con los .npz de los HMM.",
    )
    parser.add_argument(
        "--codebook-path",
        required=True,
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
        default=0.0,
        help="Umbral opcional de confianza.",
    )

    parsed_args = parser.parse_args()

    rclpy.init(args=args)

    node = VoiceCommandNode(
        model_dir=parsed_args.model_dir,
        codebook_path=parsed_args.codebook_path,
        duration=parsed_args.duration,
        samplerate=parsed_args.samplerate,
        threshold=parsed_args.threshold,
    )

    try:
        node.record_predict_and_publish()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
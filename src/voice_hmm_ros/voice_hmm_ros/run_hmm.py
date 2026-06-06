"""
================================================================================
Proyecto:    voice_hmm_ros (Módulo de Entrenamiento)
Módulo:      run_hmm.py

Descripción:
    Pipeline completo de entrenamiento y evaluación para el clasificador de
    palabras aisladas basado en MFCC + VQ + HMM discreto tipo Bakis.

    El flujo principal realiza:
        1. Carga y balanceo del dataset por palabra y locutor.
        2. Preprocesamiento de audio con VAD, normalización y pre-énfasis.
        3. Extracción de características MFCC.
        4. Cuantización vectorial mediante codebook global LBG de 256 símbolos.
        5. Entrenamiento de un HMM Bakis por comando.
        6. Refinamiento opcional con Baum-Welch y early stopping.
        7. Evaluación con accuracy, matriz de confusión, métricas por clase,
           accuracy por locutor, errores individuales y margen de confianza.

Configuración recomendada:
    La configuración final recomendada usa Baum-Welch con un máximo de 20
    iteraciones, tolerancia de early stopping de 0.05 y paciencia de 3
    iteraciones. Esta configuración mantuvo el accuracy de prueba en 98% y
    redujo iteraciones innecesarias en los modelos que ya habían saturado.

Uso en Terminal:

    1. Entrenamiento recomendado con Baum-Welch + early stopping:
       $ python3 run_hmm.py \
           --dataset-dir ./dataset_unificado_6 \
           --results-dir resultados_hmm_final \
           --refine-bw \
           --bw-iters 20 \
           --bw-tol 0.05 \
           --bw-patience 3 \
           --n-states 5 \
           --verbose

    2. Entrenamiento básico sin refinamiento Baum-Welch:
       $ python3 run_hmm.py \
           --dataset-dir ./dataset_unificado_6 \
           --results-dir resultados_hmm_baseline \
           --n-states 5

    3. Inferencia o prueba con un único archivo WAV usando modelos guardados:
       $ python3 run_hmm.py \
           --dataset-dir ./dataset_unificado_6 \
           --results-dir resultados_hmm_final \
           --load-only \
           --predict-file /ruta/audio_test.wav

Archivos generados:
    - codebook.npy
    - models/
    - training_history.json
    - run_config.json
    - accuracy_test.txt
    - confusion_matrix.csv
    - predictions_detailed.csv
    - errors_only.csv
    - classification_report_by_class.csv
    - accuracy_by_speaker.csv
    - analysis_plots/

================================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.fft import dct, rfft

try:
    # Cuando se importa desde el paquete ROS2
    from .hmm_from_scratch import (
        WordHMMRecognizer,
        accuracy_from_confusion,
        confusion_matrix,
    )
except ImportError:
    # Cuando ejecutas run_hmm.py directamente con python3
    from hmm_from_scratch import (
        WordHMMRecognizer,
        accuracy_from_confusion,
        confusion_matrix,
    )


@dataclass(frozen=True)
class Config:
    dataset_dir: str
    results_dir: str
    target_sr: int = 16000
    frame_len: int = 400
    hop_len: int = 160
    pre_emph: float = 0.97
    vad_threshold_ratio: float = 0.05
    min_segment_ms: int = 200
    n_fft: int = 512
    n_mels: int = 26
    n_mfcc: int = 12
    include_c0: bool = False
    fmin: float = 20.0
    fmax: Optional[float] = None
    codebook_size: int = 256
    lbg_split_eps: float = 0.01
    lbg_max_iter: int = 60
    lbg_tol: float = 1e-4
    random_seed: int = 42


def load_audio(path: Path, cfg: Config) -> np.ndarray:
    signal, sr = sf.read(str(path))
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
    if sr != cfg.target_sr:
        raise ValueError(f"Frecuencia de muestreo {sr}Hz != {cfg.target_sr}Hz para {path}")
    return signal.astype(np.float64)


def frame_signal(signal: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    sig = np.asarray(signal, dtype=np.float64)

    if sig.size == 0:
        return np.empty((0, frame_len), dtype=np.float64)

    if sig.size < frame_len:
        padded = np.zeros(frame_len, dtype=np.float64)
        padded[: sig.size] = sig
        return padded[None, :]

    n_frames = 1 + (sig.size - frame_len) // hop_len
    starts = hop_len * np.arange(n_frames)
    indices = starts[:, None] + np.arange(frame_len)[None, :]
    return sig[indices]


def frame_energy(frames: np.ndarray) -> np.ndarray:
    if frames.size == 0:
        return np.array([], dtype=np.float64)
    return np.mean(frames ** 2, axis=1)


def vad_trim(signal: np.ndarray, cfg: Config) -> np.ndarray:
    frames = frame_signal(signal, cfg.frame_len, cfg.hop_len)

    if frames.shape[0] == 0:
        return signal

    energy = frame_energy(frames)

    if energy.size >= 3:
        energy = np.convolve(energy, np.ones(3) / 3.0, mode="same")

    threshold = cfg.vad_threshold_ratio * np.max(energy)
    voiced = np.nonzero(energy > threshold)[0]

    if voiced.size == 0:
        return signal

    pad_frames = int(cfg.min_segment_ms * cfg.target_sr / 1000.0 / cfg.hop_len)
    start = max(0, voiced[0] - pad_frames)
    end = min(len(energy) - 1, voiced[-1] + pad_frames)

    start_idx = start * cfg.hop_len
    end_idx = min(signal.size, end * cfg.hop_len + cfg.frame_len)

    return signal[start_idx:end_idx]


def pre_emphasis(signal: np.ndarray, alpha: float) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)

    if x.size == 0:
        return x

    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - alpha * x[:-1]

    return y


def apply_hamming(frames: np.ndarray) -> np.ndarray:
    if frames.size == 0:
        return frames
    return frames * np.hamming(frames.shape[1])


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    hz = np.asarray(hz, dtype=np.float64)
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    mel = np.asarray(mel, dtype=np.float64)
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: Optional[float],
) -> np.ndarray:
    if fmax is None:
        fmax = sr / 2.0

    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), num=n_mels + 2)
    hz_points = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)

    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]

        if center <= left:
            center = left + 1

        if right <= center:
            right = center + 1

        for k in range(left, center):
            if 0 <= k < fb.shape[1]:
                fb[m - 1, k] = (k - left) / max(center - left, 1)

        for k in range(center, right):
            if 0 <= k < fb.shape[1]:
                fb[m - 1, k] = (right - k) / max(right - center, 1)

    enorm = 2.0 / np.maximum(hz_points[2:n_mels + 2] - hz_points[:n_mels], 1e-12)
    fb *= enorm[:, None]

    return fb


def extract_mfcc_from_frames(frames: np.ndarray, cfg: Config) -> np.ndarray:
    if frames.size == 0:
        return np.empty((0, cfg.n_mfcc), dtype=np.float64)

    power = np.abs(rfft(frames, n=cfg.n_fft, axis=1)) ** 2
    power /= float(cfg.n_fft)

    fb = mel_filterbank(cfg.target_sr, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax)

    mel_energies = power @ fb.T
    mel_energies = np.maximum(mel_energies, 1e-10)

    log_mel = np.log(mel_energies)
    cep = dct(log_mel, type=2, axis=1, norm="ortho")

    if cfg.include_c0:
        mfcc = cep[:, : cfg.n_mfcc]
    else:
        mfcc = cep[:, 1 : cfg.n_mfcc + 1]

    return mfcc.astype(np.float64)


def extract_mfcc_sequence(audio_path: Path, cfg: Config) -> np.ndarray:
    sig = load_audio(audio_path, cfg)

    sig = vad_trim(sig, cfg)

    mx = float(np.max(np.abs(sig))) if sig.size else 0.0
    if mx > 0.0:
        sig = sig / mx

    sig = pre_emphasis(sig, cfg.pre_emph)

    frames = frame_signal(sig, cfg.frame_len, cfg.hop_len)
    frames = apply_hamming(frames)

    return extract_mfcc_from_frames(frames, cfg)


def squared_euclidean_distance(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    x2 = np.sum(X ** 2, axis=1, keepdims=True)
    c2 = np.sum(C ** 2, axis=1, keepdims=True).T
    return np.maximum(x2 + c2 - 2.0 * X @ C.T, 0.0)


def _rebalance_empty_clusters(
    X: np.ndarray,
    C: np.ndarray,
    labels: np.ndarray,
    d2: np.ndarray,
) -> np.ndarray:
    k = C.shape[0]
    counts = np.bincount(labels, minlength=k)
    empties = np.where(counts == 0)[0]

    if empties.size == 0:
        return labels

    farthest = np.argsort(np.min(d2, axis=1))[::-1]
    ptr = 0

    for empty in empties:
        while ptr < len(farthest):
            idx = farthest[ptr]
            ptr += 1
            donor = labels[idx]

            if counts[donor] > 1:
                labels[idx] = empty
                counts[donor] -= 1
                counts[empty] += 1
                break

    return labels


def kmeans_refine(
    X: np.ndarray,
    init_C: np.ndarray,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    C = np.asarray(init_C, dtype=np.float64).copy()
    prev_dist = np.inf

    for _ in range(max_iter):
        d2 = squared_euclidean_distance(X, C)
        labels = np.argmin(d2, axis=1)
        labels = _rebalance_empty_clusters(X, C, labels, d2)

        new_C = C.copy()
        distortions = []

        for j in range(C.shape[0]):
            mask = labels == j

            if np.any(mask):
                new_C[j] = np.mean(X[mask], axis=0)
                distortions.append(np.mean(np.sum((X[mask] - new_C[j]) ** 2, axis=1)))

        avg_dist = float(np.mean(distortions)) if distortions else np.inf
        C = new_C

        if prev_dist < np.inf and abs(prev_dist - avg_dist) / max(prev_dist, 1e-12) < tol:
            break

        prev_dist = avg_dist

    return C


def lbg_train(
    X: np.ndarray,
    codebook_size: int = 256,
    split_eps: float = 0.01,
    max_iter: int = 60,
    tol: float = 1e-4,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X debe ser una matriz no vacía de shape (n_frames, dim)")

    if codebook_size < 1:
        raise ValueError("codebook_size debe ser >= 1")

    C = np.mean(X, axis=0, keepdims=True)

    while C.shape[0] < codebook_size:
        C = np.vstack([C * (1.0 + split_eps), C * (1.0 - split_eps)])

        if C.shape[0] > codebook_size:
            C = C[:codebook_size]

        C = kmeans_refine(X, C, max_iter=max_iter, tol=tol)

    return C


def quantize_sequence(X: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError("X debe tener shape (T, dim)")

    if X.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)

    d2 = squared_euclidean_distance(X, codebook)
    return np.argmin(d2, axis=1).astype(np.int64)


def split_dataset(
    paths: Sequence[Path],
    train_ratio: float,
    rng: np.random.Generator,
) -> Tuple[List[Path], List[Path]]:
    paths = list(paths)
    rng.shuffle(paths)

    if len(paths) <= 1:
        return paths, []

    split = int(round(len(paths) * train_ratio))
    split = min(max(split, 1), len(paths) - 1)

    return paths[:split], paths[split:]


def collect_mfcc_sequences(
    dataset_dir: Path,
    cfg: Config,
    min_frames_for_train: int,
    train_ratio: float,
    seed: int,
) -> Tuple[
    Dict[str, List[np.ndarray]],
    Dict[str, List[np.ndarray]],
    Dict[str, List[Path]],
    Dict[str, List[Path]],
]:
    rng = np.random.default_rng(seed)

    words = sorted([p.name for p in dataset_dir.iterdir() if p.is_dir()])

    if not words:
        raise FileNotFoundError(f"No se encontraron subdirectorios de palabras en {dataset_dir}")

    train_seq: Dict[str, List[np.ndarray]] = {}
    test_seq: Dict[str, List[np.ndarray]] = {}
    train_paths: Dict[str, List[Path]] = {}
    test_paths: Dict[str, List[Path]] = {}

    print("\n--- Analizando y balanceando por locutor ---")

    for word in words:
        word_dir = dataset_dir / word

        speakers = sorted([p.name for p in word_dir.iterdir() if p.is_dir()])

        speaker_wavs_dict = {}

        for spk in speakers:
            wavs = sorted((word_dir / spk).rglob("*.wav"))

            if wavs:
                speaker_wavs_dict[spk] = wavs

        if not speaker_wavs_dict:
            continue

        min_audios_per_speaker = min(len(wavs) for wavs in speaker_wavs_dict.values())

        print(f" -> Palabra [{word.upper()}]: balanceando a {min_audios_per_speaker} audios por locutor.")

        seqs_tr: List[np.ndarray] = []
        seqs_te: List[np.ndarray] = []
        keep_tr: List[Path] = []
        keep_te: List[Path] = []

        for _spk, wavs in speaker_wavs_dict.items():
            local_wavs = list(wavs)
            rng.shuffle(local_wavs)

            balanced_wavs = local_wavs[:min_audios_per_speaker]
            tr_paths, te_paths = split_dataset(balanced_wavs, train_ratio, rng)

            for p in tr_paths:
                X = extract_mfcc_sequence(p, cfg)

                if X.shape[0] >= min_frames_for_train:
                    seqs_tr.append(X)
                    keep_tr.append(p)

            for p in te_paths:
                X = extract_mfcc_sequence(p, cfg)

                if X.shape[0] >= min_frames_for_train:
                    seqs_te.append(X)
                    keep_te.append(p)

        if seqs_tr:
            train_seq[word] = seqs_tr
            train_paths[word] = keep_tr

        if seqs_te:
            test_seq[word] = seqs_te
            test_paths[word] = keep_te

    if not train_seq:
        raise RuntimeError("No se extrajeron secuencias de entrenamiento válidas")

    print("---------------------------------------------------------------\n")

    return train_seq, test_seq, train_paths, test_paths


def save_metrics(results_dir: Path, acc: float, cm: np.ndarray, labels: Sequence[str]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "accuracy_test.txt", "w", encoding="utf-8") as f:
        f.write(f"{acc:.6f}\n")

    with open(results_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + list(labels))

        for i, lab in enumerate(labels):
            writer.writerow([lab] + list(map(int, cm[i].tolist())))


def print_confusion(cm: np.ndarray, labels: Sequence[str]) -> None:
    header = " " * 14 + " ".join(f"{lab[:10]:>10s}" for lab in labels)
    print(header)

    for i, lab in enumerate(labels):
        row = " ".join(f"{int(v):10d}" for v in cm[i])
        print(f"{lab[:12]:>12s}  {row}")


def save_heatmap(
    matrix: np.ndarray,
    title: str,
    out_path: Path,
    xlabel: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(6, 4))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_state_b_plot(B: np.ndarray, word: str, out_path: Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(B.shape[1]), B[0])
    plt.title(f"{word}: distribución B del estado 1")
    plt.xlabel("Índice del codebook")
    plt.ylabel("Probabilidad")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_model_diagnostics(recognizer: WordHMMRecognizer, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for word, model in recognizer.models.items():
        safe = word.replace("/", "_")

        save_heatmap(
            model.A,
            f"{word}: matriz A",
            out_dir / f"{safe}_A.png",
            "Estado destino",
            "Estado origen",
        )

        save_state_b_plot(
            model.B,
            word,
            out_dir / f"{safe}_B_state1.png",
        )

        np.savetxt(out_dir / f"{safe}_A.csv", model.A, delimiter=",")
        np.savetxt(out_dir / f"{safe}_B.csv", model.B, delimiter=",")


def evaluate_detailed(
    recognizer: WordHMMRecognizer,
    test_sequences: Dict[str, List[np.ndarray]],
    test_paths: Optional[Dict[str, List[Path]]] = None,
) -> Tuple[float, np.ndarray, List[str], List[dict]]:
    labels = sorted(recognizer.models.keys())

    rows: List[dict] = []
    y_true: List[str] = []
    y_pred: List[str] = []

    for true_word in labels:
        sequences = test_sequences.get(true_word, [])

        if test_paths is not None:
            paths = test_paths.get(true_word, [None] * len(sequences))
        else:
            paths = [None] * len(sequences)

        for obs, path in zip(sequences, paths):
            scores = recognizer.score_all(obs)
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

            pred_word = ranked[0][0]
            top1_score = ranked[0][1]

            top2_word = ranked[1][0] if len(ranked) > 1 else ""
            top2_score = ranked[1][1] if len(ranked) > 1 else float("nan")

            margin = top1_score - top2_score if len(ranked) > 1 else float("nan")
            correct = int(pred_word == true_word)

            speaker = ""
            if path is not None:
                parts = Path(path).parts
                if len(parts) >= 3:
                    speaker = parts[-2]

            y_true.append(true_word)
            y_pred.append(pred_word)

            rows.append({
                "file": str(path) if path is not None else "",
                "speaker": speaker,
                "y_true": true_word,
                "y_pred": pred_word,
                "correct": correct,
                "top1_score": top1_score,
                "top2_label": top2_word,
                "top2_score": top2_score,
                "margin": margin,
                "seq_len": len(obs),
            })

    cm = confusion_matrix(y_true, y_pred, labels)
    acc = accuracy_from_confusion(cm)

    return acc, cm, labels, rows


def save_detailed_predictions(results_dir: Path, detailed_rows: List[dict]) -> None:
    fieldnames = [
        "file",
        "speaker",
        "y_true",
        "y_pred",
        "correct",
        "top1_score",
        "top2_label",
        "top2_score",
        "margin",
        "seq_len",
    ]

    with open(results_dir / "predictions_detailed.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detailed_rows)

    errors = [row for row in detailed_rows if row["correct"] == 0]

    error_fieldnames = [
        "file",
        "speaker",
        "y_true",
        "y_pred",
        "correct",
        "top1_score",
        "top2_label",
        "top2_score",
        "margin",
        "seq_len",
    ]

    with open(results_dir / "errors_only.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=error_fieldnames)
        writer.writeheader()
        writer.writerows(errors)


def save_analysis_plots(
    results_dir: Path,
    cm: np.ndarray,
    labels: Sequence[str],
    detailed_rows: List[dict],
    histories: Dict[str, List[float]],
) -> None:
    plots_dir = results_dir / "analysis_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 1. Matriz de confusión normalizada
    # ============================================================
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_norm, aspect="auto")
    plt.colorbar(label="Recall por clase")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta real")
    plt.title("Matriz de confusión normalizada")
    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrix_normalized.png", dpi=180)
    plt.close()

    # ============================================================
    # 2. Métricas por clase
    # ============================================================
    metrics = []

    for i, label in enumerate(labels):
        tp = cm[i, i]
        support = cm[i, :].sum()
        predicted = cm[:, i].sum()

        precision = tp / predicted if predicted > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics.append({
            "label": label,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support),
        })

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(results_dir / "classification_report_by_class.csv", index=False)

    x = np.arange(len(metrics_df))
    width = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, metrics_df["precision"], width, label="Precision")
    plt.bar(x, metrics_df["recall"], width, label="Recall")
    plt.bar(x + width, metrics_df["f1"], width, label="F1-score")
    plt.xticks(x, metrics_df["label"], rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Métricas por comando")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "precision_recall_f1_by_class.png", dpi=180)
    plt.close()

    # ============================================================
    # 3. Distribución del margen de confianza
    # ============================================================
    df = pd.DataFrame(detailed_rows)

    if len(df) > 0 and "margin" in df.columns:
        correct_margins = df[df["correct"] == 1]["margin"].dropna()
        wrong_margins = df[df["correct"] == 0]["margin"].dropna()

        plt.figure(figsize=(8, 5))
        plt.hist(correct_margins, bins=20, alpha=0.7, label="Correctas")

        if len(wrong_margins) > 0:
            plt.hist(wrong_margins, bins=20, alpha=0.7, label="Incorrectas")

        plt.xlabel("Margen log-likelihood: top1 - top2")
        plt.ylabel("Cantidad de audios")
        plt.title("Distribución del margen de confianza")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "confidence_margin_distribution.png", dpi=180)
        plt.close()

    # ============================================================
    # 4. Accuracy por locutor
    # ============================================================
    if len(df) > 0 and "speaker" in df.columns and df["speaker"].astype(bool).any():
        speaker_acc = df.groupby("speaker")["correct"].mean().sort_values(ascending=False)
        speaker_acc.to_csv(results_dir / "accuracy_by_speaker.csv")

        plt.figure(figsize=(8, 5))
        plt.bar(speaker_acc.index.astype(str), speaker_acc.values)
        plt.ylim(0, 1.05)
        plt.ylabel("Accuracy")
        plt.xlabel("Locutor")
        plt.title("Accuracy por locutor")
        plt.tight_layout()
        plt.savefig(plots_dir / "accuracy_by_speaker.png", dpi=180)
        plt.close()

    # ============================================================
    # 5. Longitud de secuencia después del VAD
    # ============================================================
    if len(df) > 0 and "seq_len" in df.columns:
        data = [df[df["y_true"] == label]["seq_len"].values for label in labels]

        plt.figure(figsize=(10, 5))
        plt.boxplot(data, labels=labels)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Número de símbolos / frames")
        plt.title("Longitud de secuencia después de VAD por comando")
        plt.tight_layout()
        plt.savefig(plots_dir / "sequence_length_by_word.png", dpi=180)
        plt.close()

    # ============================================================
    # 6. Curvas Baum-Welch
    # ============================================================
    if histories:
        plt.figure(figsize=(9, 5))

        plotted = False

        for word, hist in histories.items():
            if hist:
                plt.plot(range(1, len(hist) + 1), hist, marker="o", label=word)
                plotted = True

        if plotted:
            plt.xlabel("Iteración Baum-Welch")
            plt.ylabel("Log-likelihood promedio")
            plt.title("Evolución de entrenamiento por palabra")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(plots_dir / "baum_welch_loglikelihood_history.png", dpi=180)

        plt.close()


def parse_states_json(states_json: Optional[str]) -> Dict[str, int]:
    if not states_json:
        return {}

    with open(states_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: Dict[str, int] = {}

    for k, v in data.items():
        out[str(k)] = int(v)

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena un HMM discreto Bakis desde cero con MFCC + VQ(256)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--dataset-dir", required=True, help="Carpeta raíz con una subcarpeta por palabra")
    parser.add_argument("--results-dir", default="resultados_hmm", help="Carpeta de salida")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Proporción train/test")
    parser.add_argument("--random-seed", type=int, default=42, help="Semilla aleatoria")

    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--frame-ms", type=float, default=25.0)
    parser.add_argument("--hop-ms", type=float, default=10.0)
    parser.add_argument("--pre-emph", type=float, default=0.97)
    parser.add_argument("--vad-threshold-ratio", type=float, default=0.05)
    parser.add_argument("--min-segment-ms", type=int, default=200)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--n-mels", type=int, default=26)
    parser.add_argument("--n-mfcc", type=int, default=12)
    parser.add_argument("--include-c0", action="store_true")
    parser.add_argument("--fmin", type=float, default=20.0)
    parser.add_argument("--fmax", type=float, default=None)

    parser.add_argument("--codebook-size", type=int, default=256, help="Tamaño del codebook global")
    parser.add_argument("--lbg-split-eps", type=float, default=0.01)
    parser.add_argument("--lbg-max-iter", type=int, default=60)
    parser.add_argument("--lbg-tol", type=float, default=1e-4)

    parser.add_argument("--n-states", type=int, default=5, help="Estados por palabra si no se usa --states-json")
    parser.add_argument("--states-json", default=None, help="JSON opcional con estados por palabra")
    parser.add_argument("--smoothing", type=float, default=1e-6, help="Épsilon de suavizado para A y B")
    parser.add_argument("--refine-bw", action="store_true", help="Aplicar Baum-Welch después de conteos")
    parser.add_argument("--bw-iters", type=int, default=3, help="Iteraciones Baum-Welch si se activa --refine-bw")
    parser.add_argument("--bw-tol", type=float, default=1e-4)
    parser.add_argument("--bw-patience", type=int, default=3, help="Iteraciones consecutivas sin mejora antes de early stopping (default: 3)")
    parser.add_argument("--load-only", action="store_true", help="No entrenar; cargar modelos y codebook ya guardados")
    parser.add_argument("--predict-file", default=None, help="Clasificar un archivo WAV individual")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frame_len = int(round(args.target_sr * args.frame_ms / 1000.0))
    hop_len = int(round(args.target_sr * args.hop_ms / 1000.0))

    cfg = Config(
        dataset_dir=args.dataset_dir,
        results_dir=args.results_dir,
        target_sr=args.target_sr,
        frame_len=frame_len,
        hop_len=hop_len,
        pre_emph=args.pre_emph,
        vad_threshold_ratio=args.vad_threshold_ratio,
        min_segment_ms=args.min_segment_ms,
        n_fft=args.n_fft,
        n_mels=args.n_mels,
        n_mfcc=args.n_mfcc,
        include_c0=args.include_c0,
        fmin=args.fmin,
        fmax=args.fmax,
        codebook_size=args.codebook_size,
        lbg_split_eps=args.lbg_split_eps,
        lbg_max_iter=args.lbg_max_iter,
        lbg_tol=args.lbg_tol,
        random_seed=args.random_seed,
    )

    results_dir = Path(cfg.results_dir)
    model_dir = results_dir / "models"
    diag_dir = results_dir / "diagnostics"

    results_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "dataset_dir": cfg.dataset_dir,
        "results_dir": cfg.results_dir,
        "target_sr": cfg.target_sr,
        "frame_len": cfg.frame_len,
        "hop_len": cfg.hop_len,
        "pre_emph": cfg.pre_emph,
        "vad_threshold_ratio": cfg.vad_threshold_ratio,
        "min_segment_ms": cfg.min_segment_ms,
        "n_fft": cfg.n_fft,
        "n_mels": cfg.n_mels,
        "n_mfcc": cfg.n_mfcc,
        "include_c0": cfg.include_c0,
        "fmin": cfg.fmin,
        "fmax": cfg.fmax,
        "codebook_size": cfg.codebook_size,
        "lbg_split_eps": cfg.lbg_split_eps,
        "lbg_max_iter": cfg.lbg_max_iter,
        "lbg_tol": cfg.lbg_tol,
        "n_states": args.n_states,
        "states_json": args.states_json,
        "smoothing": args.smoothing,
        "refine_bw": args.refine_bw,
        "bw_iters": args.bw_iters,
        "bw_tol": args.bw_tol,
        "bw_patience": args.bw_patience,
        "random_seed": args.random_seed,
    }

    with open(results_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    states_per_word = parse_states_json(args.states_json)
    recognizer = WordHMMRecognizer()

    if args.load_only:
        recognizer = WordHMMRecognizer.load(model_dir)
        codebook = np.load(results_dir / "codebook.npy")

    else:
        min_frames = max(
            [args.n_states] + list(states_per_word.values())
            if states_per_word
            else [args.n_states]
        )

        train_mfcc, test_mfcc, train_paths, test_paths = collect_mfcc_sequences(
            dataset_dir=Path(cfg.dataset_dir),
            cfg=cfg,
            min_frames_for_train=min_frames,
            train_ratio=args.train_ratio,
            seed=args.random_seed,
        )

        print("Resumen del dataset útil")

        words = sorted(set(train_mfcc.keys()) | set(test_mfcc.keys()))

        for word in words:
            print(
                f"  {word:15s} "
                f"train={len(train_mfcc.get(word, [])):3d} "
                f"test={len(test_mfcc.get(word, [])):3d}"
            )

        all_train_frames = np.vstack([seq for seqs in train_mfcc.values() for seq in seqs])

        if args.verbose:
            print(
                f"\nEntrenando codebook global con "
                f"{all_train_frames.shape[0]} frames y dim={all_train_frames.shape[1]}"
            )

        codebook = lbg_train(
            all_train_frames,
            codebook_size=cfg.codebook_size,
            split_eps=cfg.lbg_split_eps,
            max_iter=cfg.lbg_max_iter,
            tol=cfg.lbg_tol,
        )

        np.save(results_dir / "codebook.npy", codebook)

        train_obs = {
            word: [quantize_sequence(seq, codebook) for seq in seqs]
            for word, seqs in train_mfcc.items()
        }

        test_obs = {
            word: [quantize_sequence(seq, codebook) for seq in seqs]
            for word, seqs in test_mfcc.items()
        }

        histories = recognizer.fit(
            train_sequences=train_obs,
            n_symbols=cfg.codebook_size,
            default_n_states=args.n_states,
            states_per_word=states_per_word,
            smoothing=args.smoothing,
            bw_iters=args.bw_iters if args.refine_bw else 0,
            bw_tol=args.bw_tol,
            bw_patience=args.bw_patience,
        )

        recognizer.save(model_dir)
        save_model_diagnostics(recognizer, diag_dir)

        with open(results_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(histories, f, ensure_ascii=False, indent=2)

        print("\nIteraciones Baum-Welch utilizadas por palabra:")

        for word, hist in histories.items():
            if hist:
                best_iter = int(np.argmax(hist)) + 1
                best_ll = float(np.max(hist))

                print(
                    f"  {word:15s} "
                    f"usadas={len(hist):2d} "
                    f"mejor_iter={best_iter:2d} "
                    f"best_ll={best_ll:.6f}"
                )
            else:
                print(f"  {word:15s} usadas=0")

        acc, cm, labels, detailed_rows = evaluate_detailed(
            recognizer=recognizer,
            test_sequences=test_obs,
            test_paths=test_paths,
        )

        save_metrics(results_dir, acc, cm, labels)
        save_detailed_predictions(results_dir, detailed_rows)

        save_analysis_plots(
            results_dir=results_dir,
            cm=cm,
            labels=labels,
            detailed_rows=detailed_rows,
            histories=histories,
        )

        print(f"\nAccuracy TEST: {acc:.2%}")
        print("Matriz de confusión (filas=verdadero, columnas=predicho)")
        print_confusion(cm, labels)

        print("\nArchivos de análisis guardados:")
        print(f"  {results_dir / 'predictions_detailed.csv'}")
        print(f"  {results_dir / 'errors_only.csv'}")
        print(f"  {results_dir / 'classification_report_by_class.csv'}")
        print(f"  {results_dir / 'accuracy_by_speaker.csv'}")
        print(f"  {results_dir / 'analysis_plots'}")

    if args.predict_file:
        codebook = np.load(results_dir / "codebook.npy")

        X = extract_mfcc_sequence(Path(args.predict_file), cfg)
        obs = quantize_sequence(X, codebook)

        if len(obs) == 0:
            raise ValueError("No se pudieron extraer frames válidos del archivo a predecir")

        pred, scores = recognizer.predict(obs)

        print("\nPredicción individual")
        print(f"  Archivo: {args.predict_file}")
        print(f"  Palabra reconocida: {pred}")
        print("  Log-likelihood por modelo:")

        for word, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
            print(f"    {word:15s} {score: .6f}")


if __name__ == "__main__":
    main()
"""
================================================================================
Proyecto:    voice_hmm_ros (Módulo de Entrenamiento)
Módulo:      run_hmm.py
Descripción: Pipeline completo de entrenamiento y evaluación para el clasificador
             de palabras aisladas. Realiza extracción de MFCC, cuantización vectorial
             mediante el algoritmo LBG, inicialización de HMMs Bakis por conteo
             lineal y refinamiento opcional con Baum-Welch.

Uso en Terminal:
    1. Entrenamiento básico y evaluación:
       $ python3 run_hmm.py --dataset-dir /ruta/al/dataset --results-dir resultados_hmm
       
    2. Entrenamiento avanzado con refinamiento Baum-Welch y modo detallado:
       $ python3 run_hmm.py --dataset-dir ./dataset --results-dir ./resultados --refine-bw --verbose
       
    3. Inferencia o prueba con un único archivo WAV:
       $ python3 run_hmm.py --dataset-dir ./dataset --load-only --predict-file /ruta/audio_test.wav

Argumentos Clave:
    --dataset-dir   Carpeta raíz que contiene un subdirectorio por palabra/clase.
    --results-dir   Carpeta de salida para modelos (.npz), matrices y diagnósticos.
    --refine-bw     Flag para activar la optimización por Baum-Welch tras los conteos.
    --load-only     Carga un modelo preexistente en lugar de iniciar un entrenamiento.

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
    frame_len: int = 400           # 25 ms @ 16 kHz
    hop_len: int = 160             # 10 ms @ 16 kHz
    pre_emph: float = 0.97
    vad_threshold_ratio: float = 0.05
    min_segment_ms: int = 200
    n_fft: int = 512
    n_mels: int = 26
    n_mfcc: int = 12               # 12 MFCCs como práctica común
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


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: Optional[float]) -> np.ndarray:
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

    # Normalización de área aproximada
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
    sig = vad_trim(sig, cfg)  # VAD primero
    mx = float(np.max(np.abs(sig))) if sig.size else 0.0
    if mx > 0.0:
        sig = sig / mx         # normalización por utterance
    sig = pre_emphasis(sig, cfg.pre_emph)
    frames = frame_signal(sig, cfg.frame_len, cfg.hop_len)
    frames = apply_hamming(frames)
    return extract_mfcc_from_frames(frames, cfg)


def squared_euclidean_distance(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    x2 = np.sum(X ** 2, axis=1, keepdims=True)
    c2 = np.sum(C ** 2, axis=1, keepdims=True).T
    return np.maximum(x2 + c2 - 2.0 * X @ C.T, 0.0)


def _rebalance_empty_clusters(X: np.ndarray, C: np.ndarray, labels: np.ndarray, d2: np.ndarray) -> np.ndarray:
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


def kmeans_refine(X: np.ndarray, init_C: np.ndarray, max_iter: int, tol: float) -> np.ndarray:
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


def lbg_train(X: np.ndarray, codebook_size: int = 256, split_eps: float = 0.01, max_iter: int = 60, tol: float = 1e-4) -> np.ndarray:
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


def split_dataset(paths: Sequence[Path], train_ratio: float, rng: np.random.Generator) -> Tuple[List[Path], List[Path]]:
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
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]], Dict[str, List[Path]], Dict[str, List[Path]]]:
    rng = np.random.default_rng(seed)
    words = sorted([p.name for p in dataset_dir.iterdir() if p.is_dir()])
    if not words:
        raise FileNotFoundError(f"No se encontraron subdirectorios de palabras en {dataset_dir}")

    train_seq: Dict[str, List[np.ndarray]] = {}
    test_seq: Dict[str, List[np.ndarray]] = {}
    train_paths: Dict[str, List[Path]] = {}
    test_paths: Dict[str, List[Path]] = {}

    for word in words:
        wavs = sorted((dataset_dir / word).rglob("*.wav"))
        if not wavs:
            continue
        tr_paths, te_paths = split_dataset(wavs, train_ratio, rng)

        seqs_tr: List[np.ndarray] = []
        seqs_te: List[np.ndarray] = []
        keep_tr: List[Path] = []
        keep_te: List[Path] = []

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


def save_heatmap(matrix: np.ndarray, title: str, out_path: Path, xlabel: str, ylabel: str) -> None:
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
        save_heatmap(model.A, f"{word}: matriz A", out_dir / f"{safe}_A.png", "Estado destino", "Estado origen")
        save_state_b_plot(model.B, word, out_dir / f"{safe}_B_state1.png")
        np.savetxt(out_dir / f"{safe}_A.csv", model.A, delimiter=",")
        np.savetxt(out_dir / f"{safe}_B.csv", model.B, delimiter=",")


def evaluate(recognizer: WordHMMRecognizer, test_sequences: Dict[str, List[np.ndarray]]) -> Tuple[float, np.ndarray, List[str], List[str], List[str]]:
    labels = sorted(recognizer.models.keys())
    y_true: List[str] = []
    y_pred: List[str] = []
    for word in labels:
        for obs in test_sequences.get(word, []):
            pred, _ = recognizer.predict(obs)
            y_true.append(word)
            y_pred.append(pred)
    cm = confusion_matrix(y_true, y_pred, labels)
    acc = accuracy_from_confusion(cm)
    return acc, cm, labels, y_true, y_pred


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
    parser.add_argument("--states-json", default=None, help="JSON opcional con estados por palabra, p.ej. {'izquierda':6}")
    parser.add_argument("--smoothing", type=float, default=1e-6, help="Épsilon de suavizado para A y B")
    parser.add_argument("--refine-bw", action="store_true", help="Aplicar Baum-Welch opcional después de conteos")
    parser.add_argument("--bw-iters", type=int, default=3, help="Iteraciones Baum-Welch si se activa --refine-bw")
    parser.add_argument("--bw-tol", type=float, default=1e-4)

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
        min_frames = max([args.n_states] + list(states_per_word.values()) if states_per_word else [args.n_states])

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
            print(f"  {word:15s} train={len(train_mfcc.get(word, [])):3d} test={len(test_mfcc.get(word, [])):3d}")

        all_train_frames = np.vstack([seq for seqs in train_mfcc.values() for seq in seqs])
        if args.verbose:
            print(f"\nEntrenando codebook global con {all_train_frames.shape[0]} frames y dim={all_train_frames.shape[1]}")

        codebook = lbg_train(
            all_train_frames,
            codebook_size=cfg.codebook_size,
            split_eps=cfg.lbg_split_eps,
            max_iter=cfg.lbg_max_iter,
            tol=cfg.lbg_tol,
        )
        np.save(results_dir / "codebook.npy", codebook)

        train_obs = {word: [quantize_sequence(seq, codebook) for seq in seqs] for word, seqs in train_mfcc.items()}
        test_obs = {word: [quantize_sequence(seq, codebook) for seq in seqs] for word, seqs in test_mfcc.items()}

        histories = recognizer.fit(
            train_sequences=train_obs,
            n_symbols=cfg.codebook_size,
            default_n_states=args.n_states,
            states_per_word=states_per_word,
            smoothing=args.smoothing,
            bw_iters=args.bw_iters if args.refine_bw else 0,
            bw_tol=args.bw_tol,
        )
        recognizer.save(model_dir)
        save_model_diagnostics(recognizer, diag_dir)

        with open(results_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(histories, f, ensure_ascii=False, indent=2)

        acc, cm, labels, y_true, y_pred = evaluate(recognizer, test_obs)
        save_metrics(results_dir, acc, cm, labels)

        print(f"\nAccuracy TEST: {acc:.2%}")
        print("Matriz de confusión (filas=verdadero, columnas=predicho)")
        print_confusion(cm, labels)

        with open(results_dir / "predictions_test.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["y_true", "y_pred"])
            for yt, yp in zip(y_true, y_pred):
                writer.writerow([yt, yp])

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

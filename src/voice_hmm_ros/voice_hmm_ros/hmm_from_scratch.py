"""
================================================================================
Proyecto:    voice_hmm_ros (Núcleo Matemático / Algorítmico)
Módulo:      hmm_from_scratch.py
Descripción: Implementación desde cero (utilizando únicamente NumPy y SciPy básico)
             de Modelos Ocultos de Markov Discretos con topología Bakis (Left-to-Right).
             Incluye los algoritmos de Forward (Log-space), Backward, Viterbi
             y las ecuaciones de reestimación de Baum-Welch.

Uso (Importación en otros módulos):
    Este archivo es una librería de clases y funciones. No se ejecuta directamente.
    Para integrarlo en tus scripts o nodos, utiliza:
    
    >>> from hmm_from_scratch import DiscreteBakisHMM, WordHMMRecognizer

Componentes Principales:
    - logsumexp:          Manejo robusto de subdesbordamiento (underflow) en log-space.
    - DiscreteBakisHMM:   Clase contenedora de las matrices A, B y pi para una palabra.
    - WordHMMRecognizer:  Clasificador multiclase que gestiona un HMM por cada comando.

================================================================================
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import json
import numpy as np

EPS = 1e-300


def logsumexp(a: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        raise ValueError("logsumexp recibió un arreglo vacío")
    amax = np.max(a, axis=axis, keepdims=True)
    safe = np.where(np.isfinite(amax), amax, 0.0)
    out = safe + np.log(np.sum(np.exp(a - safe), axis=axis, keepdims=True) + EPS)
    out = np.where(np.isfinite(amax), out, -np.inf)
    if axis is not None and not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def as_1d_int(seq: np.ndarray) -> np.ndarray:
    seq = np.asarray(seq)
    if seq.ndim != 1:
        raise ValueError(f"Se esperaba un vector 1D de símbolos, no {seq.shape}")
    return seq.astype(np.int64, copy=False)


@dataclass
class HMMHyperParams:
    n_states: int = 5
    n_symbols: int = 256
    smoothing: float = 1e-6
    bw_iters: int = 0
    bw_tol: float = 1e-4

    def __post_init__(self) -> None:
        if self.n_states < 2:
            raise ValueError("n_states debe ser >= 2")
        if self.n_symbols < 2:
            raise ValueError("n_symbols debe ser >= 2")
        if self.smoothing <= 0.0:
            raise ValueError("smoothing debe ser > 0")
        if self.bw_iters < 0:
            raise ValueError("bw_iters debe ser >= 0")


class DiscreteBakisHMM:
    """HMM discreto left-to-right (Bakis) con observaciones enteras 0..M-1.

    Estructura:
    - estado inicial fijo en 0
    - estados emisores [0, ..., N-1]
    - transiciones permitidas: i->i e i->i+1
    - último estado absorbente: a[N-1,N-1] = 1
    - observaciones discretas con B de forma (N, M)

    La inicialización sigue el requisito didáctico del reto:
    segmentación lineal + conteos directos de A y B + suavizado epsilon.
    """

    def __init__(self, n_states: int, n_symbols: int, smoothing: float = 1e-6) -> None:
        self.n_states = int(n_states)
        self.n_symbols = int(n_symbols)
        self.smoothing = float(smoothing)

        self.pi = np.zeros(self.n_states, dtype=np.float64)
        self.pi[0] = 1.0

        self.A = np.zeros((self.n_states, self.n_states), dtype=np.float64)
        for i in range(self.n_states - 1):
            self.A[i, i] = 0.5
            self.A[i, i + 1] = 0.5
        self.A[-1, -1] = 1.0

        self.B = np.full((self.n_states, self.n_symbols), 1.0 / self.n_symbols, dtype=np.float64)
        self._sanitize()

    def _sanitize(self) -> None:
        eps = self.smoothing
        A = np.zeros_like(self.A)
        for i in range(self.n_states):
            allowed = [i]
            if i + 1 < self.n_states:
                allowed.append(i + 1)
            vals = np.maximum(self.A[i, allowed], eps)
            vals /= np.sum(vals)
            A[i, allowed] = vals
        A[-1, :] = 0.0
        A[-1, -1] = 1.0
        self.A = A

        B = np.maximum(self.B, eps)
        B /= np.sum(B, axis=1, keepdims=True)
        self.B = B

    @staticmethod
    def _uniform_segments(T: int, n_states: int) -> List[Tuple[int, int]]:
        if T < n_states:
            raise ValueError(
                f"Secuencia demasiado corta: T={T} < n_states={n_states}. Reduce estados o filtra utterances cortos."
            )
        base = T // n_states
        rem = T % n_states
        bounds: List[Tuple[int, int]] = []
        start = 0
        for s in range(n_states):
            seg_len = base + (1 if s < rem else 0)
            end = start + seg_len
            bounds.append((start, end))
            start = end
        return bounds

    def initialize_from_counts(self, sequences: Sequence[np.ndarray]) -> None:
        sequences = [as_1d_int(seq) for seq in sequences if len(seq) > 0]
        if not sequences:
            raise ValueError("No hay secuencias para inicializar el HMM")

        A_counts = np.zeros_like(self.A)
        B_counts = np.zeros_like(self.B)

        for obs in sequences:
            T = len(obs)
            bounds = self._uniform_segments(T, self.n_states)
            for s, (a, b) in enumerate(bounds):
                seg = obs[a:b]
                if seg.size == 0:
                    continue
                binc = np.bincount(seg, minlength=self.n_symbols)
                B_counts[s] += binc

                if s < self.n_states - 1:
                    dur = len(seg)
                    A_counts[s, s] += max(dur - 1, 0)
                    A_counts[s, s + 1] += 1
                else:
                    A_counts[s, s] += max(len(seg) - 1, 0)

        eps = self.smoothing
        for i in range(self.n_states):
            allowed = [i]
            if i + 1 < self.n_states:
                allowed.append(i + 1)
            vals = A_counts[i, allowed] + eps
            vals /= np.sum(vals)
            self.A[i, allowed] = vals
        self.A[-1, :] = 0.0
        self.A[-1, -1] = 1.0

        self.B = B_counts + eps
        self.B /= np.sum(self.B, axis=1, keepdims=True)
        self._sanitize()

    @property
    def logA(self) -> np.ndarray:
        out = np.full_like(self.A, -np.inf, dtype=np.float64)
        out[self.A > 0.0] = np.log(self.A[self.A > 0.0])
        return out

    @property
    def logB(self) -> np.ndarray:
        out = np.full_like(self.B, -np.inf, dtype=np.float64)
        out[self.B > 0.0] = np.log(self.B[self.B > 0.0])
        return out

    def emission_log_probs(self, obs: np.ndarray) -> np.ndarray:
        obs = as_1d_int(obs)
        if np.any(obs < 0) or np.any(obs >= self.n_symbols):
            raise ValueError("La secuencia contiene símbolos fuera del rango del codebook")
        return self.logB[:, obs]  # (N, T)

    def forward_log(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        obs = as_1d_int(obs)
        T = len(obs)
        if T == 0:
            return np.empty((0, self.n_states), dtype=np.float64), -np.inf

        logB_t = self.emission_log_probs(obs)  # (N, T)
        alpha = np.full((T, self.n_states), -np.inf, dtype=np.float64)
        alpha[0, 0] = logB_t[0, 0]  # pi=[1,0,...]

        logA = self.logA
        for t in range(1, T):
            for j in range(self.n_states):
                terms = alpha[t - 1] + logA[:, j]
                alpha[t, j] = float(np.asarray(logsumexp(terms)).item()) + logB_t[j, t]

        loglik = float(alpha[T - 1, self.n_states - 1])
        return alpha, loglik

    def backward_log(self, obs: np.ndarray) -> np.ndarray:
        obs = as_1d_int(obs)
        T = len(obs)
        if T == 0:
            return np.empty((0, self.n_states), dtype=np.float64)

        logB_t = self.emission_log_probs(obs)
        beta = np.full((T, self.n_states), -np.inf, dtype=np.float64)
        beta[T - 1, self.n_states - 1] = 0.0
        logA = self.logA

        for t in range(T - 2, -1, -1):
            for i in range(self.n_states):
                terms = logA[i, :] + logB_t[:, t + 1] + beta[t + 1, :]
                beta[t, i] = float(np.asarray(logsumexp(terms)).item())
        return beta

    def score(self, obs: np.ndarray) -> float:
        _alpha, loglik = self.forward_log(obs)
        return loglik

    def viterbi(self, obs: np.ndarray) -> Tuple[float, np.ndarray]:
        obs = as_1d_int(obs)
        T = len(obs)
        if T == 0:
            return -np.inf, np.empty((0,), dtype=np.int64)

        logB_t = self.emission_log_probs(obs)
        delta = np.full((T, self.n_states), -np.inf, dtype=np.float64)
        psi = np.zeros((T, self.n_states), dtype=np.int64)

        delta[0, 0] = logB_t[0, 0]
        logA = self.logA

        for t in range(1, T):
            for j in range(self.n_states):
                vals = delta[t - 1] + logA[:, j]
                psi[t, j] = int(np.argmax(vals))
                delta[t, j] = vals[psi[t, j]] + logB_t[j, t]

        best = float(delta[T - 1, self.n_states - 1])
        path = np.zeros(T, dtype=np.int64)
        path[T - 1] = self.n_states - 1
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return best, path

    def baum_welch_refine(
        self,
        sequences: Sequence[np.ndarray],
        max_iters: int = 5,
        tol: float = 1e-4,
        patience: int = 3,
    ) -> List[float]:
        """Reestimación de Baum-Welch con early stopping basado en log-likelihood de entrenamiento.

        Early stopping:
            - Se detiene si la mejora absoluta en LL promedio es menor que `tol` durante
              `patience` iteraciones consecutivas sin batir el mejor valor histórico.
            - Al terminar (por patience o por max_iters), restaura los parámetros (A, B)
              de la iteración con mayor LL promedio observado.

        Args:
            sequences:  Secuencias de observaciones (enteros 0..M-1).
            max_iters:  Tope duro de iteraciones (igual que antes).
            tol:        Umbral mínimo de mejora para considerar progreso real.
            patience:   Iteraciones consecutivas sin superar el mejor LL antes de parar.
                        patience=1 reproduce el comportamiento original (para en la primera
                        iteración sin mejora).

        Returns:
            history:    Lista de LL promedios por iteración (longitud <= max_iters).
                        El índice del máximo indica la iteración óptima.
                        Ejemplo: np.argmax(history) + 1  →  mejor número de iteraciones.
        """
        sequences = [as_1d_int(seq) for seq in sequences if len(seq) >= self.n_states]
        if not sequences:
            return []

        history: List[float] = []
        eps = self.smoothing

        # ── Snapshot del mejor modelo ──────────────────────────────────────────
        best_ll: float = -np.inf
        best_A: np.ndarray = self.A.copy()
        best_B: np.ndarray = self.B.copy()
        no_improve_count: int = 0   # iteraciones consecutivas sin mejora real
        # ───────────────────────────────────────────────────────────────────────

        for iteration in range(max_iters):
            A_num = np.zeros_like(self.A)
            B_num = np.zeros_like(self.B)
            ll_total = 0.0
            used = 0

            for obs in sequences:
                alpha, loglik = self.forward_log(obs)
                if not np.isfinite(loglik):
                    continue
                beta = self.backward_log(obs)
                logB_t = self.emission_log_probs(obs)
                logA = self.logA
                T = len(obs)
                used += 1
                ll_total += loglik

                gamma = np.exp(alpha + beta - loglik)
                gamma /= np.maximum(np.sum(gamma, axis=1, keepdims=True), EPS)

                for t, sym in enumerate(obs):
                    B_num[:, sym] += gamma[t]

                for t in range(T - 1):
                    log_xi = (
                        alpha[t, :, None]
                        + logA
                        + logB_t[:, t + 1][None, :]
                        + beta[t + 1, None, :]
                        - loglik
                    )
                    xi = np.exp(log_xi)
                    denom = np.sum(xi)
                    if denom > 0.0:
                        xi /= denom
                    A_num += xi

            if used == 0:
                break

            # ── Actualizar parámetros ──────────────────────────────────────────
            for i in range(self.n_states):
                allowed = [i]
                if i + 1 < self.n_states:
                    allowed.append(i + 1)
                vals = A_num[i, allowed] + eps
                vals /= np.sum(vals)
                self.A[i, :] = 0.0
                self.A[i, allowed] = vals
            self.A[-1, :] = 0.0
            self.A[-1, -1] = 1.0

            self.B = B_num + eps
            self.B /= np.sum(self.B, axis=1, keepdims=True)
            self._sanitize()

            avg = ll_total / used
            history.append(float(avg))

            # ── Early stopping ─────────────────────────────────────────────────
            if avg > best_ll + tol:
                # Mejora real: actualizar snapshot y reiniciar contador
                best_ll = avg
                best_A = self.A.copy()
                best_B = self.B.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    # patience agotada → restaurar el mejor snapshot y salir
                    self.A = best_A
                    self.B = best_B
                    self._sanitize()
                    break
            # ───────────────────────────────────────────────────────────────────

        else:
            # Terminó el bucle por max_iters (no por break): restaurar igualmente
            self.A = best_A
            self.B = best_B
            self._sanitize()

        return history

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {
            "n_states": np.array([self.n_states], dtype=np.int64),
            "n_symbols": np.array([self.n_symbols], dtype=np.int64),
            "smoothing": np.array([self.smoothing], dtype=np.float64),
            "pi": self.pi,
            "A": self.A,
            "B": self.B,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, np.ndarray]) -> "DiscreteBakisHMM":
        n_states = int(np.asarray(payload["n_states"]).ravel()[0])
        n_symbols = int(np.asarray(payload["n_symbols"]).ravel()[0])
        smoothing = float(np.asarray(payload["smoothing"]).ravel()[0])
        model = cls(n_states=n_states, n_symbols=n_symbols, smoothing=smoothing)
        model.pi = np.asarray(payload["pi"], dtype=np.float64)
        model.A = np.asarray(payload["A"], dtype=np.float64)
        model.B = np.asarray(payload["B"], dtype=np.float64)
        model._sanitize()
        return model

    def save(self, path: str | Path) -> None:
        np.savez_compressed(str(path), **self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "DiscreteBakisHMM":
        with np.load(str(path), allow_pickle=False) as data:
            payload = {k: data[k] for k in data.files}
        return cls.from_dict(payload)


class WordHMMRecognizer:
    """Un HMM discreto por palabra; decisión por argmax de log-likelihood."""

    def __init__(self, models: Optional[Dict[str, DiscreteBakisHMM]] = None) -> None:
        self.models = models or {}

    def fit(
        self,
        train_sequences: Dict[str, Sequence[np.ndarray]],
        n_symbols: int,
        default_n_states: int = 5,
        states_per_word: Optional[Dict[str, int]] = None,
        smoothing: float = 1e-6,
        bw_iters: int = 0,
        bw_tol: float = 0.01,
        bw_patience: int = 3,
    ) -> Dict[str, List[float]]:
        self.models = {}
        histories: Dict[str, List[float]] = {}
        states_per_word = states_per_word or {}

        for word, seqs in sorted(train_sequences.items()):
            seqs = [as_1d_int(seq) for seq in seqs if len(seq) > 0]
            if not seqs:
                continue
            n_states = int(states_per_word.get(word, default_n_states))
            seqs = [seq for seq in seqs if len(seq) >= n_states]
            if not seqs:
                continue

            model = DiscreteBakisHMM(n_states=n_states, n_symbols=n_symbols, smoothing=smoothing)
            model.initialize_from_counts(seqs)
            hist: List[float] = []
            if bw_iters > 0:
                hist = model.baum_welch_refine(seqs, max_iters=bw_iters, tol=bw_tol, patience=bw_patience)
            self.models[word] = model
            histories[word] = hist
        return histories

    def score_all(self, obs: np.ndarray) -> Dict[str, float]:
        obs = as_1d_int(obs)
        return {word: model.score(obs) for word, model in self.models.items()}

    def predict(self, obs: np.ndarray) -> Tuple[str, Dict[str, float]]:
        scores = self.score_all(obs)
        if not scores:
            raise ValueError("No hay modelos entrenados")
        best_word = max(scores, key=scores.get)
        return best_word, scores

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        manifest: Dict[str, int] = {}
        for word, model in self.models.items():
            safe = word.replace("/", "_")
            path = directory / f"{safe}.npz"
            model.save(path)
            manifest[word] = model.n_states
        with open(directory / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: str | Path) -> "WordHMMRecognizer":
        directory = Path(directory)
        models: Dict[str, DiscreteBakisHMM] = {}
        for path in sorted(directory.glob("*.npz")):
            models[path.stem] = DiscreteBakisHMM.load(path)
        return cls(models=models)


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        if yt in idx and yp in idx:
            cm[idx[yt], idx[yp]] += 1
    return cm


def accuracy_from_confusion(cm: np.ndarray) -> float:
    total = int(np.sum(cm))
    if total == 0:
        return 0.0
    return float(np.trace(cm) / total)
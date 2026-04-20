"""
lpc_pipeline.py  (v2 – LSF edition)
=====================================
Pipeline de preprocesamiento y entrenamiento LPC para reconocimiento de
comandos de voz usando codebooks LBG con Line Spectral Frequencies (LSF)
y distancia Itakura-Saito.

Flujo completo:
    audio .wav
       │
       ├─ Pre-énfasis      H(z) = 1 – 0.95·z⁻¹
       ├─ VAD trim         ZCR + energía  (lógica del script MATLAB)
       ├─ Ventaneo         Hamming 320 pts, hop 128 muestras
       ├─ LPC              autocorrelación + Levinson-Durbin  (orden 12)
       ├─ LPC → LSF        raíces de los polinomios P(z) y Q(z)
       └─ LBG              codebook con distancia Itakura-Saito
                           (los LSFs se reconvierten a LPC para calcular IS)

¿Por qué LSF en lugar de coeficientes LPC crudos?
──────────────────────────────────────────────────
  • Cada par de LSFs encierra un formante → representación más interpretable.
  • Siempre ordenados  0 < ω₁ < ω₂ < … < ωₚ < π  → cuantización más estable.
  • El centroide aritmético en espacio LSF sigue siendo un LSF válido,
    lo que no está garantizado con coeficientes LPC crudos.

Sobre el orden del modelo:
──────────────────────────
  LPC_ORDER = 12  →  6 pares de LSFs  (recomendado para sistemas embebidos / 16 kHz)
  LPC_ORDER = 14  →  7 pares          (captura mejor fricativas y consonantes)
  LPC_ORDER = 16  →  8 pares          (más preciso, mayor coste computacional)
  Regla general:  p ≈ fs/1000 + 2 = 18 para cobertura espectral completa a 16 kHz,
  pero 12 es el equilibrio habitual en reconocimiento de comandos.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.signal import lfilter

# ──────────────────────────────────────────────────────────────
#  CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────

DATASET_DIR     = "dataset_coki"
OUTPUT_DIR      = "lpc_codebooks"
AUDIOS_PER_WORD = 10

PRE_EMPH_COEF   = 0.95
FRAME_LEN       = 320        # ~20 ms @ 16 kHz
HOP_LEN         = 128        # ~8  ms @ 16 kHz
LPC_ORDER       = 12         # debe ser par para la conversión LSF

ZCR_RATIO       = 0.08
ENERGY_RATIO    = 0.03
VAD_PADDING_MS  = 20

CODEBOOK_SIZE   = 16         # potencia de 2
LBG_EPSILON     = 0.01
LBG_MAX_ITER    = 100
LBG_TOL         = 1e-4


# ──────────────────────────────────────────────────────────────
#  1. PRE-ÉNFASIS
# ──────────────────────────────────────────────────────────────

def pre_emphasis(signal: np.ndarray, coef: float = PRE_EMPH_COEF) -> np.ndarray:
    """H(z) = 1 – coef·z⁻¹"""
    return lfilter([1.0, -coef], [1.0], signal)


# ──────────────────────────────────────────────────────────────
#  2. VAD
# ──────────────────────────────────────────────────────────────

def vad_trim(signal: np.ndarray, fs: int,
             zcr_ratio: float    = ZCR_RATIO,
             energy_ratio: float = ENERGY_RATIO,
             pad_ms: int         = VAD_PADDING_MS) -> np.ndarray:
    """
    Recorta silencios con ZCR + Energía (portado del MATLAB).
    Devuelve la señal original si no detecta voz.
    """
    fl = round(0.02 * fs)
    hl = round(0.01 * fs)
    n  = (len(signal) - fl) // hl

    zcr    = np.zeros(n)
    energy = np.zeros(n)
    for i in range(n):
        s         = i * hl
        frame     = signal[s : s + fl]
        zcr[i]    = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * fl)
        energy[i] = np.dot(frame, frame) / fl

    voiced = np.where(
        (zcr    > zcr_ratio    * np.max(zcr    + 1e-12)) &
        (energy > energy_ratio * np.max(energy + 1e-12))
    )[0]

    if len(voiced) == 0:
        print("    ⚠  VAD: sin voz detectada, usando señal completa")
        return signal

    pad   = int(pad_ms * fs / 1000)
    start = max(0,           voiced[0]  * hl - pad)
    end   = min(len(signal), voiced[-1] * hl + fl + pad)
    return signal[start:end]


# ──────────────────────────────────────────────────────────────
#  3. VENTANEO DE HAMMING
# ──────────────────────────────────────────────────────────────

def hamming_frames(signal: np.ndarray,
                   frame_len: int = FRAME_LEN,
                   hop_len:   int = HOP_LEN) -> np.ndarray:
    """Devuelve (N_tramas, frame_len) con ventana Hamming aplicada."""
    w  = np.hamming(frame_len)
    nf = 1 + (len(signal) - frame_len) // hop_len
    out = np.zeros((nf, frame_len))
    for i in range(nf):
        s      = i * hop_len
        out[i] = signal[s : s + frame_len] * w
    return out


# ──────────────────────────────────────────────────────────────
#  4. LPC  (autocorrelación + Levinson-Durbin)
# ──────────────────────────────────────────────────────────────

def _autocorr(frame: np.ndarray, order: int) -> np.ndarray:
    n = len(frame)
    return np.array([np.dot(frame[:n-k], frame[k:]) for k in range(order + 1)])


def levinson_durbin(r: np.ndarray, order: int):
    """
    Resuelve Yule-Walker.  Devuelve (a, E):
        a : coeficientes [a₁…aₚ]
        E : energía del error de predicción
    """
    a = np.zeros(order)
    E = r[0]
    for m in range(order):
        if abs(E) < 1e-12:
            break
        lam   = r[m + 1] + np.dot(a[:m], r[m:0:-1])
        k     = -lam / E
        a_new = a.copy()
        a_new[m] = k
        for i in range(m):
            a_new[i] = a[i] + k * a[m - 1 - i]
        a  = a_new
        E *= (1.0 - k * k)
    return a, max(E, 1e-10)


# ──────────────────────────────────────────────────────────────
#  5. CONVERSIÓN  LPC ↔ LSF
# ──────────────────────────────────────────────────────────────

def lpc_to_lsf(a: np.ndarray) -> np.ndarray:
    """
    Convierte coeficientes LPC  [a₁…aₚ]  a  LSFs  [ω₁…ωₚ]  (radianes).

    Procedimiento:
        A(z)  = 1 + a₁z⁻¹ + … + aₚz⁻ᵖ
        P(z)  = A(z) + z⁻⁽ᵖ⁺¹⁾A(z⁻¹)   →  raíz en z = −1
        Q(z)  = A(z) − z⁻⁽ᵖ⁺¹⁾A(z⁻¹)   →  raíz en z = +1
        P'(z) = P(z)/(1+z⁻¹),  Q'(z) = Q(z)/(1−z⁻¹)
        LSFs  = ángulos de las raíces de P'(z) y Q'(z) en el círculo unitario.

    Los LSFs devueltos están ordenados  0 < ω₁ < … < ωₚ < π.
    Requiere orden p par.
    """
    p     = len(a)
    A     = np.concatenate([[1.0], a])
    A_rev = A[::-1]

    P = A + A_rev          # grado p+1, raíz en z=-1
    Q = A - A_rev          # grado p+1, raíz en z=+1

    P_red = np.polydiv(P, [1.0,  1.0])[0]   # ÷ (z+1)
    Q_red = np.polydiv(Q, [1.0, -1.0])[0]   # ÷ (z-1)

    rP = np.roots(P_red)
    rQ = np.roots(Q_red)

    ang_P = np.angle(rP[np.imag(rP) >= -1e-6])
    ang_Q = np.angle(rQ[np.imag(rQ) >= -1e-6])

    lsf = np.sort(np.concatenate([ang_P, ang_Q]))
    lsf = lsf[(lsf > 1e-6) & (lsf < np.pi - 1e-6)]

    # Debe haber exactamente p LSFs
    if len(lsf) > p:
        lsf = lsf[:p]
    elif len(lsf) < p:
        lsf = np.linspace(0.05 * np.pi, 0.95 * np.pi, p)  # fallback numérico

    return lsf


def lsf_to_lpc(lsf: np.ndarray) -> np.ndarray:
    """
    Convierte LSFs  [ω₁…ωₚ]  de vuelta a coeficientes LPC  [a₁…aₚ].

    Por el teorema de entrelazado, las raíces de P' y Q' se alternan
    cuando los LSFs están ordenados:
        índices pares (0,2,4,…) → P'(z)
        índices impares (1,3,5,…) → Q'(z)

    Cada raíz compleja conjugada  e^{±jω}  aporta el factor cuadrático
    (z² − 2cos(ω)z + 1).
    """
    lsf_s = np.sort(lsf)

    P = np.array([1.0])
    for w in lsf_s[0::2]:
        P = np.convolve(P, [1.0, -2.0 * np.cos(w), 1.0])

    Q = np.array([1.0])
    for w in lsf_s[1::2]:
        Q = np.convolve(Q, [1.0, -2.0 * np.cos(w), 1.0])

    P_full = np.convolve(P, [1.0,  1.0])
    Q_full = np.convolve(Q, [1.0, -1.0])

    p = len(lsf)
    A = 0.5 * (P_full + Q_full)
    return A[1:p + 1]


# ──────────────────────────────────────────────────────────────
#  6. EXTRACCIÓN DE FEATURES POR TRAMA
# ──────────────────────────────────────────────────────────────

def extract_features(frames: np.ndarray, order: int = LPC_ORDER) -> list:
    """
    Para cada trama:
        1. Autocorrelación r[0..p]
        2. Levinson-Durbin → coeficientes LPC + error E
        3. LPC → LSF

    Devuelve lista de dicts:
        { 'lsf': array(p,),  'a': array(p,),  'E': float,  'r': array(p+1,) }

    Los campos 'a', 'E', 'r' se conservan para calcular la distancia
    Itakura-Saito directamente en espacio LPC.
    """
    features = []
    for frame in frames:
        if np.max(np.abs(frame)) < 1e-6:
            continue
        r    = _autocorr(frame, order)
        a, E = levinson_durbin(r, order)
        lsf  = lpc_to_lsf(a)
        features.append({"lsf": lsf, "a": a, "E": E, "r": r})
    return features


# ──────────────────────────────────────────────────────────────
#  7. DISTANCIA ITAKURA-SAITO  (calculada en espacio LPC)
# ──────────────────────────────────────────────────────────────

def _toeplitz(r: np.ndarray) -> np.ndarray:
    p1  = len(r)
    idx = np.abs(np.arange(p1)[:, None] - np.arange(p1))
    return r[idx]


def itakura_saito(feat_a: dict, feat_b: dict) -> float:
    """
    Distancia IS asimétrica  d(a → b):

        d = (bᵀ·Rₐ·b) / Eₐ  −  log[(bᵀ·Rₐ·b) / Eₐ]  −  1

    Si el codevector viene del codebook (cargado desde .npz), sus 'a' y 'r'
    se reconstruyen desde los LSFs almacenados.
    """
    a_b = feat_b.get("a")
    if a_b is None:
        a_b = lsf_to_lpc(feat_b["lsf"])

    r_a    = feat_a["r"]
    E_a    = feat_a["E"]
    b_full = np.concatenate([[1.0], a_b])
    R_a    = _toeplitz(r_a)
    num    = b_full @ R_a @ b_full

    ratio  = num / E_a
    return (ratio - np.log(ratio) - 1.0) if ratio > 0 else 1e9


def itakura_saito_sym(feat_a: dict, feat_b: dict) -> float:
    """IS simétrica: promedio de d(a→b) y d(b→a)."""
    return 0.5 * (itakura_saito(feat_a, feat_b) + itakura_saito(feat_b, feat_a))


# ──────────────────────────────────────────────────────────────
#  8. ALGORITMO LBG con vectores LSF
# ──────────────────────────────────────────────────────────────

def _centroid(cluster: list) -> dict:
    """
    Centroide aritmético en espacio LSF.

    Ventaja clave: el promedio de vectores LSF ordenados en (0,π) sigue
    siendo un vector LSF válido (continuo y acotado), a diferencia de los
    coeficientes LPC crudos cuyo promedio puede producir filtros inestables.
    Los campos 'a', 'r', 'E' se reconstruyen desde el LSF promedio.
    """
    lsf_mean = np.mean([f["lsf"] for f in cluster], axis=0)
    a_mean   = lsf_to_lpc(lsf_mean)
    r_mean   = np.mean([f["r"] for f in cluster], axis=0)
    E_mean   = float(np.mean([f["E"] for f in cluster]))
    return {"lsf": lsf_mean, "a": a_mean, "E": E_mean, "r": r_mean}


def lbg_train(features: list,
              codebook_size: int   = CODEBOOK_SIZE,
              epsilon:       float = LBG_EPSILON,
              max_iter:      int   = LBG_MAX_ITER,
              tol:           float = LBG_TOL) -> list:
    """
    Algoritmo LBG:
        • Vectores en espacio LSF
        • Distancia Itakura-Saito simétrica
        • Centroide como promedio de LSFs
        • Perturbación proporcional a cada frecuencia LSF (respeta el orden)

    Pasos:
        1. Inicializar con centroide global.
        2. Split ±ε·ω por codevector.
        3. K-means IS hasta convergencia.
        4. Repetir hasta codebook_size.
    """
    if not features:
        raise ValueError("Lista de features vacía.")

    codebook = [_centroid(features)]
    print(f"  → LBG start: 1 codevector | {len(features)} frames de entrenamiento")

    while len(codebook) < codebook_size:
        # ── Split ──────────────────────────────────────────
        new_cb = []
        for cv in codebook:
            lsf_p = np.sort(np.clip(cv["lsf"] * (1 + epsilon), 1e-4, np.pi - 1e-4))
            lsf_m = np.sort(np.clip(cv["lsf"] * (1 - epsilon), 1e-4, np.pi - 1e-4))
            new_cb.append({"lsf": lsf_p, "a": lsf_to_lpc(lsf_p),
                           "E": cv["E"], "r": cv["r"]})
            new_cb.append({"lsf": lsf_m, "a": lsf_to_lpc(lsf_m),
                           "E": cv["E"], "r": cv["r"]})
        codebook = new_cb

        # ── K-means IS ─────────────────────────────────────
        prev_dist = float("inf")
        for it in range(max_iter):
            clusters   = [[] for _ in range(len(codebook))]
            total_dist = 0.0

            for feat in features:
                dists = [itakura_saito_sym(feat, cv) for cv in codebook]
                best  = int(np.argmin(dists))
                clusters[best].append(feat)
                total_dist += dists[best]

            for i, cl in enumerate(clusters):
                if cl:
                    codebook[i] = _centroid(cl)

            avg_dist = total_dist / len(features)
            if abs(prev_dist - avg_dist) / (prev_dist + 1e-10) < tol:
                break
            prev_dist = avg_dist

        print(f"  CB size {len(codebook):3d} | avg IS dist = {avg_dist:.6f} | iters = {it + 1}")

    return codebook


# ──────────────────────────────────────────────────────────────
#  9. PIPELINE POR PALABRA
# ──────────────────────────────────────────────────────────────

def process_word(word_dir: Path, word: str,
                 n_audios: int = AUDIOS_PER_WORD) -> list:
    """
    Aplica el pipeline completo a los primeros n_audios de una carpeta.
    Devuelve lista de features {lsf, a, E, r}.
    """
    wav_files = sorted(word_dir.glob("*.wav"))[:n_audios]
    if not wav_files:
        print(f"  ✗  Sin archivos .wav en {word_dir}")
        return []

    print(f"\n── [{word.upper()}]  {len(wav_files)} archivos ──")
    all_features = []

    for wav_path in wav_files:
        signal, fs = sf.read(str(wav_path))
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
        signal = signal.astype(np.float64)

        signal = pre_emphasis(signal)
        signal = vad_trim(signal, fs)

        if len(signal) < FRAME_LEN:
            print(f"    ⚠  {wav_path.name}: muy corta tras VAD, saltando")
            continue

        mx = np.max(np.abs(signal))
        if mx > 0:
            signal /= mx

        frames = hamming_frames(signal)
        feats  = extract_features(frames)
        all_features.extend(feats)

        print(f"    {wav_path.name}  →  {len(frames)} tramas / {len(feats)} frames voz")

    print(f"  Total LSF features para '{word}': {len(all_features)} frames")
    return all_features


# ──────────────────────────────────────────────────────────────
#  10. CARGA DE CODEBOOKS Y RECONOCIMIENTO
# ──────────────────────────────────────────────────────────────

def load_codebooks(output_dir: Path) -> dict:
    """Carga codebooks guardados como .npz (LSF + E + r)."""
    codebooks = {}
    for npz in sorted(output_dir.glob("codebook_*.npz")):
        word = npz.stem.replace("codebook_", "")
        d    = np.load(str(npz))
        cb   = []
        for i in range(len(d["lsf"])):
            lsf_i = d["lsf"][i]
            cb.append({"lsf": lsf_i, "a": lsf_to_lpc(lsf_i),
                       "E": float(d["E"][i]), "r": d["r"][i]})
        codebooks[word] = cb
    return codebooks


def distorsion_codebook(features: list, codebook: list) -> float:
    """Distorsión IS promedio de features contra un codebook."""
    if not features:
        return float("inf")
    return sum(
        min(itakura_saito_sym(f, cv) for cv in codebook)
        for f in features
    ) / len(features)


def recognize(audio_path: str, codebooks: dict,
              lpc_order: int = LPC_ORDER) -> str:
    """
    Clasifica un audio nuevo contra todos los codebooks disponibles.
    Devuelve la palabra con menor distorsión IS promedio.
    """
    signal, fs = sf.read(audio_path)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    signal = signal.astype(np.float64)

    signal = pre_emphasis(signal)
    signal = vad_trim(signal, fs)
    mx = np.max(np.abs(signal))
    if mx > 0:
        signal /= mx

    frames   = hamming_frames(signal)
    features = extract_features(frames, lpc_order)
    scores   = {w: distorsion_codebook(features, cb) for w, cb in codebooks.items()}
    best     = min(scores, key=scores.get)

    print(f"\n  Resultado: '{best}'")
    print("  Distorsiones:")
    for w, d in sorted(scores.items(), key=lambda x: x[1]):
        mark = "  ◀" if w == best else ""
        print(f"    {w:<12}  IS = {d:.6f}{mark}")

    return best


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────

def main():
    dataset_path = Path(DATASET_DIR)
    output_path  = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)

    word_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    if not word_dirs:
        print(f"No se encontraron subcarpetas en '{DATASET_DIR}'.")
        return

    print(f"Palabras : {[d.name for d in word_dirs]}")
    print(f"Orden LPC: {LPC_ORDER}  →  {LPC_ORDER} LSFs por trama  "
          f"({LPC_ORDER // 2} pares de formantes)")
    print(f"Codebook : {CODEBOOK_SIZE} codevectores  |  LBG ε = {LBG_EPSILON}")
    print(f"Frame    : {FRAME_LEN} muestras / hop {HOP_LEN} muestras\n")

    for word_dir in word_dirs:
        word     = word_dir.name
        features = process_word(word_dir, word)

        if len(features) < CODEBOOK_SIZE * 2:
            print(f"  ✗  Insuficientes features para '{word}', saltando")
            continue

        print(f"\n  Entrenando LBG para '{word}'...")
        codebook = lbg_train(features)

        npz_path = output_path / f"codebook_{word}.npz"
        np.savez(
            str(npz_path),
            lsf = np.array([cv["lsf"] for cv in codebook]),
            E   = np.array([cv["E"]   for cv in codebook]),
            r   = np.array([cv["r"]   for cv in codebook]),
        )
        print(f"  ✓  {npz_path}   "
              f"({CODEBOOK_SIZE} codevectores × {LPC_ORDER} LSFs)")

    print(f"\n{'─'*50}")
    print(f"Pipeline completado. Codebooks en: {output_path}/")
    print("\nPara reconocer:")
    print("    cbs  = load_codebooks(Path('lpc_codebooks'))")
    print("    word = recognize('audio.wav', cbs)")


if __name__ == "__main__":
    main()
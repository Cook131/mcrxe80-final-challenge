"""
train_yolov8.py
───────────────────────────────────────────────────────────────
YOLOv8-m Training Script  —  Cookie / E80 Group R&D
Optimizado para detección de patrones con colores similares.

Instalación previa:
    pip install ultralytics roboflow matplotlib seaborn
───────────────────────────────────────────────────────────────
"""

# ── 0. Imports ────────────────────────────────────────────────
from pathlib import Path
import shutil
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────
# 1. CONFIGURACIÓN — edita solo este bloque
# ─────────────────────────────────────────────────────────────
CFG = {
    # Ruta al data.yaml exportado desde Roboflow (export → YOLOv8)
    "data_yaml"    : "dataset/data.yaml",

    # Directorio donde se guardarán runs/pesos
    "project"      : "runs/train",
    "run_name"     : "yolov8m_colorpatterns_v1",

    # Modelo base (descarga automática si no existe)
    "model_weights": "yolov8m.pt",

    # ── Hiperparámetros de entrenamiento ──────────────────────
    "epochs"       : 120,
    "imgsz"        : 640,
    "batch"        : 16,          # baja a 8 si te quedas sin VRAM
    "workers"      : 4,
    "device"       : 0,           # GPU 0; pon "cpu" si no tienes GPU
    "patience"     : 25,          # early stopping

    # ── Augmentaciones de color (claves para colores similares) ─
    # Hue, Saturation, Value: rangos de variación aleatoria
    "hsv_h"        : 0.02,        # ±2% del hue  (más = más agresivo)
    "hsv_s"        : 0.75,        # ±75% saturación
    "hsv_v"        : 0.45,        # ±45% brillo

    # Fuerza al modelo a aprender FORMA, no solo color
    "grayscale"    : 0.15,        # 15% de imágenes en escala de grises

    # Mezcla dos imágenes para regularizar clases confundibles
    "mixup"        : 0.15,

    # Mosaic: combina 4 imágenes en una → más variedad de contexto
    "mosaic"       : 1.0,
    "close_mosaic" : 10,          # desactiva mosaic en las últimas N epochs

    # ── IMPORTANTE: rect=False ─────────────────────────────────
    # True  → imágenes rectangulares (inferencia más rápida, menos contexto)
    # False → imágenes cuadradas 640x640 (más contexto espacial, mejor para
    #         patrones que se parecen — el modelo ve más entorno)
    "rect"         : False,

    # Optimizer
    "optimizer"    : "AdamW",
    "lr0"          : 0.001,
    "lrf"          : 0.01,        # lr final = lr0 * lrf
    "weight_decay" : 0.0005,
    "warmup_epochs": 5,

    # Otras augmentaciones útiles
    "fliplr"       : 0.5,
    "flipud"       : 0.1,
    "degrees"      : 5.0,         # rotación leve
    "translate"    : 0.1,
    "scale"        : 0.5,
    "shear"        : 2.0,
}

# ─────────────────────────────────────────────────────────────
# 2. ENTRENAMIENTO PRINCIPAL
# ─────────────────────────────────────────────────────────────
def train():
    print("\n" + "="*60)
    print("  YOLOv8-m  —  Entrenamiento iniciado")
    print("="*60 + "\n")

    model = YOLO(CFG["model_weights"])

    results = model.train(
        data         = CFG["data_yaml"],
        epochs       = CFG["epochs"],
        imgsz        = CFG["imgsz"],
        batch        = CFG["batch"],
        workers      = CFG["workers"],
        device       = CFG["device"],
        patience     = CFG["patience"],
        project      = CFG["project"],
        name         = CFG["run_name"],
        exist_ok     = True,

        # Augmentaciones de color
        hsv_h        = CFG["hsv_h"],
        hsv_s        = CFG["hsv_s"],
        hsv_v        = CFG["hsv_v"],
        grayscale    = CFG["grayscale"],
        mixup        = CFG["mixup"],
        mosaic       = CFG["mosaic"],
        close_mosaic = CFG["close_mosaic"],

        # Contexto espacial
        rect         = CFG["rect"],       # False → cuadrado, más contexto

        # Optimizer
        optimizer    = CFG["optimizer"],
        lr0          = CFG["lr0"],
        lrf          = CFG["lrf"],
        weight_decay = CFG["weight_decay"],
        warmup_epochs= CFG["warmup_epochs"],

        # Otras augmentaciones
        fliplr       = CFG["fliplr"],
        flipud       = CFG["flipud"],
        degrees      = CFG["degrees"],
        translate    = CFG["translate"],
        scale        = CFG["scale"],
        shear        = CFG["shear"],

        # Guardar el mejor modelo
        save         = True,
        save_period  = 10,           # guarda checkpoint cada 10 epochs
    )

    best_weights = Path(CFG["project"]) / CFG["run_name"] / "weights" / "best.pt"
    print(f"\n[OK] Entrenamiento terminado. Mejor modelo: {best_weights}\n")
    return best_weights


# ─────────────────────────────────────────────────────────────
# 3. VALIDACIÓN + CONFUSION MATRIX DETALLADA
#    → te muestra qué pares de clases se confunden entre sí
# ─────────────────────────────────────────────────────────────
def validate_and_plot_confusion(weights_path: Path):
    """
    Carga el mejor modelo, corre validación y grafica la
    confusion matrix con mAP por clase para identificar
    qué clases necesitan más datos.
    """
    print("\n" + "="*60)
    print("  Validación y análisis de clases confundibles")
    print("="*60 + "\n")

    model = YOLO(str(weights_path))

    # Validación (genera confusion_matrix.png en el run dir)
    metrics = model.val(
        data    = CFG["data_yaml"],
        imgsz   = CFG["imgsz"],
        batch   = CFG["batch"],
        device  = CFG["device"],
        plots   = True,            # genera confusion matrix, PR curve, etc.
        save_json= True,
    )

    # ── mAP50 por clase ───────────────────────────────────────
    class_names = metrics.names          # dict {0: 'clase_A', 1: 'clase_B', ...}
    map50_per_class = metrics.box.ap50   # array con mAP50 de cada clase

    print("\n[mAP50 por clase]")
    print("-" * 35)
    sorted_classes = sorted(
        zip(class_names.values(), map50_per_class),
        key=lambda x: x[1]
    )
    for name, score in sorted_classes:
        bar = "█" * int(score * 20)
        flag = "  <- necesita mas datos" if score < 0.60 else ""
        print(f"  {name:<20} {score:.3f}  {bar}{flag}")

    print(f"\n  mAP50 global : {metrics.box.map50:.3f}")
    print(f"  mAP50-95     : {metrics.box.map:.3f}\n")

    # ── Clases con bajo rendimiento ───────────────────────────
    weak_classes = [n for n, s in sorted_classes if s < 0.60]
    if weak_classes:
        print("[!] Clases con mAP50 < 0.60 (priorizar en dataset):")
        for c in weak_classes:
            print(f"     -> {c}")
        print()
    else:
        print("[OK] Todas las clases superan mAP50 = 0.60\n")

    # ── Leer confusion matrix guardada por Ultralytics ────────
    run_dir = Path(CFG["project"]) / CFG["run_name"]
    cm_path = run_dir / "confusion_matrix_normalized.png"
    if cm_path.exists():
        print(f"[DIR] Confusion matrix guardada en: {cm_path}")
    else:
        print(f"[DIR] Revisa el directorio: {run_dir}")

    return metrics, weak_classes


# ─────────────────────────────────────────────────────────────
# 4. HELPER: BALANCEO DE CLASES CONFUNDIBLES
#    Muestra las imágenes donde dos clases aparecen juntas
#    para que puedas verificar que el dataset las distingue bien
# ─────────────────────────────────────────────────────────────
def audit_cooccurrence(class_a: str, class_b: str):
    """
    Busca en el dataset de entrenamiento qué imágenes
    contienen simultáneamente class_a y class_b.

    Úsalo después de ver la confusion matrix para identificar
    imágenes donde el modelo puede confundirse.

    Parámetros:
        class_a, class_b: nombres de clases tal como están en data.yaml
    """
    with open(CFG["data_yaml"]) as f:
        data = yaml.safe_load(f)

    names = data["names"]  # lista de nombres de clases
    try:
        idx_a = names.index(class_a)
        idx_b = names.index(class_b)
    except ValueError as e:
        print(f"[ERROR] Clase no encontrada: {e}")
        print(f"   Clases disponibles: {names}")
        return

    train_labels = Path(data.get("train", "")).parent / "labels"
    if not train_labels.exists():
        # Intenta ruta relativa al yaml
        yaml_dir = Path(CFG["data_yaml"]).parent
        train_labels = yaml_dir / "train" / "labels"

    cooccurring = []
    for label_file in train_labels.glob("*.txt"):
        lines = label_file.read_text().strip().splitlines()
        classes_in_img = {int(l.split()[0]) for l in lines if l}
        if idx_a in classes_in_img and idx_b in classes_in_img:
            cooccurring.append(label_file.stem)

    print(f"\n[INFO] Imagenes con '{class_a}' Y '{class_b}' juntas: {len(cooccurring)}")
    if len(cooccurring) < 20:
        print("   [!] Pocas co-ocurrencias -- añade mas imagenes donde ambas clases")
        print("   aparezcan juntas para que el modelo aprenda a diferenciarlas.")
    for stem in cooccurring[:10]:
        print(f"   • {stem}")
    if len(cooccurring) > 10:
        print(f"   ... y {len(cooccurring)-10} más")


# ─────────────────────────────────────────────────────────────
# 5. INFERENCIA RÁPIDA (verificación visual post-entrenamiento)
# ─────────────────────────────────────────────────────────────
def quick_infer(weights_path: Path, source: str = "dataset/test/images"):
    """
    Corre inferencia sobre imágenes de test y guarda resultados
    con bounding boxes dibujados.

    source puede ser:
        - carpeta: "dataset/test/images"
        - imagen:  "mi_imagen.jpg"
        - webcam:  0
    """
    model = YOLO(str(weights_path))
    results = model.predict(
        source      = source,
        imgsz       = CFG["imgsz"],
        device      = CFG["device"],
        conf        = 0.25,
        iou         = 0.45,
        save        = True,
        save_txt    = True,
        project     = "runs/predict",
        name        = CFG["run_name"],
    )
    print(f"\n[OK] Predicciones guardadas en runs/predict/{CFG['run_name']}/")
    return results


# ─────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Paso 1: Entrenar ──────────────────────────────────────
    best = train()

    # ── Paso 2: Validar y ver qué clases fallan ───────────────
    metrics, weak = validate_and_plot_confusion(best)

    # ── Paso 3: Si hay clases confundibles, auditar co-ocurrencias
    # Descomenta y pon los nombres reales de tus clases:
    # audit_cooccurrence("patron_A", "patron_B")

    # ── Paso 4: Inferencia visual ─────────────────────────────
    # quick_infer(best, source="dataset/test/images")

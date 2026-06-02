#!/usr/bin/env python3
"""
calibrar.py — Calibración de cámara con el método de Zhang + Levenberg-Marquardt
Uso:
    python calibrar.py --input ./calib_imgs --rows 8 --cols 6 --square 25
    (rows/cols = esquinas internas del checkerboard, square = tamaño del cuadro en mm)
"""

import cv2
import numpy as np
import argparse
import glob
import json
import os
from pathlib import Path


# ─── Argumentos ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Calibración con Zhang + LM")
    p.add_argument("--input",  default="./calib_imgs", help="Carpeta con imágenes del checkerboard")
    p.add_argument("--output", default="./camera_params.json", help="Archivo de salida con parámetros")
    p.add_argument("--rows",   type=int, default=7,  help="Esquinas internas por fila")
    p.add_argument("--cols",   type=int, default=5,  help="Esquinas internas por columna")
    p.add_argument("--square", type=float, default=30.0, help="Tamaño del cuadro en mm")
    p.add_argument("--ext",    default="jpg,jpeg,png,JPG,JPEG,PNG", help="Extensiones a buscar")
    p.add_argument("--show",   action="store_true", help="Mostrar imágenes con esquinas detectadas")
    return p.parse_args()


# ─── Utilidades ──────────────────────────────────────────────────────────────

def find_images(folder: str, extensions: str) -> list[str]:
    imgs = []
    for ext in extensions.split(","):
        imgs += glob.glob(os.path.join(folder, f"*.{ext}"))
    imgs.sort()
    return imgs


def draw_reprojection_error(K, dist, rvecs, tvecs, obj_pts, img_pts):
    """Calcula y muestra el error de reproyección por imagen."""
    errors = []
    for i in range(len(obj_pts)):
        proj, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(img_pts[i], proj, cv2.NORM_L2) / len(proj)
        errors.append(err)
    mean_err = np.mean(errors)
    print(f"\n  Errores de reproyección por imagen:")
    for i, e in enumerate(errors):
        flag = "  ✓" if e < 1.0 else "  ⚠"
        print(f"    Img {i+1:02d}: {e:.4f} px{flag}")
    print(f"\n  Error medio total: {mean_err:.4f} px")
    if mean_err > 1.5:
        print("  ⚠  Error alto — considera descartar imágenes con patrón mal enfocado o distorsionado")
    else:
        print("  ✓  Calibración dentro de parámetros aceptables")
    return mean_err


# ─── Pipeline principal ───────────────────────────────────────────────────────

def calibrate(args):
    PATTERN = (args.rows, args.cols)
    SQUARE_SIZE = args.square

    print(f"\n{'='*60}")
    print("  CALIBRACIÓN DE CÁMARA — Método de Zhang + LM")
    print(f"{'='*60}")
    print(f"  Patrón:      {PATTERN[0]} x {PATTERN[1]} esquinas internas")
    print(f"  Cuadro:      {SQUARE_SIZE} mm")
    print(f"  Input:       {args.input}")

    # Puntos 3D del tablero en coordenadas del mundo (Z=0)
    obj_p = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2)
    obj_p *= SQUARE_SIZE

    obj_points = []  # puntos 3D reales
    img_points = []  # puntos 2D en imagen
    image_size = None

    images = find_images(args.input, args.ext)
    if not images:
        print(f"\n  ✗ No se encontraron imágenes en '{args.input}'")
        return

    print(f"\n  Imágenes encontradas: {len(images)}")
    print(f"\n  Detectando esquinas del checkerboard...")

    success_count = 0
    for path in images:
        img = cv2.imread(path)
        if img is None:
            print(f"    ✗ No se pudo leer: {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        ret, corners = cv2.findChessboardCorners(gray, PATTERN, None)

        if ret:
            # Refinar esquinas con sub-pixel accuracy (prepara para LM)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            obj_points.append(obj_p)
            img_points.append(corners_refined)
            success_count += 1
            print(f"    ✓ {Path(path).name}")

            if args.show:
                vis = img.copy()
                cv2.drawChessboardCorners(vis, PATTERN, corners_refined, ret)
                cv2.imshow(f"Esquinas — {Path(path).name}", vis)
                cv2.waitKey(500)
        else:
            print(f"    ✗ {Path(path).name}  (no se detectó el patrón)")

    if args.show:
        cv2.destroyAllWindows()

    if success_count < 3:
        print(f"\n  ✗ Se necesitan al menos 3 imágenes válidas (encontradas: {success_count})")
        return

    print(f"\n  Imágenes usadas para calibración: {success_count}/{len(images)}")
    print(f"\n  Ejecutando calibración Zhang + Levenberg-Marquardt...")

    # Flags de OpenCV: cv2.CALIB_USE_LU fuerza LM internamente en calibrateCamera
    # Por defecto ya usa LM; CALIB_RATIONAL_MODEL añade k3,k4,k5,k6 (no necesario si la distorsión es baja)
    flags = 0  # puedes añadir cv2.CALIB_FIX_K3 si sabes que la distorsión es baja

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size,
        None, None,
        flags=flags
    )

    print(f"\n  {'='*40}")
    print(f"  MATRIZ INTRÍNSECA K:")
    print(f"  {'='*40}")
    print(f"  fx = {K[0,0]:.4f} px")
    print(f"  fy = {K[1,1]:.4f} px")
    print(f"  cx = {K[0,2]:.4f} px  (centro óptico x)")
    print(f"  cy = {K[1,2]:.4f} px  (centro óptico y)")
    print(f"\n  COEFICIENTES DE DISTORSIÓN:")
    print(f"  k1={dist[0,0]:.6f}  k2={dist[0,1]:.6f}  p1={dist[0,2]:.6f}  p2={dist[0,3]:.6f}", end="")
    if dist.shape[1] > 4:
        print(f"  k3={dist[0,4]:.6f}")
    else:
        print()

    mean_err = draw_reprojection_error(K, dist, rvecs, tvecs, obj_points, img_points)

    # Guardar parámetros
    params = {
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.tolist(),
        "image_size": list(image_size),
        "reprojection_error_px": float(mean_err),
        "calibration_images_used": success_count,
        "checkerboard_pattern": list(PATTERN),
        "square_size_mm": SQUARE_SIZE
    }

    with open(args.output, "w") as f:
        json.dump(params, f, indent=2)

    # También guardar en .npz para carga rápida
    npz_path = args.output.replace(".json", ".npz")
    np.savez(npz_path, K=K, dist=dist, image_size=np.array(image_size))

    print(f"\n  Parámetros guardados en:")
    print(f"    → {args.output}")
    print(f"    → {npz_path}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    calibrate(parse_args())

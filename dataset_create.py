import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import os

# CONFIGURACIÓN
SAMPLE_RATE = 16000   
DURATION = 2          # segundos
SAMPLES_PER_COMMAND = 25

nombre_usuario = "coki"

comandos = [
    "avanza", "detente", "izquierda", "derecha", "atras",
    "arriba", "abajo", "gira", "toma", "suelta"
]

# CREAR ESTRUCTURA
base_folder = f"dataset_{nombre_usuario}"
os.makedirs(base_folder, exist_ok=True)

for comando in comandos:
    os.makedirs(os.path.join(base_folder, comando), exist_ok=True)

print("\n=== GENERADOR DE DATASET ===\n")
input("Presiona ENTER para comenzar...")

# GRABACIÓN
for comando in comandos:

    print(f"\n=== COMANDO: {comando} ===")

    for i in range(1, SAMPLES_PER_COMMAND + 1):

        filename = f"{comando}_{nombre_usuario}_{i}.wav"
        filepath = os.path.join(base_folder, comando, filename)

        print(f"\nGrabación {i}/{SAMPLES_PER_COMMAND}")

        # Instrucciones claras al usuario
        print("Prepárate...")
        time.sleep(0.5)

        print("Silencio...")
        time.sleep(0.5)

        print(f">>> DI: {comando} <<<")

        # Grabar audio
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32'
        )
        sd.wait()

        print("Fin de grabación")

        # Normalizar 
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # Guardar
        sf.write(filepath, audio, SAMPLE_RATE)

        print(f"Guardado: {filepath}")

print("\n=== DATASET COMPLETADO ===")
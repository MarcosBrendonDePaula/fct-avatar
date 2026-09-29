"""Baixa os modelos do MediaPipe para models/.

Eles nao ficam no git (tem ~11 MB somados), entao quem clonar o repo roda
isto uma vez antes do primeiro preview.
"""

import urllib.request
from pathlib import Path

BASE = "https://storage.googleapis.com/mediapipe-models"
MODELOS = {
    "face_landmarker.task":
        f"{BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker.task":
        f"{BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}

destino = Path(__file__).resolve().parent.parent / "models"
destino.mkdir(exist_ok=True)

for nome, url in MODELOS.items():
    alvo = destino / nome
    if alvo.exists():
        print(f"{nome}: ja existe")
        continue
    print(f"{nome}: baixando...")
    urllib.request.urlretrieve(url, alvo)
    print(f"{nome}: {alvo.stat().st_size / 1e6:.1f} MB")

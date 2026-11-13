import hashlib
import os
import json
import argparse
from pathlib import Path

def huella(ruta: Path, algoritmo: str = "sha256", bloque: int = 1 << 20) -> str:
    """Devuelve el resumen hexadecimal del archivo , leyendolo de a 1 MiB."""
    h = hashlib.new(algoritmo)
    with open(ruta, "rb") as f:
        for trozo in iter(lambda: f.read(bloque), b""):
            h.update(trozo)
    return h.hexdigest()




def create_checksum(json_path: Path, datos_path: Path):
    checksum_data = {}
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                checksum_data = json.load(f)
        except json.JSONDecodeError:
            checksum_data = {}

    checksum_data[datos_path.name] = huella(datos_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(checksum_data, f, indent=4)


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


parser = argparse.ArgumentParser(description="Se crea el checksum del dataset_consolidado.csv")
parser.add_argument("--path",type=str,
                    help='direccion del dataset')

args = parser.parse_args()

datos_path = Path(args.path)

checksum_data = {
    datos_path.name : huella(datos_path),
} 

with open(r"data\checksum.json", "w") as f:
    json.dump(checksum_data, f)


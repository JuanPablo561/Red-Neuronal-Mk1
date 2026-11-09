"""Servidor web que sirve la interfaz y clasifica prendas de ropa con la CNN.

Uso: python servidor.py   y abre http://127.0.0.1:8000 en el navegador.
Antes debes haber entrenado el modelo con: python entrenar.py
"""

from pathlib import Path

import torch
from flask import Flask, jsonify, request, send_from_directory

from modelo import CNN, DESVIACION, MEDIA

CARPETA = Path(__file__).parent
RUTA_MODELO = CARPETA / "fashion_cnn.pt"

if not RUTA_MODELO.exists():
    raise SystemExit("No se encontró fashion_cnn.pt. Primero ejecuta: python entrenar.py")

# Cargar el modelo entrenado (una sola vez, al iniciar el servidor)
modelo = CNN()
modelo.load_state_dict(torch.load(RUTA_MODELO, map_location="cpu", weights_only=True))
modelo.eval()  # desactiva el dropout

app = Flask(__name__)


@app.get("/")
def inicio():
    return send_from_directory(CARPETA, "index.html")


@app.post("/predecir")
def predecir():
    datos = request.get_json(silent=True)
    pixeles = datos.get("pixeles") if isinstance(datos, dict) else None

    if not isinstance(pixeles, list) or len(pixeles) != 784:
        return jsonify(error="Se esperaban 784 píxeles (imagen de 28x28)."), 400

    try:
        x = torch.tensor(pixeles, dtype=torch.float32).reshape(1, 1, 28, 28)
    except (TypeError, ValueError):
        return jsonify(error="Los píxeles deben ser números."), 400

    x = (x - MEDIA) / DESVIACION  # misma normalización del entrenamiento

    with torch.no_grad():
        probabilidades = torch.softmax(modelo(x), dim=1)[0]

    return jsonify(
        digito=int(probabilidades.argmax()),
        probabilidades=[round(p, 4) for p in probabilidades.tolist()],
    )


if __name__ == "__main__":
    print("Abre http://127.0.0.1:8000 en tu navegador")
    app.run(host="127.0.0.1", port=8000)
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from modelo import CNN, DESVIACION, MEDIA

EPOCAS = 15
TAMANO_LOTE = 128
TASA_MAXIMA = 0.003
CARPETA = Path(__file__).parent


def main():
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando: {dispositivo}")

    transformar_entrenamiento = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((MEDIA,), (DESVIACION,)),
        transforms.RandomErasing(p=0.25),
    ])
    transformar_prueba = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MEDIA,), (DESVIACION,)),
    ])

    carpeta_datos = CARPETA / "datos"
    entrenamiento = datasets.FashionMNIST(carpeta_datos, train=True, download=True,
                                          transform=transformar_entrenamiento)
    prueba = datasets.FashionMNIST(carpeta_datos, train=False, download=True,
                                   transform=transformar_prueba)

    cargador_entrenamiento = DataLoader(entrenamiento, batch_size=TAMANO_LOTE, shuffle=True)
    cargador_prueba = DataLoader(prueba, batch_size=1000)

    modelo = CNN().to(dispositivo)
    funcion_perdida = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=TASA_MAXIMA)
    # La tasa de aprendizaje sube al principio y baja suavemente hasta casi cero:
    # así la red afina los pesos al final y suele ganar 1-2 puntos de precisión.
    programador = torch.optim.lr_scheduler.OneCycleLR(
        optimizador, max_lr=TASA_MAXIMA,
        steps_per_epoch=len(cargador_entrenamiento), epochs=EPOCAS,
    )

    def evaluar():
        modelo.eval()
        correctas = 0
        with torch.no_grad():
            for imagenes, etiquetas in cargador_prueba:
                imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)
                correctas += (modelo(imagenes).argmax(dim=1) == etiquetas).sum().item()
        return correctas / len(prueba)

    mejor = 0.0
    ruta = CARPETA / "fashion_cnn.pt"

    for epoca in range(1, EPOCAS + 1):
        modelo.train()
        for i, (imagenes, etiquetas) in enumerate(cargador_entrenamiento):
            imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)

            perdida = funcion_perdida(modelo(imagenes), etiquetas)
            optimizador.zero_grad()
            perdida.backward()
            optimizador.step()
            programador.step()

            if i % 100 == 0:
                print(f"Época {epoca}/{EPOCAS} | lote {i:4d} | error: {perdida.item():.4f}")

        precision = evaluar()
        print(f"==> Época {epoca} terminada | precisión en prueba: {precision:.2%}")
        # Guardamos solo cuando mejora, así nos quedamos con la mejor versión
        if precision > mejor:
            mejor = precision
            torch.save(modelo.state_dict(), ruta)
            print(f"    (nuevo mejor modelo guardado)\n")
        else:
            print()

    print(f"Mejor precisión: {mejor:.2%}")
    print(f"Modelo guardado en {ruta}")
    print("Ahora ejecuta: python servidor.py")


if __name__ == "__main__":
    main()
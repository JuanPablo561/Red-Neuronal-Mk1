from torch import nn

# Valores de normalización de Fashion-MNIST (media y desviación de todos los píxeles)
MEDIA = 0.2860
DESVIACION = 0.3530


def bloque(entrada, salida):
    return nn.Sequential(
        nn.Conv2d(entrada, salida, kernel_size=3, padding=1),
        nn.BatchNorm2d(salida),   # estabiliza y acelera el entrenamiento
        nn.ReLU(),
        nn.Conv2d(salida, salida, kernel_size=3, padding=1),
        nn.BatchNorm2d(salida),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Parte convolucional: detecta bordes, texturas y formas
        self.convolucion = nn.Sequential(
            bloque(1, 32),    # 1x28x28  -> 32x14x14
            bloque(32, 64),   # -> 64x7x7
            bloque(64, 128),  # -> 128x3x3
        )
        # Parte clasificadora: decide qué prenda es
        self.clasificador = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 3 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),  # 10 salidas, una por tipo de prenda
        )

    def forward(self, x):
        return self.clasificador(self.convolucion(x))
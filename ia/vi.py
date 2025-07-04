import torch
import torch.nn as nn
import torch.optim as optim

# 6 muestras con 2 características cada una
x = torch.tensor([
    [1.0, 2.0],   # clase 0
    [2.0, 1.0],   # clase 0
    [3.0, 3.0],   # clase 1
    [4.0, 3.0],   # clase 1
    [5.0, 5.0],   # clase 2
    [6.0, 5.0]    # clase 2
])

# Etiquetas (0, 1 o 2)
y = torch.tensor([0, 0, 1, 1, 2, 2])

class ClasificadorMulticlase(nn.Module):
    def __init__(self):
        super().__init__()
        self.oculta = nn.Linear(2, 10)
        self.salida = nn.Linear(10, 3)

    def forward(self, x):
        x = torch.relu(self.oculta(x))
        return self.salida(x)

criterio = nn.CrossEntropyLoss()
modelo = ClasificadorMulticlase()
optimizador = optim.SGD(modelo.parameters(), lr=0.1)

for epoca in range(4001):
    salida = modelo(x)
    loss = criterio(salida, y)

    optimizador.zero_grad()
    loss.backward()
    optimizador.step()

    if epoca % 10 == 0:
        print(f"Época {epoca}, pérdida: {loss.item():.4f}")

con_nueva = torch.tensor([
        [1.5,2.0],
        [3.5,3.0],
        [5.5,5.0]
    ])

salida = modelo(con_nueva)


predicciones = torch.argmax( salida, dim=1 )

for entrada, pred in zip(con_nueva,predicciones):
    print(f"Entrada: {entrada.tolist()} clase predicha {pred.item()}")


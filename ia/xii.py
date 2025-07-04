import torch
import torch.nn as nn
import torch.optim as optim

# Modelo de clasificación binaria
class ModeloComida(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)  # Entradas: energía, tiempo libre, dinero

    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # Activación sigmoide

def main():
    # Datos simulados: [energía, tiempo libre, dinero] => 1 = pedir comida, 0 = cocinar
    X_train = torch.tensor([
        [3, 10, 50],
        [7, 60, 10],
        [2, 5, 100],
        [9, 40, 5],
        [4, 30, 80],
        [1, 10, 150],
        [8, 70, 0]
    ], dtype=torch.float32)

    y_train = torch.tensor([1, 0, 1, 0, 1, 1, 0], dtype=torch.float32).view(-1, 1)

    # Modelo y configuración
    model = ModeloComida()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Entrenamiento
    for epoca in range(3000):
        print(epoca)
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Entrada del usuario
    energia = float(input("Nivel de energía (1 a 10): "))
    tiempo = float(input("Tiempo libre disponible (en minutos): "))
    dinero = float(input("Dinero disponible ($): "))

    entrada = torch.tensor([[energia, tiempo, dinero]], dtype=torch.float32)
    salida = model(entrada).item()

    decision = "Pedir comida a domicilio" if salida > 0.5 else "Cocinar tú mismo"
    print(f"Decisión del sistema: {decision} (probabilidad: {salida:.2f})")

if __name__ == "__main__":
    main()

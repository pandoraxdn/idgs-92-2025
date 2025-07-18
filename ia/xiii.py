import torch
import torch.nn as nn
import torch.optim as optim

x = torch.tensor([[1.0],[2.0],[3.0],[4.0]]) # Prompt
y = torch.tensor([[5.0],[6.0],[7.0],[8.0]]) # Output

class ModeloSimple(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lineal = nn.Linear(1,1)

    def forward(self,x):
        return self.lineal(x)

modelo = ModeloSimple()

criterio = nn.MSELoss()

optimizador = optim.SGD(modelo.parameters(), lr=0.01)

# Cargar entrenamiento
checkpoint = torch.load("./modelo_entrado.pth")

# Restaurar estados
modelo.load_state_dict(checkpoint["model_state_dict"])
optimizador.load_state_dict(checkpoint["optimizer_state_dict"])
epoca = checkpoint["epoch"]
modelo.eval()

"""
cantidad_epoca = 2401

for epoca in range(cantidad_epoca):
    y_prediccion = modelo(x) # Predicciones
    loss = criterio(y_prediccion,y) # Calculo de perdida

    optimizador.zero_grad() # Limpiar grandientes anteriores
    loss.backward() # Calcular nuevo grandientes
    optimizador.step() # Ajustar pesos

    if epoca % 10 ==  0:
        print(f"Época: {epoca} pérdida: {loss.item():.6f}")

# Almacenar entrenamiento
torch.save({
    "epoch": cantidad_epoca,
    "model_state_dict": modelo.state_dict(),
    "optimizer_state_dict": optimizador.state_dict()
}, "modelo_entrado.pth")
"""

nueva_entrada = torch.tensor([[5.0]])

salida = modelo(nueva_entrada)

print(f"Predicción para 5.0: {salida.item():.2f}")

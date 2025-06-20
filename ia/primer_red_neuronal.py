import torch
import torch.nn as nn
import torch.optim as optim

entradas = torch.tensor([[2.0,3.0],[1.0,4.0],[5.0,5.0]]) # Prompt

salidas = torch.tensor([[5.0],[5.0],[10.0]]) # output

class PrimerRed(nn.Module):
    def __init__(self):
        super().__init__()
        # Crear una capa lineal: 2 entrada y 1 salida
        self.fc = nn.Linear(2,1)

    def forward(self, x):
        return self.fc(x)

# Creación de instancia de red neuronal
modelo = PrimerRed()

# Función de perdida y optimización
critero = nn.MSELoss()
optimzacion = optim.SGD(modelo.parameters(),lr=0.01)

# Entrenamiento
for epoca in range(1000):  # Repetimos 1000 veces
    prediccion = modelo(entradas)          # Paso 1: hace una predicción
    perdida = critero(prediccion, salidas)  # Paso 2: calcula el error
    optimzacion.zero_grad()                # Paso 3: limpia errores anteriores
    perdida.backward()                     # Paso 4: calcula derivadas
    optimzacion.step()                     # Paso 5: ajusta los pesos

    if epoca % 200 == 0:
        print(f"Época {epoca}: pérdida = {perdida.item():.4f}")

nueva_entrada = torch.tensor([[4.0,6.0]]) # Quiero que aprendas cuanto es 4 + 6 = 10
resultado = modelo(nueva_entrada)
print(f"Predicción: {resultado.item()}")

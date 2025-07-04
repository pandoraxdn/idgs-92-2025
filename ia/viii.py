import torch
import torch.nn as nn
import torch.optim as optim

calorias_dict = {
    "hamburguesa": 1500,
    "papas": 1000,
    "refresco": 230,
    "helado": 250,
    "ensalada": 120,
    "pollo frito": 800,
    "nuggets": 350,
    "pizza": 400,
    "agua": 0
}

class ModeloCalorias(nn.Module):
    def __init__(self,num_items) -> None:
        super().__init__()
        self.linear = nn.Linear(num_items,1)

    def forward(self, x):
        return self.linear(x)

# Entrenamiento basico
items = list(calorias_dict)
num_items = len(items)

x_train = torch.eye(num_items)
y_train = torch.tensor([calorias_dict[item] for item in items], dtype=torch.float32).view(-1,1)

modelo = ModeloCalorias(num_items)
criterio = nn.MSELoss()
optimizar = optim.SGD(modelo.parameters(),lr=0.01)

for epoca in range(3000):
    print(epoca)
    optimizar.zero_grad()
    output = modelo(x_train)
    loss = criterio(output,y_train)
    loss.backward()
    optimizar.step()

print("Opciones disponibles:")
for item in items:
    print(item)

seleccion = input("Que comiste?, separa por comas: ").lower().split(",")
seleccion = [ s.strip() for s in seleccion ]

# Codificar la seleccion
x_usuario = torch.zeros(num_items)
for idx, item in enumerate(items):
    if item in seleccion:
        x_usuario[idx] = 1

prediccion = modelo(x_usuario.view(1, -1)).item()
print(f"Calorias estimadas de tu comida: {prediccion:.2f} kcal")






































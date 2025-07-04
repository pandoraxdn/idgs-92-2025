import torch
import torch.nn as nn
import torch.optim as optim

class ModeloGasolina(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3,1)

    def forward(self,x):
        return self.linear(x)

# [dias,km/d,tipo_coche] => listros estimados por semana
x_train = torch.tensor([
        [5,10,1], # día 5, litros 10, automatico = 1, 2=standard, 3=hibrido
        [7,20,2],
        [3,30,3],
        [6,12,2],
        [2,8,1],
        [4,25,3]
    ],dtype=torch.float32)

y_train = torch.tensor([15,45,60,40,8,50],dtype=torch.float32).view(-1,1)

modelo = ModeloGasolina()
criterio = nn.MSELoss()
optimizador = optim.SGD(modelo.parameters(),lr=0.001)

for epoca in range(3000):
    print(epoca)
    optimizador.zero_grad()
    output = modelo(x_train)
    loss = criterio(output,y_train)
    loss.backward()
    optimizador.step()


dias = int(input("Cuantos días usas el auto a la semana: "))
km_dia = float(input("Cuantos kilometros manejas por día: "))
tipo = int(input("Tipo de auto: (1 - automatico, 2 - standard, 3 - hibrido): "))

entrada = torch.tensor([dias,km_dia,tipo],dtype=torch.float32)
prediccion = modelo(entrada).item()
print(f"Estmacion de gasolina semanal: {prediccion:.2f} litros")




















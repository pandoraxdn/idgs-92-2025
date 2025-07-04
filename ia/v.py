import torch
import torch.nn as nn
import torch.optim as optim

x = torch.tensor([[1.0],[2.0],[3.0],[6.0],[7.0],[8.0]]) # prompt
y = torch.tensor([[0.0],[0.0],[0.0],[1.0],[1.0],[1.0]]) # output

class ClasificadorBinario( nn.Module ):
    def __init__(self) -> None:
        super().__init__()
        self.lineal = nn.Linear(1,1)

    def forward(self,x):
        return torch.sigmoid(self.lineal(x))

modelo = ClasificadorBinario()

criterio = nn.BCELoss() # Binary Cross Entropy Loss

optmizador = optim.SGD( modelo.parameters(), lr=0.01 )


for epoca in range(30000):
    y_prediccion = modelo(x)
    loss = criterio(y_prediccion,y)

    optmizador.zero_grad()
    loss.backward()
    optmizador.step()
    
    if epoca % 10 == 0:
        print(f"Época: {epoca}, perdida: {loss.item():.4f}")

nueva_entrada = torch.tensor([[4.0],[5.0],[6.5]])

salida = modelo(nueva_entrada)


for i, valor in enumerate(salida):
    pred = [0,1][valor >= 0.5]
    print(f"Entrada: {nueva_entrada[i].item()}, Probabilidad: {valor.item():.2f}, Preducción: {pred}")


















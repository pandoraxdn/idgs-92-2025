import torch

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# Datos de entrada
x = torch.tensor([2.0])
y = torch.tensor([5.0])

y_pred = w * x + b

loss = ( y_pred - y ) ** 2

# Calcular el grandiente
loss.backward()

print(f"Grandiente w: {w.grad}")
print(f"Grandiente b: {b.grad}")

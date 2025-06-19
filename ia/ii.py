import torch
import numpy as np

arr = np.array([1.0,2.0,3.0])
print(arr)
tensor = torch.from_numpy(arr)
print(tensor)
nuevo_arr = tensor.numpy()
print(nuevo_arr)

# autogrand calcular derivas y grandientes
# Loss perdida que hizo mal el modelo
# backward Calcula las grandientes (direccion correcta para corregir errores)
# grad Contiene el valor de corrección

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x
y.backward()
print(y)
print(x.grad)

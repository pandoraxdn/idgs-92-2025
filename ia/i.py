import torch

# Tipos de tensor escalar
a = torch.tensor(5.0)

# Tipo de tensor de vectores
b = torch.tensor([1.0,2.0,3.0])

# Tipo de tensor matriz
c = torch.tensor([[1.0,2.0],[3.0,4.0]])


x = torch.tensor([2.0,4.0])
y = torch.tensor([1.0,3.0])

print( x + y ) # Suma
print( x * y ) # Multiplicación de elemento a elemento
print(x.mean()) # Promedio de valores
print(x.max()) # El valor max de mi vector

# Formas y tamaños de un tensor
a = torch.tensor([[1,2],[3,4],[5,6]])
print(a.shape) # Resultado (3 filas y 2 columnas)

# Cambiar forma
print(a.view(2,3)) # Reorganizarlo como 2 filas y 3 columnas

# Cortar y seleccionar partes de un tensor (index)
x = torch.tensor([
    [10,20,30],
    [40,50,60]])

# Primera fila
print(x[0])

# Segunda fila
print(x[:,1])

# Fila 1, columna 2 
print( x[1,2] )







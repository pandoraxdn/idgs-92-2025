import torch

# Tensor de enteros
tensor_entero = torch.tensor([1,2,3], dtype=torch.int32 )

# Tensor flotante
tensor_flotante = torch.tensor([1.5,2.5,3.5], dtype=torch.float32 )

# Tensor de boolean
tensor_boolean = torch.tensor([ True, False, True ], dtype=torch.bool)

print(tensor_entero, type(tensor_entero))
print(tensor_flotante, type(tensor_flotante))
print(tensor_boolean, type(tensor_boolean))


# Conversion de tensores
x = torch.tensor([1,2,3]) # int64
print(x)
x_float = x.float()
print(x_float)
x_bool = x.bool()
print(x_bool)


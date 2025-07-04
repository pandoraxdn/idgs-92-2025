import torch

tensor = torch.tensor(1.0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tensor_gpu = tensor.to(device)

print(tensor_gpu,device)

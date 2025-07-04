import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import cv2
import numpy as np
import os

# Red neuronal entrada
class RedMNIST(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28*28,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)

    def forward(self,x):
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

modelo = RedMNIST()
modelo.eval()

# FUnciones de la imagen
to_tensor = transforms.ToTensor()

def cargar_y_procesar_imagen(ruta):
    imagen = Image.open(ruta).convert("L")
    tensor = to_tensor(imagen)
    reducida = transforms.Resize((14,14),antialias=True)(tensor)
    reescalada = transforms.Resize((28,28),antialias=True)(reducida)
    return tensor, reducida, reescalada

def tensor_a_cv2(imagen_tensor):
    img_array = imagen_tensor.squeeze().numpy() * 255
    return img_array.astype(np.uint8)

def mostrar_con_cv2(tensor_img,nombre="imagen"):
    img = tensor_a_cv2(tensor_img)
    cv2.imshow(nombre,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def guardar_imagen(tensor_img, nombre_archivo):
    save_image(tensor_img,nombre_archivo)
    print(f"Imagen guardada como: {nombre_archivo}")

def predecir(tensor_img):
    entrada = tensor_img.squeeze(0)
    with torch.no_grad():
        salida = modelo(entrada)
        pred = torch.argmax(salida,dim=1).item()
    return pred

# Lógica principal
def procesar_y_mostrar(ruta_img, nombre="Imagen"):
    print(f"\n Procesando {nombre} desde: {ruta_img}")

    original, reducida, reescalada = cargar_y_procesar_imagen(ruta_img)

    try:
        mostrar_con_cv2(original, f"{nombre} - Original 28 x 28")
        mostrar_con_cv2(reducida, f"{nombre} - Reducida 14 x 14")
    except:
        # En caso de no haber GUI
        guardar_imagen(original,f"{nombre}_original.png")
        guardar_imagen(reducida,f"{nombre}_reducida.png")

    prediccion = predecir(reescalada)
    print(f"Predicción del modelo para {nombre} : {prediccion}")

# Ejecutar imágenes
ruta_img1 = './assets/I.png'
ruta_img2 = './assets/II.png'

if not os.path.exists(ruta_img1):
    print("Error verificar ruta de imagen")
else:
    procesar_y_mostrar(ruta_img1,"Imagen 1")
    procesar_y_mostrar(ruta_img2,"Imagen 2")

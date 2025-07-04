import easyocr
import cv2
import matplotlib.pyplot as plt
import os

def procesa_imagen(ruta_imagen,idiomas=["es","en"]):
    if not os.path.exists(ruta_imagen):
        print("Archivo no encontrado")
        return

    imagen = cv2.imread(ruta_imagen)

    if imagen is None:
        print("La imagen no es compatible")
        return

    print("Cargar modelo")
    lector = easyocr.Reader(idiomas,gpu=False)

    resultado = lector.readtext(ruta_imagen)

    print("Texto extraido...")
    if not resultado:
        print("No se encontro texto")

    for (bbox,texto,prob) in resultado:
        print(f"Texto: {texto}, probabilidad: {prob:.4f}")

    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    for (bbox,texto,prob) in resultado:
        (top_left,_,bottom_right,_) = bbox
        top_left = tuple(map(int,top_left))
        bottom_right = tuple(map(int,bottom_right))

        cv2.rectangle(imagen_rgb, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(imagen_rgb, texto, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
    print("Imagen: ")
    plt.figure(figsize=(12,12))
    plt.imshow(imagen_rgb)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    ruta_imagen = './ab.jpg'
    procesa_imagen(ruta_imagen)








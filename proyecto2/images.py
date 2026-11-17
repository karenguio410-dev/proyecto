import cv2


def obtener_matriz(ruta_imagen="Data/unNumerooo.png"):
    img_matriz = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

    # Si la ruta esta mal, imread devuelve None
    if img_matriz is None:
        print("No se pudo leer la imagen:", ruta_imagen)
        return None

    # INTER_AREA promedia los pixeles al achicar (mejor que el valor por defecto)
    nueva_matriz = cv2.resize(img_matriz, (8, 8), interpolation=cv2.INTER_AREA)

    # Pasamos a una lista de Python para trabajar con numeros normales
    matriz = nueva_matriz.tolist()

    i = 0
    while i < 8:
        j = 0
        while j < 8:
            # Reducir de 0-255 a 0-16
            valor = matriz[i][j] / 255 * 16
            # Invertir: el fondo blanco pasa a 0 y la tinta negra a 16
            valor = 16 - valor
            # El papel fotografiado no es blanco puro: quitamos el "ruido"
            if valor < 2:
                valor = 0
            matriz[i][j] = round(valor)
            j = j + 1
        i = i + 1

    return matriz

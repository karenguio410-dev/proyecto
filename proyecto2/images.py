import cv2
import numpy as np


def leer_imagen_gris(ruta_imagen):
    # np.fromfile + cv2.imdecode en vez de cv2.imread: cv2.imread falla en
    # Windows si la ruta tiene tildes o ñ (por ejemplo "Introducción")
    datos = np.fromfile(ruta_imagen, dtype=np.uint8)
    img = cv2.imdecode(datos, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if len(img.shape) == 2:
        # Ya estaba en escala de grises
        return img

    if img.shape[2] == 4:
        # PNG con transparencia: lo transparente se vuelve papel blanco
        alfa = img[:, :, 3]
        gris = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        gris[alfa < 128] = 255
        return gris

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def recortar_numero(binaria):
    # binaria: 0 = tinta, 255 = papel.
    # Busca la primera y la ultima fila/columna que tienen tinta, recorta ahi
    # y completa con papel blanco para que la imagen quede cuadrada (item b).
    filas = binaria.tolist()
    alto = len(filas)
    ancho = len(filas[0])

    # Una fila o columna cuenta como "con tinta" si tiene al menos 1% de
    # pixeles negros. Asi un puntito de suciedad no cambia el recorte.
    minimo_fila = max(1, ancho // 100)
    minimo_col = max(1, alto // 100)

    arriba = -1
    abajo = -1
    i = 0
    while i < alto:
        contador = 0
        j = 0
        while j < ancho:
            if filas[i][j] == 0:
                contador = contador + 1
            j = j + 1
        if contador >= minimo_fila:
            if arriba == -1:
                arriba = i
            abajo = i
        i = i + 1

    izquierda = -1
    derecha = -1
    j = 0
    while j < ancho:
        contador = 0
        i = 0
        while i < alto:
            if filas[i][j] == 0:
                contador = contador + 1
            i = i + 1
        if contador >= minimo_col:
            if izquierda == -1:
                izquierda = j
            derecha = j
        j = j + 1

    if arriba == -1 or izquierda == -1:
        # No se encontro tinta: se deja la imagen como esta
        return binaria

    recorte = binaria[arriba:abajo + 1, izquierda:derecha + 1]

    # Completar con blanco a los lados (o arriba/abajo) hasta que sea cuadrada,
    # dejando el numero centrado
    alto_r = abajo - arriba + 1
    ancho_r = derecha - izquierda + 1
    lado = max(alto_r, ancho_r)
    sobra_alto = lado - alto_r
    sobra_ancho = lado - ancho_r

    cuadrada = cv2.copyMakeBorder(recorte,
                                  sobra_alto // 2, sobra_alto - sobra_alto // 2,
                                  sobra_ancho // 2, sobra_ancho - sobra_ancho // 2,
                                  cv2.BORDER_CONSTANT, value=255)
    return cuadrada


def obtener_matriz(ruta_imagen="Data/unNumerooo.png"):
    # 1. Leer la foto en escala de grises (0 = negro, 255 = blanco)
    img_matriz = leer_imagen_gris(ruta_imagen)
    if img_matriz is None:
        print("No se pudo leer la imagen:", ruta_imagen)
        return None

    # 2. Blanco y negro puro: cada pixel pasa a 0 (tinta) o 255 (papel).
    #    THRESH_OTSU elige solo el limite entre tinta y papel, asi el papel
    #    gris de una foto queda blanco y no se confunde con tinta.
    limite, binaria = cv2.threshold(img_matriz, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Recortar el numero y dejar la imagen cuadrada
    binaria = recortar_numero(binaria)

    # 4. Engrosar el trazo. El lapicero hace lineas finas y los numeros del
    #    dataset tienen trazos gruesos. erode agranda lo negro.
    grosor = max(1, len(binaria) // 15)
    nucleo = np.ones((grosor, grosor), np.uint8)
    binaria = cv2.erode(binaria, nucleo)

    # 5. Reducir a 8x8. INTER_AREA promedia todos los pixeles de cada bloque
    nueva_matriz = cv2.resize(binaria, (8, 8), interpolation=cv2.INTER_AREA)

    # Pasamos a una lista de Python para trabajar con numeros normales
    matriz = nueva_matriz.tolist()

    i = 0
    while i < 8:
        j = 0
        while j < 8:
            # 6. Reducir de 0-255 a 0-16
            valor = matriz[i][j] / 255 * 16
            # 7. Invertir: el papel pasa a 0 y la tinta a 16, como en el dataset
            valor = 16 - valor
            matriz[i][j] = round(valor)
            j = j + 1
        i = i + 1

    return matriz

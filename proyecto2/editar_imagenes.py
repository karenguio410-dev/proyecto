import os
import cv2
import numpy as np
from images import leer_imagen_gris, obtener_matriz

# ---------------------------------------------------------------
# PROGRAMA DE RECORTE Y FILTRO
# Toma las fotos originales, las limpia y las guarda listas para
# main_lectura_dibujo_nuevo.py con nombres tipo 4_5.jpg
#
#   Fotos/        -> tus fotos originales (no se modifican nunca)
#   Data/         -> las imagenes editadas (las que lee el main)
#   Comparacion/  -> antes | despues | 8x8, para revisar y para el informe
#
# La primera vez, si no existe Fotos/, la carpeta Data/ actual se
# renombra a Fotos/ (no se borra nada) y se crea una Data/ nueva.
# ---------------------------------------------------------------

CARPETA_PROYECTO = os.path.dirname(os.path.abspath(__file__))
CARPETA_FOTOS = os.path.join(CARPETA_PROYECTO, "Fotos")
CARPETA_DATA = os.path.join(CARPETA_PROYECTO, "Data")
CARPETA_COMPARACION = os.path.join(CARPETA_PROYECTO, "Comparacion")

TAMANO_TRABAJO = 800   # las fotos del celular se achican a esto para ir rapido
TAMANO_FINAL = 256     # tamano de la imagen editada (cuadrada)
EXTENSIONES = [".png", ".jpg", ".jpeg"]


def preparar_carpetas():
    if not os.path.isdir(CARPETA_FOTOS):
        if os.path.isdir(CARPETA_DATA):
            os.rename(CARPETA_DATA, CARPETA_FOTOS)
            print("Tus fotos originales se movieron a la carpeta Fotos/ (no se borro nada)")
        else:
            print("No existe la carpeta Fotos/ ni Data/ con tus fotos.")
            return False
    if not os.path.isdir(CARPETA_DATA):
        os.mkdir(CARPETA_DATA)
    if not os.path.isdir(CARPETA_COMPARACION):
        os.mkdir(CARPETA_COMPARACION)
    return True


def achicar(gris):
    # Una foto de celular puede tener 4000 x 3000 pixeles: la achicamos
    alto = gris.shape[0]
    ancho = gris.shape[1]
    lado_mayor = max(alto, ancho)
    if lado_mayor <= TAMANO_TRABAJO:
        return gris
    escala = TAMANO_TRABAJO / lado_mayor
    nuevo_ancho = round(ancho * escala)
    nuevo_alto = round(alto * escala)
    return cv2.resize(gris, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)


def quitar_sombras(gris):
    # 1. Estimar como se ve el papel SIN tinta. MORPH_CLOSE borra todo lo
    #    oscuro que sea mas delgado que k pixeles (las lineas de lapicero) y
    #    deja igual lo oscuro grande (sombras, borde de la mesa). Asi queda
    #    una "foto del papel solo", con sus sombras pero sin el numero.
    # (antes se suaviza un poco el ruido de la camara con GaussianBlur)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)
    lado = max(gris.shape[0], gris.shape[1])
    k = max(3, lado // 6)
    papel = cv2.morphologyEx(gris, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    # 2. Dividir la foto entre el papel: donde solo hay papel (o sombra, o
    #    mesa) queda 255, blanco parejo, y solo la tinta queda oscura.
    return cv2.divide(gris, papel, scale=255)


def binarizar(gris):
    # Pasar a blanco y negro puro.
    # Otsu elige solo el limite entre tinta (0) y papel (255).
    limite, binaria = cv2.threshold(gris, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binaria


def bordes_que_toca(x, y, ancho, alto, ancho_img, alto_img):
    # Cuenta cuantos de los 4 bordes de la foto toca una mancha
    cuenta = 0
    if x == 0:
        cuenta = cuenta + 1
    if y == 0:
        cuenta = cuenta + 1
    if x + ancho == ancho_img:
        cuenta = cuenta + 1
    if y + alto == alto_img:
        cuenta = cuenta + 1
    return cuenta


def quitar_manchas(binaria):
    # Separa la tinta en "manchas" sueltas (pedazos de tinta que se tocan).
    # 1. Una mancha que toca 3 o 4 bordes de la foto es el borde de la mesa,
    #    de la hoja o una sombra: no es el numero.
    # 2. El numero es la mancha mas grande que queda; se borran las que miden
    #    menos del 10% de esa (puntitos, polvo, ruido).
    # Devuelve la imagen limpia y el rectangulo donde esta el numero.
    tinta = 255 - binaria   # connectedComponents busca lo blanco
    cantidad, etiquetas, datos, centros = cv2.connectedComponentsWithStats(tinta)
    alto_img = binaria.shape[0]
    ancho_img = binaria.shape[1]

    # Mancha mas grande de todas (la etiqueta 0 es el fondo: empezamos en 1)
    mayor_total = 0
    k = 1
    while k < cantidad:
        if datos[k][cv2.CC_STAT_AREA] > mayor_total:
            mayor_total = datos[k][cv2.CC_STAT_AREA]
        k = k + 1

    # valida[k] = True si la mancha k puede ser parte del numero
    valida = [False]
    hay_validas = False
    k = 1
    while k < cantidad:
        toca = bordes_que_toca(datos[k][cv2.CC_STAT_LEFT], datos[k][cv2.CC_STAT_TOP],
                               datos[k][cv2.CC_STAT_WIDTH], datos[k][cv2.CC_STAT_HEIGHT],
                               ancho_img, alto_img)
        if toca >= 3:
            valida.append(False)
        else:
            valida.append(True)
            # Solo cuenta si no es un puntito (al menos 2% de la mas grande)
            if datos[k][cv2.CC_STAT_AREA] >= mayor_total * 0.02:
                hay_validas = True
        k = k + 1

    if not hay_validas:
        # Todo lo importante toca los bordes (por ejemplo un 0 recortado
        # muy justo): entonces se acepta todo
        k = 1
        while k < cantidad:
            valida[k] = True
            k = k + 1

    mayor = 0
    k = 1
    while k < cantidad:
        area = datos[k][cv2.CC_STAT_AREA]
        if valida[k] and area > mayor:
            mayor = area
        k = k + 1

    if mayor == 0:
        return None, None   # la imagen no tiene tinta

    limpia = np.full(binaria.shape, 255, np.uint8)
    arriba = binaria.shape[0]
    izquierda = binaria.shape[1]
    abajo = 0
    derecha = 0

    k = 1
    while k < cantidad:
        x = datos[k][cv2.CC_STAT_LEFT]
        y = datos[k][cv2.CC_STAT_TOP]
        ancho = datos[k][cv2.CC_STAT_WIDTH]
        alto = datos[k][cv2.CC_STAT_HEIGHT]
        area = datos[k][cv2.CC_STAT_AREA]

        if valida[k] and area >= mayor * 0.10:
            limpia[etiquetas == k] = 0      # pintar esta mancha de negro
            if y < arriba:
                arriba = y
            if x < izquierda:
                izquierda = x
            if y + alto > abajo:
                abajo = y + alto
            if x + ancho > derecha:
                derecha = x + ancho
        k = k + 1

    return limpia, [arriba, abajo, izquierda, derecha]


def recortar_cuadrado(limpia, caja):
    # Recorta justo donde empieza y termina el numero (toca arriba y abajo)
    # y completa con blanco a los costados para que quede cuadrada (item b)
    arriba = caja[0]
    abajo = caja[1]
    izquierda = caja[2]
    derecha = caja[3]
    recorte = limpia[arriba:abajo, izquierda:derecha]

    alto = abajo - arriba
    ancho = derecha - izquierda
    lado = max(alto, ancho)
    sobra_alto = lado - alto
    sobra_ancho = lado - ancho
    cuadrada = cv2.copyMakeBorder(recorte,
                                  sobra_alto // 2, sobra_alto - sobra_alto // 2,
                                  sobra_ancho // 2, sobra_ancho - sobra_ancho // 2,
                                  cv2.BORDER_CONSTANT, value=255)
    return cv2.resize(cuadrada, (TAMANO_FINAL, TAMANO_FINAL), interpolation=cv2.INTER_AREA)


def editar(gris):
    gris = achicar(gris)
    sin_sombra = quitar_sombras(gris)
    binaria = binarizar(sin_sombra)
    limpia, caja = quitar_manchas(binaria)
    if limpia is None:
        return None
    return recortar_cuadrado(limpia, caja)


def armar_comparacion(original, editada, ruta_editada):
    # Tres cuadros de 256 x 256: foto original | imagen editada | como la ve el
    # programa en 8x8 (lo que devuelve obtener_matriz)
    original = achicar(original)
    alto = original.shape[0]
    ancho = original.shape[1]
    escala = TAMANO_FINAL / max(alto, ancho)
    original = cv2.resize(original, (max(1, round(ancho * escala)), max(1, round(alto * escala))),
                          interpolation=cv2.INTER_AREA)
    alto = original.shape[0]
    ancho = original.shape[1]
    cuadro1 = cv2.copyMakeBorder(original,
                                 (TAMANO_FINAL - alto) // 2, TAMANO_FINAL - alto - (TAMANO_FINAL - alto) // 2,
                                 (TAMANO_FINAL - ancho) // 2, TAMANO_FINAL - ancho - (TAMANO_FINAL - ancho) // 2,
                                 cv2.BORDER_CONSTANT, value=255)

    matriz = obtener_matriz(ruta_editada)
    pequena = np.zeros((8, 8), np.uint8)
    i = 0
    while i < 8:
        j = 0
        while j < 8:
            # 16 (tinta) se dibuja negro y 0 (fondo) blanco
            pequena[i][j] = 255 - matriz[i][j] * 255 // 16
            j = j + 1
        i = i + 1
    cuadro3 = cv2.resize(pequena, (TAMANO_FINAL, TAMANO_FINAL), interpolation=cv2.INTER_NEAREST)

    separador = np.full((TAMANO_FINAL, 10), 128, np.uint8)
    return np.hstack([cuadro1, separador, editada, separador, cuadro3])


def preguntar_numero(nombre, comparacion):
    # Para fotos cuyo nombre no empieza con el numero (ej. IMG_2045.jpg):
    # se muestra la imagen y se presiona la tecla del numero.
    print("¿Que numero es", nombre, "? Presiona la tecla 0-9 en la ventana "
          "(s = saltar esta foto)")
    try:
        cv2.imshow("Que numero es? (0-9, s = saltar)", comparacion)
        tecla = cv2.waitKey(0)
        cv2.destroyAllWindows()
        letra = chr(tecla & 255)
    except cv2.error:
        # Si tu OpenCV no puede abrir ventanas, se pregunta por la terminal
        ruta_vista = os.path.join(CARPETA_COMPARACION, "_revisar.jpg")
        cv2.imwrite(ruta_vista, comparacion)
        letra = input("   (abre Comparacion/_revisar.jpg para verla) Numero: ").strip()

    if len(letra) == 1 and letra.isdigit():
        return int(letra)
    return None


def main():
    if not preparar_carpetas():
        return

    archivos = sorted(os.listdir(CARPETA_FOTOS))

    # contador[d] = cuantas imagenes del numero d ya se guardaron
    contador = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    problemas = []

    for nombre in archivos:
        extension = os.path.splitext(nombre)[1].lower()
        if extension not in EXTENSIONES:
            continue

        original = leer_imagen_gris(os.path.join(CARPETA_FOTOS, nombre))
        if original is None:
            problemas.append(nombre + " (no se pudo leer)")
            continue

        editada = editar(original)
        if editada is None:
            problemas.append(nombre + " (no se encontro tinta)")
            continue

        # El numero real: primer caracter del nombre, o se pregunta
        if nombre[0].isdigit():
            numero = int(nombre[0])
        else:
            ruta_temp = os.path.join(CARPETA_COMPARACION, "_temp.png")
            cv2.imwrite(ruta_temp, editada)
            numero = preguntar_numero(nombre, armar_comparacion(original, editada, ruta_temp))
            os.remove(ruta_temp)
            if numero is None:
                problemas.append(nombre + " (saltada)")
                continue

        contador[numero] = contador[numero] + 1
        nuevo_nombre = str(numero) + "_" + str(contador[numero]) + ".jpg"
        ruta_nueva = os.path.join(CARPETA_DATA, nuevo_nombre)
        cv2.imwrite(ruta_nueva, editada, [cv2.IMWRITE_JPEG_QUALITY, 95])

        comparacion = armar_comparacion(original, editada, ruta_nueva)
        cv2.imwrite(os.path.join(CARPETA_COMPARACION, nuevo_nombre), comparacion)

        print(nombre, "->", "Data/" + nuevo_nombre)

    print()
    print("RESUMEN")
    total = 0
    d = 0
    while d < 10:
        aviso = ""
        if contador[d] < 3:
            aviso = "   <- pocas imagenes de este numero"
        print("   Numero", d, ":", contador[d], "imagenes" + aviso)
        total = total + contador[d]
        d = d + 1
    print("   Total guardadas en Data/:", total)

    if len(problemas) > 0:
        print()
        print("No se guardaron:")
        for p in problemas:
            print("   ", p)

    print()
    print("Revisa la carpeta Comparacion/ para ver el antes y el despues.")
    print("Ahora puedes correr main_lectura_dibujo_nuevo.py")


main()

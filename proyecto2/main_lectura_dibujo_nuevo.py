import os
from sklearn import datasets
from images import obtener_matriz

base_numeros = datasets.load_digits()

# Carpeta con tus dibujos. Cada archivo se llama  <numero_real>_<n>.png
# Ejemplo: 0_1.png, 0_2.png, 5_1.png, 5_2.png ...
CARPETA = "Data"


def medir_similitud(img_entrada, img_base):
    acumulado = 0

    for y in range(8):
        for x in range(8):
            cuadrado = (img_entrada[y][x] - img_base[y][x]) ** 2
            acumulado += cuadrado

    return acumulado ** 0.5


def identificar_digito(matriz_prueba):
    # c) 1797 distancias
    resultados_dist = []

    for idx in range(len(base_numeros["images"])):
        valor_d = medir_similitud(matriz_prueba, base_numeros["images"][idx])
        resultados_dist.append(valor_d)

    # d) los 3 mas cercanos
    ubicaciones = []
    distancias = []

    for puesto in range(3):
        menor = min(resultados_dist)
        lugar = resultados_dist.index(menor)

        ubicaciones.append(lugar)
        distancias.append(menor)

        resultados_dist[lugar] = 420  # mayor que cualquier distancia posible (max 128)

    posibles = []

    for u in ubicaciones:
        posibles.append(int(base_numeros["target"][u]))

    # f) votacion
    frecuencias = []

    for i in posibles:
        contador = 0
        for j in posibles:
            if i == j:
                contador += 1
        frecuencias.append(contador)

    rep_maxima = max(frecuencias)

    if rep_maxima >= 2:
        # 2 o 3 targets iguales -> ese es el numero
        digito_final = posibles[frecuencias.index(rep_maxima)]
    else:
        # Los 3 son diferentes -> nos quedamos con el vecino MAS cercano
        # (el de menor distancia, que es el primero de la lista)
        digito_final = posibles[0]

    return posibles, distancias, digito_final


def crear_matriz_ceros(filas, columnas):
    matriz = []
    for i in range(filas):
        fila = []
        for j in range(columnas):
            fila.append(0)
        matriz.append(fila)
    return matriz


# ---------------------------------------------------------------
# Leer todos los dibujos de la carpeta y clasificarlos
# ---------------------------------------------------------------
reales = []
predichos = []

archivos = sorted(os.listdir(CARPETA))

for nombre in archivos:
    if not nombre.endswith(".png") and not nombre.endswith(".jpg"):
        continue

    numero_real = int(nombre[0])  # la etiqueta verdadera sale del nombre
    matriz = obtener_matriz(CARPETA + "/" + nombre)
    if matriz is None:
        continue

    posibles, distancias, digito_final = identificar_digito(matriz)

    reales.append(numero_real)
    predichos.append(digito_final)

    # e) imprimir targets de los 3 vecinos y el numero real
    print("Imagen:", nombre)
    print("   Numero real:", numero_real)
    print("   Targets de los 3 mas cercanos:", posibles)
    print("   Distancias:", [round(float(d), 2) for d in distancias])
    # f) mensaje pedido
    print("   Soy la inteligencia artificial, y he detectado que el digito "
          "ingresado corresponde al numero", digito_final)
    if digito_final == numero_real:
        print("   -> ACIERTO")
    else:
        print("   -> ERROR")
    print()

# ---------------------------------------------------------------
# g) Matriz de confusion de 10 clases
#    filas = numero real, columnas = numero predicho
# ---------------------------------------------------------------
confusion = crear_matriz_ceros(10, 10)

for k in range(len(reales)):
    confusion[reales[k]][predichos[k]] += 1

print("MATRIZ DE CONFUSION 10 CLASES (filas = real, columnas = predicho)")
encabezado = "real\\pred"
for j in range(10):
    encabezado += f"{j:>4}"
print(encabezado)
for i in range(10):
    linea = f"{i:>9}"
    for j in range(10):
        linea += f"{confusion[i][j]:>4}"
    print(linea)

aciertos = 0
for i in range(10):
    aciertos += confusion[i][i]
print("Aciertos totales:", aciertos, "de", len(reales))
print()

# ---------------------------------------------------------------
# h) Matriz de confusion de 2 clases para cada numero dibujado
#    Positivo = "es el numero d", Negativo = "no es el numero d"
# ---------------------------------------------------------------
numeros_dibujados = []
for r in reales:
    if r not in numeros_dibujados:
        numeros_dibujados.append(r)
numeros_dibujados.sort()

for d in numeros_dibujados:
    VP = 0  # era d y dijo d
    FN = 0  # era d y dijo otro
    FP = 0  # no era d y dijo d
    VN = 0  # no era d y no dijo d

    for k in range(len(reales)):
        if reales[k] == d and predichos[k] == d:
            VP += 1
        elif reales[k] == d and predichos[k] != d:
            FN += 1
        elif reales[k] != d and predichos[k] == d:
            FP += 1
        else:
            VN += 1

    total = VP + FN + FP + VN

    if VP + FP > 0:
        precision = VP / (VP + FP)
    else:
        precision = 0
    if VP + FN > 0:
        sensibilidad = VP / (VP + FN)
    else:
        sensibilidad = 0
    exactitud = (VP + VN) / total

    print(f"MATRIZ DE CONFUSION 2 CLASES PARA EL NUMERO {d}  ({VP + FN} dibujos de {d})")
    print(f"                 pred: {d}   pred: no {d}")
    print(f"   real: {d}      {VP:>6}   {FN:>10}")
    print(f"   real: no {d}   {FP:>6}   {VN:>10}")
    print(f"   Precision    = VP/(VP+FP)         = {precision:.2f}")
    print(f"   Sensibilidad = VP/(VP+FN)         = {sensibilidad:.2f}")
    print(f"   Exactitud    = (VP+VN)/total      = {exactitud:.2f}")
    print()

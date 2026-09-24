import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction import text

# ============================================================
# CARGA DEL DATASET
# ============================================================
df = pd.read_csv('smogon.csv')
print("Columnas del CSV:", df.columns.tolist())
print("Total de Pokémon:", len(df))

df['moves'] = df['moves'].fillna('')

# ============================================================
# PREGUNTA 1: Agrupamiento mediante TF-IDF con n-gramas
# ============================================================
print("\n" + "=" * 60)
print("PREGUNTA 1: Agrupamiento con TF-IDF y n-gramas")
print("=" * 60)
mis_stop_words = ["power", "chance", "stage", "turn", "attack", "turns",
                  "ivs", "hidden", "user", "target", "based", "accuracy",
                  "pp", "pokemon", "type", "moves", "types"]
stop_words_ingles = list(text.ENGLISH_STOP_WORDS)
todas_las_stop_words = stop_words_ingles + mis_stop_words

# ngram_range=(1, 2): unigramas y bigramas
vectorizer_p1 = TfidfVectorizer(ngram_range=(1, 2),
                                stop_words= todas_las_stop_words,
                                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")

tfidf_matrix_p1 = vectorizer_p1.fit_transform(df['moves'])
vocabulario_p1 = sorted(vectorizer_p1.get_feature_names_out())

print(f"Total de columnas (vocabulario): {len(vectorizer_p1.vocabulary_)}")
print("Elementos del vocabulario:")
print(vocabulario_p1)

df_tfidf_p1 = pd.DataFrame(tfidf_matrix_p1.toarray(), index=df['Pokemon'], columns=vocabulario_p1)
print("\nMatriz TF-IDF")
print(df_tfidf_p1)

# Primer agrupamiento: 18 clusters (uno por cada tipo de Pokémon)
N_CLUSTERS = 18
kmeans_p1 = KMeans(n_clusters=N_CLUSTERS, random_state=10, n_init=10)
kmeans_p1.fit(tfidf_matrix_p1)
df['cluster_p1'] = kmeans_p1.labels_

# Ver qué Pokémon quedaron en cada cluster, para la reflexión
# (¿hay clusters que no corresponden a un tipo, como el de los Arceus?)
for c in range(N_CLUSTERS):
    pokemones = df[df['cluster_p1'] == c]['Pokemon'].tolist()
    print(f"\nCluster {c} ({len(pokemones)} Pokémon): {pokemones}")

df[['Pokemon', 'cluster_p1']].to_csv('resultado_p1.csv', index=False)
print("\nArchivo 'resultado_p1.csv' guardado.")

# ============================================================
# PREGUNTA 2: Verificación del agrupamiento por tipos
# ============================================================
print("\n" + "=" * 60)
print("PREGUNTA 2: Verificación de agrupamiento por tipos")
print("=" * 60)

# Los 18 tipos de Pokémon (en inglés, como aparecen en el CSV)
TIPOS_POKEMON = [
    'Normal', 'Fire', 'Water', 'Electric', 'Grass', 'Ice',
    'Fighting', 'Poison', 'Ground', 'Flying', 'Psychic', 'Bug',
    'Rock', 'Ghost', 'Dragon', 'Dark', 'Steel', 'Fairy'
]

# Nombre en español de cada tipo, para nombrar los clusters
TIPOS_ES = {
    'normal': 'Normal', 'fire': 'Fuego', 'water': 'Agua', 'electric': 'Eléctrico',
    'grass': 'Planta', 'ice': 'Hielo', 'fighting': 'Lucha', 'poison': 'Veneno',
    'ground': 'Tierra', 'flying': 'Volador', 'psychic': 'Psíquico', 'bug': 'Bicho',
    'rock': 'Roca', 'ghost': 'Fantasma', 'dragon': 'Dragón', 'dark': 'Siniestro',
    'steel': 'Acero', 'fairy': 'Hada', 'unknown': 'Desconocido'
}

# ============================================================
# CONTEO: en 'moves' el tipo aparece pegado al nombre del movimiento
# y seguido de un espacio especial (\xa0) antes de la palabra Power.
# Ejemplo: "Ancient PowerRock\xa0Power60" → movimiento de tipo Rock
# Por eso contamos "Rock\xa0": así solo contamos el tipo de cada movimiento
# y no la palabra "Rock" de "Rock Smash" (que es de tipo Fighting).
# ============================================================
def contar_tipos(texto, tipos):
    """Cuenta cuántos movimientos de cada tipo aprende un Pokémon."""
    conteos = {}
    for tipo in tipos:
        conteos[tipo.lower()] = texto.count(tipo + '\xa0')
    return conteos

conteos_por_pokemon = df['moves'].apply(lambda x: contar_tipos(x, TIPOS_POKEMON))
df_conteos = pd.DataFrame(conteos_por_pokemon.tolist(), index=df.index)

print("\nConteo de tipos por Pokémon - primeras 8 filas:")
print(pd.concat([df['Pokemon'], df_conteos], axis=1).head(8))

def tipo_n(fila, n=1, excluir=None):
    """Retorna el n-ésimo tipo más frecuente (ignora conteo 0 y tipos excluidos)."""
    excluir = excluir or []
    ordenados = sorted(fila.items(), key=lambda x: x[1], reverse=True)
    validos = [(t, c) for t, c in ordenados if c > 0 and t not in excluir]
    return validos[n - 1][0] if len(validos) >= n else 'unknown'

# Columnas de verificación: tipo_1 y tipo_2 (conteo total incluyendo normal)
df['tipo_1'] = df_conteos.apply(lambda f: tipo_n(f, 1), axis=1)
df['tipo_2'] = df_conteos.apply(lambda f: tipo_n(f, 2), axis=1)

# Cada movimiento tiene un solo tipo, así que la suma de los conteos
# es el número de movimientos que aprende el Pokémon
df['num_movimientos'] = df_conteos.sum(axis=1)

print("\ntipo_1 más frecuentes:", df['tipo_1'].value_counts().head(5).to_dict())
print("Casi todos los Pokémon aprenden muchos movimientos Normal, por eso")
print("usamos tipo_2 para ponerle nombre a los clusters.")

# ============================================================
# NOMBRE DE CADA CLUSTER
# ============================================================
def nombrar_cluster(grupo):
    """Nombra un cluster según el tipo_2 que más se repite entre sus Pokémon."""
    # Clusters que no corresponden a un tipo
    if grupo['Pokemon'].str.contains('Arceus').mean() > 0.5:
        return 'Arceus'
    if grupo['num_movimientos'].mean() < 50:
        return 'Pocos movimientos'

    conteo = grupo['tipo_2'].value_counts()
    nombre = TIPOS_ES[conteo.index[0]]
    # Si el tipo más común no llega a la mitad del cluster, es un cluster mixto
    if conteo.iloc[0] < len(grupo) / 2:
        nombre = nombre + ' / ' + TIPOS_ES[conteo.index[1]]
    return nombre

def analizar_clusters(columna, n_clusters, archivo_resumen):
    """Muestra un resumen de cada cluster, lo guarda en un CSV y retorna sus nombres."""
    filas = []
    for c in range(n_clusters):
        grupo = df[df[columna] == c]
        nombre = nombrar_cluster(grupo)
        print(f"\nCluster {c}: {nombre} ({len(grupo)} Pokémon, "
              f"{grupo['num_movimientos'].mean():.0f} movimientos en promedio)")
        print("   tipo_2 más comunes:", grupo['tipo_2'].value_counts().head(3).to_dict())
        print("   Ejemplos:", grupo['Pokemon'].tolist()[:10])
        filas.append({'cluster': c, 'nombre': nombre, 'cantidad': len(grupo),
                      'pokemon': ', '.join(grupo['Pokemon'])})

    resumen = pd.DataFrame(filas)
    resumen.to_csv(archivo_resumen, index=False, encoding='utf-8-sig')
    print(f"\nArchivo '{archivo_resumen}' guardado.")

    # Si un nombre se repite, ese tipo quedó separado en más de un cluster
    repetidos = resumen['nombre'].value_counts()
    print("Tipos separados en más de un cluster:", repetidos[repetidos > 1].to_dict())
    return dict(zip(resumen['cluster'], resumen['nombre']))

# ============================================================
# VERIFICACIÓN DE LOS 18 CLUSTERS DE LA PREGUNTA 1
# ============================================================
print("\n--- Verificación de los 18 clusters de la Pregunta 1 ---")
nombres_p1 = analizar_clusters('cluster_p1', N_CLUSTERS, 'resumen_clusters_p1.csv')
df['nombre_cluster_p1'] = df['cluster_p1'].map(nombres_p1)

columnas_verificacion = ['Pokemon', 'cluster_p1', 'nombre_cluster_p1',
                         'tipo_1', 'tipo_2', 'num_movimientos']
pd.concat([df[columnas_verificacion], df_conteos], axis=1).to_csv(
    'resultado_p1_verificado.csv', index=False, encoding='utf-8-sig')
print("Archivo 'resultado_p1_verificado.csv' guardado.")

# ============================================================
# NUEVO AGRUPAMIENTO CON UN NÚMERO DE CLUSTERS MÁS ADECUADO
# Con 18 clusters, Arceus y los Pokémon con pocos movimientos (Wynaut,
# Caterpie...) ocupan 2 clusters, así que solo quedan 16 para los 18 tipos
# (por ejemplo, Fuego no tiene cluster propio y Agua sale partido en dos).
# Por eso usamos 20 = 18 tipos + 1 para Arceus + 1 para pocos movimientos.
# ============================================================
N_CLUSTERS_FINAL = 20
kmeans_final = KMeans(n_clusters=N_CLUSTERS_FINAL, random_state=10, n_init=10)
kmeans_final.fit(tfidf_matrix_p1)
df['cluster_final'] = kmeans_final.labels_

print(f"\n--- Agrupamiento final con {N_CLUSTERS_FINAL} clusters ---")
nombres_final = analizar_clusters('cluster_final', N_CLUSTERS_FINAL, 'resumen_clusters_final.csv')
df['nombre_cluster_final'] = df['cluster_final'].map(nombres_final)

# CSV final ordenado por cluster, para mostrarlo en Google Sheets o Excel
# (encoding='utf-8-sig' para que Excel muestre bien las tildes)
df_final = df[['cluster_final', 'nombre_cluster_final', 'Pokemon', 'tipo_1', 'tipo_2']]
df_final.sort_values(['cluster_final', 'Pokemon']).to_csv(
    'resultado_final.csv', index=False, encoding='utf-8-sig')
print("Archivo 'resultado_final.csv' guardado.")

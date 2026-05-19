#Rover Marciano — Descenso al cráter sin nodo meta
#  En este ejercicio se pide implementar dos algoritmos de descenso sin nodo meta:
#  Implementación y comparación del Simulated Annealing (SA) con el Descenso de Colina (Greedy).
#  Consta de 2 etapas, la primer implementación del Greedy y luego la implementación del SA

import heapq
import math
import random
import numpy as np
import plotly.graph_objects as go
from skimage.transform import downscale_local_mean


# 1. Carga del archivo del cráter .IMG

input_file = r"C:\Users\jeher\Downloads\Semestre 4\Diseño de agentes inteligentes\crater_map.IMG"

data_file = open(input_file, "rb")

endHeader = False
while not endHeader:
    line     = data_file.readline().rstrip().lower()
    sep_line = line.split(b'=')
    if len(sep_line) == 2:
        itemName  = sep_line[0].rstrip().lstrip()
        itemValue = sep_line[1].rstrip().lstrip()
        if   itemName == b'valid_maximum': maxV      = float(itemValue)
        elif itemName == b'valid_minimum': minV      = float(itemValue)
        elif itemName == b'lines':         n_rows    = int(itemValue)
        elif itemName == b'line_samples':  n_columns = int(itemValue)
        elif itemName == b'map_scale':
            scale_str = itemValue.split()
            if len(scale_str) > 1:
                scale = float(scale_str[0])
    elif line == b'end':
        endHeader = True
        char = 0
        while char == 0 or char == 32:
            char = data_file.read(1)[0]
        data_file.seek(-1, 1)

image_size = n_rows * n_columns
data       = data_file.read(4 * image_size)
data_file.close()

image_data = np.frombuffer(data, dtype=np.dtype('f'))
image_data = image_data.reshape((n_rows, n_columns))
image_data = np.array(image_data, dtype='float64')
image_data = image_data - minV
image_data[image_data < -10000] = -1


# 2. SUBSAMPLING

sub_rate   = round(10 / scale)
image_data = downscale_local_mean(image_data, (sub_rate, sub_rate))
image_data[image_data < 0] = -1
# El nuevo tamaño del pixel es el original multiplicado por el factor de submuestreo
new_scale  = scale * sub_rate
PIXEL_SIZE = new_scale
ROWS, COLS = image_data.shape

print(f"Resolución: {new_scale:.4f} m/píxel")
print(f"Tamaño del mapa: {ROWS} x {COLS} píxeles\n")


# 3. Parámetros globales

MAX_DH    = 2.0    # metros — restricción dura de altura entre pixeles adyacentes
MAX_SLOPE = 0.30   # 30%   — pendiente maxima permitida para evitar daños en el Rover


# 4. Funciones de utilidad

DIRS_8 = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
# Convierte coordenadas del mundo real (en metros) a coordenadas de pixel en la imagen
def world_to_grid(x, y):
    r = int(np.clip(ROWS - round(y / PIXEL_SIZE), 0, ROWS-1))
    c = int(np.clip(round(x / PIXEL_SIZE),         0, COLS-1))
    return (r, c)

def is_valid(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS and image_data[r, c] != -1

def hdist(a, b):
    dx = (b[1] - a[1]) * PIXEL_SIZE
    dy = (b[0] - a[0]) * PIXEL_SIZE
    return math.sqrt(dx*dx + dy*dy)
# Diferencia de altura real entre dos pixeles (puede ser negativa al bajar).
def delta_h(a, b):
    return float(image_data[b] - image_data[a])
# Devuelve los 8 vecinos que están cercanos a pos (incluyendo diagonales) y cumplen las restricciones de altura.
def vecinos_validos(pos):
    resultado = []
    for dr, dc in DIRS_8:
        nb = (pos[0]+dr, pos[1]+dc)
        if not is_valid(*nb):
            continue
        dh   = delta_h(pos, nb)
        if abs(dh) > MAX_DH:          # restriccion dura — no se puede cruzar
            continue
        dist = hdist(pos, nb)
        resultado.append((nb, dh, dist))
    return resultado


# 5. Etapa 1 — Descenso al cráter (Greedy sin nodo meta)

# El Descenso de Colina siempre elige el vecino más bajo (menor altura) en cada paso.
def hill_descent(start):
    current = start
    path    = [start]

    while True:
        vecinos = vecinos_validos(current)

        # Busca el vecino con la altura mas baja
        mejor_nb  = None
        mejor_h   = image_data[current]   # referencia: altura actual

        for nb, dh, dist in vecinos:
            h_nb = image_data[nb]
            if h_nb < mejor_h:            # solo si es mas profundo
                mejor_h  = h_nb
                mejor_nb = nb

        if mejor_nb is None:
            # Ningun vecino es mas bajo → minimo local alcanzado
            break

        current = mejor_nb
        path.append(current)

    return path, float(image_data[current])


# 6. Etapa 2 — Descenso al cráter (Simulated Annealing sin nodo meta)
# Inicia desde el punto establecido y explora aleatoriamente los vecinos, evita caer en mínimos locales

# Parámetros clave:
# T0       — temperatura inicial (alta = mucha exploración al principio)
# cooling  — factor de enfriamiento por iteración
# max_iter — iteraciones máximas
# patience — parada anticipada si no se encuentra punto más bajo en N pasos
def sa_descent(start,
               T0=3000,
               cooling=0.9997,
               max_iter=150000,
               patience=25000):
    current    = start
    trayectoria = [start]           # recorrido completo del SA

    mejor_h    = float(image_data[start])
    mejor_pos  = start
    # Guardamos el camino al mejor punto en el momento en que lo encontramos
    camino_mejor = [start]

    T     = T0
    stag  = 0                       # pasos sin encontrar nuevo minimo

    print(f"  SA Descenso: T0={T0} | enfriamiento={cooling} | max {max_iter:,} iter.")

    for it in range(max_iter):

        vecinos = vecinos_validos(current)

        if not vecinos:
            # Sin vecinos válidos: el explorador esta completamente rodeado
            stag += 1
        else:
            nb, dh, _ = random.choice(vecinos)   # vecino aleatorio válido

            # Criterio de aceptación:
            #   dh < 0 → bajamos → aceptar siempre
            #   dh > 0 → subimos → aceptar con prob exp(-dh/T)
            if dh < 0 or random.random() < math.exp(-dh / max(T, 1e-9)):
                current = nb
                trayectoria.append(current)

                h_actual = float(image_data[current])
                if h_actual < mejor_h:
                    mejor_h       = h_actual
                    mejor_pos     = current
                    camino_mejor  = list(trayectoria)   # snapshot del camino
                    stag          = 0
            else:
                stag += 1

        T *= cooling

        if stag >= patience:
            print(f"  SA: parada anticipada en iteración {it:,} "
                  f"(sin nuevo mínimo en {patience:,} pasos)")
            break
        if T < 1e-9:
            print(f"  SA: temperatura mínima en iteración {it:,}")
            break

    print(f"  SA: punto más profundo en píxel {mejor_pos}  "
          f"|  altura {mejor_h:.2f} m")

    return camino_mejor, mejor_h


# 7. Métricas de evaluación

# Calcula y muestra métricas completas de un camino, incluyendo distancia real 3D y pendiente promedio.
def compute_metrics(path, nombre="Ruta", mass=200, gravity=3.71, rolling=0.02):
    # Parámetros físicos teóricos establecidos para estimar energía gastada del Rover:
    total_horiz   = 0.0
    total_real    = 0.0    # distancia 3D real (horizontal + desnivel)
    total_dh      = 0.0    # cambio acumulado de altura (solo abs)
    total_descenso = 0.0   # metros descendidos en total
    total_energy  = 0.0
    max_slope     = 0.0
    violations    = 0

    for i in range(len(path)-1):
        a, b  = path[i], path[i+1]
        dist  = hdist(a, b)
        dh    = delta_h(a, b)              # negativo = bajando
        slope = abs(dh) / max(dist, 1e-9)

        # Distancia real 3D: hipotenusa entre dist horizontal y desnivel
        real = math.sqrt(dist**2 + dh**2)

        if abs(dh) > MAX_DH:
            violations += 1

        max_slope      = max(max_slope, slope)
        total_horiz   += dist
        total_real    += real
        total_dh      += abs(dh)
        total_descenso += max(-dh, 0)      # solo los tramos que bajan
        total_energy  += (mass * gravity * max(dh, 0)   # energía por subir
                         + rolling * mass * gravity * dist)

    # Pendiente promedio = cambio total de altura / distancia horizontal
    avg_slope = total_dh / max(total_horiz, 1e-9)

    inicio_h = float(image_data[path[0]])
    fin_h    = float(image_data[path[-1]])
    descenso_neto = inicio_h - fin_h       # positivo = el explorador bajo


    print(f"  Métricas — {nombre}")
    print(f"  {'─'*45}")
    print(f"  Nodos recorridos              : {len(path)}")
    print(f"  Distancia horizontal (m)      : {total_horiz:.2f}")
    print(f"  Distancia real 3D (m)         : {total_real:.2f}")
    print(f"  Pendiente promedio            : {avg_slope:.4f}  ")
    print(f"  Pendiente máxima              : {max_slope:.4f}")
    print(f"  Cambio acumulado de altura (m): {total_dh:.2f}")
    print(f"  Descenso neto (m)             : {descenso_neto:.2f}")
    print(f"  Altura inicio (m)             : {inicio_h:.2f}")
    print(f"  Altura final alcanzada (m)    : {fin_h:.2f}")
    print(f"  Energía estimada (J)          : {total_energy:.1f}")
    #print(f"  Violaciones |dh| > {MAX_DH} m   : {violations}")

    return {
        "nodos"          : len(path),
        "dist_horiz"     : round(total_horiz,    2),
        "dist_real_3d"   : round(total_real,     2),
        "pendiente_prom" : round(avg_slope,       4),
        "pendiente_max"  : round(max_slope,       4),
        "descenso_neto"  : round(descenso_neto,  2),
        "altura_final"   : round(fin_h,           2),
        "energia_J"      : round(total_energy,    1),
        "violaciones"    : violations,
    }


# 8. Creación del mapa y trazado de ruta en 3D

def plot_3d(path_hd, path_sa, start):
    x = PIXEL_SIZE * np.arange(COLS)
    y = PIXEL_SIZE * np.arange(ROWS)
    X, Y = np.meshgrid(x, y)

    fig = go.Figure()

    # Superficie del terreno
    fig.add_trace(go.Surface(
        x=X, y=Y,
        z=np.flipud(image_data),
        colorscale='hot',
        cmin=0,
        opacity=1.0,
        lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.5, specular=0.2),
        showscale=True,
        name='Terreno'
    ))

def plot_3d(path_hd, path_sa, path_optima, start):
    x = PIXEL_SIZE * np.arange(COLS)
    y = PIXEL_SIZE * np.arange(ROWS)
    X, Y = np.meshgrid(x, y)

    fig = go.Figure()

    # Superficie del terreno
    fig.add_trace(go.Surface(
        x=X, y=Y,
        z=np.flipud(image_data),
        colorscale='hot',
        cmin=0,
        opacity=1.0,
        lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.5, specular=0.2),
        showscale=True,
        name='Terreno'
    ))

    def add_path(path, color, name, width=4, offset=8):
        arr = np.array(path)
        px  = arr[:,1] * PIXEL_SIZE
        py  = (ROWS - arr[:,0]) * PIXEL_SIZE
        pz  = image_data[arr[:,0], arr[:,1]] + offset
        fig.add_trace(go.Scatter3d(
            x=px, y=py, z=pz,
            mode='lines',
            line=dict(color=color, width=width),
            name=name
        ))
        return px, py, pz

    # Punto de inicio compartido
    sr, sc = start
    fig.add_trace(go.Scatter3d(
        x=[sc * PIXEL_SIZE],
        y=[(ROWS - sr) * PIXEL_SIZE],
        z=[float(image_data[sr, sc]) + 20],
        mode='markers+text',
        marker=dict(size=10, color='lime'),
        text=['Inicio'], textposition='top center',
        name='Inicio'
    ))

    # Ruta Greedy (naranja, tenue)
    if path_hd:
        px, py, pz = add_path(path_hd, 'orange', 'Greedy', width=3, offset=5)
        fig.add_trace(go.Scatter3d(
            x=[px[-1]], y=[py[-1]], z=[pz[-1] + 12],
            mode='markers+text',
            marker=dict(size=8, color='darkorange', symbol='diamond'),
            text=['Mínimo Greedy'], textposition='top center',
            name='Mínimo Greedy'
        ))

    # Trayectoria completa SA (cian, delgada — muestra la exploración)
    if path_sa:
        add_path(path_sa, 'cyan', 'Exploración SA',
                 width=2, offset=10)

    # Ruta óptima SA (verde brillante, gruesa)
    if path_optima:
        px, py, pz = add_path(path_optima, 'darkgreen', 'Ruta Óptima (SA)',
                              width=6, offset=18)
        # Marcador en el punto más profundo
        fig.add_trace(go.Scatter3d(
            x=[px[-1]], y=[py[-1]], z=[pz[-1] + 15],
            mode='markers+text',
            marker=dict(size=11, color='darkgreen', symbol='diamond',
                        line=dict(color='white', width=2)),
            text=['Punto más profundo (SA)'], textposition='top center',
            name='Punto más profundo SA'
        ))

    fig.update_layout(
        title="Rover en Marte — Descenso al Cráter (Greedy vs SA)",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Altura (m)"
        ),
        legend=dict(x=0.01, y=0.99)
    )
    fig.show()


# 9. Ejecución


# Solo se define el punto de INICIO — no hay nodo meta
start_world = (4409, 3365)
start = world_to_grid(*start_world)

print(f"Punto de inicio : {start_world} m  ->  píxel {start}")
print(f"Altura en inicio: {image_data[start]:.2f} m\n")

# Primer algoritmo: Greedy

print("Algoritmo Greedy")
print("__" * 55 + "\n")

path_hd, h_hd = hill_descent(start)

print(f"  Mínimo local alcanzado en pixel {path_hd[-1]}")
print(f"  Altura final: {h_hd:.2f} m")
metrics_hd = compute_metrics(path_hd, "Greedy")

# Segundo algoritmo: Simulated Annealing

print("\nAlgoritmo Simulated Annealing")
print("__" * 55 + "\n")

path_sa, h_sa = sa_descent(start)

metrics_sa = compute_metrics(path_sa, "Simulated Annealing")

# Tabla comparativa 
print("\n")
print("Greedy vs Simulated Annealing")
print(f"{'Tópico':<30} {'Greedy':>12} {'SA':>12}")
print("-" * 55)
comparativas = [
    ("Altura final (m)",       metrics_hd['altura_final'],   metrics_sa['altura_final']),
    ("Descenso neto (m)",      metrics_hd['descenso_neto'],  metrics_sa['descenso_neto']),
    ("Distancia horiz. (m)",   metrics_hd['dist_horiz'],     metrics_sa['dist_horiz']),
    ("Distancia real 3D (m)",  metrics_hd['dist_real_3d'],   metrics_sa['dist_real_3d']),
    ("Pendiente promedio",     metrics_hd['pendiente_prom'], metrics_sa['pendiente_prom']),
    ("Pendiente máxima",       metrics_hd['pendiente_max'],  metrics_sa['pendiente_max']),
    ("Energía estimada (J)",   metrics_hd['energia_J'],      metrics_sa['energia_J']),
    ("Nodos recorridos",       metrics_hd['nodos'],          metrics_sa['nodos']),
]
for label, v_hd, v_sa in comparativas:
    print(f"  {label:<28} {v_hd:>12} {v_sa:>12}")

ganador = "SA" if h_sa < h_hd else "Greedy"
diferencia = abs(h_hd - h_sa)
print(f"\n  Algoritmo que llegó más profundo : {ganador}")
print(f"  Diferencia de profundidad        : {diferencia:.2f} m")

# Ruta corta (extrae y suaviza el camino SA hacia el punto más profundo)

# Reduce el camino SA (elimina el zig-zag)
def smooth_path(path, n_waypoints=80):
    if len(path) <= n_waypoints:
        return list(path)
    idx = np.linspace(0, len(path)-1, n_waypoints, dtype=int)
    return [path[i] for i in idx]

# Construye la ruta óptima recomendada a partir del camino SA:
def extract_optimal_route(path_sa):
    waypoints = smooth_path(path_sa, n_waypoints=80)

    # Re-densificar con Bresenham entre waypoints consecutivos
    ruta_densa = []
    for i in range(len(waypoints)-1):
        a, b   = waypoints[i], waypoints[i+1]
        steps  = max(abs(b[0]-a[0]), abs(b[1]-a[1]), 1)
        seg    = [(int(round(a[0]+(b[0]-a[0])*k/steps)),
                   int(round(a[1]+(b[1]-a[1])*k/steps)))
                  for k in range(steps+1)]
        ruta_densa.extend(seg if i == 0 else seg[1:])

    return ruta_densa

# Extraer y mostrar ruta óptima SA
ruta_optima = extract_optimal_route(path_sa)

print(f"\n  Ruta óptima SA: {len(ruta_optima)} nodos "
      f"(simplificada desde {len(path_sa)} nodos de exploración)")

plot_3d(path_hd, path_sa, ruta_optima, start)
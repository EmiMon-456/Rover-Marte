# Rover Marciano A* + Simulated Annealing vs Greedy
#  El algoritmo A* encuentra la ruta óptima respetando las restricciones duras de altura y pendiente.
#  El algoritmo Simulated Annealing mejora la ruta inicial del A* buscando vecinos.
#  El algoritmo Greedy es mucho más rápido pero puede encontrar rutas subóptimas.

import heapq
import math
import random
import numpy as np
import plotly.graph_objects as go
from skimage.transform import downscale_local_mean


# 1. Carga el archivo .IMG

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
# El nuevo tamaño del pixel es el original multiplicado por el sub_rate
new_scale  = scale * sub_rate
PIXEL_SIZE = new_scale
ROWS, COLS = image_data.shape

print(f"Resolución: {new_scale:.2f} m/píxel")
print(f"Tamaño del mapa: {ROWS} x {COLS} pixeles\n")


# 3. Parámetros globales

MAX_DH    = 2.0    # metros — restricción dura de altura entre pixeles adyacentes
MAX_SLOPE = 0.30   # 30% — pendiente maxima permitida


# 4. UTILIDADES

# Convierte coordenadas del mundo real (metros) a coordenadas de grilla (pixeles).
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
# Costo de altura entre dos pixeles (positivo si sube, negativo si baja).
def delta_h(a, b):
    return float(image_data[b] - image_data[a])
# Bresenham: algoritmo clásico para obtener los pixeles entre dos puntos sin saltos (8-conectividad).
def bresenham(a, b):
    steps = max(abs(b[0]-a[0]), abs(b[1]-a[1]), 1)
    return [(int(round(a[0] + (b[0]-a[0])*i/steps)),
             int(round(a[1] + (b[1]-a[1])*i/steps)))
            for i in range(steps+1)]
# Convierte una lista de waypoints (pocos nodos) en un camino denso interpolado pixel a pixel.
def dense_path(waypoints):
    if len(waypoints) < 2:
        return list(waypoints)
    out = []
    for i in range(len(waypoints)-1):
        seg = bresenham(waypoints[i], waypoints[i+1])
        out.extend(seg if i == 0 else seg[1:])
    return out
# Extrae n waypoints igualmente espaciados del camino denso.
def downsample_path(dense, n=60):
    if len(dense) <= n:
        return list(dense)
    idx = np.linspace(0, len(dense)-1, n, dtype=int)
    return [dense[i] for i in idx]


# 5. Implementación de los algoritmos de búsqueda
#  GREEDY A* (Mejorado con Simulated Annealing)
#  Ambos respetan las mismas restricciones duras de la altura y pendiente entre pixeles adyacentes.

DIRS_8 = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
# Recontruye el camino denso desde el diccionario came_from (usado por ambos algoritmos).
def _reconstruct(came_from, goal):
    wp, node = [], goal
    while node is not None:
        wp.append(node)
        node = came_from[node]
    wp.reverse()
    return dense_path(wp)
# Devuelve vecinos que respetan restricciones duras de altura y pendiente.
def _neighbors_validos(cur):
    validos = []
    for dr, dc in DIRS_8:
        nb = (cur[0]+dr, cur[1]+dc)
        if not is_valid(*nb):
            continue
        dh = delta_h(cur, nb)
        if abs(dh) > MAX_DH:
            continue
        dist  = hdist(cur, nb)
        slope = abs(dh) / max(dist, 1e-9)
        if slope > MAX_SLOPE * 1.5:
            continue
        validos.append((nb, dh, dist))
    return validos


# GREEDY
def greedy(start, goal):
    def h(n):
        return math.sqrt((n[0]-goal[0])**2 + (n[1]-goal[1])**2) * PIXEL_SIZE

    heap      = [(h(start), start)]
    visited   = {start}
    came_from = {start: None}
    g_real    = {start: 0.0}

    while heap:
        _, cur = heapq.heappop(heap)

        if cur == goal:
            path = _reconstruct(came_from, goal)
            return path, g_real[goal]

        for nb, dh, dist in _neighbors_validos(cur):
            if nb in visited:
                continue
            visited.add(nb)
            came_from[nb] = cur
            g_real[nb]    = g_real[cur] + dist + 3.0 * max(dh, 0)
            heapq.heappush(heap, (h(nb), nb))

    return None, float('inf')

# A*
def astar(start, goal):
    def h(n):
        return math.sqrt((n[0]-goal[0])**2 + (n[1]-goal[1])**2) * PIXEL_SIZE

    heap      = [(h(start), 0.0, start)]
    g         = {start: 0.0}
    came_from = {start: None}

    while heap:
        _, g_cur, cur = heapq.heappop(heap)

        if cur == goal:
            return _reconstruct(came_from, goal), g_cur

        if g_cur > g.get(cur, float('inf')) + 1e-9:
            continue

        for nb, dh, dist in _neighbors_validos(cur):
            step_g = g[cur] + dist + 3.0 * max(dh, 0)
            if step_g < g.get(nb, float('inf')):
                g[nb]         = step_g
                came_from[nb] = cur
                heapq.heappush(heap, (step_g + h(nb), step_g, nb))

    return None, float('inf')

# 6. Costo del segmento

# El costo del segmento a→b se valida pixel a pixel usando Bresenham.
def segment_cost(a, b, slope_weight=60.0):
    pixels = bresenham(a, b)
    total  = 0.0
    for i in range(len(pixels)-1):
        p, q = pixels[i], pixels[i+1]
        if not is_valid(*p) or not is_valid(*q):
            return float('inf')
        dh    = delta_h(p, q)
        if abs(dh) > MAX_DH:
            return float('inf')
        dist  = hdist(p, q)
        slope = abs(dh) / max(dist, 1e-9)
        if slope > MAX_SLOPE * 2:
            return float('inf')
        pen    = slope_weight * (slope - MAX_SLOPE)**2 if slope > MAX_SLOPE else 0.0
        total += dist + pen
    return total
# El costo total de una lista de waypoints es la suma de los costos de los segmentos entre ellos.
def waypoints_cost(wps):
    total = 0.0
    for i in range(len(wps)-1):
        c = segment_cost(wps[i], wps[i+1])
        if c == float('inf'):
            return float('inf')
        total += c
    return total


# 7. Simulated Annealing (sobre waypoints dispersos del camino de A*)

# Tres operaciones sobre waypoints con validacion rapida (fail-fast).
# Retorna None si el vecino viola restricciones (descartado sin copiar lista).
def neighbor_wp(wps, max_delta=10):
    n  = len(wps)
    op = random.random()

    # A) Mover waypoint intermedio
    if op < 0.50:
        idx  = random.randint(1, n-2)
        r, c = wps[idx]
        nr   = int(np.clip(r + random.randint(-max_delta, max_delta), 0, ROWS-1))
        nc   = int(np.clip(c + random.randint(-max_delta, max_delta), 0, COLS-1))
        if not is_valid(nr, nc):
            return None
        # Validación rápida solo de los dos segmentos afectados
        if (segment_cost(wps[idx-1], (nr, nc)) == float('inf') or
            segment_cost((nr, nc),   wps[idx+1]) == float('inf')):
            return None
        new      = list(wps)
        new[idx] = (nr, nc)
        return new

    # B) Insertar waypoint de desvío
    elif op < 0.75:
        idx    = random.randint(0, n-2)
        r1, c1 = wps[idx]
        r2, c2 = wps[idx+1]
        mr = int(np.clip((r1+r2)//2 + random.randint(-max_delta, max_delta), 0, ROWS-1))
        mc = int(np.clip((c1+c2)//2 + random.randint(-max_delta, max_delta), 0, COLS-1))
        if not is_valid(mr, mc):
            return None
        if (segment_cost(wps[idx],   (mr, mc)) == float('inf') or
            segment_cost((mr, mc), wps[idx+1]) == float('inf')):
            return None
        new = list(wps)
        new.insert(idx+1, (mr, mc))
        return new

    # C) Eliminar waypoint redundante
    else:
        if n <= 4:
            return None
        idx = random.randint(1, n-2)
        if segment_cost(wps[idx-1], wps[idx+1]) == float('inf'):
            return None
        new = list(wps)
        del new[idx]
        return new

# Función principal de Simulated Annealing
def simulated_annealing(dense_init,
                        n_waypoints=100,
                        T0=1500,
                        cooling=0.995,
                        max_iter=30000,
                        patience=10000):
    wps  = downsample_path(dense_init, n_waypoints)
    cur  = list(wps)
    c_g  = waypoints_cost(cur)
    best = list(cur)
    b_g  = c_g
    T    = T0
    stag = 0

    print(f"  SA: {len(cur)} waypoints | {max_iter:,} iteraciones max.")

    for it in range(max_iter):
        cand = neighbor_wp(cur)

        if cand is None:
            stag += 1
        else:
            cand_g = waypoints_cost(cand)
            delta  = cand_g - c_g

            if delta < 0 or (T > 1e-9 and random.random() < math.exp(-delta / T)):
                cur, c_g = cand, cand_g
                if c_g < b_g:
                    best, b_g = list(cur), c_g
                    stag = 0
            else:
                stag += 1

        T *= cooling
        if stag >= patience:
            print(f"  SA: parada anticipada en iteración {it:,}")
            break
        if T < 1e-9:
            print(f"  SA: temperatura minima en iteración {it:,}")
            break

    # Devuelve camino denso final para visualizacion y métricas
    return dense_path(best), b_g

# 8. Métricas de evaluación

def compute_metrics(path, mass=200, gravity=3.71, rolling=0.02):
    total_dist = total_energy = total_dh = max_slope = 0.0
    violations = 0
    for i in range(len(path)-1):
        a, b   = path[i], path[i+1]
        dist   = hdist(a, b)
        dh     = delta_h(a, b)
        slope  = abs(dh) / max(dist, 1e-9)
        if abs(dh) > MAX_DH:
            violations += 1
        max_slope     = max(max_slope, slope)
        total_dist   += dist
        total_dh     += abs(dh)
        total_energy += mass*gravity*max(dh,0) + rolling*mass*gravity*dist
    return {
        "Distancia horizontal (m)"       : round(total_dist,   2),
        "Cambio acumulado de altura (m)" : round(total_dh,     2),
        "Energía estimada (J)"           : round(total_energy, 1),
        "Pendiente máxima"               : round(max_slope,    4),
        "Nodos en ruta"                  : len(path),
        #"Violaciones dh > 2 m"          : violations,
    }

# 9. Visualización 3D  (muestra las 3 rutas superpuestas)

def plot_3d(path_greedy, path_astar, path_sa):
    x = PIXEL_SIZE * np.arange(COLS)
    y = PIXEL_SIZE * np.arange(ROWS)
    X, Y = np.meshgrid(x, y)

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=X, y=Y,
        z=np.flipud(image_data),
        colorscale='hot',
        cmin=0,
        opacity=1.0,
        lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.5, specular=0.2),
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

    if path_greedy:
        add_path(path_greedy, 'orange', 'Greedy',    width=3, offset=5)
    if path_astar:
        add_path(path_astar,  'green', 'A*',     width=3, offset=10)
    if path_sa:
        px, py, pz = add_path(path_sa, 'cyan', 'SA', width=5, offset=15)
        for i, label, color in [(0,'Inicio','lime'), (-1,'Meta','red')]:
            fig.add_trace(go.Scatter3d(
                x=[px[i]], y=[py[i]], z=[pz[i]+12],
                mode='markers+text',
                marker=dict(size=9, color=color),
                text=[label], textposition='top center',
                name=label
            ))

    fig.update_layout(
        title="Rover en Marte — Greedy vs A* vs SA",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Altura (m)"
        ),
        legend=dict(x=0.01, y=0.99)
    )
    fig.show()

# 10. Ejecución y comparación

start_world = (3350, 5800)
goal_world  = (2943, 4962)

start = world_to_grid(*start_world)
goal  = world_to_grid(*goal_world)

print(f"Inicio : {start_world} m  ->  pixel {start}")
print(f"Meta   : {goal_world} m  ->  pixel {goal}\n")

# --- GREEDY ------------------------------------------------------------------
print("\n"+"="*10+"GREEDY"+"="*10)

path_greedy, cost_greedy = greedy(start, goal)

if path_greedy is None:
    print("Greedy no encontro ruta valida.")
    print("Prueba aumentar MAX_DH o MAX_SLOPE al inicio del script.")
else:
    print(f"Ruta Greedy: {len(path_greedy)} nodos")#  |  costo {cost_greedy:.2f}
    for k, v in compute_metrics(path_greedy).items():
        print(f"  {k}: {v}")

# --- A* ----------------------------------------------------------------------
print("\n"+"="*10+"A*"+"="*10)

path_astar, cost_astar = astar(start, goal)

if path_astar is None:
    print("A* no encontro ruta valida.")
else:
    print(f"Ruta A*:     {len(path_astar)} nodos") # costo {cost_astar:.2f}
    for k, v in compute_metrics(path_astar).items():
        print(f"  {k}: {v}")

# --- SA sobre la ruta Greedy -------------------------------------------------
seed_path = path_astar if path_astar is not None else path_greedy

if seed_path is None:
    print("\nNo hay ruta válida para ejecutar SA.")
else:
    print("\n"+"="*10+"SIMULATED ANNEALING"+"="*10)

    path_sa, cost_sa = simulated_annealing(seed_path)

    print(f"Ruta SA:     {len(path_sa)} nodos") #costo {cost_sa:.2f}
    for k, v in compute_metrics(path_sa).items():
        print(f"  {k}: {v}")

    # Tabla comparativa
    print("\n"+"="*10+"TABLA COMPARATIVA"+"="*10)
    print(f"{'Algoritmo':<12} {'Nodos':>8} {'Costo':>12}")
    print("-" * 35)
    if path_greedy: print(f"{'Greedy':<12} {len(path_greedy):>8} {cost_greedy:>12.2f}")
    if path_astar:  print(f"{'A*':<12} {len(path_astar):>8} {cost_astar:>12.2f}")
    print(f"{'SA':<12} {len(path_sa):>8} {cost_sa:>12.2f}")

    if path_greedy:
        mejora_vs_greedy = 100*(cost_greedy - cost_sa) / max(cost_greedy, 1e-9)
        print(f"\nMejora SA vs Greedy : {mejora_vs_greedy:.1f}%")
    if path_astar:
        diff_greedy_astar = 100*(cost_greedy - cost_astar) / max(cost_astar, 1e-9) if path_greedy else 0
        print(f"Greedy vs A* (costo): {diff_greedy_astar:+.1f}%")

    plot_3d(path_greedy, path_astar, path_sa)
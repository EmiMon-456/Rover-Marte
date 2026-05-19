
import copy
import numpy as np
import heapq
import random
import math
import matplotlib.pyplot as plt
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import plotly.graph_objects as go




input_file = r"C:\Users\jeher\Downloads\Semestre 4\Diseño de agentes inteligentes\mars_map.IMG"

data_file = open(input_file, "rb")

endHeader = False
while not endHeader:
    line = data_file.readline().rstrip().lower()
    sep_line = line.split(b'=')

    if len(sep_line) == 2:
        itemName = sep_line[0].strip()
        itemValue = sep_line[1].strip()

        if itemName == b'valid_maximum':
            maxV = float(itemValue)
        elif itemName == b'valid_minimum':
            minV = float(itemValue)
        elif itemName == b'lines':
            n_rows = int(itemValue)
        elif itemName == b'line_samples':
            n_columns = int(itemValue)
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
data = data_file.read(4 * image_size)

image_data = np.frombuffer(data, dtype=np.dtype('f'))
image_data = image_data.reshape((n_rows, n_columns))
image_data = image_data.astype('float64')

image_data = image_data - minV
image_data[image_data < -10000] = -1



sub_rate = round(10 / scale)
image_data = downscale_local_mean(image_data, (sub_rate, sub_rate))
image_data[image_data < 0] = -1

new_scale = scale * sub_rate
print("Sub-sampling:", sub_rate)
print("Nueva escala:", new_scale, "metros por píxel")



MAX_HEIGHT_DIFF = 0.25
PIXEL_SIZE = new_scale

def world_to_grid(x, y, n_rows, scale):
    r = n_rows - round(y / scale)
    c = round(x / scale)
    return (int(r), int(c))

def path_cost(path, heights):

    total_cost = 0.0

    for i in range(len(path) - 1):

        a = path[i]
        b = path[i+1]

        dx = (b[0] - a[0]) * PIXEL_SIZE
        dy = (b[1] - a[1]) * PIXEL_SIZE
        horizontal = np.sqrt(dx**2 + dy**2)

        height_diff = heights[b] - heights[a]

        if heights[b] == -1:
            return float("inf")

        if abs(height_diff) > MAX_HEIGHT_DIFF:
            return float("inf")

        slope_penalty = abs(height_diff) / max(horizontal, 1e-6)

        total_cost += horizontal + 5.0 * slope_penalty

    return total_cost


def initial_path(start, goal):

    path = []
    steps = max(abs(goal[0]-start[0]), abs(goal[1]-start[1]))

    for i in range(steps+1):
        r = int(start[0] + (goal[0]-start[0]) * i/steps)
        c = int(start[1] + (goal[1]-start[1]) * i/steps)
        path.append((r,c))

    return path

def generate_neighbor(path, heights):

    new_path = path.copy()

    if len(new_path) < 3:
        return new_path

    idx = random.randint(1, len(new_path)-2)

    r, c = new_path[idx]

    moves = [(1,0),(-1,0),(0,1),(0,-1)]
    dr, dc = random.choice(moves)

    nr = r + dr
    nc = c + dc

    rows, cols = heights.shape

    if 0 <= nr < rows and 0 <= nc < cols:
        new_path[idx] = (nr, nc)

    return new_path

def simulated_annealing(start, goal, heights,
                        T0=1000, cooling=0.995, max_iter=5000):

    current_path = initial_path(start, goal)
    current_cost = path_cost(current_path, heights)

    best_path = current_path
    best_cost = current_cost

    T = T0

    for i in range(max_iter):

        candidate = generate_neighbor(current_path, heights)
        candidate_cost = path_cost(candidate, heights)

        delta = candidate_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / T):
            current_path = candidate
            current_cost = candidate_cost

            if current_cost < best_cost:
                best_path = current_path
                best_cost = current_cost

        T *= cooling

        if T < 1e-3:
            break

    return best_path, best_cost


start_world = (2850, 6400) 
goal_world  = (3150, 6800)

start = world_to_grid(start_world[0], start_world[1], image_data.shape[0], new_scale)
goal  = world_to_grid(goal_world[0],  goal_world[1],  image_data.shape[0], new_scale)

print("Píxel inicial:", start)
print("Píxel final:", goal)

sa_path, sa_cost = simulated_annealing(start, goal, image_data)

print("Nodos en ruta SA:", len(sa_path))


path, cost = simulated_annealing(start, goal, image_data)

if path is None:
    print("No se encontró camino válido.")
else:
    print("Camino encontrado con", len(path), "nodos recorridos (pixeles).")


x = new_scale * np.arange(image_data.shape[1])
y = new_scale * np.arange(image_data.shape[0])
X, Y = np.meshgrid(x, y)

fig = go.Figure()

fig.add_trace(go.Surface(
    x=X,
    y=Y,
    z=np.flipud(image_data),
    colorscale='hot',
    cmin=0
))

if path is not None:
    path = np.array(path)
    path_x = path[:,1] * new_scale
    path_y = (image_data.shape[0] - path[:,0]) * new_scale
    path_z = image_data[path[:,0], path[:,1]]

    fig.add_trace(go.Scatter3d(
        x=path_x,
        y=path_y,
        z=path_z,
        mode='lines',
        line=dict(color='cyan', width=5),
        name='Ruta óptima'
    ))

fig.update_layout(
    title="Exploración Rover en Marte (SA)",
    scene=dict(
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        zaxis_title="Altura (m)"
    )
)

fig.show()

path, cost = simulated_annealing(start, goal, image_data)


if path is not None:

    total_distance = 0.0
    total_horizontal = 0.0
    total_height_change = 0.0

    for i in range(len(path) - 1):

        a = path[i]
        b = path[i + 1]

        dx_pix = b[0] - a[0]
        dy_pix = b[1] - a[1]

        dx = dx_pix * PIXEL_SIZE
        dy = dy_pix * PIXEL_SIZE

        horizontal_dist = np.sqrt(dx**2 + dy**2)
        height_diff = image_data[b] - image_data[a]

        segment_dist = np.sqrt(horizontal_dist**2 + height_diff**2)

        total_distance += segment_dist
        total_horizontal += horizontal_dist
        total_height_change += abs(height_diff)

    num_pixels = len(path)

    avg_slope = total_height_change / max(total_horizontal, 1e-6)

    print("Distancia horizontal total: {:.2f} m".format(total_horizontal))
    print("Distancia real recorrida (3D): {:.2f} m".format(total_distance))
    print("Cambio acumulado de altura: {:.2f} m".format(total_height_change))
    print("Pendiente promedio:", round(avg_slope, 4))
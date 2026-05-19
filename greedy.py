import numpy as np
from simpleai.search import SearchProblem, greedy
import time

mars_map = np.load(r"C:\Users\tamar\Downloads\mars_map.npy")


nr, nc = mars_map.shape 
escala = 10.0174

start_pos = (int(nr - round(6400 / escala)), int(round(2850 / escala)))
goal_pos = (int(nr - round(6800 / escala)), int(round(3150 / escala)))

MAX_SLOPE = 0.25 

altura_inicio = mars_map[start_pos[0], start_pos[1]]
print(f"La altura en el punto de inicio {start_pos} es: {altura_inicio}")


r, c = start_pos
print("Alturas de los vecinos:")
for dr in [-1, 0, 1]:
    for dc in [-1, 0, 1]:
        vecino_r, vecino_c = r + dr, c + dc
        alt = mars_map[vecino_r, vecino_c]
        diff = abs(alt - altura_inicio) if alt != -1 else "N/A"
        print(f"Vecino ({vecino_r}, {vecino_c}): Altura={alt}, Dif={diff}")

class MarteAgente(SearchProblem):
    def __init__(self, initial_state):
        super().__init__(initial_state=initial_state)
        self.count = 0 

    def actions(self, state):
        self.count += 1
        
        if self.count % 100 == 0:
            print(f"Nodos explorados: {self.count} | Posición actual: {state}")
            
        r, c = state
        acciones_validas = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                new_r, new_c = r + dr, c + dc
                if 0 <= new_r < nr and 0 <= new_c < nc:
                    h_actual = mars_map[r, c]
                    h_vecino = mars_map[new_r, new_c]
                    if h_vecino != -1 and abs(h_vecino - h_actual) < MAX_SLOPE:
                        acciones_validas.append((new_r, new_c))
        return acciones_validas

    def result(self, state, action):
        return action

    def is_goal(self, state):
        return state == goal_pos

    def heuristic(self, state): #eucladiana 
        return np.sqrt((state[0] - goal_pos[0])**2 + (state[1] - goal_pos[1])**2)


print(f"Iniciando greedy de {start_pos} a {goal_pos}...")
problema = MarteAgente(initial_state=start_pos)

start_time = time.time() 
resultado = greedy(problema)
end_time = time.time()  

total_time = end_time - start_time

print(total_time)
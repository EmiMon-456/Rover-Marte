import numpy as np
import time


ruta_mapa = r"C:\Users\tamar\Downloads\mars_map.npy"
mars_map = np.load(ruta_mapa)
nr, nc = mars_map.shape
escala = 10.0174
MAX_SLOPE = 0.25

start_pos = (int(nr - round(6400 / escala)), int(round(2850 / escala)))
goal_pos = (int(nr - round(6800 / escala)), int(round(3150 / escala)))


def es_meta(estado):
    return estado == goal_pos

def heuristica(estado): #eucladiana 
    return np.sqrt((estado[0] - goal_pos[0])**2 + (estado[1] - goal_pos[1])**2)

def obtener_vecinos_validos(estado):
    r, c = estado
    vecinos = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0: continue
            nr_n, nc_n = r + dr, c + dc
            if 0 <= nr_n < nr and 0 <= nc_n < nc:
                h_actual = mars_map[r, c]
                h_vecino = mars_map[nr_n, nc_n]
                if h_vecino != -1 and abs(h_vecino - h_actual) < MAX_SLOPE:
                    vecinos.append((nr_n, nc_n))
    return vecinos


def local_beam_search(inicio, meta, k):
   
    fringe = [inicio]
    padres = {inicio: None} 
    
    nivel = 0
    while fringe:
        nivel += 1
        if nivel % 200 == 0:
            print(f"Buscando... Profundidad: {nivel} | Mejor pos actual: {fringe[0]}")
            
        
        for estado in fringe:
            if es_meta(estado):
                ruta = []
                actual = estado
                while actual is not None:
                    ruta.append(actual)
                    actual = padres[actual]
                ruta.reverse()
                return ruta
                
       
        sucesores = []
        for estado in fringe:
            for vecino in obtener_vecinos_validos(estado):
                if vecino not in padres: 
                    padres[vecino]= estado 
                    sucesores.append(vecino)
                    
        if not sucesores:
            return None 
            
        
        sucesores.sort(key=heuristica)
        
        
        fringe = sucesores[:k]
        
    return None

if __name__ == "__main__":
    print("Iniciando local beam search ")
    print(f"Desde: {start_pos} ---> hasta: {goal_pos}")
    
    k_haz = 30 #num exploradores simultaneos 
    print(f"Lanzando {k_haz} exploradores simultáneos...\n")
    
    tiempo_inicio = time.time()
    ruta_final = local_beam_search(start_pos, goal_pos, k=k_haz)
    tiempo_total = time.time() - tiempo_inicio

    if ruta_final:
        print(f"Punto objetivo encontrado")
        print(f"Tiempo total: {tiempo_total:.2f} segundos")
        print(f"Longitud de la ruta real: {len(ruta_final)} pasos")
        print("nodos expandidos/explorados:",k_haz*len(ruta_final) )
    else:
        print("Exploradores atorados")
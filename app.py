from flask import Flask, render_template, request, jsonify
import concurrent.futures
from itertools import permutations
import numpy as np
import random
import os
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

#-------------------------------------------------reglas------------------------------------------------------------------
def check_cond_ini(matrix):
    n = len(matrix)
    matrixt = list(zip(*matrix))
    for col in matrixt:
        if (col.count(1) > n/2) or (col.count(0) > n/2):
            return False
        for i in range(n-2):
            if col[i] == col[i+1] == col[i+2] != -1:
                return False
    for row in matrix:
        if (row.count(1) > n/2) or (row.count(0) > n/2):
            return False
        for i in range(n-2):
            if row[i] == row[i+1] == row[i+2] != -1:
                return False
    filled = [row for row in matrix if sum(row) == (n)/2]
    for i in range(len(filled)):
        for j in range(i+1, n):
            if matrix[i] == matrix[j]:
                return False
    return True

def check_rows(matrix):
    n = len(matrix)
    for row in matrix:
        for i in range(n-2):
            if row[i] == row[i+1] == row[i+2]:
                return False
        if sum(row) != int((n)/2):
            return False
    return True

def check_cols(matrix):
    n = len(matrix)
    matrixt = list(zip(*matrix))
    for col in matrixt:
        for i in range(n-2):
            if col[i] == col[i+1] == col[i+2]:
                return False
        if sum(col) != int((n)/2):
            return False
    return True

def check_duplicates(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i+1, n):
            if matrix[i] == matrix[j]:
                return False
            if list(zip(*matrix))[i] == list(zip(*matrix))[j]:
                return False
    return True

def rules(matrix):
    return check_rows(matrix) and check_cols(matrix) and check_duplicates(matrix)

#-------------------------------------------------verificación------------------------------------------------------------------
def has_solution(matrix, combinaciones, row, col):
    num_posiciones = len(row)
    c = 0
    sols = []
    for combinacion in combinaciones:
        count = 0
        for i in range(num_posiciones):
            if matrix[row[i]][col[i]] == combinacion[row[i]][col[i]]:
                count = count + 1
        if count == num_posiciones:
            if rules(combinacion):
                sols.append(combinacion)
                c += 1
        del count
    return c, sols

def no_three_consecutive(perm):
    for i in range(len(perm) - 2):
        if perm[i] == perm[i+1] == perm[i+2]:
            return False
    return True

#---------------------------------------backtracking-------------------------------------------------------
steps = []
def backtracking(board, row=0, col=0):
    n = len(board)

    if row == n:  # Se completó el tablero
        return board if rules(board) else 0

    next_row, next_col = (row, col + 1) if col + 1 < n else (row + 1, 0)

    if board[row][col] != -1:  # Si la celda ya está rellenada, avanza
        return backtracking(board, next_row, next_col)

    # Prueba con 1
    board[row][col] = 0
    aux = [row[:] for row in board]
    steps.append(aux)
    if check_cond_ini(board):
        solved = backtracking(board, next_row, next_col)
        if solved:
            return solved

    # Prueba con 2
    board[row][col] = 1
    aux = [row[:] for row in board]
    steps.append(aux)
    if check_cond_ini(board):
        solved = backtracking(board, next_row, next_col)
        if solved:
            return solved

    # Backtrack
    board[row][col] = -1
    aux = [row[:] for row in board]
    steps.append(aux)
    return 0


#---------------------------------------------cumpliendo reglas-----------------------------------------------------
def similar_col(index,sudoku,fila_completada):
    n = len(sudoku)
    sudoku = [list(fila) for fila in zip(*sudoku)]
    sudoku[index] = fila_completada
    for i, fila in enumerate(sudoku):
        if i != index and sum(fila) == n/2:
            contador = 0
            for j in range(len(fila)):
                if fila[j] == fila_completada[j] != -1:
                    contador += 1
            if contador == n-2:
                return True
            del contador
    return False

def repeat_col(fila_index,matriz,fila_completada):    
    # Compara la fila con todas las demás filas
    matriz = [list(fila) for fila in zip(*matriz)]
    for i, fila in enumerate(matriz):
        if i != fila_index and fila_completada.tolist() == fila:
            return True  # Hay una fila duplicada
    return False

def similar_row(index,sudoku,fila_completada):
    n = len(sudoku)
    sudoku[index] = fila_completada
    for i, fila in enumerate(sudoku):
        if i != index and sum(fila) == n/2:
            contador = 0
            for j in range(len(fila)):
                if fila[j] == fila_completada[j] != -1:
                    contador += 1
            if contador == n-2:
                return True
            del contador
    return False

def repeat_row(fila_index,matriz,fila_completada):    
    # Compara la fila con todas las demás filas
    for i, fila in enumerate(matriz):
        if i != fila_index and fila_completada.tolist() == fila.tolist():
            return True  # Hay una fila duplicada
    return False

def aplicar_reglas_basicas(sudoku):
    n = len(sudoku)
    cambios = True
    
    while cambios:
        cambios = False

        # Aplicar reglas fila por fila
        for i in range(n):
            fila = sudoku[i]

            # Evitar tres números consecutivos en filas
            for j in range(1, n - 1):
                if fila[j - 1] == fila[j] != -1 and fila[j + 1] == -1:
                    sudoku[i][j + 1] = 1 - fila[j]
                    steps.append(sudoku.tolist())
                    cambios = True
                elif fila[j + 1] == fila[j] != -1 and fila[j - 1] == -1:
                    sudoku[i][j - 1] = 1 - fila[j]
                    steps.append(sudoku.tolist())
                    cambios = True
                elif fila[j - 1] == fila[j + 1] != -1 and fila[j] == -1:
                    sudoku[i][j] = 1 - fila[j - 1]
                    steps.append(sudoku.tolist())
                    cambios = True
            
            # Si hay exactamente la mitad de ceros o unos, rellenar con el valor opuesto
            if np.count_nonzero(fila == 0) == n // 2:
                if sum(fila) != n/2:
                    sudoku[i] = np.where(fila == -1, 1, fila)
                    steps.append(sudoku.tolist())
                    cambios = True
            elif np.count_nonzero(fila == 1) == n // 2:
                if sum(fila) != n/2:
                    sudoku[i] = np.where(fila == -1, 0, fila)
                    steps.append(sudoku.tolist())
                    cambios = True
                    
            # Evitar filas duplicadas
            if fila.tolist().count(-1) == 2:
                if similar_row(i,sudoku.copy(),fila):
                    aux = fila.copy()
                    vacia = [j for j in range(n) if fila[j] == -1]
                    combinaciones_validas = [(0, 1), (1, 0)]
                    for valor1, valor2 in combinaciones_validas:
                        aux[vacia[0]] = valor1
                        aux[vacia[1]] = valor2
                        # Verifica si la fila completada es única
                        if not repeat_row(i,sudoku.copy(),aux):
                            sudoku[i] = aux.copy()
                            steps.append(sudoku.tolist())
                            cambios = True
                            
        # Aplicar reglas columna por columna
        for j in range(n):
            columna = sudoku[:,j]

            # Evitar tres números consecutivos en columnas
            for i in range(1, n - 1):
                if columna[i - 1] == columna[i] != -1 and columna[i + 1] == -1:
                    sudoku[i + 1][j] = 1 - columna[i]
                    steps.append(sudoku.tolist())
                    cambios = True
                elif columna[i + 1] == columna[i] != -1 and columna[i - 1] == -1:
                    sudoku[i - 1][j] = 1 - columna[i]
                    steps.append(sudoku.tolist())
                    cambios = True
                elif columna[i - 1] == columna[i + 1] != -1 and columna[i] == -1:
                    sudoku[i][j] = 1 - columna[i - 1]
                    steps.append(sudoku.tolist())
                    cambios = True
            
            # Si hay exactamente la mitad de ceros o unos, rellenar con el valor opuesto
            if np.count_nonzero(columna == 0) == n // 2:
                if sum(columna) != n/2:
                    sudoku[:, j] = np.where(columna == -1, 1, columna)
                    steps.append(sudoku.tolist())
                    cambios = True
            elif np.count_nonzero(columna == 1) == n // 2:
                if sum(columna) != n/2:
                    sudoku[:, j] = np.where(columna == -1, 0, columna)
                    steps.append(sudoku.tolist())
                    cambios = True
        
            # Evitar columnas duplicadas
            if columna.tolist().count(-1) == 2:
                if similar_col(j,sudoku.copy(),columna):
                    aux = columna.copy()
                    vacia = [j for j in range(n) if columna[j] == -1]
                    combinaciones_validas = [(0, 1), (1, 0)]
                    for valor1, valor2 in combinaciones_validas:
                        aux[vacia[0]] = valor1
                        aux[vacia[1]] = valor2
                        # Verifica si la fila completada es única
                        if not repeat_col(j,sudoku.copy(),aux):
                            sudoku[:,j] = aux.copy()
                            steps.append(sudoku.tolist())
                            cambios = True

#-------------------------------------------renderizar imagen-------------------------------------------------------

lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
lower_blue = np.array([95, 190, 160])
upper_blue = np.array([110, 255, 215])

# Función para procesar la imagen y generar la matriz
def process_image(image, n):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    matrix = []
    height, width, _ = image.shape
    cell_height = height // n
    cell_width = width // n

    for i in range(n):
        row = []
        for j in range(n):
            red_count = cv2.countNonZero(red_mask[i * cell_height: (i + 1) * cell_height, j * cell_width: (j + 1) * cell_width])
            blue_count = cv2.countNonZero(blue_mask[i * cell_height: (i + 1) * cell_height, j * cell_width: (j + 1) * cell_width])

            if red_count > blue_count and red_count > 100:
                row.append(0)
            elif blue_count > red_count and blue_count > 100:
                row.append(1)
            else:
                row.append(-1)
        matrix.append(row)

    return matrix

# Ruta para procesar la imagen y devolver la matriz
@app.route('/process_image', methods=['POST'])
def process_image_route():
    file = request.files['image']
    n = int(request.form['matrix_size'])

    img = Image.open(file.stream)
    img = img.convert('RGB')
    open_cv_image = np.array(img)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convertir de RGB a BGR

    matrix = process_image(open_cv_image, n)
    return jsonify(matrix)

#--------------------------------------------------------------------------------------------------------------------
@app.route('/')
def menu():
    return render_template('menu.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/solve')
def solve_page():
    return render_template('solve.html')

@app.route('/play')
def play_page():
    return render_template('play.html')

@app.route('/solve/solve_fuerza/matrix', methods=['POST'])
def solve_matrix():
    matrix = request.json['matrix']
    if check_cond_ini(matrix):
        col = []
        row = []
        n = len(matrix)
        for i in range(n):
            for j in range(n): 
                if matrix[i][j] != -1:
                    row.append(i)
                    col.append(j)

        num_posiciones = len(row)
        num_hilos = os.cpu_count()

        op = [1 if i < n/2 else 0 for i in range(n)]
        
        permutaciones = list(set(permutations(op)))
        permutaciones = [perm for perm in permutaciones if no_three_consecutive(perm)]
        combinaciones = list(permutations(permutaciones, n))
        comb = len(combinaciones)
        chunks = [combinaciones[int(comb*i/num_hilos):int(comb*(i+1)/num_hilos)] for i in range(num_hilos)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_hilos) as executor:
            futures = [executor.submit(has_solution, matrix, chunk, row, col) for chunk in chunks]

        results = [result.result() for result in concurrent.futures.as_completed(futures)]

        solutions_count = sum(list(zip(*results))[0])
        solutions = [sol for sublist in list(zip(*results))[1] for sol in sublist]

        return jsonify({'solutions_count': solutions_count, 'solutions': solutions})
    else:
        return jsonify({'solutions_count': 0, 'solutions': []})

@app.route('/solve/solve_backtracking/matrix', methods=['POST'])
def backtracking1():
    board = request.json['matrix']
    sol = backtracking(board)
    aux = steps.copy()
    steps.clear()
    return jsonify({'solution': sol, 'steps': aux})

@app.route('/solve/solve_by_rules/matrix', methods=['POST'])
def resolver_sudoku_binario():
    sudoku = np.array(request.json['matrix'], dtype=np.int32)
    aplicar_reglas_basicas(sudoku)
    aux = steps.copy()
    steps.clear()
    if rules([list(row) for row in sudoku]):
        return jsonify({'solution': 1, 'steps': aux})
    else:
        return jsonify({'solution': 0, 'steps': aux})

@app.route('/play/matrix', methods=['POST'])
def play_matrix():
    # Configuración del juego
    n = 4  # Tamaño de la matriz
    while True:
        cond_ini = []
        for i in range(5):
            while True:
                a = random.choices(range(n), k=2)
                if a not in cond_ini:
                    break
            cond_ini.append(a)

        matrix = [[-1] * n for _ in range(n)]  # Matriz para llevar el control de los colores
        matrix[cond_ini[0][0]][cond_ini[0][1]], matrix[cond_ini[1][0]][cond_ini[1][1]], matrix[cond_ini[2][0]][cond_ini[2][1]] = 0, 0, 0
        matrix[cond_ini[3][0]][cond_ini[3][1]], matrix[cond_ini[4][0]][cond_ini[4][1]] = 1, 1
        if check_cond_ini(matrix):
            col = []
            row = []
            n = len(matrix)
            for i in range(n):
                for j in range(n):
                    if matrix[i][j] != -1:
                        row.append(i)
                        col.append(j)

            num_posiciones = len(row)
            num_hilos = os.cpu_count()

            op = [1 if i < n/2 else 0 for i in range(n)]
            
            permutaciones = list(set(permutations(op)))
            permutaciones = [perm for perm in permutaciones if no_three_consecutive(perm)]
            combinaciones = list(permutations(permutaciones, n))
            comb = len(combinaciones)
            chunks = [combinaciones[int(comb*i/num_hilos):int(comb*(i+1)/num_hilos)] for i in range(num_hilos)]

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_hilos) as executor:
                futures = [executor.submit(has_solution, matrix, chunk, row, col) for chunk in chunks]

            results = [result.result() for result in concurrent.futures.as_completed(futures)]

            if sum(list(zip(*results))[0]) == 1:
                break

    return jsonify({'matrix': matrix})

if __name__ == "__main__":
    app.run(debug=True)

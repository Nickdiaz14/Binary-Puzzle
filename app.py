from flask import Flask, render_template, request, jsonify
from itertools import permutations
from constraint import Problem
import concurrent.futures
import numpy as np
import threading
import psycopg2
import random
import json
import ast
import os

app = Flask(__name__)

#------------------------------------------------varibles globales-------------------------------------------------------
result_found = False
result_matrix = None
result_lock = threading.Lock()
steps = []
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

def generate_permutations(n):
    if n % 2 != 0:
        return []  # No es posible si n no es par

    def backtrack(seq, zeros_left, ones_left, last_two):
        # Caso base: hemos generado una secuencia de longitud n
        if len(seq) == n:
            yield seq
            return

        # Intentamos agregar un '0' si quedan ceros y no hay dos ceros consecutivos
        if zeros_left > 0 and (len(last_two) < 2 or last_two != [0, 0]):
            yield from backtrack(seq + [0], zeros_left - 1, ones_left, (last_two[-1:] + [0])[-2:])

        # Intentamos agregar un '1' si quedan unos y no hay dos unos consecutivos
        if ones_left > 0 and (len(last_two) < 2 or last_two != [1, 1]):
            yield from backtrack(seq + [1], zeros_left, ones_left - 1, (last_two[-1:] + [1])[-2:])

    # Inicializamos la búsqueda con secuencia vacía, n/2 ceros y n/2 unos
    return backtrack([], n // 2, n // 2, [])

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
        permutaciones = list(generate_permutations(n))
        combinaciones = list(permutations(permutaciones, n))
        comb = len(combinaciones)
        chunks = [combinaciones[int(comb*i/num_hilos):int(comb*(i+1)/num_hilos)] for i in range(num_hilos)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_hilos) as executor:
            futures = [executor.submit(has_solution, matrix, chunk, row, col) for chunk in chunks]

        results = [result.result() for result in concurrent.futures.as_completed(futures)]

        solutions_count = sum(list(zip(*results))[0])
        solutions = [sol for sublist in list(zip(*results))[1] for sol in sublist]
        if len(solutions) == 0:
            return jsonify({'solution': 0})
        else:
            return jsonify({'solution': solutions})
    else:
        return jsonify({'solution': 0})

#---------------------------------------backtracking-------------------------------------------------------
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

@app.route('/solve/solve_backtracking/matrix', methods=['POST'])
def backtracking1():
    board = request.json['matrix']
    sol = backtracking(board)
    aux = steps.copy()
    steps.clear()
    return jsonify({'solution': sol, 'steps': aux})


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


#------------------------------------------------algoritmo genetico-------------------------------------------------------

def check_rows_ga(matrix):
    check = []
    n = len(matrix)
    for row in matrix:
        for i in range(n-2):
            if row[i] == row[i+1] == row[i+2]:
                check.append(2)
        if sum(row) != int((n)/2):
            check.append(3)
    return check

def check_cols_ga(matrix):
    check = []
    n = len(matrix)
    matrixt = list(zip(*matrix))
    for col in matrixt:
        for i in range(n-2):
            if col[i] == col[i+1] == col[i+2]:
                check.append(2)
        if sum(col) != int((n)/2):
            check.append(3)
    return check

def check_duplicates_ga(matrix):
    check = []
    n = len(matrix)
    for i in range(n):
        for j in range(i+1, n):
            if list(matrix[i]) == list(matrix[j]):
                check.append(4)
            if list(zip(*matrix))[i] == list(zip(*matrix))[j]:
                check.append(4)
    return check

def enforce_constraints(individual,mtx,ubi):
    check = []
    for r,c in ubi:
        if individual[r][c] != mtx[r][c]:
            check.append(5)
    return check

# Fitness function example: lower is better (goal is 0 fitness)
def fitness_fn(individual,mtx,ubi):
    violations = count_rule_violations(individual,mtx,ubi)
    # 2: No more than two consecutive zeros and two consecutive ones are allowed
    # 3: The number of zeros and ones in every row and every column are equal.
    # 4: Each row has a distinct permutation of n/2 zeros and n/2 ones.
    # 5: Follows the initial condition.
    return (10 * violations.count(2) + 8 * violations.count(3) + 7 * violations.count(4) + 15 * violations.count(5)) ** 2

# Function to randomly select an individual based on fitness
def random_selection(population, fitness_fn,mtx,ubi):
    weights = [1 / (fitness_fn(ind,mtx,ubi) + 1e-6) for ind in population]
    return random.choices(population, weights=weights, k=1)[0]

# Function to perform crossover between two individuals
def crossover(x, y):
    if random.random() < 0.6:  # 60% chance to transpose
        x, y = np.transpose(x), np.transpose(y)
    
    n = len(x)
    point = random.randint(1, n - 1)  # Crossover en un solo punto
    child = np.vstack((x[:point], y[point:]))
    return child


# Function to mutate a child with some probability
def mutate(child, mutation_rate):
    if random.random() < mutation_rate:
        row1, row2 = random.sample(range(len(child)), 2)  # Intercambia dos filas
        child[[row1, row2]] = child[[row2, row1]]
    a = random.choices(range(len(child)),k=2)
    child[a[0]][a[1]] = 1 - random.randint(0,1)
    return child

# Elitism: preserve the top N fittest individuals
def elitism(population, fitness_fn, elitism_count,mtx,ubi):
    return sorted(population, key=lambda individual: fitness_fn(individual,mtx,ubi))[:elitism_count]

# Genetic Algorithm function
def genetic_algorithm(population, fitness_fn, mutation_rate, elitism_count, mtx, ubi):
    global result_found, result_matrix
    cont = 0
    
    while not result_found:  # Continuar solo si no se ha encontrado una solución
        new_population = []
        
        # Generar nueva población
        for _ in range(len(population)):
            x = random_selection(population, fitness_fn, mtx, ubi)
            y = random_selection(population, fitness_fn, mtx, ubi)
            child = crossover(x, y)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        
        # Aplicar elitismo
        elite = elitism(population, fitness_fn, elitism_count, mtx, ubi)
        population = new_population + elite
        
        # Verificar la mejor solución de la nueva población
        best_individual = min(population, key=lambda individual: fitness_fn(individual, mtx, ubi))
        best_fitness = fitness_fn(best_individual, mtx, ubi)
        print(f"Gen: {cont}, Hilo: {threading.current_thread().name[-1]}, BFS: {best_fitness} FTS: {count_rule_violations(best_individual,mtx,ubi)}")
        # Si se encuentra una solución con fitness 0
        if best_fitness == 0:
            with result_lock:  # Proteger el acceso a las variables compartidas
                if not result_found:
                    result_found = True
                    result_matrix = best_individual
        
        cont += 1
        if cont == 20:  # Limitar a 20 iteraciones
            break
    
    return None  # Si no se encuentra una solución válida

# Example rule violation counter (you need to implement the actual rules)
def count_rule_violations(individual,mtx,ubi):
    violations = check_cols_ga(individual)
    violations.extend(check_duplicates_ga(individual))
    violations.extend(check_rows_ga(individual))
    violations.extend(enforce_constraints(individual,mtx,ubi))
    return violations

@app.route('/solve/solve_genetic_algorithm/matrix', methods=['POST'])
def genetic_algorithm1():
    global result_found, result_matrix
    MUTATION_RATE = 0.01
    ELITISM_COUNT = 2
    POPULATION_SIZE = 50
    inicial_matrix = np.array(request.json['matrix'], dtype=np.int32)
    chromosome_length = len(inicial_matrix)
    
    # Ubicaciones no -1 en la matriz inicial
    ubi = [(i, j) for i in range(chromosome_length) for j in range(chromosome_length) if inicial_matrix[i][j] != -1]
    
    # Generar vectores de permutaciones y dividir la población entre los hilos
    vectors = list(generate_permutations(chromosome_length))
    num_hilos = os.cpu_count()
    populations = [[np.vstack(random.sample(vectors, chromosome_length)) for _ in range(POPULATION_SIZE)] for _ in range(num_hilos)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_hilos) as executor:
        futures = [executor.submit(genetic_algorithm, population, fitness_fn, MUTATION_RATE, ELITISM_COUNT, inicial_matrix, ubi) for population in populations]

        for future in concurrent.futures.as_completed(futures):
            if result_found:
                break  # Terminar inmediatamente si se encontró una solución válida
    
    if result_matrix is not None:
        aux = result_matrix.copy()
        result_found = False
        result_matrix = None
        return jsonify({'solution': 1, 'steps': [aux.tolist() for i in range(2)]})
    else:
        return jsonify({'solution': 0, 'steps': []})

#------------------------------constraint problem---------------------------------------------------------------------
def binary_puzzle_solver(n, matrix, ubi):
    if n % 2 != 0:
        raise ValueError("Puzzle size must be even for equal 0s and 1s.")
    
    # Create a constraint problem
    problem = Problem()
    
    # Define the grid variables (each cell of the grid)
    grid = [[f"x{i}-{j}" for j in range(n)] for i in range(n)]
    
    # Add variables: each cell can be either 0 or 1
    for row in grid:
        for cell in row:
            problem.addVariable(cell, [0, 1])
    
    # Add constraints to ensure equal number of 0s and 1s in each row
    for row in grid:
        problem.addConstraint(lambda *row: row.count(0) == row.count(1), row)
    
    # Add constraints to ensure equal number of 0s and 1s in each column
    for col in range(n):
        problem.addConstraint(lambda *col: col.count(0) == col.count(1), [grid[row][col] for row in range(n)])
    
    # Add constraints to prevent more than two consecutive 0s or 1s in rows
    for row in grid:
        for i in range(n - 2):
            problem.addConstraint(lambda x, y, z: not (x == y == z), (row[i], row[i+1], row[i+2]))
    
    # Add constraints to prevent more than two consecutive 0s or 1s in columns
    for col in range(n):
        for i in range(n - 2):
            problem.addConstraint(lambda x, y, z: not (x == y == z), (grid[i][col], grid[i+1][col], grid[i+2][col]))
    
    # Add constraints for predefined values from 'ubi'
    for i, j in ubi:
        problem.addConstraint(lambda x, v=matrix[i][j]: x == v, [f"x{i}-{j}"])
    
    # Ensuring that all rows are unique
    for i in range(n):
        for j in range(i + 1, n):
            problem.addConstraint(lambda *row_cells: row_cells[:n] != row_cells[n:], 
                                  [grid[i][k] for k in range(n)] + [grid[j][k] for k in range(n)])

    # Ensuring that all columns are unique
    for i in range(n):
        for j in range(i + 1, n):
            problem.addConstraint(lambda *col_cells: col_cells[:n] != col_cells[n:], 
                                  [grid[k][i] for k in range(n)] + [grid[k][j] for k in range(n)])
    
    # Solve the puzzle (returns all solutions)
    solutions = problem.getSolutions()
    
    # If solutions are found, format and return all of them
    all_solutions = []
    if solutions:
        for solution in solutions:
            solution_grid = [[solution[f"x{i}-{j}"] for j in range(n)] for i in range(n)]
            all_solutions.append(solution_grid)
        return all_solutions
    else:
        return []

@app.route('/solve/solve_constraint_problem/matrix', methods=['POST'])
def constraint_solver ():
    matrix = request.json['matrix']
    n = len(matrix)  # Puzzle size (must be even)

    ubi = [(i, j) for i in range(n) for j in range(n) if matrix[i][j] != -1]
    solutions = binary_puzzle_solver(n, matrix, ubi)
    return jsonify({'solution': solutions})
#--------------------------------------------------------------------------------------------------------------------
@app.route('/')
def index():
    ch = request.args.get('ch')
    if ch:
        return render_template('register.html', ind = "si")
    return render_template('register.html')

@app.route('/solve')
def solve_page():
    return render_template('solve.html')

@app.route('/play')
def play_page():
    return render_template('play.html')

@app.route('/levels')
def levels_page():
    return render_template('levels.html')

@app.route('/menu')
def menu():
    id = request.args.get('userID')
    nom = request.args.get('nickname')
    ch = request.args.get('ch')
    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute("""
    SELECT userid FROM nickname WHERE userid = %s;
    """,(id,))
    flag = cursor.fetchone()

    if flag:
        if ch:
            cursor.execute("""
            SELECT nickname FROM nickname WHERE nickname = %s AND userid != %s;
            """, (nom,id))
            prueba = cursor.fetchone()

            if prueba:
                return render_template('register.html',message = "Este nombre de usuario ya está en uso", ind = "si")
            
            cursor.execute("""
            UPDATE nickname
            SET nickname = %s
            WHERE userid = %s;
            """, (nom,id))

        cursor.execute("""
        SELECT nickname FROM nickname WHERE userid = %s;
        """, (id,))
        final_nom = cursor.fetchone()[0]
    else:
        cursor.execute("""
        INSERT INTO nickname (userid, nickname)
        VALUES (%s, %s);
        """, (id, nom))
        final_nom = nom

    # Guardar los cambios en la base de datos
    connection.commit()

    # Cerrar la conexión
    cursor.close()
    connection.close()

    return render_template('menu.html', nickname = final_nom)
#----------------------------------------------------------------------------------------------------------------------------------------
# Conectar a Supabase
def connect_db():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port="5432"
    )

# Insertar o actualizar un registro en la base de datos
def update_leaderboard(board, userid, time_string, total_time):
    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute("""
    SELECT COUNT(*) FROM leaderboard WHERE board = %s;
    """, (board,))
    count = cursor.fetchone()[0]

    # Comprobar si el registro con el mismo nombre ya existe
    cursor.execute("""
    SELECT total_time FROM leaderboard WHERE board = %s AND userid = %s;
    """, (board, userid))

    existing_entry = cursor.fetchone()      

    if existing_entry:
        # Si ya existe, comparar el total_time
        existing_time = existing_entry[0]
        if total_time < existing_time:
            # Si el nuevo total_time es menor, actualiza el registro
            cursor.execute("""
            UPDATE leaderboard
            SET time_string = %s, total_time = %s
            WHERE board = %s AND userid = %s;
            """, (time_string, total_time, board, userid))
    else:
        if count >= 10:
            cursor.execute("""
            SELECT id, total_time FROM leaderboard WHERE board = %s ORDER BY total_time DESC LIMIT 1;
            """, (board,))
            max_entry = cursor.fetchone()

            # Si el nuevo tiempo es menor que el máximo, reemplazar el registro con el tiempo más alto
            if total_time < max_entry[1]:
                cursor.execute("""
                DELETE FROM leaderboard WHERE id = %s;
                """, (max_entry[0],))

                # Insertar el nuevo registro
                cursor.execute("""
                INSERT INTO leaderboard (board, userid, time_string, total_time)
                VALUES (%s, %s, %s, %s);
                """, (board, userid, time_string, total_time))
        else:
            # Si no existe, insertar un nuevo registro
            cursor.execute("""
            INSERT INTO leaderboard (board, userid, time_string, total_time)
            VALUES (%s, %s, %s, %s);
            """, (board, userid, time_string, total_time))

    # Guardar los cambios en la base de datos
    connection.commit()

    # Cerrar la conexión
    cursor.close()
    connection.close()

# Ejemplo de uso

def get_top_scores(board):
    connection = connect_db()
    cursor = connection.cursor()

    # Consultar los 5 mejores registros
    cursor.execute("""
    SELECT b.nickname, a.time_string
    FROM nickname b
    INNER JOIN leaderboard a
    ON b.userid = a.userid
    WHERE board = %s ORDER BY total_time ASC LIMIT 10;
    """, (board,))

    results = cursor.fetchall()

    cursor.close()
    connection.close()
    
    return results

@app.route('/leaderboard')
def leader_page():
    tsecs = int(request.args.get('totaltime'))
    id = request.args.get('userID')
    n = int(request.args.get('n'))
    board = f'T{n}'
    update_leaderboard(board, id,f'{(tsecs//6000):02}:{((tsecs%6000)//100):02}.{(tsecs%100):02}', tsecs)
    return render_template('leaderboard.html', board=f'{n}x{n}', data=json.dumps(get_top_scores(board)))

@app.route('/leaderboards')
def leaders_page():
    dictionary = {'T4':list(get_top_scores('T4')),'T6':list(get_top_scores('T6')),'T8':list(get_top_scores('T8')),'T10':list(get_top_scores('T10'))}
    return render_template('leaderboards.html', data=json.dumps(dictionary))

@app.route('/display_level')
def display_levels_page():
    level = int(request.args.get('level'))
    n = int(request.args.get('n'))
    with open(f'retos/aleatorios{n}.txt', 'r', encoding='utf-8') as file:
        lineas = file.readlines()
        linea_especifica = lineas[level - 1].strip()  # Remueve espacios en blanco y saltos de línea
    matrix = eval(linea_especifica)
    return render_template('display_level.html', matrix=json.dumps(matrix), n=n, level = level)

@app.route('/play/matrix', methods=['POST'])
def play_matrix():
    n = request.json['matrix'] # Tamaño de la matriz
    with open(f'retos/aleatorios{n}.txt', 'r', encoding='utf-8') as file:
        lineas = file.readlines()
        linea_especifica = lineas[random.randint(0, len(lineas)-1)].strip()  # Remueve espacios en blanco y saltos de línea
    matrix = eval(linea_especifica)

    return jsonify({'matrix': matrix})

if __name__ == "__main__":
    app.run()

#%% ------------------------- 0hh1 -------------------------
import random
import numpy as np

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
                    cambios = True
                elif fila[j + 1] == fila[j] != -1 and fila[j - 1] == -1:
                    sudoku[i][j - 1] = 1 - fila[j]
                    cambios = True
                elif fila[j - 1] == fila[j + 1] != -1 and fila[j] == -1:
                    sudoku[i][j] = 1 - fila[j - 1]
                    cambios = True
            
            # Si hay exactamente la mitad de ceros o unos, rellenar con el valor opuesto
            if np.count_nonzero(fila == 0) == n // 2:
                if sum(fila) != n/2:
                    sudoku[i] = np.where(fila == -1, 1, fila)
                    cambios = True
            elif np.count_nonzero(fila == 1) == n // 2:
                if sum(fila) != n/2:
                    sudoku[i] = np.where(fila == -1, 0, fila)
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
                            cambios = True
                            
        # Aplicar reglas columna por columna
        for j in range(n):
            columna = sudoku[:,j]

            # Evitar tres números consecutivos en columnas
            for i in range(1, n - 1):
                if columna[i - 1] == columna[i] != -1 and columna[i + 1] == -1:
                    sudoku[i + 1][j] = 1 - columna[i]
                    cambios = True
                elif columna[i + 1] == columna[i] != -1 and columna[i - 1] == -1:
                    sudoku[i - 1][j] = 1 - columna[i]
                    cambios = True
                elif columna[i - 1] == columna[i + 1] != -1 and columna[i] == -1:
                    sudoku[i][j] = 1 - columna[i - 1]
                    cambios = True
            
            # Si hay exactamente la mitad de ceros o unos, rellenar con el valor opuesto
            if np.count_nonzero(columna == 0) == n // 2:
                if sum(columna) != n/2:
                    sudoku[:, j] = np.where(columna == -1, 1, columna)
                    cambios = True
            elif np.count_nonzero(columna == 1) == n // 2:
                if sum(columna) != n/2:
                    sudoku[:, j] = np.where(columna == -1, 0, columna)
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
                            cambios = True

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


for ii in range(2,6):
    n = ii*2
    print(n)
    if n == 4:
        cells = 5
    elif n == 6:
        cells = 11
    elif n == 8:
        cells = 24
    else:
        cells = 40
    cont = 0
    while True:
        matrix = [[-1] * n for _ in range(n)]  # Matriz para llevar el control de los colores
        for i in range(cells):
            a = random.choices(range(n), k=2)
            matrix[a[0]][a[1]] = random.randint(0,1)
        matrix_solution = np.array(matrix, dtype=np.int32)
        if check_cond_ini(matrix):
            aplicar_reglas_basicas(matrix_solution)
            if rules([list(row) for row in matrix_solution]):
                level = ""
                with open(f'aleatorios{n}.txt', 'a', encoding='utf-8') as file:
                    for row in matrix:
                        level += str(row) + ","
                    # Agrega información al archivo
                    file.write(f'{level[:-1]}\n')
                cont += 1
        if cont == 400:
            break

#%%
for ii in range(2,6):
    n = ii*2
    output_file = f"aleatorios{n}.txt"

    # Usar un conjunto para eliminar duplicados
    lineas_unicas = set()

    # Leer archivo y almacenar líneas únicas
    with open(output_file, "r") as archivo:
        for linea in archivo:
            lineas_unicas.add(linea)

    # Escribir líneas únicas en el nuevo archivo
    with open(output_file, "w") as archivo:
        archivo.writelines(sorted(lineas_unicas))  # Puedes usar sorted si quieres que las líneas estén ordenadas

    print(f"Archivo sin duplicados guardado como {output_file}")

#%% ------------------------- 0h-n0 -------------------------


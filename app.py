from flask import Flask, render_template, request, jsonify
import psycopg2
import random
import json
import os

app = Flask(__name__)

@app.route('/play/matrix', methods=['POST'])
def play_matrix():
    n = request.json['matrix'] # Tamaño de la matriz
    with open(f'retos/aleatorios{n}.txt', 'r', encoding='utf-8') as file:
        lineas = file.readlines()
        linea_especifica = lineas[random.randint(0, len(lineas)-1)].strip()  # Remueve espacios en blanco y saltos de línea
    matrix = eval(linea_especifica)

    return jsonify({'matrix': matrix})

@app.route('/memory', methods=['POST'])
def memory():
    cells = int(request.json['cells']) + 3
    n = int(request.json['size'])
    matrix = [[-1] * n for _ in range(n)]  # Matriz para llevar el control de los colores
    inds = []
    for i in range(cells):
        while True:
            a = random.choices(range(n), k=2)
            if a not in inds:
                inds.append(a)
                if n == 6:
                    matrix[a[0]][a[1]] = 0
                else:
                    matrix[a[0]][a[1]] = random.randint(0,1)
                break
    return jsonify({'matrix': matrix})

@app.route('/sequence', methods=['POST'])
def sequence():
    mode = request.json['mode']
    if mode:
        cells = 40
    else:
        cells = int(request.json['cells']) + 3
    n = int(request.json['size'])
    inds = [random.choices(range(n)) for _ in range(cells*2)]
    return jsonify({'places': inds})

@app.route('/speed', methods=['POST'])
def speed():
    n = int(request.json['n'])
    matrix = [[-1] * n for _ in range(n)]
    cells = [(i,j) for i in range(n) for j in range(n)]
    random.shuffle(cells)
    for index, (i,j) in enumerate(cells):
        matrix[i][j] = index+1
    return jsonify({'matrix': matrix})

@app.route('/leaderboard/update', methods=['POST'])
def leader_update():
    game = request.json['game']
    point = request.json['score']
    id = request.json['userID']
    if game == '0hh1':
        score = int(point)
        n = int(request.json['n'])
        board = f'T{n}'
        better = update_leaderboard(board, id,f'{(score//6000):02}:{((score%6000)//100):02}.{(score%100):02}', score)
        return jsonify({'better': better,'score':score,'board':board})
    elif game == 'TT0hh1':
        score = int(point)
        board = "TContrareloj"
        better = update_leaderboard(board, id,f'{score} tabs', score)
        return jsonify({'better': better,'score':score,'board':board})
    elif game == 'mindgrid1':
        score = int(point)
        board = "TUnicolor"
        better = update_leaderboard(board, id,f'{score} tabs', score)
        return jsonify({'better': better,'score':score,'board':board})
    elif game == 'mindgrid2':
        score = int(point)
        board = "TBicolor"
        better = update_leaderboard(board, id,f'{score} tabs', score)
        return jsonify({'better': better,'score':score,'board':board})
    elif game == 'speed':
        score = int(point)
        board = "TSpeed"
        better = update_leaderboard(board, id,f'{(score//6000):02}:{((score%6000)//100):02}.{(score%100):02}', score)
        return jsonify({'better': better,'score':score,'board':board})
    elif game == 'knight':
        score = float(point)
        board = "TKnight"
        better = update_leaderboard(board, id, round(score,2), score*100)
        return jsonify({'better': better,'score':score,'board':board})
    else:
        mode = request.json['mode']
        board = f'T{mode}'
        better = update_leaderboard(board, id,f'{score} tabs', score)
        return jsonify({'better': better,'score':score,'board':board})
#--------------------------------------------------------------------------------------------------------------------
@app.route('/')
def index():
    id = request.args.get('userID')
    ch = request.args.get('ch')
    if ch:
        return render_template('register.html', ind = "si")
    if id:
        return render_template('register.html', ind = id)
    return render_template('register.html')

@app.route('/solve')
def solve_page():
    return render_template('solve.html')

@app.route('/knight')
def knight_page():
    return render_template('knight.html')

@app.route('/tutorial')
def tutorial_page():
    return render_template('tutorial.html')

@app.route('/sequence')
def sequence_page():
    return render_template('sequence.html')

@app.route('/speed')
def speed_page():
    return render_template('speed.html')

@app.route('/time_trial_0hh1')
def time_trial_page():
    return render_template('time_trial_0hh1.html')

@app.route('/0hh1')
def ohhi_page():
    return render_template('0hh1.html')

@app.route('/0hn0')
def ohno_page():
    return render_template('0hn0.html')

@app.route('/memory')
def memory_page():
    return render_template('memory.html')

@app.route('/levels')
def levels_page():
    return render_template('levels.html')

@app.route('/leaderboard')
def leader_page():
    id = request.args.get('userID')
    better = request.args.get('better')
    board = request.args.get('board')
    finish = request.args.get('finished')
    if finish is None:
        if better:
            if len(board) <= 3:
                return render_template('leaderboard.html', board=f'{board[1:]}x{board[1:]}', data=json.dumps(get_top_scores(board)), best = True, message = "¡Superaste tu record!")
            else:
                return render_template('leaderboard.html', board=board[1:] if board != 'TSpeed' else "CuentaManía", data=json.dumps(get_top_scores(board)), best = True, message = "¡Superaste tu record!")
        elif board in ['T4', 'T6', 'T8', 'T10', 'TSpeed']:
            score = int(request.args.get('score'))
            if board == 'TSpeed':
                aux = 'CuentaManía'
            else:
                aux = f'{board[1:]}x{board[1:]}'
            return render_template('leaderboard.html', board= aux, data=json.dumps(get_top_scores(board)), best = better, message = f'¡Hiciste {(score//6000):02}:{((score%6000)//100):02}.{(score%100):02}, bien hecho!')
        elif board in ['TKnight']:
            score = round(float(request.args.get('score')),2)
            return render_template('leaderboard.html', board= board[1:], data=json.dumps(get_top_scores(board)), best = better, message = f'¡Hiciste {score} puntos, bien hecho!')
        else:
            score = int(request.args.get('score'))
            return render_template('leaderboard.html', board=board[1:], data=json.dumps(get_top_scores(board)), best = better, message = f'¡Hiciste {score} tableros, bien hecho!')
    else:
        return render_template('leaderboard.html', board="CuentaManía", data=json.dumps(get_top_scores(board)), message = f'¡Intentalo nuevamente!')

@app.route('/leaderboards')
def leaders_page():
    game = request.args.get('game')
    dictionary = {}
    if game == '0h-h1':
        boards = ['T4', 'T6', 'T8', 'T10', 'TContrareloj']
    elif game == 'MindGrid':
        boards = ['TUnicolor', 'TBicolor', 'TProgresivo', 'TAleatorio','TSpeed']
    else:
        boards = ['TKnight']
    for board in boards:
        dictionary[board] = list(get_top_scores(board))
    return render_template('leaderboards.html', data=json.dumps(dictionary), game = game)

@app.route('/menu')
def menu():
    id = request.args.get('userID')
    nom = str(request.args.get('nickname'))
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
            SELECT nickname FROM nickname WHERE LOWER(nickname) = %s AND userid != %s;
            """, (nom.lower(),id))
            prueba = cursor.fetchone()
            if prueba:
                if prueba[0].lower() == nom.lower():
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
        SELECT nickname FROM nickname WHERE LOWER(nickname) = %s AND userid != %s;
        """, (nom.lower(),id))
        prueba = cursor.fetchone()
        if prueba:
            if prueba[0].lower() == nom.lower():
                return render_template('register.html',message = "Este nombre de usuario ya está en uso", ind = "si")
            
        cursor.execute("""
        INSERT INTO nickname (userid, nickname)
        VALUES (%s, %s);
        """, (id, nom))
        final_nom = nom

        connection.commit()
        cursor.close()
        connection.close()   
        return render_template('tutorial.html', init = True)     

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
    comp = board in ['T4', 'T6', 'T8', 'T10', 'TSpeed']
    better = False
    connection = connect_db()
    cursor = connection.cursor()

    # Comprobar si el registro con el mismo nombre ya existe
    cursor.execute("""
    SELECT total_time FROM leaderboard WHERE board = %s AND userid = %s;
    """, (board, userid))

    existing_entry = cursor.fetchone()      

    if existing_entry:
        # Si ya existe, comparar el total_time
        existing_time = existing_entry[0]
        if (comp and total_time < existing_time) or ((not comp) and total_time > existing_time):
            better = True
            # Si el nuevo total_time es menor, actualiza el registro
            cursor.execute("""
            UPDATE leaderboard
            SET time_string = %s, total_time = %s
            WHERE board = %s AND userid = %s;
            """, (time_string, total_time, board, userid))
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
    return better

# Ejemplo de uso

def get_top_scores(board):
    comp = board in ['T4', 'T6', 'T8', 'T10', 'TSpeed']
    connection = connect_db()
    cursor = connection.cursor()

    # Consultar los 5 mejores registros
    order = "ASC" if comp else "DESC"
    query = f"""
    SELECT nickname,time_string, userid FROM public.leader_final_view
    WHERE board = %s ORDER BY total_time {order};
    """
    cursor.execute(query,(board,))

    results = cursor.fetchall()

    cursor.close()
    connection.close()
    
    return results

if __name__ == "__main__":
    app.run()

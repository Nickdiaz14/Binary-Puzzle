from flask import Flask, render_template, request, jsonify
import psycopg2
import random
import json
import os

app = Flask(__name__)

@app.route('/games', methods=['POST'])
def tendencia():
    game = str(request.json['game'])
    num = float(request.json['num'])
    valid = True
    aproach = -1
    connection = connect_db()
    cursor = connection.cursor()

    
    cursor.execute("""
        SELECT number FROM public.tendencias
        WHERE aproach = 1 AND board = %s""",(game,))
    
    flag = cursor.fetchone()

    if not flag:
        valid = False
        cursor.execute("""
            SELECT number FROM public.tendencias
            WHERE aproach = -1 AND board = %s""",(game,))
        
        oficial_num = cursor.fetchone()

        if not oficial_num:
            cursor.execute("""
                INSERT INTO tendencias (number, aproach, board)
                VALUES (%s, %s, %s);""",(random.randint(0,10000),-1,game))
            
            cursor.execute("""
                SELECT number FROM public.tendencias
                WHERE aproach = -1 AND board = %s""",(game,))
            
            oficial_num = cursor.fetchone()
            print(oficial_num)

        oficial_num = int(oficial_num[0])
        aproach = (num)/oficial_num
        percentage = f'{round(aproach,5)}'

        if aproach == 1:
            color = '#f1c40f'
        elif aproach > 1:
            color = '#237a04'
        else:
            color = '#f00'

        cursor.execute("""
            INSERT INTO tendencias (number, aproach, percentage, color, board)
            VALUES (%s, %s, %s, %s, %s);""",(int(num), aproach, percentage, color,game))

    connection.commit()
    cursor.close()
    connection.close()
    return jsonify({'flag': valid, 'aproach': aproach})
#--------------------------------------------------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('menu.html')

@app.route('/win')
def win_page():
    return render_template('win.html')

@app.route('/input')
def input_page():
    game = request.args.get('game')
    return render_template('input.html',game=game)

@app.route('/leaderboard')
def leader_page():
    game = request.args.get('game')
    return render_template('leaderboard.html',data=json.dumps(get_top_scores(game)))

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

def get_top_scores(game):
    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT subquery.number, subquery.percentage, subquery.color
        FROM (
            SELECT CAST(number AS INTEGER) AS number, percentage, color, aproach
            FROM public.tendencias
            WHERE aproach != -1 AND board = %s
            ORDER BY ABS(aproach-1) ASC
            LIMIT 10
        ) AS subquery
        ORDER BY subquery.aproach ASC;
    """,(game,))

    results = cursor.fetchall()

    cursor.close()
    connection.close()
    
    return results

if __name__ == "__main__":
    app.run()

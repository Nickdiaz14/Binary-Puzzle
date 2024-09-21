let matrix = [];

function createTable(n) {
    matrix = Array.from({ length: n }, () => Array(n).fill(0));
    const table = document.getElementById('matrix');
    table.innerHTML = '';
    for (let i = 0; i < n; i++) {
        const row = document.createElement('tr');
        for (let j = 0; j < n; j++) {
            const cell = document.createElement('td');
            cell.className = 'grey';
            cell.onclick = () => toggleColor(i, j, cell);
            row.appendChild(cell);
        }
        table.appendChild(row);
    }
}

function toggleColor(row, col, cell) {
    if (cell.className === 'grey') {
        cell.className = 'red';
        matrix[row][col] = 1;
    } else if (cell.className === 'red') {
        cell.className = 'blue';
        matrix[row][col] = 2;
    } else {
        cell.className = 'grey';
        matrix[row][col] = 0;
    }
}

function solveMatrix() {
    fetch('/solve/matrix', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ matrix: matrix }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.solutions_count > 0) {
            alert(`La condición inicial tiene ${data.solutions_count} soluciones`);
            if (data.solutions_count > 0) {
                displaySolution(data.solutions[0]);
            }
        } else {
            alert("La condición inicial no tiene solución");
        }
    });
}

function displaySolution(solution) {
    const table = document.getElementById('matrix');
    for (let i = 0; i < solution.length; i++) {
        for (let j = 0; j < solution[i].length; j++) {
            const cell = table.rows[i].cells[j];
            if (solution[i][j] === 1) {
                cell.className = 'red';
            } else if (solution[i][j] === 2) {
                cell.className = 'blue';
            } else {
                cell.className = 'grey';
            }
        }
    }
}

function startGame() {
    fetch('/play/matrix', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        const gameMatrix = data.matrix;
        const table = document.getElementById('matrix');
        for (let i = 0; i < gameMatrix.length; i++) {
            for (let j = 0; j < gameMatrix[i].length; j++) {
                const cell = table.rows[i].cells[j];
                if (gameMatrix[i][j] === 1) {
                    cell.className = 'red';
                } else if (gameMatrix[i][j] === 2) {
                    cell.className = 'blue';
                } else {
                    cell.className = 'grey';
                }
                matrix[i][j] = gameMatrix[i][j];
            }
        }
    });
}

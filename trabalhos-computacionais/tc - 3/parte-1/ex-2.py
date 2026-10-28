dados = [[1.0, 14.5], [2.0, 19.5], [3.0, 30.5], [4.0, 53.5], [5.0, 94.5], [6.0, 159.5]]

def newton_diferencas_divididas(x, y):
    if len(x) != len(y):
        raise ValueError("x e y precisam ter as mesmas dimensões")

    n = len(x)

    F = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        F[i][0] = y[i]
        
    for j in range(1, n):
        for i in range(n - j):
            F[i][j] = (F[i + 1][j - 1] - F[i][j - 1]) / (x[i + j] - x[i])

    coeficientes = [F[0][j] for j in range(n)]
    
    return coeficientes

def calcular_y(x, coeficientes, x_alvo):
    n = len(x)
    resultado = coeficientes[n - 1]
    
    for k in range(n - 2, -1, -1):
        resultado = resultado * (x_alvo - x[k]) + coeficientes[k]
        
    return resultado

def ex2():
    x = []
    y = []

    [x.append(dados[i][0]) for i in range(len(dados))]
    [y.append(dados[i][1]) for i in range(len(dados))]

    coeficientes = newton_diferencas_divididas(x, y)
    print("Coeficientes do Polinômio Interpolador:", coeficientes)

    x_alvo = 4.5
    y = calcular_y(x, coeficientes, x_alvo)
    print(f"f({x_alvo}) = {y}")

if __name__ == "__main__":
    ex2()
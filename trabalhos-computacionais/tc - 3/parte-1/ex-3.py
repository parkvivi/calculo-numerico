#EXERCÍCIO 3 (TC 3 - PARTE 1)
dados = [
    [0.0, 0.0061],
    [10.0, 0.0123],
    [20.0, 0.0234],
    [30.0, 0.0424],
    [40.0, 0.0738],
    [60.0, 0.1992],
    [80.0, 0.4736],
    [100.0, 1.0133]
]

def diferencas_divididas_de_newton (x, y):
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

def ex3():
    x = []
    y = []

    [x.append(dados[i][0]) for i in range(len(dados))]
    [y.append(dados[i][1]) for i in range(len(dados))]

    coeficientes = diferencas_divididas_de_newton (x, y)
    print("Coeficientes do Polinômio Interpolador:", coeficientes)

    alvos = [
        [5.0, 0.008721],
        [45.0, 0.095848],
        [95.0, 0.84528]
    ]

    for x_alvo, valor_conhecido in alvos:
        y_estimado = calcular_y(x, coeficientes, x_alvo)
        erro = abs(y_estimado - valor_conhecido)
        print(f"P({x_alvo}) estimado = {y_estimado:.6f} | Conhecido = {valor_conhecido} | Erro = {erro:.6f}")

if __name__ == "__main__":
    ex3()
    
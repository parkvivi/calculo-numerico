#EXERCÍCIO 4 (TC3 - PARTE 1)
dados = [
    [605.0, 0.622],
    [685.0, 0.655],
    [725.0, 0.668],
    [765.0, 0.679],
    [825.0, 0.730],
    [855.0, 0.907],
    [875.0, 1.336]
]

def newton_diferencas_divididas (x, y):
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

def ex4():
    x = []
    y = []

    [x.append(dados[i][0]) for i in range(len(dados))]
    [y.append(dados[i][1]) for i in range(len(dados))]

    coeficientes = newton_diferencas_divididas (x, y)
    print("Coeficientes do Polinômio Interpolador:", coeficientes)

    alvos = [
        [645.0, 0.639],
        [795.0, 0.694],
        [845.0, 0.812]
    ]

    for x_alvo, valor_conhecido in alvos:
        y_estimado = calcular_y(x, coeficientes, x_alvo)
        erro = abs(y_estimado - valor_conhecido)
        print(f"C({x_alvo}) estimado = {y_estimado:.6f} | Conhecido = {valor_conhecido} | Erro = {erro:.6f}")

if __name__ == "__main__":
    ex4()
    
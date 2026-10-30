import numpy as np

dados_ex5 = [[1.2, 2.3], [3.0, 3.3], [4.5, 5.6], [5.6, 15.4], [6.9, 38.2], [8.1, 59.1]]
dados_ex6 = [[1.0, 0.1], [2.3, 1.5], [3.8, 7.4], [4.9, 15.2], [6.1, 26.2], [7.2, 27.9]]

def resolve_sistema(A, b):
    return np.linalg.solve(A, b)

def polinomio_2ograu(x, y):
    if len(x) != len(y):
        raise ValueError("x e y precisam ter as mesmas dimensões")

    if len(x) < 2:
        raise ValueError("são precisos ao menos dois pontos para o ajuste")
    
    xi_0 = xi_1 = xi_2 = xi_3 = xi_4 = 0
    yi_0 = yi_1 = yi_2 = 0

    for i in range(len(x)):
        # calculando somatórios da matriz A
        xi_0 = xi_0 + (x[i])**0
        xi_1 = xi_1 + (x[i])**1
        xi_2 = xi_2 + (x[i])**2
        xi_3 = xi_3 + (x[i])**3
        xi_4 = xi_4 + (x[i])**4

        # calculando somatórios da matriz b
        yi_0 = yi_0 + (y[i]*((x[i])**0))
        yi_1 = yi_1 + (y[i]*((x[i])**1))
        yi_2 = yi_2 + (y[i]*((x[i])**2))

    A = np.array([
        [xi_4, xi_3, xi_2],
        [xi_3, xi_2, xi_1],
        [xi_2, xi_1, xi_0]
    ], dtype=float)

    b = np.array([yi_2, yi_1, yi_0], dtype=float)

    coeficientes = resolve_sistema(A, b)

    return coeficientes

def avalia_ponto(coeficientes, x):
    grau = len(coeficientes) - 1
    resultado = 0

    for i in range(grau + 1):
        resultado = resultado + coeficientes[grau - i]*(x**i)

    print(f"f({x}) = {resultado}")

def ex5():
    x = [val[0] for val in dados_ex5]
    y = [val[1] for val in dados_ex5]
    coeficientes = polinomio_2ograu(x, y)
    print("EXERCÍCIO 5: [a, b, c] = ", coeficientes)

    x_alvo = 6.0
    avalia_ponto(coeficientes, x_alvo)

def ex6():
    x = [val[0] for val in dados_ex6]
    y = [val[1] for val in dados_ex6]
    coeficientes = polinomio_2ograu(x, y)
    print("EXERCÍCIO 6: [a, b, c] = ", coeficientes)


if __name__ == "__main__":
    ex5()
    ex6()
import numpy as np

dados = [[1.2, 2.3], [3.0, 3.3], [4.5, 5.6], [5.6, 15.4], [6.9, 38.2], [8.1, 59.1]]

def resolve_sistema(A, b):
    return np.linalg.solve(A, b)

def ex5(x, y):
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

    print("[a, b, c, d] = ", coeficientes)

if __name__ == "__main__":
    x = [val[0] for val in dados]
    y = [val[1] for val in dados]

    ex5(x, y)
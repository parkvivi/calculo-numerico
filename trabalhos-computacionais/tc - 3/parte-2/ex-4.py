import numpy as np
from math import sin, pi

def resolve_sistema(A, b):
    return np.linalg.solve(A, b)

def ex4(x, y):
    if len(x) != len(y):
        raise ValueError("x e y precisam ter as mesmas dimensões")

    if len(x) < 2:
        raise ValueError("são precisos ao menos dois pontos para o ajuste")
    
    xi_0 = xi_1 = xi_2 = xi_3 = xi_4 = xi_5 = xi_6 = 0
    yi_0 = yi_1 = yi_2 = yi_3 = 0

    for i in range(len(x)):
        # calculando somatórios da matriz A
        xi_0 = xi_0 + (x[i])**0
        xi_1 = xi_1 + (x[i])**1
        xi_2 = xi_2 + (x[i])**2
        xi_3 = xi_3 + (x[i])**3
        xi_4 = xi_4 + (x[i])**4
        xi_5 = xi_5 + (x[i])**5
        xi_6 = xi_6 + (x[i])**6

        # calculando somatórios da matriz b
        yi_0 = yi_0 + (y[i]*((x[i])**0))
        yi_1 = yi_1 + (y[i]*((x[i])**1))
        yi_2 = yi_2 + (y[i]*((x[i])**2))
        yi_3 = yi_3 + (y[i]*((x[i])**3))

    A = np.array([
        [xi_6, xi_5, xi_4, xi_3],
        [xi_5, xi_4, xi_3, xi_2],
        [xi_4, xi_3, xi_2, xi_1],
        [xi_3, xi_2, xi_1, xi_0]
    ], dtype=float)

    b = np.array([yi_3, yi_2, yi_1, yi_0], dtype=float)

    coeficientes = resolve_sistema(A, b)

    print("[a, b, c, d] = ", coeficientes)

if __name__ == "__main__":
    x = [-1, -0.5, 0, 0.5, 1]
    y = [round(sin(pi*val), 10) for val in x]

    ex4(x, y)
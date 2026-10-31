import numpy as np

dados = [[1.2, 0.7], [2.1, 8.1], [3.0, 27.7], [3.6, 45.1]]

A = np.array([[dados[j][0]**i for i in range(len(dados))] for j in range(len(dados))], dtype=float)
b = np.array([dados[i][1] for i in range(len(dados))], dtype=float)

def ex1():
    coeficientes = np.linalg.solve(A, b)
    print("Coeficientes do Polinômio Interpolador:", coeficientes)

if __name__ == "__main__":
    ex1();
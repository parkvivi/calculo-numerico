import matplotlib.pyplot as plt

def newton_diferencas_divididas(x, y):
    n = len(x)
    F = [[0.0] * n for _ in range(n)]
    for i in range(n):
        F[i][0] = y[i]
    for j in range(1, n):
        for i in range(n - j):
            F[i][j] = (F[i + 1][j - 1] - F[i][j - 1]) / (x[i + j] - x[i])
    return [F[0][j] for j in range(n)]

def calcular_y(x, coeficientes, x_alvo):
    n = len(x)
    resultado = coeficientes[n - 1]
    for k in range(n - 2, -1, -1):
        resultado = resultado * (x_alvo - x[k]) + coeficientes[k]
    return resultado

def gerar_todos_os_graficos():
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))

    # --- EXERCÍCIO 1 ---
    d1 = [[1.2, 0.7], [2.1, 8.1], [3.0, 27.7], [3.6, 45.1]]
    x1, y1 = [p[0] for p in d1], [p[1] for p in d1]
    c1 = newton_diferencas_divididas(x1, y1)
    xc1 = [1.2 + i * 0.02 for i in range(121)]
    yc1 = [calcular_y(x1, c1, v) for v in xc1]
    axs[0, 0].plot(xc1, yc1, 'b-', label='Polinômio (Grau 3)')
    axs[0, 0].plot(x1, y1, 'ko', label='Dados')
    axs[0, 0].set_title('Exercício 1')
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)
    axs[0, 0].legend()

    # --- EXERCÍCIO 2 ---
    d2 = [[1.0, 14.5], [2.0, 19.5], [3.0, 30.5], [4.0, 53.5], [5.0, 94.5], [6.0, 159.5]]
    x2, y2 = [p[0] for p in d2], [p[1] for p in d2]
    c2 = newton_diferencas_divididas(x2, y2)
    xc2 = [1.0 + i * 0.05 for i in range(101)]
    yc2 = [calcular_y(x2, c2, v) for v in xc2]
    y2_est = calcular_y(x2, c2, 4.5)
    axs[0, 1].plot(xc2, yc2, 'b-', label='Polinômio (Grau 5)')
    axs[0, 1].plot(x2, y2, 'ko', label='Dados')
    axs[0, 1].plot([4.5], [71.375], 'g*', markersize=10, label='Exato f(4.5)')
    axs[0, 1].plot([4.5], [y2_est], 'rx', markersize=8, markeredgewidth=2, label='Estimado f(4.5)')
    axs[0, 1].set_title('Exercício 2')
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)
    axs[0, 1].legend()

    # --- EXERCÍCIO 3 ---
    d3 = [[0.0, 0.0061], [10.0, 0.0123], [20.0, 0.0234], [30.0, 0.0424],
          [40.0, 0.0738], [60.0, 0.1992], [80.0, 0.4736], [100.0, 1.0133]]
    x3, y3 = [p[0] for p in d3], [p[1] for p in d3]
    c3 = newton_diferencas_divididas(x3, y3)
    xc3 = [i * 0.5 for i in range(201)]
    yc3 = [calcular_y(x3, c3, v) for v in xc3]
    xa3, yr3 = [5.0, 45.0, 95.0], [0.008721, 0.095848, 0.84528]
    ye3 = [calcular_y(x3, c3, v) for v in xa3]
    axs[1, 0].plot(xc3, yc3, 'b-', label='Polinômio (Grau 7)')
    axs[1, 0].plot(x3, y3, 'ko', label='Dados')
    axs[1, 0].plot(xa3, yr3, 'g*', markersize=10, label='Conhecidos')
    axs[1, 0].plot(xa3, ye3, 'rx', markersize=8, markeredgewidth=2, label='Estimados')
    axs[1, 0].set_title('Exercício 3: Pressão da Água P(T)')
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)
    axs[1, 0].legend()

    # --- EXERCÍCIO 4 ---
    d4 = [[605.0, 0.622], [685.0, 0.655], [725.0, 0.668], [765.0, 0.679],
          [825.0, 0.730], [855.0, 0.907], [875.0, 1.336]]
    x4, y4 = [p[0] for p in d4], [p[1] for p in d4]
    c4 = newton_diferencas_divididas(x4, y4)
    xc4 = [605.0 + i for i in range(271)]
    yc4 = [calcular_y(x4, c4, v) for v in xc4]
    xa4, yr4 = [645.0, 795.0, 845.0], [0.639, 0.694, 0.812]
    ye4 = [calcular_y(x4, c4, v) for v in xa4]
    axs[1, 1].plot(xc4, yc4, 'b-', label='Polinômio (Grau 6)')
    axs[1, 1].plot(x4, y4, 'ko', label='Dados')
    axs[1, 1].plot(xa4, yr4, 'g*', markersize=10, label='Conhecidos')
    axs[1, 1].plot(xa4, ye4, 'rx', markersize=8, markeredgewidth=2, label='Estimados')
    axs[1, 1].set_title('Exercício 4: Propriedade do Titânio C(T)')
    axs[1, 1].grid(True, linestyle='--', alpha=0.6)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    gerar_todos_os_graficos()
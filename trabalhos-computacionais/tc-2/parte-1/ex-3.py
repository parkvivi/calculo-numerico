# EXERCÍCIO 3 - MP

import numpy as np
import matplotlib.pyplot as plt

def newton_sistemas(F, JF, x0, TOL=1e-5, max_iter=200):
    x = np.copy(x0).astype('double')
    historico = []

    for k in range(1, max_iter + 1):
        Fx = F(x)
        Jx = JF(x)
        
        delta = -np.linalg.inv(Jx).dot(Fx)
        x = x + delta
        erro = np.linalg.norm(delta, np.inf)

        historico.append({"iteração": k, "solução intermediária": np.copy(x), "erro": erro})

        if erro < TOL:
            return x, historico

    print("Aviso: Número máximo de iterações excedido.")
    return x, historico

def F_ex3(X):
    x, y = X[0], X[1]
    return np.array([
        x**2 - y + 1,
        x**2 + (y**2)/4 - 1
    ])

def JF_ex3(X):
    x, y = X[0], X[1]
    return np.array([
        [2*x, -1],
        [2*x, y/2]
    ])

def esboco():
    x = np.linspace(-1.5, 1.5, 400)
    x_elipse = np.linspace(-1, 1, 400)

    # Equação da Parábola (y = x^2 + 1)
    y_parabola = x**2 + 1

    # Equação da Elipse (x^2 + y^2/4 = 1  =>  y = +- 2*sqrt(1 - x^2)
    y_elipse_sup = 2 * np.sqrt(1 - x_elipse**2) # Parte de cima da elipse
    y_elipse_inf = -2 * np.sqrt(1 - x_elipse**2) # Parte de baixo da elipse

    plt.figure(figsize=(8, 6))

    plt.plot(x, y_parabola, label='Parábola: $y = x^2 + 1$', color='blue')
    plt.plot(x_elipse, y_elipse_sup, label='Elipse: $x^2 + y^2/4 = 1$', color='orange')
    plt.plot(x_elipse, y_elipse_inf, color='orange') # Completando a elipse

    plt.scatter([0.68125, -0.68125], [1.4641, 1.4641], color='red', zorder=5, label='Pontos de Intersecção')

    plt.title('Esboço do Exercício 3: Intersecção entre Parábola e Elipse')
    plt.axhline(0, color='black',linewidth=1)
    plt.axvline(0, color='black',linewidth=1)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.axis('equal')

    plt.show()

def main():
    print("===== EXERCÍCIO 3: PARÁBOLA E ELIPSE =====")
    # Esboço das curvas
    esboco()

    # Ponto no Primeiro Quadrante
    x0_q1 = np.array([0.7, 1.5])
    raiz_q1, hist_q1 = newton_sistemas(F_ex3, JF_ex3, x0_q1, TOL=1e-5)
    print("Intersecção 1 (1º Quadrante):")
    print(f"Coordenadas: {raiz_q1}")
    print(f"Iterações gastas: {len(hist_q1)}\n")

    # Ponto no Segundo Quadrante
    x0_q2 = np.array([-0.7, 1.5])
    raiz_q2, hist_q2 = newton_sistemas(F_ex3, JF_ex3, x0_q2, TOL=1e-5)
    print("Intersecção 2 (2º Quadrante):")
    print(f"Coordenadas: {raiz_q2}")
    print(f"Iterações gastas: {len(hist_q2)}\n")

if __name__ == "__main__":
    main()
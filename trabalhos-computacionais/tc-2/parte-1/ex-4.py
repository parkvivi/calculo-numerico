# EXERCÍCIO 4 - MP

import numpy as np

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

def F_ex4(X):
    x, y, z = X[0], X[1], X[2]
    return np.array([
        6*x - 2*y + np.exp(z) - 2,
        np.sin(x) - y + z,
        np.sin(x) + 2*y + 3*z - 1
    ])

def JF_ex4(X):
    x, y, z = X[0], X[1], X[2]
    return np.array([
        [6,         -2, np.exp(z)],
        [np.cos(x), -1, 1],
        [np.cos(x),  2, 3]
    ])

def main():
    print("===== EXERCÍCIO 4: SISTEMA 3D =====")
    # Solução próxima da origem
    x0_ex4 = np.array([0.0, 0.0, 0.0])
    raiz_ex4, hist_ex4 = newton_sistemas(F_ex4, JF_ex4, x0_ex4, TOL=1e-5)
    
    print("Solução aproximada:")
    print(f"Coordenadas (x, y, z): {raiz_ex4}")
    
    print("\n===== HISTÓRICO DE ITERAÇÕES (EX 4) =====")
    for h in hist_ex4:
        print(h)

if __name__ == "__main__":
    main()
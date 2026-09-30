# ITENS a-f
import numpy as np

G = 9.81
Z = np.array([100.0, 85.0, 60.0])         
L = np.array([1200.0, 900.0, 1500.0])     
D = np.array([0.30, 0.25, 0.25])          
F_ATRITO = np.array([0.022, 0.024, 0.024])
Q_DEMANDA = 0.200        

def F(x, K, z=Z, q=Q_DEMANDA):
    
    Q, H = x[:3], x[3]
    r = np.empty(4)
    r[:3] = K * Q * np.abs(Q) + H - z
    r[3] = Q.sum() - q
    return r

# Questão 1 - a)
def calcular_K(f=F_ATRITO, L=L, D=D, g=G):
    return 8 * f * L / (np.pi**2 * g * D**5)
 
 
def J(x, K):
    Jm = np.zeros((4, 4))
    Jm[:3, :3] = np.diag(2 * K * np.abs(x[:3]))
    Jm[:3, 3] = 1.0
    Jm[3, :3] = 1.0
    return Jm
 
 
def J_diferencas_finitas(x, K, h=1e-7):
    n = len(x)
    Jn = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        Jn[:, j] = (F(x + e, K) - F(x - e, K)) / (2 * h)
    return Jn
 
 
def letra_a():
    print("=== Letra (a) ===")
    K = calcular_K()
    for i, Ki in enumerate(K, start=1):
        print(f"K{i} = {Ki:.4f} s2/m5")
 
    x0 = np.array([0.10, 0.10, -0.05, 70.0])
    print("\nJ(x0) analitica:")
    print(J(x0, K))
    dif = np.abs(J(x0, K) - J_diferencas_finitas(x0, K)).max()
    print(f"\nmax|J - J_fd| = {dif:.2e}")
    return K  

# Questão 1 - b)

def newton(F_fun, J_fun, x0, tol=1e-12, max_iter=50):
    
    x = np.array(x0, dtype=float)
    hist = [x.copy()]
    for _ in range(max_iter):
        try:
            delta = np.linalg.solve(J_fun(x), -F_fun(x))
        except np.linalg.LinAlgError as err:
            raise RuntimeError(f"Jacobiana singular em x = {x}") from err
        x = x + delta
        hist.append(x.copy())
        if np.linalg.norm(delta, np.inf) < tol:
            return np.array(hist)
    raise RuntimeError("num. max. iter. excedido")
 
 
def erros(hist, x_star):
    return np.array([np.linalg.norm(x - x_star, np.inf) for x in hist])
 
 
def ordem_estimada(e):
    p = {}
    for k in range(1, len(e) - 1):
        if e[k - 1] > 0 and e[k] > 0 and e[k + 1] > 0:
            p[k + 1] = np.log(e[k + 1] / e[k]) / np.log(e[k] / e[k - 1])
    return p
 
 
def razoes_quadraticas(e):
    return {k: e[k + 1] / e[k]**2 for k in range(len(e) - 1) if e[k + 1] > 0}
 
 
def imprimir_tabela(hist, F_fun, x_star):
    e = erros(hist, x_star)
    print(f"{'k':>2} {'Q1':>12} {'Q2':>12} {'Q3':>12} {'H':>12} "
          f"{'||F||inf':>11} {'||x-x*||inf':>12}")
    for k, x in enumerate(hist):
        nf = np.linalg.norm(F_fun(x), np.inf)
        print(f"{k:>2} {x[0]:12.8f} {x[1]:12.8f} {x[2]:12.8f} {x[3]:12.6f} "
              f"{nf:11.3e} {e[k]:12.3e}")
 
 
def imprimir_convergencia(hist, x_star):
    e = erros(hist, x_star)
    print("\nOrdem estimada p_k (confiavel so perto da raiz):")
    for k, p in ordem_estimada(e).items():
        print(f"  k = {k}: p = {p:.3f}")
    print("\nRazao e_{k+1}/e_k^2:")
    for k, r in razoes_quadraticas(e).items():
        print(f"  k = {k}: {r:.4f}")
 
 
def letra_b(K):
    print("\n=== Letra (b) ===")
    x0 = np.array([0.10, 0.10, -0.05, 70.0])
    F_fun = lambda x: F(x, K)
    J_fun = lambda x: J(x, K)
 
    hist = newton(F_fun, J_fun, x0)
    x_star = hist[-1]
 
    imprimir_tabela(hist, F_fun, x_star)
    imprimir_convergencia(hist, x_star)
 
    print("\nSolucao:")
    print(f"  Q1 = {x_star[0]:.6f} m3/s")
    print(f"  Q2 = {x_star[1]:.6f} m3/s")
    print(f"  Q3 = {x_star[2]:.6f} m3/s")
    print(f"  H  = {x_star[3]:.4f} m")
    return x_star

# Questão 1 - c)

def det_J_formula(Q, K):
    a = 2 * K * np.abs(Q)
    return -sum(np.prod(np.delete(a, i)) for i in range(3))
 
 
def tentar_newton(x0, K, max_iter=50):
    try:
        hist = newton(lambda x: F(x, K), lambda x: J(x, K), x0, max_iter=max_iter)
        return hist
    except RuntimeError as err:
        print(f"  Newton falhou: {err}")
        return None
 
 
def letra_c(K):
    print("\n=== Letra (c) ===")
    x0 = np.array([0.0, 0.0, 0.0, 70.0])
    Jx0 = J(x0, K)
    print("F(x0) =", F(x0, K))
    print("J(x0) =")
    print(Jx0)
    print(f"det J(x0) = {np.linalg.det(Jx0):.3g},  posto = {np.linalg.matrix_rank(Jx0)}")
    tentar_newton(x0, K)
 
    print("\nOutros pontos (H = 70): det numerico vs formula, posto")
    casos = [(0, 0, 0), (0, 0, 0.2), (0, 0, -0.05), (0, 0.1, 0.1), (0.1, 0.1, 0.1)]
    for Q in casos:
        x = np.array([*Q, 70.0])
        print(f"  Q = {Q}: det = {np.linalg.det(J(x, K)):.6g}, "
              f"formula = {det_J_formula(np.array(Q), K):.6g}, "
              f"posto = {np.linalg.matrix_rank(J(x, K))}")
 
    print("\nPartida quase singular x0 = (1e-8, 1e-8, 1e-8, 70):")
    xq = np.array([1e-8, 1e-8, 1e-8, 70.0])
    hist = tentar_newton(xq, K, max_iter=100)
    if hist is not None:
        print(f"  convergiu em {len(hist) - 1} iteracoes; passo 1 gerou "
              f"Q1 = {hist[1][0]:.3g} (gigante)")

# Questão 1 - letra d)

# Questão 1 - letra e)

# Questão 1 - letra f)

#Função Main!!!
def main():
    K = letra_a()
    letra_b(K)
    letra_c(K)
 
 
if __name__ == "__main__":
    main()
"""
TC4 - Parte 3: Aedes aegypti, sazonalidade e controle  --  arquivo-base
Complete as funcoes marcadas com TODO. Todo numero do relatorio deve sair deste codigo.
"""
import csv
import math
import numpy as np
import matplotlib.pyplot as plt

T0 = 70            # semana de inicio da soltura
OMEGA = 2 * np.pi / 52
L = 5.0            # limiar de baixo risco (ovos por armadilha)


def le_dados(arquivo):
    """Devolve (sem, imo, sem_gene, n, k). Semanas sem coleta NAO entram em sem/imo."""
    sem, imo, sg, n, k = [], [], [], [], []
    with open(arquivo) as f:
        for linha in csv.DictReader(f):
            s = int(linha["semana"])
            if linha["imo"] != "":
                sem.append(s); imo.append(float(linha["imo"]))
            if linha["k_gene"] != "":
                sg.append(s); n.append(int(linha["n_amostra"])); k.append(int(linha["k_gene"]))
    return (np.array(sem, float), np.array(imo), np.array(sg, float),
            np.array(n, float), np.array(k, float))


def gauss_pivoteamento(A, b):
    """Eliminacao de Gauss com pivotamento parcial + substituicao regressiva."""
    M = np.array(A, dtype=float)
    rhs = np.array(b, dtype=float)
    n = len(rhs)

    for k in range(n - 1):
        p = k + np.argmax(np.abs(M[k:, k]))
        
        if p != k:
            M[[k, p]] = M[[p, k]]
            rhs[[k, p]] = rhs[[p, k]]

        for i in range(k + 1, n):
            fator = M[i, k] / M[k, k]
            M[i, k:] = M[i, k:] - fator * M[k, k:]
            rhs[i] = rhs[i] - fator * rhs[k]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        soma = np.dot(M[i, i+1:], x[i+1:])
        x[i] = (rhs[i] - soma) / M[i, i]

    return x


def newton(f, df, x0, eps=1e-8, imprimir_historico=False, max_iter=200):
    """
    Determina a raiz de f(x) = 0 pelo Metodo de Newton-Raphson.
    Retorna uma tupla: (raiz, historico)
    """
    x = x0
    fx = f(x)

    if imprimir_historico == True:
        print("Histórico do Método de Newton:")

    for k in range(1, max_iter + 1):
        dfx = df(x)

        if dfx == 0:
            raise ValueError(f"Iteracao {k}: derivada nula em x = {x}.")

        xn = x - fx / dfx
        fxn = f(xn)
        erro = abs(xn - x)

        convergiu = (erro < eps) or (abs(fxn) < eps)

        if imprimir_historico == True:
            print({"k": k, "x": float(xn), "fx": float(fxn), "erro": float(erro), "convergiu": convergiu})

        if convergiu:
            return xn, k

        x, fx = xn, fxn

    print("Aviso: max_iter atingido no Newton sem convergencia completa.")
    return x, k


def s_chapeu(t, a):
    return a[0] + a[1] * np.cos(OMEGA * t) + a[2] * np.sin(OMEGA * t)


def ds_chapeu(t, a):
    return -a[1] * OMEGA * np.sin(OMEGA * t) + a[2] * OMEGA * np.cos(OMEGA * t)


def g(t, a, b):
    y_chapeu = s_chapeu(t, a) * np.exp(b[0] + b[1] * (t - T0))
    return y_chapeu - L


def dg(t, a, b):
    exp_termo = np.exp(b[0] + b[1] * (t - T0))
    return (ds_chapeu(t, a) + b[1] * s_chapeu(t, a)) * exp_termo


if __name__ == "__main__":
    t, y, tg, n, k = le_dados("aedes_dados.csv")
    print(f"{len(t)} semanas com IMO; {len(tg)} amostras geneticas")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(t, y, ".", ms=4)
    ax1.axvline(T0, ls="--", c="gray")
    ax1.set_ylabel("IMO (ovos/armadilha)")
    ax2.plot(tg, k / n, "o")
    ax2.set_ylabel("fração com o gene")
    ax2.set_xlabel("semana")
    plt.tight_layout(); plt.show()

    # Questao 1:
    # montagem da matriz X
    X_1 = []
    for semana in t:
        if semana >= T0:
            break
    
        X_1.append([1, math.cos(OMEGA*semana), math.sin(OMEGA*semana)])

    X_1 = np.array(X_1)

    # montagem das equações normais X^T * X * a = X^T * y
    y_antes = y[t < T0]

    A_1 = X_1.T @ X_1
    b_1 = X_1.T @ y_antes

    # resolução da equação e diferença máxima
    a = gauss_pivoteamento(A_1, b_1)
    a_lstsq = np.linalg.lstsq(X_1, y_antes)[0]

    print(f"Coeficientes a (Pivoteamento de Gauss): {a}")
    print(f"Coeficientes a (np.linalg.lstsq): {a_lstsq}")

    dif = np.max(np.abs(a - a_lstsq))

    print(f"Diferença Máxima entre os Coeficientes: {dif}")

    # cálculo de S
    residuos = y_antes - (X_1 @ a)
    S = np.sum(residuos**2)

    print(f"S = {S}")

    # t pico
    t_pico = math.atan2(a[2], a[1]) / OMEGA
    print(f"t_pico = {t_pico} semanas")

    # gráfico dados e s_chapeu x tempo
    s_chapeu_dados = s_chapeu(t, a)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, y, "o", color="tab:blue", markersize=4, label="Dados observados (IMO)")
    ax.plot(t, s_chapeu_dados, "-", color="tab:red", linewidth=2, label="Modelo sazonal $\hat{s}(t)$")
    ax.axvline(T0, linestyle="--", color="gray", label="Início da soltura ($t_0$)")
    ax.set_xlabel("Semana")
    ax.set_ylabel("IMO (ovos/armadilha)")
    ax.legend()
    plt.title("Ex. 1d: Dados e modelo sazonal pelo tempo")
    plt.tight_layout()
    plt.show()

    # Questao 2:
    # ajuste de z(t) = b0 + b1*t por mmq
    z = np.log(np.divide(y, s_chapeu_dados))

    X_2 = []
    for semana in t:
        if semana < T0:
            continue

        X_2.append([1, semana - T0])

    X_2 = np.array(X_2)

    z_depois = z[t >= T0]

    A_2 = X_2.T @ X_2
    b_2 = X_2.T @ z_depois

    b = gauss_pivoteamento(A_2, b_2)

    print(f"Coeficientes b (Pivoteamento de Gauss): {b}")

    lamb = -b[1]
    meia_vida = math.log(2) / lamb

    print(f"Lambda: {lamb} por semana")
    print(f"Meia-vida: {meia_vida} semanas")

    # raizes de g(t)
    semanas_int = np.arange(T0, 401)
    valores_g = g(semanas_int, a, b)

    intervalos = []
    for i in range(len(semanas_int) - 1):
        if valores_g[i] * valores_g[i + 1] <= 0:
            intervalos.append((semanas_int[i], semanas_int[i + 1]))

    raizes = []
    for t_esq, t_dir in intervalos:
        chute = 0.5 * (t_esq + t_dir)
        raiz, n_iter = newton(lambda t_val: g(t_val, a, b), lambda t_val: dg(t_val, a, b), chute, 1e-10)
        raizes.append((raiz, n_iter))

    print("Raizes de g(t) / Número de Iterações")
    print(raizes)

    # t_env
    A = math.sqrt(a[1]**2 + a[2]**2)
    t_env = T0 + ((math.log(L / (a[0] + A)) - b[0]) / b[1])

    print(f"t_env = {t_env}")

    # método de Newton com chute inicial = T0
    newton(lambda t_val: g(t_val, a, b), lambda t_val: dg(t_val, a, b), T0, 1e-10, True)

    # Questao 3: TODO
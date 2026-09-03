def bisseccao (f, a, b, eps=1e-8, max_iter=200):
    # Retorna (raiz, historico)
    pass

def newton (f, df, x0, eps=1e-8, max_iter=200):

    x = x0

    historico = []

    for k in range(1, max_iter + 1):

        fx = f(x)
        dfx = df(x)

        if dfx == 0:
            raise ValueError(f"iteração {k} :: dfx assume 0")

        xn = x - fx/dfx

        e = abs(xn - x)

        historico.append({"k": k, "x": x, "fx": fx, "erro": e})

        if e < eps or abs(fx) < eps:
            return xn, historico

        x = xn

    print("Aviso: Não houve convergência.")
    return x, historico

def secante (f, x0, x1, eps=1e-8, max_iter=200):
    # Retorna (raiz, historico)
    pass
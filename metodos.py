def bisseccao (f, a, b, eps=1e-8, max_iter=200):
    # Retorna (raiz, historico)

def newton (f, df, x0, eps=1e-8, max_iter=200):

    x = x0

    for k in range(max_iter):

        fx = f(x)
        dfx = df(x)

        if dfx == 0:
            raise Exception(f"ERRO :: iteração {k} :: dfx assume 0")

        xn = x - fx/dfx

        e = xn - x

        print(f"[\"k\": {k}, \"x\": {xn}, \"fx\": {fx}, \"erro\": {e}]")   # retorna historico

        if abs(xn-x) < eps or abs(xn) < eps:
            return xn                                                       # retorna raiz

        x = xn

    print(f"não houve convergência :: x={x} foi o melhor valor encontrado")
    return x

def secante (f, x0, x1, eps=1e-8, max_iter=200):
    # Retorna (raiz, historico)

    # teste aaaaaaaaaaaaaaa
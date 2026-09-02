def bisseccao (f, a, b, eps=1e-8, max_iter=200):
    # Retorna (raiz, historico)

def newton (f, df, x0, eps=1e-8, max_iter=200):

    x = x0

    for k in range(max_iter):

        fx = f(x)
        dfx = df(x)

        if dfx == 0:
            raise Exception(f"ATENÇÃO :: iteração {k} :: dfx assume 0")

        xn = x - fx/dfx

        # falta calcular erro aqui

        print(f"[\"k\": {k}, \"x\": {xn}, \"fx\": {fx}, \"erro\": erro]")   # retorna historico

        if abs(xn) < eps or xn == x:
            return xn                                                       # retorna raiz

        x = xn

def secante (f, x0, x1, eps=1e-8, max_iter=200):
    # Retorna (raiz, historico)

    # teste aaaaaaaaaaaaaaa
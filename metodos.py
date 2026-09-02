def bisseccao (f, a, b, eps=1e-8, max_iter=200):
    # Retorna (raiz, historico)

def newton (f, df, x0, eps=1e-8, max_iter=200):

    x = x0

    cont_f = 0
    cont_df = 0

    for k in range(max_iter):

        fx = f(x)
        cont_f += 1

        dfx = df(x)
        cont_df += 1

        if dfx == 0:
            raise Exception(f"ERRO :: iteração {k} :: dfx assume 0")

        xn = x - fx/dfx

        e = xn - x

        print(f"[\"k\": {k}, \"x\": {xn}, \"fx\": {fx}, \"erro\": {e}]")   # retorna historico

        if abs(e) < eps or abs(xn) < eps:
            print(f"\"f\" chamada {cont_f} vezes :: \"df\" chamada {cont_df} vezes")
            return xn                                                       # retorna raiz

        x = xn

    print(f"não houve convergência :: x={x} foi o melhor valor encontrado")
    print(f"\"f\" chamada {cont_f} vezes :: \"df\" chamada {cont_df} vezes")
    return x

def secante (f, x0, x1, eps=1e-8, max_iter=200):
    # Retorna (raiz, historico)

    # teste aaaaaaaaaaaaaaa
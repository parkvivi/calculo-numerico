
def contador(f):
    """
    Decorator auxiliar para contar quantas vezes uma funcao foi chamada.
    """
    def wrapper(x):
        wrapper.n += 1
        return f(x)
    wrapper.n = 0
    return wrapper


def bisseccao(f, a, b, eps=1e-8, max_iter=200):
    """
    Determina a raiz de f(x) = 0 pelo Metodo da Bisseccao.
    Retorna uma tupla: (raiz, historico)
    """
    fa, fb = f(a), f(b)

    if (fa > 0) == (fb > 0):
        raise ValueError("f(a) e f(b) devem ter sinais opostos (f(a)*f(b) < 0).")

    historico = []

    for k in range(1, max_iter + 1):
        # Erro teoricamente correto da bisseccao: raio do intervalo ATUAL,
        # calculado ANTES de atualizar a ou b nesta iteracao.
        erro = abs(b - a) / 2.0
        x = (a + b) / 2.0
        fx = f(x)

        convergiu = (erro < eps) or (abs(fx) < eps)
        historico.append({"k": k, "x": float(x), "fx": float(fx), "erro": float(erro), "convergiu": convergiu})

        if convergiu:
            return x, historico

        if (fa > 0) != (fx > 0):
            b = x
            fb = fx
        else:
            a = x
            fa = fx

    print("Aviso: max_iter atingido na Bisseccao sem convergencia completa.")
    return x, historico


def newton(f, df, x0, eps=1e-8, max_iter=200):
    """
    Determina a raiz de f(x) = 0 pelo Metodo de Newton-Raphson.
    Retorna uma tupla: (raiz, historico)
    """
    historico = []
    x = x0
    fx = f(x)

    for k in range(1, max_iter + 1):
        dfx = df(x)

        if dfx == 0:
            raise ValueError(f"Iteracao {k}: derivada nula em x = {x}.")

        xn = x - fx / dfx
        fxn = f(xn)
        erro = abs(xn - x)

        convergiu = (erro < eps) or (abs(fxn) < eps)
        historico.append({"k": k, "x": float(xn), "fx": float(fxn), "erro": float(erro), "convergiu": convergiu})

        if convergiu:
            return xn, historico

        x, fx = xn, fxn

    print("Aviso: max_iter atingido no Newton sem convergencia completa.")
    return x, historico


def secante(f, x0, x1, eps=1e-8, max_iter=200):
    """
    Determina a raiz de f(x) = 0 pelo Metodo da Secante.
    Retorna uma tupla: (raiz, historico)
    """
    historico = []
    fx0 = f(x0)
    fx1 = f(x1)

    for k in range(1, max_iter + 1):
        den = fx1 - fx0

        if den == 0:
            raise ValueError(f"Iteracao {k}: denominador nulo na Secante.")

        x2 = x1 - fx1 * (x1 - x0) / den
        fx2 = f(x2)
        erro = abs(x2 - x1)

        convergiu = (erro < eps) or (abs(fx2) < eps)
        historico.append({"k": k, "x": float(x2), "fx": float(fx2), "erro": float(erro), "convergiu": convergiu})

        if convergiu:
            return x2, historico

        x0, x1 = x1, x2
        fx0, fx1 = fx1, fx2

    print("Aviso: max_iter atingido na Secante sem convergencia completa.")
    return x1, historico
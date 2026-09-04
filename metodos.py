# MÉTODO DA BISSEÇÃO  
def bisseccao (f, a, b, eps=1e-8, max_iter=200):
    """
    Determina a raiz de f(x) = 0 pelo Método da Bissecção.
    Retorna uma tupla: (raiz, historico)
    """
    if (f(a) > 0) == (f(b) > 0) >= 0:

        raise ValueError ("f(a) e f(b) devem ter sinais opostos")

    historico = []
    x_ant = a

    for k in range(1, max_iter + 1):
        x = (a + b) / 2.0
        fx = f(x)

        erro = abs(x - x_ant)

        historico.append({"k": k, "x": x, "fx": fx, "erro": erro})

        if erro < eps or abs(fx) < eps:
            return x, historico

        if (f(a) > 0) != (fx > 0):
            b = x
        else:
            a = x
            
        x_ant = x

    print("Aviso: Número máximo de iterações atingido.")
    return x, historico


# MÉTODO DE NEWTON
def newton (f, df, x0, eps=1e-8, max_iter=200):
    """
    Determina a raiz pelo Método da Newton.
    Retorna uma tupla: (raiz, historico)
    """
    historico = []
    x = x0

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


# MÉTODO DA SECANTE
def secante (f, x0, x1, eps=1e-8, max_iter=200):
    """
    Determina a raiz de f(x) = 0 pelo Método da Secante.
    Retorna uma tupla: (raiz, historico)
    """
    historico = []
    
    fx0 = f(x0)
    fx1 = f(x1)

    for k in range(1, max_iter + 1):
        den = fx1 - fx0
        
        if den == 0:
            raise ValueError(f"iteração {k} :: Denominador nulo na secante.")

        x2 = x1 - fx1 * (x1 - x0) / den
        fx2 = f(x2)
        
        erro = abs(x2 - x1)

        historico.append({"k": k, "x": x2, "fx": fx2, "erro": erro})


        if erro < eps or abs(fx2) < eps:
            return x2, historico

        # Atualizando os valores para a próxima iteração
        x0, x1 = x1, x2
        fx0, fx1 = fx1, fx2

    print("Aviso: Número máximo de iterações atingido. Não houve convergência na Secante.")
    return x1, historico

if __name__ == "__main__":
  
    def f_padrao(x):
        return x**3 - 9*x + 3

    print("=== EXECUTANDO TESTE DAS REGRAS DO TRABALHO ===")
    try:
        raiz, hist = bisseccao(f_padrao, a=0, b=1, eps=1e-8)
        
        print(f"Raiz obtida: {raiz:.8f}")
        print(f"Total de iterações (k): {len(hist)}")
        print("\nEstrutura do Dicionário (Última iteração):")
        print(hist[-1])
        
    except ValueError as e:
        print(f"Erro capturado na validação: {e}")
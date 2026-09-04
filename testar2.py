
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metodos import bisseccao, newton, secante, contador


# Funcao-teste padrao usada em quase toda a Parte 2
def f_padrao(x):
    return x**3 - 9 * x + 3


def df_padrao(x):
    return 3 * x**2 - 9


RAIZ_EXATA = 0.3376089559658377  # valor dado no enunciado (exercicio 2.4)


# =====================================================================
# 2.1 -- Isolamento de raizes
# =====================================================================
def tabelar_sinais(f, a, b, n):
    """
    Avalia f em n pontos igualmente espacados em [a, b] e retorna a
    lista de subintervalos (x_i, x_{i+1}) onde ha mudanca de sinal.
    """
    x_pontos = np.linspace(a, b, n)
    intervalos = []
    for i in range(len(x_pontos) - 1):
        x_atual, x_prox = x_pontos[i], x_pontos[i + 1]
        if (f(x_atual) > 0) != (f(x_prox) > 0):
            intervalos.append((x_atual, x_prox))
    return intervalos


def exercicio_2_1():
    print("\n" + "=" * 70)
    print("EXERCICIO 2.1 -- Isolamento de raizes")
    print("=" * 70)

    print("\n(a) f(x) = x^3 - 9x + 3 em [-5, 5]")
    for n in [21, 11, 6, 4]:
        ints = tabelar_sinais(f_padrao, -5, 5, n)
        print(f"  n = {n:3d} -> {len(ints)} intervalo(s) com mudanca de sinal")

    def g(x):
        return (x - 1.05) * (x - 1.15) * (x - 3.0)

    print("\n(b) g(x) = (x-1.05)(x-1.15)(x-3) em [0, 4]  (raizes reais conhecidas: 3)")
    for n in [9, 17, 41, 401]:
        ints = tabelar_sinais(g, 0, 4, n)
        print(f"  n = {n:3d} -> {len(ints)} intervalo(s) com mudanca de sinal")

    print(
        "\n(c) Discussao: g tem duas raizes muito proximas (1.05 e 1.15,\n"
        "  separadas por apenas 0.10), entao malhas grosseiras (n=9, 17) tem\n"
        "  passo maior que essa distancia e 'pulam' por cima das duas raizes\n"
        "  sem detectar troca de sinal. f nao sofre disso porque suas tres\n"
        "  raizes reais estao bem espacadas dentro de [-5,5]. Sem saber a\n"
        "  priori onde estao as raizes, a estrategia mais segura e refinar a\n"
        "  malha progressivamente (dobrar n) ate o numero de intervalos\n"
        "  encontrados estabilizar."
    )


# =====================================================================
# 2.2 -- Previsao x realidade na bissecao
# =====================================================================
def exercicio_2_2():
    print("\n" + "=" * 70)
    print("EXERCICIO 2.2 -- Previsao x realidade na bisseccao")
    print("=" * 70)

    a0, b0 = 0.0, 1.0
    tolerancias = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
    linhas = []
    for eps in tolerancias:
        k_previsto = math.ceil((math.log(b0 - a0) - math.log(eps)) / math.log(2))
        _, hist = bisseccao(f_padrao, a0, b0, eps=eps)
        linhas.append({"eps": eps, "k_previsto": k_previsto, "k_efetivo": len(hist)})

    df = pd.DataFrame(linhas)
    print(df.to_string(index=False))
    print(
        "\nDiscussao: a previsao teorica bate exatamente com o numero real de\n"
        "iteracoes em todos os casos, porque a bisseccao reduz o intervalo de\n"
        "forma deterministica pela metade a cada passo -- a formula nao e uma\n"
        "estimativa, e uma igualdade exata (a menos de arredondamento no teto)."
    )


# =====================================================================
# 2.3 -- Custo real: avaliacoes de funcao
# =====================================================================
def exercicio_2_3():
    print("\n" + "=" * 70)
    print("EXERCICIO 2.3 -- Custo real: avaliacoes de funcao")
    print("=" * 70)

    eps = 1e-8
    linhas = []

    # Bisseccao em [0, 1]
    f_c = contador(f_padrao)
    _, hist = bisseccao(f_c, 0.0, 1.0, eps=eps)
    linhas.append({
        "Metodo": "Bisseccao",
        "Iteracoes": len(hist),
        "Avaliacoes de f": f_c.n,
        "Avaliacoes de f'": 0,
    })

    # Newton, x0 = 0.5 (dentro do mesmo intervalo de busca)
    f_c = contador(f_padrao)
    df_c = contador(df_padrao)
    _, hist = newton(f_c, df_c, x0=0.5, eps=eps)
    linhas.append({
        "Metodo": "Newton",
        "Iteracoes": len(hist),
        "Avaliacoes de f": f_c.n,
        "Avaliacoes de f'": df_c.n,
    })

    # Secante, x0=0, x1=1 (mesmos extremos do intervalo)
    f_c = contador(f_padrao)
    _, hist = secante(f_c, x0=0.0, x1=1.0, eps=eps)
    linhas.append({
        "Metodo": "Secante",
        "Iteracoes": len(hist),
        "Avaliacoes de f": f_c.n,
        "Avaliacoes de f'": 0,
    })

    df_resultado = pd.DataFrame(linhas)
    print(df_resultado.to_string(index=False))
    print(
        "\nDiscussao: Newton costuma gastar menos iteracoes, mas cada iteracao\n"
        "custa 1 avaliacao de f + 1 de f' -- ou seja, 'menos iteracoes' nao e\n"
        "sinonimo de 'menos avaliacoes de funcao'. A secante evita calcular f',\n"
        "o que e uma vantagem quando f' e cara ou dificil de obter analiticamente\n"
        "(caso do Problema B, Colebrook-White)."
    )


# =====================================================================
# 2.4 -- Ordem empirica de convergencia
# =====================================================================
def calcular_ordem_p(historico, xi=RAIZ_EXATA, limite_precisao=1e-14):
    """
    Estima a ordem de convergencia p_k = ln(e_k+1/e_k) / ln(e_k/e_k-1).

    limite_precisao: erros abaixo desse valor sao descartados da estimativa,
    pois se aproximam da precisao de maquina (float64 ~ 2.2e-16) e o log da
    razao passa a refletir ruido de arredondamento, nao a taxa de convergencia
    real -- isso e esperado e deve ser comentado no relatorio, nao "corrigido".
    """
    erros = [abs(h["x"] - xi) for h in historico]
    resultado = []
    for k in range(2, len(erros)):
        e_k, e_k1, e_k2 = erros[k], erros[k - 1], erros[k - 2]
        if (
            e_k > limite_precisao
            and e_k1 > limite_precisao
            and e_k2 > limite_precisao
            and e_k1 != e_k2
        ):
            p = math.log(e_k / e_k1) / math.log(e_k1 / e_k2)
            resultado.append({"k": historico[k]["k"], "erro": e_k, "p_estimado": p})
    return resultado


def exercicio_2_4():
    print("\n" + "=" * 70)
    print("EXERCICIO 2.4 -- Ordem empirica de convergencia")
    print("=" * 70)

    _, hist_newton = newton(f_padrao, df_padrao, x0=0.5, eps=1e-14, max_iter=50)
    print("\nNewton (teorico p = 2.0):")
    print(pd.DataFrame(calcular_ordem_p(hist_newton)).to_string(index=False))

    _, hist_secante = secante(f_padrao, x0=0.0, x1=1.0, eps=1e-14, max_iter=50)
    print("\nSecante (teorico p ~= 1.618):")
    print(pd.DataFrame(calcular_ordem_p(hist_secante)).to_string(index=False))

    print(
        "\nObservacao: estimativas com erro abaixo de ~1e-14 foram descartadas\n"
        "de proposito -- perto da precisao de maquina (float64 ~ 2.2e-16), o\n"
        "log da razao entre erros passa a refletir ruido de arredondamento, e\n"
        "nao mais a taxa de convergencia teorica do metodo."
    )


# =====================================================================
# 2.5 -- Os modos de falha de Newton
# =====================================================================
def exercicio_2_5():
    print("\n" + "=" * 70)
    print("EXERCICIO 2.5 -- Os modos de falha de Newton")
    print("=" * 70)

    # (a) f(x) = x^3 - 2x + 2, x0 = 0 -- rodar 10 iteracoes, converge?
    print("\n(a) f(x) = x^3 - 2x + 2, x0 = 0, 10 iteracoes")

    def fa(x):
        return x**3 - 2 * x + 2

    def dfa(x):
        return 3 * x**2 - 2

    _, hist = newton(fa, dfa, x0=0.0, eps=1e-12, max_iter=10)
    for h in hist:
        print(f"  k={h['k']}: x = {h['x']:.6f}, f(x) = {h['fx']:.6f}")
    print(
        "  -> Nao converge: x oscila entre 0 e 1 indefinidamente (f(0)=2,\n"
        "     f'(0)=-2 leva a x=1; f(1)=1, f'(1)=1 leva de volta a x=0).\n"
        "     Geometricamente, a reta tangente em x=0 aponta para x=1 e a\n"
        "     tangente em x=1 aponta de volta para x=0 -- um ciclo de\n"
        "     periodo 2 que nunca se aproxima da raiz real (~ -1.7693)."
    )

    # (b) f(x) = arctan(x): x0=2.0 e x0=1.0; achar o valor-limite de x0
    print("\n(b) f(x) = arctan(x)")

    def fb(x):
        return math.atan(x)

    def dfb(x):
        return 1.0 / (1.0 + x**2)

    for x0_teste in [2.0, 1.0]:
        try:
            raiz, hist = newton(fb, dfb, x0=x0_teste, eps=1e-10, max_iter=30)
            print(f"  x0={x0_teste}: convergiu para {raiz:.10f} em {len(hist)} iteracoes")
        except (ValueError, OverflowError) as e:
            print(f"  x0={x0_teste}: falhou ({e})")

    # Busca binaria sobre x0 para achar o valor-limite de convergencia.
    def newton_diverge(x0, max_iter=50):
        x = x0
        for _ in range(max_iter):
            dfx = dfb(x)
            if dfx == 0:
                return True
            xn = x - fb(x) / dfx
            if abs(xn) > 1e6:
                return True
            if abs(xn - x) < 1e-12:
                return False
            x = xn
        return False

    lo, hi = 1.0, 2.0  # sabe-se que lo converge e hi diverge
    for _ in range(60):
        meio = (lo + hi) / 2
        if newton_diverge(meio):
            hi = meio
        else:
            lo = meio
    print(f"  Valor-limite estimado de x0 (fronteira converge/diverge): {lo:.6f}")
    print(
        "  (o valor teorico conhecido para arctan(x) e x0* ~= 1.3917 --\n"
        "   ponto onde a tangente cruza o eixo exatamente em -x0, gerando\n"
        "   oscilacao de amplitude constante na fronteira)."
    )

    # (c) f(x) = x^3 - 9x + 3, x0 = sqrt(3)  (f'(sqrt(3)) = 0)
    print("\n(c) f(x) = x^3 - 9x + 3, x0 = sqrt(3)")
    x0c = math.sqrt(3)
    print(f"  f'(sqrt(3)) = {df_padrao(x0c):.6f}  (deveria ser 0)")
    try:
        newton(f_padrao, df_padrao, x0=x0c, eps=1e-10, max_iter=10)
    except ValueError as e:
        print(f"  Newton levanta ValueError corretamente: {e}")
    print(
        "  -> Geometricamente, a reta tangente em x0=sqrt(3) e horizontal\n"
        "     (pico/vale local de f), entao ela nunca cruza o eixo x, e o\n"
        "     passo de Newton (divisao por f'=0) e indefinido."
    )


# =====================================================================
# 2.6 -- Raiz multipla
# =====================================================================
def newton_modificado(f, df, x0, m, eps=1e-8, max_iter=200):
    """
    Newton modificado para raizes de multiplicidade m:
    x_{k+1} = x_k - m * f(x_k) / f'(x_k)
    Implementado aqui (fora de metodos.py) porque e uma variante pedida
    especificamente pelo exercicio 2.6, fora do escopo fixo da Parte 1.
    """
    historico = []
    x = x0
    for k in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            raise ValueError(f"iteracao {k}: derivada nula em x = {x}")
        xn = x - m * fx / dfx
        fxn = f(xn)
        erro = abs(xn - x)
        convergiu = (erro < eps) or (abs(fxn) < eps)
        historico.append({"k": k, "x": xn, "fx": fxn, "erro": erro, "convergiu": convergiu})
        if convergiu:
            return xn, historico
        x = xn
    print("Aviso: max_iter atingido no Newton modificado.")
    return x, historico


def exercicio_2_6():
    print("\n" + "=" * 70)
    print("EXERCICIO 2.6 -- Raiz multipla")
    print("=" * 70)

    def f(x):
        return (x - 2) ** 2 * (x + 1)

    def df(x):
        # derivada de (x-2)^2 (x+1) = (x-2) * [2(x+1) + (x-2)] = (x-2)*3x
        return (x - 2) * (3 * x)

    print("\nNewton padrao, x0 = 3, raiz dupla em x = 2 (m = 2):")
    _, hist = newton(f, df, x0=3.0, eps=1e-14, max_iter=10)
    xi = 2.0
    linhas = []
    for h in hist:
        e = abs(h["x"] - xi)
        linhas.append({"k": h["k"], "x": h["x"], "erro e_k": e})
    df_hist = pd.DataFrame(linhas)
    df_hist["e_(k+1)/e_k"] = df_hist["erro e_k"].shift(-1) / df_hist["erro e_k"]
    print(df_hist.to_string(index=False))
    print(
        "\n  -> A convergencia NAO e quadratica: o erro cai por um fator\n"
        "     praticamente constante a cada iteracao (convergencia LINEAR),\n"
        "     e a razao e_(k+1)/e_k se aproxima de (m-1)/m = 1/2 = 0.5,\n"
        "     como previsto pela teoria para raiz de multiplicidade m=2."
    )

    print("\nNewton MODIFICADO (m=2), x0 = 3:")
    _, hist_mod = newton_modificado(f, df, x0=3.0, m=2, eps=1e-14, max_iter=10)
    linhas_mod = []
    for h in hist_mod:
        e = abs(h["x"] - xi)
        linhas_mod.append({"k": h["k"], "x": h["x"], "erro e_k": e})
    print(pd.DataFrame(linhas_mod).to_string(index=False))
    print(
        "  -> Com o fator de multiplicidade m=2 explicito na formula, a\n"
        "     convergencia quadratica e restaurada -- o erro cai muito mais\n"
        "     rapido (poucas iteracoes para atingir precisao de maquina)."
    )


# =====================================================================
# 2.7 -- A armadilha do residuo
# =====================================================================
def bisseccao_criterio_isolado(f, df, a, b, eps, max_iter, criterio):
    """
    Variante da bisseccao para o exercicio 2.7, que isola cada criterio
    de parada (em vez do OR combinado de metodos.py), exatamente para
    comparar o comportamento de cada um separadamente, como pedido.

    IMPORTANTE: f(x) = (x-1)^10 nunca muda de sinal (potencia par, f >= 0
    sempre), entao a bisseccao classica (que exige f(a)*f(b) < 0) e
    IMPOSSIVEL de aplicar diretamente sobre f. A solucao numericamente
    correta e aplicar a bisseccao sobre f'(x), que MUDA de sinal em x=1
    (f' < 0 para x<1, f' > 0 para x>1) -- ou seja, localizamos a raiz de
    f' (o minimo de f). Os criterios de parada (residuo e incremento)
    continuam sendo avaliados sobre f(x) e x, como pede o enunciado.

    criterio: "residuo" usa so |f(x)| < eps
              "incremento" usa so |x_k - x_{k-1}| < eps
    """
    dfa, dfb = df(a), df(b)
    if (dfa > 0) == (dfb > 0):
        raise ValueError("f'(a) e f'(b) devem ter sinais opostos")

    historico = []
    for k in range(1, max_iter + 1):
        erro = abs(b - a) / 2.0
        x = (a + b) / 2.0
        fx = f(x)
        dfx = df(x)

        if criterio == "residuo":
            parar = abs(fx) < eps
        elif criterio == "incremento":
            parar = erro < eps
        else:
            raise ValueError("criterio deve ser 'residuo' ou 'incremento'")

        historico.append({"k": k, "x": x, "fx": fx, "erro": erro})
        if parar:
            return x, historico

        if (dfa > 0) != (dfx > 0):
            b = x
            dfb = dfx
        else:
            a = x
            dfa = dfx

    return x, historico


def exercicio_2_7():
    print("\n" + "=" * 70)
    print("EXERCICIO 2.7 -- A armadilha do residuo")
    print("=" * 70)

    def f(x):
        return (x - 1) ** 10

    def df(x):
        return 10 * (x - 1) ** 9

    print(f"\nf(1.1) = {f(1.1):.6e}")
    print(f"f(1.3) = {f(1.3):.6e}")
    print(
        "  -> f(1.1), apesar de x estar a 0.1 da raiz real, ja vale ~1e-10 --\n"
        "     o RESIDUO fica minusculo muito antes de x se aproximar de fato\n"
        "     da raiz, porque a funcao e extremamente achatada perto de x=1\n"
        "     (expoente 10)."
    )

    print(
        "\nObservacao importante: f(x)=(x-1)^10 NUNCA muda de sinal (potencia\n"
        "par, f>=0 sempre), entao a bisseccao classica sobre f e impossivel\n"
        "(f(a)*f(b) nunca e negativo). A saida correta e aplicar a bisseccao\n"
        "sobre f'(x), que muda de sinal em x=1, localizando o minimo de f\n"
        "(=raiz de multiplicidade 10). Os criterios de parada continuam\n"
        "avaliados sobre f(x) e sobre x, como pede o enunciado."
    )

    raiz_res, hist_res = bisseccao_criterio_isolado(
        f, df, 0.0, 1.5, eps=1e-8, max_iter=200, criterio="residuo"
    )
    print(f"\nCriterio do RESIDUO |f(x)|<1e-8: raiz = {raiz_res:.10f}, "
          f"iteracoes = {len(hist_res)}, erro real |x-1| = {abs(raiz_res - 1):.6e}")

    raiz_inc, hist_inc = bisseccao_criterio_isolado(
        f, df, 0.0, 1.5, eps=1e-8, max_iter=200, criterio="incremento"
    )
    print(f"Criterio do INCREMENTO |xk+1-xk|<1e-8: raiz = {raiz_inc:.10f}, "
          f"iteracoes = {len(hist_inc)}, erro real |x-1| = {abs(raiz_inc - 1):.6e}")

    print(
        "\nConclusao: o criterio do residuo para MUITO antes (poucas iteracoes)\n"
        "mas com erro real em x ainda grande -- perigoso para funcoes muito\n"
        "'achatadas' perto da raiz (derivadas de ordem alta nulas, como\n"
        "(x-1)^10). Ja o criterio do incremento continua ate o intervalo\n"
        "ficar realmente pequeno, dando um erro real muito menor. O residuo\n"
        "so e o criterio mais adequado quando f e 'ingreme' perto da raiz\n"
        "(f' grande), caso em que um |f(x)| pequeno de fato implica x proximo\n"
        "da raiz -- ou quando o que importa fisicamente e o valor de f, nao x."
    )


if __name__ == "__main__":
    exercicio_2_1()
    exercicio_2_2()
    exercicio_2_3()
    exercicio_2_4()
    exercicio_2_5()
    exercicio_2_6()
    exercicio_2_7()
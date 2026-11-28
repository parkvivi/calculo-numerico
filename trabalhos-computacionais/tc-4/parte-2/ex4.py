from fractions import Fraction
import math

# Dados do caso de 30/10 (Versão A, rampa linear) para verificação numérica da ordem 2
k = 0.196
Ti = 18.0
b = 3.0
L = 4.0
TL = 27.4


def f(t, T):
    """EDO da Lei do Resfriamento de Newton na rampa linear: Ta(t) = Ti + b*t."""
    return -k * (T - (Ti + b * t))


def solucao_exata_T0():
    """Solução analítica exata em t = 0."""
    C = (TL - (Ti + b * L - b / k)) * math.exp(k * L)
    return Ti - b / k + C


def ponto_medio(f, t0, y0, h, n):
    """Método do Ponto Médio (RK2: c = 1/2, b1 = 0, b2 = 1)."""
    t, y = t0, y0
    for _ in range(n):
        k1 = f(t, y)
        k2 = f(t + (h / 2.0), y + (h / 2.0) * k1)
        y = y + h * k2
        t = t + h
    return y


def ex4():
    print("===== EXERCÍCIO 4 — CONDIÇÕES DE ORDEM 2 NO MÉTODO DO PONTO MÉDIO =====\n")

    # 1. Verificação algébrica exata das condições b1 + b2 = 1 e b2 * c = 1/2
    c = Fraction(1, 2)
    b1 = Fraction(0, 1)
    b2 = Fraction(1, 1)

    cond1 = b1 + b2
    cond2 = b2 * c

    print("--- Verificação algébrica (aritmética exata com frações) ---")
    print(f"Parâmetros do Ponto Médio: c = {c}, b1 = {b1}, b2 = {b2}")
    print(f"Condição 1 (b1 + b2 = 1):   {b1} + {b2} = {cond1}  -> {'SATISFEITA' if cond1 == 1 else 'FALHOU'}")
    print(f"Condição 2 (b2 * c  = 1/2): {b2} * {c} = {cond2} -> {'SATISFEITA' if cond2 == Fraction(1, 2) else 'FALHOU'}\n")

    # 2. Confirmação numérica da ordem 2 (erro global O(h^2) => razão ~ 4 ao dividir h por 2)
    T0_exato = solucao_exata_T0()
    print("--- Verificação numérica da ordem 2 no caso de 30/10 (Versão A) ---")
    print(f"Solução exata: T(0) = {T0_exato:.6f} °C\n")

    cabecalho = f"{'h':>6} | {'n':>3} | {'T(0) Ponto Médio':>16} | {'Erro em T(0)':>12} | {'Razão E(2h)/E(h)':>16} | {'Ordem p':>7}"
    print(cabecalho)
    print("-" * len(cabecalho))

    erro_anterior = None
    for h in [0.5, 0.25, 0.125]:
        n = round(L / h)
        T0_pm = ponto_medio(f, t0=L, y0=TL, h=-h, n=n)
        erro = abs(T0_exato - T0_pm)

        if erro_anterior is None:
            razao_str = "       ---      "
            ordem_str = "  ---  "
        else:
            razao = erro_anterior / erro
            ordem = math.log2(razao)
            razao_str = f"{razao:16.4f}"
            ordem_str = f"{ordem:7.4f}"

        print(f"{h:6.3f} | {n:3d} | {T0_pm:16.6f} | {erro:12.6e} | {razao_str} | {ordem_str}")
        erro_anterior = erro


if __name__ == "__main__":
    ex4()
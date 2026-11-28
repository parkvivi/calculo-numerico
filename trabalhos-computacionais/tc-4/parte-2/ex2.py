import math

# Dados do caso de 30/10 (Versão A, rampa linear)
k = 0.196    # constante de resfriamento (h^-1)
Ti = 18.0    # temperatura com ar-condicionado ligado (°C)
b = 3.0      # taxa de aquecimento da sala durante o apagão (°C/h)
L = 4.0      # duração do apagão (h)
TL = 27.4    # temperatura do corpo quando a luz voltou (°C)


def Ta_linear(t):
    """Temperatura ambiente da rampa linear: Ta(t) = Ti + b*t."""
    return Ti + b * t


def f(t, T):
    """EDO da Lei do Resfriamento de Newton: T'(t) = -k * (T - Ta(t))."""
    return -k * (T - Ta_linear(t))


def solucao_exata_T0():
    """Solução analítica da EDO linear em t = 0 via fator integrante."""
    C = (TL - (Ti + b * L - b / k)) * math.exp(k * L)
    return Ti - b / k + C


def euler(f, t0, y0, h, n, verbose=False):
    """Método de Euler (passo negativo h para andar para trás no tempo)."""
    t, y = t0, y0
    for passo in range(1, n + 1):
        inclinacao = f(t, y)
        y_novo = y + h * inclinacao
        t_prox = t + h

        if verbose:
            print(
                f"Passo {passo:2d} | t = {t:5.3f} -> {t_prox:5.3f} | "
                f"Ta = {Ta_linear(t):6.3f} | T = {y:7.4f} | "
                f"f = {inclinacao:+8.4f} | T_prox = {y_novo:7.4f}"
            )

        y = y_novo
        t = t_prox
    return y


def calcular_tm_e_relogio(T0):
    """Calcula o instante da morte tm (em horas) e o horário HHhMM."""
    tm = -math.log((37.0 - Ti) / (T0 - Ti)) / k
    total_min = round((24.0 + tm) * 60)
    horas, minutos = divmod(total_min, 60)
    return tm, f"{horas:02d}h{minutos:02d}"


def ex2():
    print("===== EXERCÍCIO 2 — CONVERGÊNCIA DO MÉTODO DE EULER (CASO 30/10) =====\n")

    T0_exato = solucao_exata_T0()
    tm_exato, hora_exata = calcular_tm_e_relogio(T0_exato)
    print(f"Solução exata: T(0) = {T0_exato:.4f} °C | tm = {tm_exato:.4f} h | morte às {hora_exata}\n")

    passos_h = [0.5, 0.25, 0.125]
    resultados = []

    for h in passos_h:
        n = round(L / h)
        # Mostra o passo a passo completo para h = 0.5 (troque para True se quiser imprimir todos)
        mostrar_passos = (h == 0.5)
        if mostrar_passos:
            print(f"--- Detalhamento conta por conta para h = {h} ({n} passos) ---")

        T0_num = euler(f, t0=L, y0=TL, h=-h, n=n, verbose=mostrar_passos)

        if mostrar_passos:
            print()

        erro_T = abs(T0_exato - T0_num)
        tm_num, hora_num = calcular_tm_e_relogio(T0_num)
        erro_min = abs(tm_exato - tm_num) * 60.0

        resultados.append((h, n, T0_num, erro_T, hora_num, erro_min))

    print("===== TABELA DE CONVERGÊNCIA (EULER) =====")
    cabecalho = (
        f"{'h':>6} | {'n':>3} | {'T(0) Euler':>10} | {'Erro T(0)':>10} | "
        f"{'Razão E':>8} | {'Ordem p':>7} | {'Hora':>6} | {'Erro (min)':>10}"
    )
    print(cabecalho)
    print("-" * len(cabecalho))

    erro_anterior = None
    for h, n, T0_num, erro_T, hora_num, erro_min in resultados:
        if erro_anterior is None:
            razao_str = "   ---  "
            ordem_str = "  ---  "
        else:
            razao = erro_anterior / erro_T
            ordem = math.log2(razao)
            razao_str = f"{razao:8.4f}"
            ordem_str = f"{ordem:7.4f}"

        print(
            f"{h:6.3f} | {n:3d} | {T0_num:10.4f} | {erro_T:10.4f} | "
            f"{razao_str} | {ordem_str} | {hora_num:>6} | {erro_min:10.2f}"
        )
        erro_anterior = erro_T


if __name__ == "__main__":
    ex2()
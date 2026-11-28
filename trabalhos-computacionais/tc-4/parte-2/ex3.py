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


def calcular_tm(T0):
    """Retorna o instante da morte tm (em horas, negativo antes da meia-noite)."""
    return -math.log((37.0 - Ti) / (T0 - Ti)) / k


def formatar_relogio(tm):
    """Converte tm (em horas) para o horário HHhMM."""
    total_min = round((24.0 + tm) * 60)
    horas, minutos = divmod(total_min, 60)
    return f"{horas:02d}h{minutos:02d}"


def euler(f, t0, y0, h, n):
    """Método de Euler (1 avaliação de f por passo)."""
    t, y = t0, y0
    for _ in range(n):
        y = y + h * f(t, y)
        t = t + h
    return y


def rk4(f, t0, y0, h, n, verbose=False):
    """Método de Runge-Kutta clássico de 4ª ordem (4 avaliações de f por passo)."""
    t, y = t0, y0
    for passo in range(1, n + 1):
        k1 = f(t, y)
        k2 = f(t + h / 2.0, y + (h / 2.0) * k1)
        k3 = f(t + h / 2.0, y + (h / 2.0) * k2)
        k4 = f(t + h, y + h * k3)
        y_novo = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if verbose:
            print(
                f"  Passo {passo}: t = {t:.1f} -> {t+h:.1f} | "
                f"k1={k1:+.4f}, k2={k2:+.4f}, k3={k3:+.4f}, k4={k4:+.4f} | "
                f"T({t+h:.1f}) = {y_novo:.6f} °C"
            )

        y = y_novo
        t = t + h
    return y


def ex3():
    print("===== EXERCÍCIO 3 — PASSOS DE EULER (< 1 MIN) vs. CUSTO DO RK4 (h = 1) =====\n")

    T0_exato = solucao_exata_T0()
    tm_exato = calcular_tm(T0_exato)
    print(f"Solução exata: T(0) = {T0_exato:.6f} °C | morte às {formatar_relogio(tm_exato)}\n")

    # 1. Estimativa teórica pela ordem 1 usando n = 32 (h = 0.125) do Exercício 2
    T0_e32 = euler(f, L, TL, -L / 32, 32)
    erro_min_32 = abs(tm_exato - calcular_tm(T0_e32)) * 60.0
    n_estimado = math.ceil(32 * erro_min_32)
    print(f"Estimativa teórica (a partir de n=32, erro={erro_min_32:.4f} min): ~{n_estimado} passos\n")

    # 2. Busca exata do menor n inteiro tal que erro na hora da morte < 1.0 min
    n_euler = 1
    while True:
        h = L / n_euler
        T0_num = euler(f, L, TL, -h, n_euler)
        erro_min = abs(tm_exato - calcular_tm(T0_num)) * 60.0
        if erro_min < 1.0:
            break
        n_euler += 1

    # Confere o ponto imediatamente anterior (n_euler - 1) para evidenciar a transição
    n_ant = n_euler - 1
    T0_ant = euler(f, L, TL, -L / n_ant, n_ant)
    erro_T_ant = abs(T0_exato - T0_ant)
    erro_min_ant = abs(tm_exato - calcular_tm(T0_ant)) * 60.0

    T0_min = euler(f, L, TL, -L / n_euler, n_euler)
    erro_T_min = abs(T0_exato - T0_min)
    erro_min_min = abs(tm_exato - calcular_tm(T0_min)) * 60.0

    print("--- Transição exata do método de Euler para erro < 1 minuto ---")
    print(f"  n = {n_ant} (h = {L/n_ant:.5f} h): T(0) = {T0_ant:.4f} °C | erro T = {erro_T_ant:.4f} °C | erro hora = {erro_min_ant:.4f} min")
    print(f"  n = {n_euler} (h = {L/n_euler:.5f} h): T(0) = {T0_min:.4f} °C | erro T = {erro_T_min:.4f} °C | erro hora = {erro_min_min:.4f} min\n")

    # 3. Execução do RK4 com h = 1 (n = 4 passos, 16 avaliações de f)
    print("--- Execução do RK4 com h = 1 (4 passos) ---")
    n_rk4 = 4
    T0_rk4 = rk4(f, L, TL, -1.0, n_rk4, verbose=True)
    erro_T_rk4 = abs(T0_exato - T0_rk4)
    erro_min_rk4 = abs(tm_exato - calcular_tm(T0_rk4)) * 60.0
    print()

    # 4. Tabela comparativa de custo (avaliações de f) e precisão
    aval_euler = n_euler * 1
    aval_rk4 = n_rk4 * 4

    print("===== COMPARAÇÃO DE CUSTO E PRECISÃO =====")
    cabecalho = f"{'Método':<18} | {'Passos':>6} | {'h (h)':>7} | {'Aval. f':>7} | {'Erro T(0)':>10} | {'Erro hora (min)':>15}"
    print(cabecalho)
    print("-" * len(cabecalho))
    for n_ref in [8, 32, 200, n_ant, n_euler]:
        h_ref = L / n_ref
        T0_ref = euler(f, L, TL, -h_ref, n_ref)
        eT = abs(T0_exato - T0_ref)
        em = abs(tm_exato - calcular_tm(T0_ref)) * 60.0
        rotulo = "Euler (< 1 min)" if n_ref == n_euler else "Euler"
        print(f"{rotulo:<18} | {n_ref:6d} | {h_ref:7.4f} | {n_ref:7d} | {eT:10.6f} | {em:15.4f}")

    print(f"{'RK4 (h = 1)':<18} | {n_rk4:6d} | {1.0:7.4f} | {aval_rk4:7d} | {erro_T_rk4:10.6f} | {erro_min_rk4:15.4f}")

    print("\n===== RESUMO DA COMPARAÇÃO =====")
    print(f"Passos mínimos de Euler (< 1 min): {n_euler} passos (h = {L/n_euler:.5f} h) -> {aval_euler} avaliações de f")
    print(f"RK4 com h = 1:                     {n_rk4} passos (h = 1.00000 h) -> {aval_rk4} avaliações de f")
    print(f"Razão de custo (Euler / RK4):      {aval_euler / aval_rk4:.2f}x mais avaliações de f no Euler")
    print(f"Razão de precisão (Euler / RK4):   RK4 é {erro_min_min / erro_min_rk4:.1f}x mais preciso na hora da morte ({erro_min_rk4*60:.2f} s)")


if __name__ == "__main__":
    ex3()
import math

# Dados da Versão A (Caso N.º 2 - O Recurso)
k = 0.196   # constante de resfriamento (h^-1)
Ti = 18.0   # temperatura com ar-condicionado ligado (°C)
L = 4.0     # instante em que a luz voltou (h)
TL = 27.4   # temperatura do corpo às 4h00 (°C)

# Tabela do registrador dos livros raros: t (h) -> Ta(t) (°C)
Ta = {
    0.0: 18.0,
    0.5: 19.8,
    1.0: 21.2,
    1.5: 22.1,
    2.0: 22.8,
    2.5: 23.3,
    3.0: 23.6,
    3.5: 23.9,
    4.0: 24.0,
}


def f(t, T):
    """EDO da Lei do Resfriamento de Newton: T'(t) = -k * (T - Ta(t))."""
    return -k * (T - Ta[t])


def euler(f, t0, y0, h, n):
    """Método de Euler (passo negativo h para andar para trás no tempo)."""
    t, y = t0, y0
    for _ in range(n):
        y = y + h * f(t, y)
        t = t + h
    return y


def rk4(f, t0, y0, h, n, verbose=True):
    """Método de Runge-Kutta clássico de 4ª ordem (imprimindo conta por conta)."""
    t, y = t0, y0
    for passo in range(1, n + 1):
        t_meio = t + h / 2.0
        t_prox = t + h

        k1 = f(t, y)
        y_k2 = y + (h / 2.0) * k1
        k2 = f(t_meio, y_k2)
        y_k3 = y + (h / 2.0) * k2
        k3 = f(t_meio, y_k3)
        y_k4 = y + h * k3
        k4 = f(t_prox, y_k4)

        y_novo = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if verbose:
            print(f"--- Passo {passo}: de t = {t:.1f}h até t = {t_prox:.1f}h ---")
            print(f"  T({t:.1f}) = {y:.4f} °C | Ta = [{Ta[t]}, {Ta[t_meio]}, {Ta[t_prox]}] °C")
            print(f"  k1 = {k1:.4f}")
            print(f"  k2 = {k2:.4f}  (T + (h/2)*k1 = {y_k2:.4f})")
            print(f"  k3 = {k3:.4f}  (T + (h/2)*k2 = {y_k3:.4f})")
            print(f"  k4 = {k4:.4f}  (T + h*k3     = {y_k4:.4f})")
            print(f"  -> T({t_prox:.1f}) = {y_novo:.4f} °C\n")

        y = y_novo
        t = t_prox
    return y


def calcular_hora_morte(T0):
    """Converte T(0) no instante tm (h), total em minutos e horário HHhMM."""
    tm = -math.log((37.0 - Ti) / (T0 - Ti)) / k
    total_minutos = round((24.0 + tm) * 60)
    horas, minutos = divmod(total_minutos, 60)
    return tm, total_minutos, f"{horas:02d}h{minutos:02d}"


def identificar_culpado(total_minutos):
    """Retorna o suspeito sem álibi confirmado no horário informado."""
    janelas_sem_alibi = [
        ("Fantasma",  20 * 60 + 40, 21 * 60 + 15),
        ("Vampiro",   21 * 60 + 35, 22 * 60 + 10),
        ("Múmia",     22 * 60 + 25, 23 * 60 + 0),
        ("Lobisomem", 23 * 60 + 20, 25 * 60 + 0),
    ]
    for suspeito, inicio, fim in janelas_sem_alibi:
        if inicio <= total_minutos <= fim:
            return suspeito
    return "Nenhum suspeito sem álibi nesse horário"


def ex1():
    print("===== EXERCÍCIO 1 — O RECURSO (VERSÃO A) COM RK4 (h = 1) =====\n")

    # RK4 para trás: h = -1.0, 4 passos (de t = 4.0 até t = 0.0)
    T0_rk4 = rk4(f, t0=L, y0=TL, h=-1.0, n=4, verbose=True)
    tm_rk4, min_rk4, hora_rk4 = calcular_hora_morte(T0_rk4)
    culpado_rk4 = identificar_culpado(min_rk4)

    # Referência anterior: Euler para trás com h = -0.5, 8 passos
    T0_euler = euler(f, t0=L, y0=TL, h=-0.5, n=8)
    tm_euler, min_euler, hora_euler = calcular_hora_morte(T0_euler)
    culpado_euler = identificar_culpado(min_euler)

    dif_min = min_rk4 - min_euler

    print("===== COMPARAÇÃO FINAL (VERSÃO A) =====")
    print(f"Euler (h = 0.5, 8 aval.):  T(0) = {T0_euler:.3f} °C | tm = {tm_euler:.4f} h | morte às {hora_euler} | Culpado: {culpado_euler}")
    print(f"RK4   (h = 1.0, 16 aval.): T(0) = {T0_rk4:.3f} °C | tm = {tm_rk4:.4f} h | morte às {hora_rk4} | Culpado: {culpado_rk4}")
    print(f"Diferença na hora da morte: {dif_min:+d} minutos | Culpado muda? {'Sim' if culpado_rk4 != culpado_euler else 'Não'}")


if __name__ == "__main__":
    ex1()
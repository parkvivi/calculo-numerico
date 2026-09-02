import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metodos import bisseccao, newton, secante


# ==============================================================================
# PROBLEMA A: RESERVATÓRIO ESFÉRICO
# ==============================================================================
def problema_A():
  print("\n==================================================")
  print("=== PROBLEMA A: RESERVATÓRIO ESFÉRICO ===")
  print("==================================================")

  R = 3.0  # Raio (m)
  V_alvo = 40.0  # Volume desejado (m^3)

  # Equação: V(h) = pi * h^2 * (3R - h) / 3 - V_alvo = 0
  def f(h):
    return (math.pi * (h**2) * (3 * R - h)) / 3.0 - V_alvo

  def df(h):
    return math.pi * (2 * R * h - h**2)

  # A.1: Resolver em [0, 2R] -> [0, 6.0]
  try:
    raiz_h, hist_h = bisseccao(f, 0.0, 2 * R, eps=1e-4)
    print(
        f"\nA.1: Altura necessária h = {raiz_h:.4f} m (Iterações:"
        f" {len(hist_h)})"
    )
  except Exception as e:
    print(f"Erro no A.1: {e}")

  # A.2: Encontrar as 3 raízes reais da cúbica
  print("\nA.2: Procurando todas as 3 raízes reais...")
  intervalos = [(-3.0, 0.0), (0.0, 6.0), (6.0, 9.0)]
  for i, (a, b) in enumerate(intervalos, 1):
    try:
      r, _ = bisseccao(f, a, b, eps=1e-6)
      valido = (
          "VÁLIDA (Física: 0 <= h <= 2R)"
          if 0 <= r <= 2 * R
          else "ESPÚRIA (Fora do reservatório)"
      )
      print(f" Raiz {i}: h = {r:9.4f} m  -> {valido}")
    except Exception as e:
      print(f" Raiz {i}: Não encontrada no intervalo [{a}, {b}]")

  # A.3: Tabela h x V e Gráfico
  volumes = np.arange(10, 120, 10)
  alturas = []
  for v in volumes:
    f_v = lambda h: (math.pi * (h**2) * (3 * R - h)) / 3.0 - v
    r, _ = bisseccao(f_v, 0.0, 6.0, eps=1e-4)
    alturas.append(r)

  plt.figure(figsize=(7, 4))
  plt.plot(volumes, alturas, "o-b", label="h(V)")
  plt.title("Problema A: Altura da Água vs Volume (Reservatório Esférico)")
  plt.xlabel("Volume $V$ (m³)")
  plt.ylabel("Altura $h$ (m)")
  plt.grid(True, linestyle="--", alpha=0.7)
  plt.legend()
  plt.tight_layout()
  plt.show()


# ==============================================================================
# PROBLEMA B: PERDA DE CARGA EM TUBULAÇÃO
# ==============================================================================
def problema_B():
  print("\n==================================================")
  print("=== PROBLEMA B: PERDA DE CARGA EM TUBULAÇÃO ===")
  print("==================================================")

  D = 0.100  # m
  eps_rug = 4.5e-5  # m
  Re = 2.0e5
  L = 500.0  # m
  Q = 0.050  # m^3/s
  g = 9.81  # m/s^2
  A = math.pi * (D**2) / 4.0
  V_vel = Q / A

  # Colebrook-White: 1/sqrt(f) + 2*log10( eps/(3.7D) + 2.51/(Re*sqrt(f)) ) = 0
  def f_cw(f):
    if f <= 0:
      return 1e9
    term = (eps_rug / (3.7 * D)) + (2.51 / (Re * math.sqrt(f)))
    return (1.0 / math.sqrt(f)) + 2.0 * math.log10(term)

  # B.3: Swamee-Jain (Chute inicial)
  arg_sj = (eps_rug / (3.7 * D)) + (5.74 / (Re**0.9))
  f0_sj = 0.25 / ((math.log10(arg_sj)) ** 2)
  print(f"B.3: Chute de Swamee-Jain (f0) = {f0_sj:.6f}")

  # B.1 & B.2: Secante
  try:
    f_sol, hist_sj = secante(f_cw, f0_sj, f0_sj * 1.1, eps=1e-8)
    print(f"B.1: Fator de Atrito f = {f_sol:.6f} (Iterações SJ: {len(hist_sj)})")

    _, hist_arb = secante(f_cw, 0.05, 0.06, eps=1e-8)
    print(
        f"     Iterações com chute arbitrário (f0=0.05): {len(hist_arb)} (Economia:"
        f" {len(hist_arb) - len(hist_sj)} iterações)"
    )

    # B.4: Darcy-Weisbach
    hf = f_sol * (L / D) * (V_vel**2 / (2 * g))
    print(f"B.4: Perda de carga h_f = {hf:.3f} m")

    # B.5: Sensibilidade com f = 0.02
    hf_aprox = 0.02 * (L / D) * (V_vel**2 / (2 * g))
    erro_pct = abs(hf_aprox - hf) / hf * 100
    print(
        f"B.5: h_f aproximado (f=0.02) = {hf_aprox:.3f} m -> Erro:"
        f" {erro_pct:.2f}%"
    )

  except Exception as e:
    print(f"Aguardando função Secante: {e}")


# ==============================================================================
# PROBLEMA C: EQUAÇÃO DE VAN DER WAALS
# ==============================================================================
def problema_C():
  print("\n==================================================")
  print("=== PROBLEMA C: EQUAÇÃO DE VAN DER WAALS ===")
  print("==================================================")

  R = 8.314  # J/(mol K)
  a = 0.3640  # Pa m^6 / mol^2
  b = 4.267e-5  # m^3 / mol
  T = 300.0  # K
  P = 5.0e6  # Pa (5.0 MPa)

  # C.1: Gás Ideal vs van der Waals
  v_ideal = (R * T) / P

  def f_vdw(v):
    return (P + (a / (v**2))) * (v - b) - R * T

  def df_vdw(v):
    return P - (a / (v**2)) + (2 * a * b / (v**3))

  try:
    v_vdw, _ = newton(f_vdw, df_vdw, x0=v_ideal, eps=1e-8)
    erro_pct = abs(v_ideal - v_vdw) / v_vdw * 100
    print(f"C.1: Volume Gás Ideal:     {v_ideal:.6e} m³/mol")
    print(f"     Volume van der Waals: {v_vdw:.6e} m³/mol")
    print(f"     Erro do Modelo Ideal: {erro_pct:.2f}%")
  except Exception as e:
    print(f"Aguardando função Newton: {e}")

  # C.4: Isoterma P x v
  pressoes_mpa = np.linspace(1.0, 10.0, 19)
  v_vdw_lista = []
  v_ideal_lista = []

  for p_m in pressoes_mpa:
    p_pa = p_m * 1e6
    v_i = (R * T) / p_pa
    v_ideal_lista.append(v_i)

    f_p = lambda v: (p_pa + (a / (v**2))) * (v - b) - R * T
    try:
      r_v, _ = bisseccao(f_p, b * 1.01, 1e-2, eps=1e-8)
      v_vdw_lista.append(r_v)
    except Exception:
      v_vdw_lista.append(v_i)

  plt.figure(figsize=(7, 4))
  plt.plot(
      [v * 1e3 for v in v_ideal_lista],
      pressoes_mpa,
      "--r",
      label="Gás Ideal",
  )
  plt.plot(
      [v * 1e3 for v in v_vdw_lista],
      pressoes_mpa,
      "-b",
      label="van der Waals",
  )
  plt.title("Problema C: Isoterma P x v para CO2 (T = 300 K)")
  plt.xlabel("Volume Molar $v$ (L/mol)")
  plt.ylabel("Pressão $P$ (MPa)")
  plt.grid(True, linestyle="--", alpha=0.7)
  plt.legend()
  plt.tight_layout()
  plt.show()


# ==============================================================================
# PROBLEMA D: TAXA INTERNA DE RETORNO (TIR)
# ==============================================================================
def problema_D():
  print("\n==================================================")
  print("=== PROBLEMA D: TAXA INTERNA DE RETORNO (TIR) ===")
  print("==================================================")

  fluxo1 = [-1000, 300, 350, 400, 450]

  def vpl1(i):
    return sum(cf / ((1 + i) ** k) for k, cf in enumerate(fluxo1))

  # D.1: Cálculo da TIR
  try:
    tir1, _ = bisseccao(vpl1, 0.0, 0.5, eps=1e-6)
    print(f"D.1: TIR do Projeto 1 = {tir1 * 100:.4f}%")

    # D.3: Decisão com WACC 15% e 20%
    print(f"D.3: VPL a 15% = R$ {vpl1(0.15):.2f} mil -> Aceitar (VPL > 0)")
    print(f"     VPL a 20% = R$ {vpl1(0.20):.2f} mil -> Rejeitar (VPL < 0)")
  except Exception as e:
    print(f"Erro no D.1: {e}")

  # D.4: Múltiplas TIRs (Projeto 2)
  fluxo2 = [-1000, 2500, -1540]

  def vpl2(i):
    return sum(cf / ((1 + i) ** k) for k, cf in enumerate(fluxo2))

  try:
    tir2_a, _ = bisseccao(vpl2, 0.0, 0.2, eps=1e-6)
    tir2_b, _ = bisseccao(vpl2, 0.3, 0.6, eps=1e-6)
    print(
        f"\nD.4: Projeto 2 possui DUAS TIRs: i1 = {tir2_a * 100:.2f}% e i2 ="
        f" {tir2_b * 100:.2f}%"
    )
  except Exception as e:
    print(f"Erro no D.4: {e}")


# ==============================================================================
# PROBLEMA E: EQUAÇÃO DE KEPLER
# ==============================================================================
def problema_E():
  print("\n==================================================")
  print("=== PROBLEMA E: EQUAÇÃO DE KEPLER ===")
  print("==================================================")

  # E.1: Cometa Halley (e = 0.967, M = 0.2 rad)
  e_halley, M_halley = 0.967, 0.2

  def f_halley(E):
    return E - e_halley * math.sin(E) - M_halley

  def df_halley(E):
    return 1.0 - e_halley * math.cos(E)

  try:
    E_halley, hist_e1 = newton(
        f_halley, df_halley, x0=M_halley + e_halley * math.sin(M_halley)
    )
    print(
        f"E.1: Anomalia Excêntrica Halley E = {E_halley:.6f} rad (Iterações:"
        f" {len(hist_e1)})"
    )
  except Exception as e:
    print(f"Aguardando função Newton: {e}")


# ==============================================================================
# PROBLEMA F (BÔNUS): DEFLEXÃO DE VIGA
# ==============================================================================
def problema_F():
  print("\n==================================================")
  print("=== PROBLEMA F (BÔNUS): DEFLEXÃO DE VIGA ===")
  print("==================================================")

  L = 600.0  # cm
  E_mod = 50000.0  # kN/cm^2
  I_mom = 30000.0  # cm^4
  w0 = 2.5  # kN/cm
  C = w0 / (120.0 * L * E_mod * I_mom)

  # dy/dx = C * (-5*x^4 + 6*L^2*x^2 - L^4) = 0
  def dy_dx(x):
    return C * (-5 * (x**4) + 6 * (L**2) * (x**2) - (L**4))

  try:
    # F.1: Bissecção no intervalo [0.4L, 0.8L] para ignorar a raiz x = L
    x_max, _ = bisseccao(dy_dx, 0.4 * L, 0.8 * L, eps=1e-6)

    # F.2: Deflexão máxima
    y_max = (
        C
        * (
            -(x_max**5)
            + 2 * (L**2) * (x_max**3)
            - (L**4) * x_max
        )
    )

    print(
        f"F.1: Ponto de deflexão máxima x = {x_max:.2f} cm (Teórico: x ="
        f" L/sqrt(5) = {L / math.sqrt(5):.2f} cm)"
    )
    print(f"F.2: Deflexão máxima y(x_max) = {y_max:.4f} cm")
  except Exception as e:
    print(f"Erro no Problema F: {e}")


# ==============================================================================
# BLOCO PRINCIPAL DE EXECUÇÃO
# ==============================================================================
if __name__ == "__main__":
  problema_A()
  problema_B()
  problema_C()
  problema_D()
  problema_E()
  problema_F()
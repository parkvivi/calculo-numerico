import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Importa os métodos implementados pelo grupo no metodos.py
from metodos import bisseccao, newton, secante


# Decorator de contagem
def contador(f):
  def wrapper(x):
    wrapper.n += 1
    return f(x)

  wrapper.n = 0
  return wrapper


# Função teste padrão da Parte 2
def f_padrao(x):
  return x**3 - 9 * x + 3


# --- ESTRUTURA DO EXERCÍCIO 2.1 ---
def tabelar_sinais(f, a, b, n):
  """Avalia f em n pontos igualmente espaçados no intervalo [a, b]

  e retorna a lista de subintervalos onde ocorre mudança de sinal.
  """
  x_pontos = np.linspace(a, b, n)  # n pontos no total
  intervalos_encontrados = []

  for i in range(len(x_pontos) - 1):
    x_atual = x_pontos[i]
    x_proximo = x_pontos[i + 1]

    # Teste de mudança de sinal
    if (f(x_atual) > 0) != (f(x_proximo) > 0):
      intervalos_encontrados.append((x_atual, x_proximo))

  return intervalos_encontrados


def exercicio_2_1():
  print("\n=== Exercício 2.1: Isolamento de Raízes ===")

  # Item (a)
  def f_a(x):
    return x**3 - 9 * x + 3

  print("\n--- Item (a): f(x) em [-5, 5] ---")
  malhas_a = [21, 11, 6, 4]
  for n in malhas_a:
    ints = tabelar_sinais(f_a, -5, 5, n)
    print(f"n = {n:2d} -> Raízes (mudanças de sinal) encontradas: {len(ints)}")

  # Item (b)
  def g_b(x):
    return (x - 1.05) * (x - 1.15) * (x - 3.0)

  print("\n--- Item (b): g(x) em [0, 4] ---")
  malhas_b = [9, 17, 41, 401]
  for n in malhas_b:
    ints = tabelar_sinais(g_b, 0, 4, n)
    print(f"n = {n:3d} -> Raízes (mudanças de sinal) encontradas: {len(ints)}")


# --- ESTRUTURA DO EXERCÍCIO 2.2 ---
def exercicio_2_2():
  print("\n=== Exercício 2.2: Previsão x Realidade na Bissecção ===")
  tolerancias = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
  a0, b0 = 0.0, 1.0

  resultados = []
  for eps in tolerancias:
    # Previsão teórica k > (log2(b0 - a0) - log2(eps))
    k_previsto = math.ceil(math.log2(b0 - a0) - math.log2(eps))

    _, hist = bisseccao(f_padrao, a0, b0, eps=eps)
    k_efetivo = len(hist)

    resultados.append({
        "eps": eps,
        "k_previsto": k_previsto,
        "k_efetivo": k_efetivo,
        "diferenca": k_previsto - k_efetivo,
    })

  df = pd.DataFrame(resultados)
  print(df.to_string(index=False))


# --- ESTRUTURA DO EXERCÍCIO 2.3 ---
def exercicio_2_3():
  print("\n=== Exercício 2.3: Custo Real (Avaliações de Função) ===")
  eps = 1e-8
  a0, b0 = 0.0, 1.0
  x0, x1 = 0.5, 1.0

  def df_padrao(x):
    return 3 * (x**2) - 9

  dados = []

  # 1. Bissecção
  try:
    f_biss = contador(f_padrao)
    _, hist_biss = bisseccao(f_biss, a0, b0, eps=eps)
    dados.append({
        "Método": "Bissecção",
        "Iterações": len(hist_biss),
        "Avaliações f": f_biss.n,
        "Avaliações f'": 0,
    })
  except Exception as e:
    print(f"[Bissecção]: Erro na execução: {e}")

  # 2. Newton
  try:
    f_newt, df_newt = contador(f_padrao), contador(df_padrao)
    res_newt = newton(f_newt, df_newt, x0, eps=eps)
    if res_newt is not None:
      _, hist_newt = res_newt
      dados.append({
          "Método": "Newton",
          "Iterações": len(hist_newt),
          "Avaliações f": f_newt.n,
          "Avaliações f'": df_newt.n,
      })
  except Exception:
    print("[Newton]: Função aguardando implementação do grupo.")

  # 3. Secante
  try:
    f_sec = contador(f_padrao)
    res_sec = secante(f_sec, x0, x1, eps=eps)
    if res_sec is not None:
      _, hist_sec = res_sec
      dados.append({
          "Método": "Secante",
          "Iterações": len(hist_sec),
          "Avaliações f": f_sec.n,
          "Avaliações f'": 0,
      })
  except Exception:
    print("[Secante]: Função aguardando implementação do grupo.")

  if dados:
    df = pd.DataFrame(dados)
    print("\n", df.to_string(index=False))


# --- ESTRUTURA DO EXERCÍCIO 2.4 ---
def exercicio_2_4():
  print("\n=== Exercício 2.4: Ordem Empírica de Convergência ===")
  xi = 0.337608955965837  # Raiz exata fornecida no PDF
  eps = 1e-10
  x0, x1 = 0.5, 1.0

  def df_padrao(x):
    return 3 * (x**2) - 9

  def calcular_ordem_p(historico):
    if len(historico) < 3:
      return []

    erros = [abs(h["x"] - xi) for h in historico]
    p_valores = []

    for k in range(2, len(erros)):
      e_k = erros[k]
      e_k_1 = erros[k - 1]
      e_k_2 = erros[k - 2]

      if e_k_1 > 0 and e_k_2 > 0 and e_k > 0 and e_k_1 != e_k_2:
        numerador = math.log(e_k / e_k_1)
        denominador = math.log(e_k_1 / e_k_2)
        p_k = numerador / denominador
        p_valores.append({"k": historico[k]["k"], "e_k": e_k, "p_k": p_k})

    return p_valores

  # Teste de Newton
  try:
    res_newt = newton(f_padrao, df_padrao, x0, eps=eps)
    if res_newt is not None:
      _, hist_newt = res_newt
      p_newt = calcular_ordem_p(hist_newt)
      print("\n--- Método de Newton (Teórico p = 2.0) ---")
      print(pd.DataFrame(p_newt).to_string(index=False))
  except Exception:
    print("\n[Newton]: Função aguardando implementação do grupo.")

  # Teste da Secante
  try:
    res_sec = secante(f_padrao, x0, x1, eps=eps)
    if res_sec is not None:
      _, hist_sec = res_sec
      p_sec = calcular_ordem_p(hist_sec)
      print("\n--- Método da Secante (Teórico p ≈ 1.618) ---")
      print(pd.DataFrame(p_sec).to_string(index=False))
  except Exception:
    print("\n[Secante]: Função aguardando implementação do grupo.")

# ESTRUTURA DO EXERCÍCIO 2.5 
def exercicio_2_5():
  print("\n=== Exercício 2.5: Modos de Falha do Método de Newton ===")

  # Caso 1: Derivada nula no chute (f(x) = x^2 - 2, x0 = 0)
  print("\n--- Caso 1: Derivada Nula (f(x) = x^2 - 2, x0 = 0) ---")
  try:
    res = newton(lambda x: x**2 - 2, lambda x: 2 * x, 0.0)
    if res is not None:
      raiz, _ = res
      print(f"Resultado: {raiz}")
  except Exception as e:
    print(f"Tratamento correto da falha: {e}")

  # Caso 2: Oscilação / Ciclo Infinito (f(x) = x^3 - 2x + 2, x0 = 0)
  print("\n--- Caso 2: Oscilação (f(x) = x^3 - 2x + 2, x0 = 0) ---")
  try:
    res = newton(
        lambda x: x**3 - 2 * x + 2,
        lambda x: 3 * (x**2) - 2,
        0.0,
        max_iter=10,
    )
    if res is not None:
      _, hist = res
      print("Primeiros passos da oscilação (ciclo entre 0 e 1):")
      for item in hist[:6]:
        print(f"k={item['k']}: x = {item['x']:.4f}, f(x) = {item['fx']:.4f}")
  except Exception as e:
    print(f"Observação: {e}")

  # Caso 3: Derivada próxima de zero / Salto astronômico (f(x) = x^3 - 9x + 3, x0 = 1.73)
  print(
      "\n--- Caso 3: Derivada Próxima de Zero (f(x) = x^3 - 9x + 3, x0 = 1.73)"
      " ---"
  )
  try:
    res = newton(
        lambda x: x**3 - 9 * x + 3,
        lambda x: 3 * (x**2) - 9,
        1.73,
        max_iter=15,
    )
    if res is not None:
      _, hist = res
      print("Observando os saltos nos valores de x devido a f'(x) ≈ 0:")
      for item in hist:
        print(f"k={item['k']}: x = {item['x']:.4f}, f(x) = {item['fx']:.4f}")
  except Exception as e:
    print(f"Observação: {e}")

# --- ESTRUTURA DO EXERCÍCIO 2.6 ---
def exercicio_2_6():
  print("\n=== Exercício 2.6: Sensibilidade a Chutes na Secante ===")
  eps = 1e-8
  # Pares de teste: normais e próximos de pontos críticos (ex: x ≈ 1.732 onde f'(x) = 0)
  pares_chute = [(0.0, 1.0), (0.5, 1.0), (1.7, 1.8), (-1.0, 0.0)]

  dados = []
  for x0, x1 in pares_chute:
    try:
      res = secante(f_padrao, x0, x1, eps=eps)
      if res is not None:
        raiz, hist = res
        dados.append({
            "x0": x0,
            "x1": x1,
            "Iterações": len(hist),
            "Raiz Encontrada": round(raiz, 8),
            "Status": "Convergiu",
        })
    except Exception as e:
      dados.append({
          "x0": x0,
          "x1": x1,
          "Iterações": "-",
          "Raiz Encontrada": "-",
          "Status": f"Falha/Pendente ({e})",
      })

  df = pd.DataFrame(dados)
  print(df.to_string(index=False))


# --- ESTRUTURA DO EXERCÍCIO 2.7 ---
def exercicio_2_7():
  print("\n=== Exercício 2.7: Comparação Global e Gráfico de Convergência ===")
  eps = 1e-10
  a0, b0 = 0.0, 1.0
  x0, x1 = 0.5, 1.0

  def df_padrao(x):
    return 3 * (x**2) - 9

  plt.figure(figsize=(8, 5))

  # 1. Plot Bissecção
  try:
    _, hist_biss = bisseccao(f_padrao, a0, b0, eps=eps)
    k_biss = [h["k"] for h in hist_biss]
    log_fx_biss = [
        math.log10(abs(h["fx"])) if abs(h["fx"]) > 0 else -16
        for h in hist_biss
    ]
    plt.plot(k_biss, log_fx_biss, label="Bissecção", marker="o", markersize=3)
  except Exception:
    pass

  # 2. Plot Newton
  try:
    res_newt = newton(f_padrao, df_padrao, x0, eps=eps)
    if res_newt is not None:
      _, hist_newt = res_newt
      k_newt = [h["k"] for h in hist_newt]
      log_fx_newt = [
          math.log10(abs(h["fx"])) if abs(h["fx"]) > 0 else -16
          for h in hist_newt
      ]
      plt.plot(k_newt, log_fx_newt, label="Newton", marker="s", markersize=4)
  except Exception:
    pass

  # 3. Plot Secante
  try:
    res_sec = secante(f_padrao, x0, x1, eps=eps)
    if res_sec is not None:
      _, hist_sec = res_sec
      k_sec = [h["k"] for h in hist_sec]
      log_fx_sec = [
          math.log10(abs(h["fx"])) if abs(h["fx"]) > 0 else -16
          for h in hist_sec
      ]
      plt.plot(k_sec, log_fx_sec, label="Secante", marker="^", markersize=4)
  except Exception:
    pass

  plt.title("ConverVector: $\log_{10}|f(x_k)|$ vs Iteração ($k$)")
  plt.xlabel("Iteração ($k$)")
  plt.ylabel("$\log_{10}|f(x_k)|$")
  plt.grid(True, linestyle="--", alpha=0.7)
  plt.legend()
  plt.tight_layout()
  plt.show()


# --- BLOCO PRINCIPAL DE EXECUÇÃO ---
if __name__ == "__main__":
  exercicio_2_1()
  exercicio_2_2()
  exercicio_2_3()
  exercicio_2_4()
  exercicio_2_5()
  exercicio_2_6()
  exercicio_2_7()
"""
testar_metodos.py
Script de verificacao do metodos.py -- roda sanity checks manuais
(nao precisa de pytest, so precisa do metodos.py na mesma pasta).

Como rodar:
    python testar_metodos.py
"""

import math
from metodos import bisseccao, newton, secante, contador


def f_padrao(x):
    """Funcao-teste padrao do TC1: f(x) = x^3 - 9x + 3"""
    return x**3 - 9*x + 3


def df_padrao(x):
    """Derivada de f_padrao"""
    return 3*x**2 - 9


RAIZ_EXATA = 0.3376089559658377  # dada no enunciado (exercicio 2.4)
testes_ok = 0
testes_total = 0


def checar(nome, condicao):
    global testes_ok, testes_total
    testes_total += 1
    status = "OK " if condicao else "FALHOU"
    print(f"[{status}] {nome}")
    if condicao:
        testes_ok += 1


print("=" * 60)
print("TESTE 1: Bisseccao encontra a raiz correta")
print("=" * 60)
raiz, hist = bisseccao(f_padrao, 0, 1, eps=1e-8)
checar("Bisseccao converge perto da raiz exata",
       abs(raiz - RAIZ_EXATA) < 1e-6)
checar("Historico tem as chaves k, x, fx, erro",
       all(k in hist[0] for k in ("k", "x", "fx", "erro")))
checar("Numero de iteracoes bate com a previsao teorica (27 para eps=1e-8, [0,1])",
       len(hist) == math.ceil((math.log(1) - math.log(1e-8)) / math.log(2)))
print()

print("=" * 60)
print("TESTE 2: Newton encontra a raiz correta e converge rapido")
print("=" * 60)
raiz, hist = newton(f_padrao, df_padrao, 0.5, eps=1e-8)
checar("Newton converge perto da raiz exata",
       abs(raiz - RAIZ_EXATA) < 1e-6)
checar("Newton converge em poucas iteracoes (< 10, esperado ~3-6)",
       len(hist) < 10)
print()

print("=" * 60)
print("TESTE 3: Secante encontra a raiz correta")
print("=" * 60)
raiz, hist = secante(f_padrao, 0, 1, eps=1e-8)
checar("Secante converge perto da raiz exata",
       abs(raiz - RAIZ_EXATA) < 1e-6)
print()

print("=" * 60)
print("TESTE 4: Validacao de entrada (Requisito 1)")
print("=" * 60)
try:
    bisseccao(f_padrao, 0, 0.1)  # f(0)=3, f(0.1)~2.1 -> mesmo sinal
    checar("Bisseccao levanta erro se f(a)*f(b) >= 0", False)
except ValueError:
    checar("Bisseccao levanta ValueError se f(a)*f(b) >= 0", True)

try:
    # f(x) = x^3 - 3x + 2 tem derivada nula em x=1 (raiz dupla)
    newton(lambda x: x**3 - 3*x + 2, lambda x: 3*x**2 - 3, x0=1.0, max_iter=5)
    checar("Newton levanta erro se derivada = 0", False)
except ValueError:
    checar("Newton levanta ValueError se derivada = 0", True)

try:
    # Forcar fx1 == fx0 (denominador nulo): usar dois pontos com mesma imagem
    secante(lambda x: 0.0, 0, 1)
    checar("Secante levanta erro se denominador = 0", False)
except ValueError:
    checar("Secante levanta ValueError se denominador = 0", True)
print()

print("=" * 60)
print("TESTE 5: Criterio de parada combinado (OR, nao AND)")
print("=" * 60)
# Se o criterio fosse "and" em vez de "or", a bisseccao com eps MUITO
# frouxo (ex: 0.1) ainda deveria parar rapido, pois abs(fx) tende a ficar
# pequeno perto da raiz mesmo com erro de intervalo ainda grande.
raiz, hist = bisseccao(f_padrao, 0, 1, eps=0.1, max_iter=200)
checar("Com eps frouxo (0.1), bisseccao para bem antes de max_iter",
       len(hist) < 20)
print()

print("=" * 60)
print("TESTE 6: Nunca laco infinito (Requisito 4)")
print("=" * 60)
# Funcao sem raiz facil de atingir com max_iter pequeno para newton:
# f(x) = arctan(x), x0=2.0 diverge/oscila para valores grandes de x0
raiz, hist = newton(math.atan, lambda x: 1 / (1 + x**2), x0=2.0, max_iter=5)
checar("Newton respeita max_iter e nao trava (retornou apos poucas iteracoes)",
       len(hist) <= 5)
checar("Historico guarda a flag 'convergiu'",
       "convergiu" in hist[0])
print()

print("=" * 60)
print("TESTE 7: Contador de avaliacoes (Requisito 3)")
print("=" * 60)
g = contador(f_padrao)
raiz, hist = bisseccao(g, 0, 1, eps=1e-8)
checar("Contador contou pelo menos 1 avaliacao por iteracao + 2 iniciais",
       g.n >= len(hist) + 2)
print(f"    (avaliacoes de f: {g.n}, iteracoes: {len(hist)})")
print()

print("=" * 60)
print(f"RESULTADO FINAL: {testes_ok}/{testes_total} testes passaram")
print("=" * 60)
if testes_ok == testes_total:
    print(">>> Todos os testes passaram! O arquivo parece estar correto.")
else:
    print(">>> Alguns testes falharam -- revise o metodos.py.")
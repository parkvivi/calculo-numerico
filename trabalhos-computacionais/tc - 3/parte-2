import numpy as np

np.set_printoptions(precision=6, suppress=True)


def mmq_poly(x, y, grau):
   
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    M = np.array([[np.sum(x ** (i + j)) for j in range(grau + 1)]
                  for i in range(grau + 1)])
    b = np.array([np.sum(x ** i * y) for i in range(grau + 1)])
    coef = np.linalg.solve(M, b)
    erro = np.sum((y - avalia(coef, x)) ** 2)
    return coef, erro, M, b


def avalia(coef, x):
    
    x = np.asarray(x, dtype=float)
    r = np.zeros_like(x)
    for a in coef[::-1]:
        r = r * x + a
    return r


def resolve(x, y, grau, titulo):
    coef, erro, M, b = mmq_poly(x, y, grau)
    print(f"--- {titulo} (grau {grau}) ---")
    print("Matriz normal M:\n", M)
    print("Vetor b:", b)
    print("Coeficientes [a0, a1, ...]:", coef)
    print("Valores ajustados:", avalia(coef, x))
    print(f"Erro E = {erro:.6f}   |  sqrt(E) = {np.sqrt(erro):.6f}")
    print("Confere com np.polyfit?",
          np.allclose(coef, np.polyfit(x, y, grau)[::-1], atol=1e-6))
    print()
    return coef


#                 Exercício 2 
print("=" * 50, "\nEXERCÍCIO 2\n", "=" * 50, sep="")
x2 = [1.0, 1.2, 1.5, 1.8, 2.1, 2.4, 2.8]
y2 = [5.5, 7.3, 14.1, 26.3, 48.2, 80.1, 111.6]
resolve(x2, y2, 2, "Ex. 2(a)")
resolve(x2, y2, 3, "Ex. 2(b)")

#                 Exercício 3 

print("=" * 50, "\nEXERCÍCIO 3\n", "=" * 50, sep="")
x3 = np.array([0, 1, 2, 3])
exato = (x3 ** 2 + 3 * x3 + 7).astype(float)   
for g in (2, 3):
    coef = resolve(x3, exato, g, "Ex. 3")
    p = avalia(coef, x3)
    print(f"Comparação com o exato (grau {g}):")
    print("  x      :", x3)
    print("  exato  :", exato)
    print("  p(x)   :", p)
    print("  |erro| :", np.abs(exato - p))
    print()

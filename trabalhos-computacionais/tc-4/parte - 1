# Parte - 1
import math
import os
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.integrate import quad
except ImportError:  
    quad = None

# figuras sao salvas na mesma pasta deste arquivo, de onde quer que ele seja rodado
PASTA = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# Regras de integracao (implementacao propria)
# --------------------------------------------------------------------------
def trapezio(f, a, b, n):
    """Regra do trapezio composta com n subintervalos iguais."""
    h = (b - a) / n
    x = a + h * np.arange(n + 1)
    y = f(x)
    return h * (0.5 * y[0] + y[1:-1].sum() + 0.5 * y[-1])


def simpson(f, a, b, n):
    """Regra 1/3 de Simpson composta; n (numero de subintervalos) deve ser par."""
    if n % 2 != 0:
        raise ValueError("Simpson 1/3 exige numero par de subintervalos")
    h = (b - a) / n
    x = a + h * np.arange(n + 1)
    y = f(x)
    return simpson_tabela(y, h)


def simpson_tabela(y, h):
    """Simpson 1/3 composta sobre valores tabelados igualmente espacados (len(y)-1 par)."""
    y = np.asarray(y, float)
    n = len(y) - 1
    if n % 2 != 0:
        raise ValueError("Simpson 1/3 exige numero par de subintervalos")
    return h / 3 * (y[0] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum() + y[-1])


def simpson38_tabela(y, h):
    """Simpson 3/8 composta sobre valores tabelados (len(y)-1 multiplo de 3)."""
    y = np.asarray(y, float)
    n = len(y) - 1
    if n % 3 != 0:
        raise ValueError("Simpson 3/8 exige numero de subintervalos multiplo de 3")
    s = y[0] + y[-1]
    for i in range(1, n):
        s += (2 if i % 3 == 0 else 3) * y[i]
    return 3 * h / 8 * s


def salva(nome):
    """Salva a figura atual em PDF (relatorio LaTeX) e PNG (Google Docs/Word)."""
    plt.savefig(os.path.join(PASTA, nome + ".pdf"))
    plt.savefig(os.path.join(PASTA, nome + ".png"), dpi=200)


def linha(titulo):
    print("\n" + "=" * 70 + f"\n{titulo}\n" + "=" * 70)


if __name__ == "__main__":

    # ----------------------------------------------------------------------
    # Exercicio 1: trapezio composto, n = 30
    # ----------------------------------------------------------------------
    linha("Exercicio 1  -  int_{-2}^{3} (x^4 - 3x^3 + 2x^2 - 3) dx, trapezio n = 30")
    f1 = lambda x: x**4 - 3 * x**3 + 2 * x**2 - 3
    a, b, n = -2.0, 3.0, 30
    I1 = trapezio(f1, a, b, n)
    I1_exato = 175 / 12
    h = (b - a) / n
    print(f"h = {h:.6f}")
    print(f"T_30      = {I1:.10f}")
    print(f"exato     = {I1_exato:.10f}")
    print(f"erro      = {I1 - I1_exato:.4e}   (erro relativo {abs(I1 - I1_exato) / I1_exato:.4e})")

    # Estimativa assintotica do erro: E ~ h^2/12 [f'(b) - f'(a)]
    df1 = lambda x: 4 * x**3 - 9 * x**2 + 4 * x
    E_ass = h**2 / 12 * (df1(b) - df1(a))
    print(f"estimativa h^2/12 [f'(b)-f'(a)] = {E_ass:.4e}")
    # Cota: |E| <= (b-a) h^2/12 max|f''|,  f'' = 12x^2 - 18x + 4
    xs = np.linspace(a, b, 100001)
    M2 = np.max(np.abs(12 * xs**2 - 18 * xs + 4))
    print(f"max|f''| em [-2,3] = {M2:.4f}  ->  cota |E| <= {(b - a) * h**2 / 12 * M2:.4e}")

    # Ordem observada: dobra n e ve o erro cair ~4x
    ns = [15, 30, 60, 120]
    erros1 = [abs(trapezio(f1, a, b, m) - I1_exato) for m in ns]
    print("ordem observada:")
    for m, e, e_ant in zip(ns, erros1, [None] + erros1[:-1]):
        razao = f"{e_ant / e:.4f}" if e_ant else "  -   "
        print(f"   n = {m:4d}   erro = {e:.4e}   razao = {razao}")

    # ----------------------------------------------------------------------
    # Exercicio 2: e^{-x^2} em [0,1], trapezio h = 0.25
    # ----------------------------------------------------------------------
    linha("Exercicio 2  -  int_0^1 e^{-x^2} dx, trapezio h = 0.25")
    f2 = lambda x: np.exp(-x**2)
    n2 = 4
    x2 = np.linspace(0, 1, n2 + 1)
    for xi, yi in zip(x2, f2(x2)):
        print(f"   x = {xi:.2f}   f(x) = {yi:.10f}")
    I2 = trapezio(f2, 0, 1, n2)
    I2_ref = math.sqrt(math.pi) / 2 * math.erf(1)
    print(f"T(h=0.25) = {I2:.10f}")
    print(f"referencia (sqrt(pi)/2 erf(1)) = {I2_ref:.10f}")
    print(f"erro      = {I2 - I2_ref:.4e}")
    # Cota: f'' = (4x^2 - 2) e^{-x^2}, |f''| <= 2 em [0,1] (maximo em x = 0)
    print(f"cota |E| <= (b-a) h^2/12 * 2 = {1 * 0.25**2 / 12 * 2:.4e}")

    # ----------------------------------------------------------------------
    # Exercicio 3: Simpson, int_1^2 (2x^5 + 5x^3 - 3x + 1) dx
    # ----------------------------------------------------------------------
    linha("Exercicio 3  -  int_1^2 (2x^5 + 5x^3 - 3x + 1) dx, Simpson")
    f3 = lambda x: 2 * x**5 + 5 * x**3 - 3 * x + 1
    I3_exato = 21 + 18.75 - 4.5 + 1  # = 145/4
    print(f"exato = 145/4 = {I3_exato:.10f}")
    resultados3 = []
    for m in [2, 4, 8, 16]:
        S = simpson(f3, 1, 2, m)
        hm = 1 / m
        # f^(4) = 240 x (so o termo 2x^5 contribui) -> erro = -(b-a) h^4/180 * 240 xi
        cota = 1 * hm**4 / 180 * 480
        resultados3.append((m, S, S - I3_exato, cota))
        print(f"   n = {m:3d}   S = {S:.10f}   erro = {S - I3_exato:.4e}   cota = {cota:.4e}")
    # Erro exato do Simpson simples (n=2) pela formula com f^(4) = 240x:
    # E = -h^5/90 f^(4)(xi), h = 1/2  ->  xi = ?
    e2 = resultados3[0][2]
    xi = e2 / (0.5**5 / 90 * 240)
    print(f"   n = 2: erro = h^5/90 * 240 xi  ->  xi = {xi:.6f} (deve estar em [1,2])")

    # ----------------------------------------------------------------------
    # Exercicio 4: Simpson, int_0^pi dx/(4 + sin 20x), n = 10, 20, 30
    # ----------------------------------------------------------------------
    linha("Exercicio 4  -  int_0^pi dx/(4 + sin 20x), Simpson n = 10, 20, 30")
    f4 = lambda x: 1 / (4 + np.sin(20 * x))
    I4_exato = math.pi / math.sqrt(15)  # 10 periodos de 2pi/sqrt(a^2-b^2)/20 ... ver relatorio
    if quad is not None:
        I4_quad, _ = quad(f4, 0, math.pi, limit=500)
        print(f"exato pi/sqrt(15) = {I4_exato:.10f}   (quad: {I4_quad:.10f})")
    else:
        print(f"exato pi/sqrt(15) = {I4_exato:.10f}   (scipy nao instalado: conferencia com quad pulada)")
    resultados4 = []
    for m in [10, 20, 30, 60, 120, 240]:
        S = simpson(f4, 0, math.pi, m)
        xs4 = np.linspace(0, math.pi, m + 1)
        maxsen = np.max(np.abs(np.sin(20 * xs4)))
        resultados4.append((m, S, S - I4_exato))
        print(f"   n = {m:4d}   S = {S:.10f}   erro = {S - I4_exato:+.4e}   max|sin(20 x_i)| = {maxsen:.2e}")
    print(f"   pi/4 = {math.pi / 4:.10f}")

    # ----------------------------------------------------------------------
    # Exercicio 5: distancia do corredor (Simpson sobre tabela)
    # ----------------------------------------------------------------------
    linha("Exercicio 5  -  distancia = int_0^4 v(t) dt, Simpson sobre a tabela")
    t5 = np.arange(0, 4.01, 0.5)
    v5 = np.array([0, 4.66, 7.72, 8.76, 9.72, 10.10, 10.53, 10.77, 10.91])
    d_simp = simpson_tabela(v5, 0.5)
    d_trap = 0.5 * (0.5 * v5[0] + v5[1:-1].sum() + 0.5 * v5[-1])
    print(f"n = {len(v5) - 1} subintervalos, h = 0.5 s")
    print(f"Simpson 1/3 : d = {d_simp:.6f} m")
    print(f"(trapezio, so para comparar: {d_trap:.6f} m)")
    # Estimativa de erro por Richardson: Simpson com h = 1 usando so os pontos pares
    d_simp_h1 = simpson_tabela(v5[::2], 1.0)
    print(f"Simpson com h = 1 (pontos pares): {d_simp_h1:.6f} m")
    print(f"estimativa de erro (S_h - S_2h)/15 = {(d_simp - d_simp_h1) / 15:.4e} m")

    # ----------------------------------------------------------------------
    # Exercicio 6: temperatura media (8 pontos -> 7 subintervalos, impar!)
    # ----------------------------------------------------------------------
    linha("Exercicio 6  -  temperatura media de 9h a 16h, Simpson")
    t6 = np.arange(9, 17)
    T6 = np.array([75.3, 77.0, 83.2, 84.8, 86.5, 86.4, 81.1, 78.6])
    h6 = 1.0
    n6 = len(T6) - 1
    print(f"n = {n6} subintervalos (impar) -> Simpson 1/3 puro nao se aplica")
    # Opcao A: Simpson 1/3 em [9,13] (4 subint.) + Simpson 3/8 em [13,16] (3 subint.)
    IA = simpson_tabela(T6[:5], h6) + simpson38_tabela(T6[4:], h6)
    # Opcao B: Simpson 3/8 em [9,12] + Simpson 1/3 em [12,16]
    IB = simpson38_tabela(T6[:4], h6) + simpson_tabela(T6[3:], h6)
    # Comparacao: trapezio
    IT = h6 * (0.5 * T6[0] + T6[1:-1].sum() + 0.5 * T6[-1])
    dur = t6[-1] - t6[0]
    print(f"A: 1/3 [9,13] + 3/8 [13,16]: integral = {IA:.6f}  -> T_med = {IA / dur:.6f} F")
    print(f"B: 3/8 [9,12] + 1/3 [12,16]: integral = {IB:.6f}  -> T_med = {IB / dur:.6f} F")
    print(f"(trapezio: integral = {IT:.6f} -> T_med = {IT / dur:.6f} F)")
    print(f"diferenca A - B em T_med: {(IA - IB) / dur:.4e} F")
    Tm = 0.5 * (IA + IB) / dur
    print(f"T_med adotada (A) = {IA / dur:.4f} F = {(IA / dur - 32) * 5 / 9:.4f} C")

    # ----------------------------------------------------------------------
    # Graficos
    # ----------------------------------------------------------------------
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

    # Fig 1: convergencia do trapezio (Ex.1) e do Simpson (Ex.4)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    ns_t = np.array([5, 10, 15, 30, 60, 120, 240])
    err_t = [abs(trapezio(f1, a, b, m) - I1_exato) for m in ns_t]
    ax1.loglog(ns_t, err_t, "o-", label="trapézio (Ex. 1)")
    ax1.loglog(ns_t, err_t[3] * (30 / ns_t) ** 2, "k--", lw=0.8, label=r"$\propto n^{-2}$")
    ax1.loglog([30], [abs(I1 - I1_exato)], "s", ms=9, mfc="none", mec="C3", label="$n=30$")
    ax1.set_xlabel("$n$"); ax1.set_ylabel("|erro|"); ax1.legend(fontsize=8)
    ax1.set_title("Ex. 1: convergência do trapézio")

    ns_s = np.arange(10, 202, 2)
    err_s = np.maximum([abs(simpson(f4, 0, math.pi, m) - I4_exato) for m in ns_s], 1e-16)
    ax2.semilogy(ns_s, err_s, "o", ms=3, c="C0")
    for m in (10, 20, 30, 40):
        e = abs(simpson(f4, 0, math.pi, m) - I4_exato)
        ax2.semilogy([m], [e], "o", ms=7, mfc="none", mec="C3")
    for txt, m, dx, dy in [("$n=10,\\,20$", 20, 6, 6), ("$n=30$", 30, 8, -3), ("$n=40$", 40, 8, -8)]:
        e = abs(simpson(f4, 0, math.pi, m) - I4_exato)
        ax2.annotate(txt, (m, e), xytext=(dx, dy), textcoords="offset points",
                     fontsize=8, color="C3")
    ax2.set_ylim(5e-17, 1)
    ax2.set_xlabel("$n$"); ax2.set_ylabel("|erro|")
    ax2.set_title("Ex. 4: Simpson para $1/(4+\\sin 20x)$")
    plt.tight_layout()
    salva("p1_convergencia")

    # Fig 2: integrando do Ex. 4 e os nos de Simpson para n = 10, 20, 30
    fig, ax = plt.subplots(figsize=(9, 3.0))
    xx = np.linspace(0, math.pi, 4000)
    ax.plot(xx, f4(xx), lw=0.8, c="gray", label=r"$1/(4+\sin 20x)$")
    for m, mk, c in [(10, "o", "C0"), (30, "^", "C3")]:
        xn = np.linspace(0, math.pi, m + 1)
        ax.plot(xn, f4(xn), mk, c=c, ms=5, label=f"nós, $n={m}$")
    ax.axhline(0.25, c="C0", ls=":", lw=0.8)
    ax.set_xlabel("$x$"); ax.set_xlim(0, math.pi)
    ax.legend(fontsize=8, loc="upper right", ncol=3)
    ax.set_title("Ex. 4: com $n=10$ ou $20$ todos os nós caem em zeros de $\\sin 20x$")
    plt.tight_layout()
    salva("p1_ex4_nos")

    # Fig 3: dados tabelados dos Ex. 5 e 6
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
    ax1.plot(t5, v5, "o-", ms=4)
    ax1.fill_between(t5, v5, alpha=0.15)
    ax1.set_xlabel("$t$ (s)"); ax1.set_ylabel("$v$ (m/s)")
    ax1.set_title(f"Ex. 5: $d \\approx {d_simp:.3f}$ m")
    ax2.plot(t6, T6, "o-", ms=4, c="C1")
    ax2.axvspan(9, 13, color="C0", alpha=0.08, label="Simpson 1/3")
    ax2.axvspan(13, 16, color="C2", alpha=0.10, label="Simpson 3/8")
    ax2.axhline(IA / dur, c="k", ls="--", lw=0.8, label=f"$T_{{av}}={IA / dur:.2f}$ °F")
    ax2.set_xlabel("hora"); ax2.set_ylabel("$T$ (°F)")
    ax2.legend(fontsize=8, loc="lower center")
    ax2.set_title("Ex. 6: temperatura do rio")
    plt.tight_layout()
    salva("p1_tabelas")
    print(f"\nfiguras tambem salvas automaticamente (.pdf e .png) em:\n   {PASTA}")
    print("Abrindo as 3 figuras na tela (use o icone de disquete da janela para salvar).")
    print("Feche as janelas para encerrar o programa.")
    plt.show()

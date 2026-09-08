import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metodos import bisseccao, newton, secante


def fmt_sig(x, n=4):
    return f"{x:.{n}g}"


def fase_I(nome, f, pontos, unidade_x="x", unidade_f=""):
    print("\n2. FASE I - ISOLAMENTO DA RAIZ (TABELAMENTO)")
    print("-" * 70)

    dados = []
    for x in pontos:
        y = f(x)
        dados.append((x, y))
        print(f"x = {x:12.6g} {unidade_x:>4}   f(x) = {y:14.6g} {unidade_f}")

    intervalos = []
    for (x1, y1), (x2, y2) in zip(dados[:-1], dados[1:]):
        if y1 == 0:
            intervalos.append((x1, x1))
        elif y1 * y2 < 0:
            intervalos.append((x1, x2))

    if intervalos:
        print("\nMudanças de sinal encontradas:")
        for a, b in intervalos:
            if a == b:
                print(f"  -> Raiz exata no ponto x = {a:g}")
            else:
                print(
                    f"  -> f({a:g}) * f({b:g}) < 0  =>  raiz isolada no"
                    f" intervalo [{a:g}, {b:g}]"
                )
    else:
        print(
            "\nNenhuma mudança de sinal foi encontrada nos pontos tabelados."
        )

    return intervalos


def verificar(nome, f, raiz, unidade_x="", unidade_f="", tol=None):
    residual = f(raiz)
    print("\n5. VERIFICAÇÃO (SUBSTITUIÇÃO DE VOLTA)")
    print("-" * 70)
    print(f"Substituindo a raiz x = {raiz:.6g} {unidade_x} na função f(x):")
    print(f"Resíduo f(raiz) = {residual:.6e} {unidade_f}")

    if tol is not None:
        print(f"Critério de verificação: |f(x)| < {tol:.1e}")
        if abs(residual) < tol:
            print("Resultado: OK - Resíduo satisfaz a tolerância estipulada.")
        else:
            print("Resultado: Revisar precisão da raiz.")


# PROBLEMA A: RESERVATÓRIO ESFÉRICO
def problema_A():
    print("\n" + "=" * 70)
    print("PROBLEMA A: RESERVATÓRIO ESFÉRICO")
    print("=" * 70)

    R = 3.0
    V_alvo = 40.0

    print("\n1. DEDUÇÃO DA FUNÇÃO f(x):")
    print("   Volume da calota esférica: V(h) = (pi * h^2 * (3R - h)) / 3")
    print("   Substituindo R = 3.0 m e o volume desejado V_alvo = 40.0 m^3:")
    print("   f(h) = (pi * h^2 * (9 - h)) / 3 - 40.0 = 0")

    def f(h):
        return math.pi * h**2 * (3 * R - h) / 3.0 - V_alvo

    def df(h):
        return math.pi * h * (2 * R - h)

    # A.1 - Isolamento e solução
    intervalos = fase_I("altura da água", f, [0, 1, 2, 3, 4, 5, 6], "m", "m³")
    a, b = 2.0, 3.0

    print("\n3. JUSTIFICATIVA DO MÉTODO E DO CHUTE INICIAL:")
    print(
        "   - Chute/Intervalo [2, 3] m: Escolhido com base na Fase I, onde"
    )
    print(
        "     f(2) < 0 e f(3) > 0. É o único intervalo fisicamente possível"
        " (0 <= h <= 2R)."
    )
    print(
        "   - Método: Bissecção. Garantia absoluta de convergência para o"
        " intervalo isolado."
    )

    # Reduzido eps para 1e-8 garantindo resíduo < 1e-6
    raiz_h, hist_h = bisseccao(f, a, b, eps=1e-8)

    print("\n4. RESULTADO COM UNIDADE FÍSICA E ALGARISMOS SIGNIFICATIVOS:")
    print(
        f"   Raiz física: h = {fmt_sig(raiz_h)} m (em {len(hist_h)} iterações)"
    )

    verificar("Problema A", f, raiz_h, "m", "m³", 1e-6)

    # A.2 - Todas as três raízes
    print("\nA.2 - TODAS AS RAÍZES REAIS E ANÁLISE FÍSICA")
    intervalos_raizes = [(-2.0, -1.0), (2.0, 3.0), (8.0, 9.0)]
    for i, (ia, ib) in enumerate(intervalos_raizes, 1):
        r_biss, _ = bisseccao(f, ia, ib, eps=1e-8)
        r_newt, _ = newton(f, df, x0=(ia + ib) / 2, eps=1e-8)
        r_sec, _ = secante(f, ia, ib, eps=1e-8)

        status = (
            "VÁLIDA (Física)" if 0 <= r_biss <= 2 * R else "ESPÚRIA (Matemática)"
        )
        print(f"Raiz {i}: h = {fmt_sig(r_biss)} m | {status}")
        print(
            f"   - Bissecção: {r_biss:.6f} | Newton: {r_newt:.6f} | Secante:"
            f" {r_sec:.6f}"
        )

    print("\nJustificativa das Raízes Espúrias:")
    print(
        " 1) h < 0 (h ≈ -1.87 m): Representa uma calota matemática invertida"
        " sem sentido físico."
    )
    print(
        " 2) h > 2R (h ≈ 8.47 m): Supera o diâmetro total da esfera (6m),"
        " extrapolando a geometria real."
    )

    # A.3 - Tabela e gráfico h x V
    volumes = np.arange(10, 120, 10)
    alturas = []
    for v in volumes:
        f_v = lambda h, v=v: math.pi * h**2 * (3 * R - h) / 3.0 - v
        r, _ = bisseccao(f_v, 0.0, 6.0, eps=1e-6)
        alturas.append(r)

    print("\nTabela h x V:")
    print(
        pd.DataFrame({"V (m³)": volumes, "h (m)": alturas}).to_string(
            index=False
        )
    )

    plt.figure(figsize=(7, 4))
    plt.plot(volumes, alturas, "o-b", label="h(V)")
    plt.title("Problema A: Altura da Água vs Volume")
    plt.xlabel("Volume V (m³)")
    plt.ylabel("Altura h (m)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


# PROBLEMA B: PERDA DE CARGA EM TUBULAÇÃO
def problema_B():
    print("\n" + "=" * 70)
    print("PROBLEMA B: PERDA DE CARGA EM TUBULAÇÃO")
    print("=" * 70)

    D, eps_rug, Re, L, Q, g = 0.100, 4.5e-5, 2.0e5, 500.0, 0.050, 9.81
    A = math.pi * D**2 / 4.0
    V_vel = Q / A

    print("\n1. DEDUÇÃO DA FUNÇÃO f(x):")
    print("   Equação implícita de Colebrook-White:")
    print("   1/sqrt(f) = -2 * log10( (eps / 3.7D) + (2.51 / (Re * sqrt(f))) )")
    print(
        "   f(f_atrito) = 1/sqrt(f) + 2 * log10( (eps / 3.7D) + (2.51 / (Re *"
        " sqrt(f))) ) = 0"
    )

    # Proteção contra valores negativos de f durante iterações numéricas
    def f_cw(f_val):
        if f_val <= 0:
            return 1e6  # Retorna penalidade para evitar erro de domínio
        termo = eps_rug / (3.7 * D) + 2.51 / (Re * math.sqrt(f_val))
        return 1.0 / math.sqrt(f_val) + 2.0 * math.log10(termo)

    def df_cw(f_val):
        if f_val <= 0:
            return 1.0
        termo = eps_rug / (3.7 * D) + 2.51 / (Re * math.sqrt(f_val))
        dt_df = -1.255 / (Re * (f_val**1.5))
        return -0.5 / (f_val**1.5) + (2.0 / math.log(10)) * (dt_df / termo)

    # B.1 / B.2 - Comparação
    fase_I("fator de atrito", f_cw, [0.015, 0.02, 0.025], "f", "")

    arg_sj = eps_rug / (3.7 * D) + 5.74 / (Re**0.9)
    f0_sj = 0.25 / (math.log10(arg_sj) ** 2)

    print("\n3. JUSTIFICATIVA DO MÉTODO E DO CHUTE INICIAL:")
    print(
        "   - Chute Inicial: f0 ≈"
        f" {fmt_sig(f0_sj)} (Fórmula explícita de Swamee-Jain, alta precisão)."
    )
    print(
        "   - Método: Secante / Newton. A secante é ideal por evitar a derivada"
        " complexa de Colebrook."
    )

    print("\nB.2 - COMPARAÇÃO ENTRE OS TRÊS MÉTODOS")
    r_biss, h_biss = bisseccao(f_cw, 0.015, 0.025, eps=1e-8)
    r_newt, h_newt = newton(f_cw, df_cw, x0=0.02, eps=1e-8)
    r_sec, h_sec = secante(f_cw, 0.015, 0.025, eps=1e-8)

    df_comp = pd.DataFrame([
        {"Método": "Bissecção", "f Encontrado": r_biss, "Iterações": len(h_biss)},
        {"Método": "Newton", "f Encontrado": r_newt, "Iterações": len(h_newt)},
        {"Método": "Secante", "f Encontrado": r_sec, "Iterações": len(h_sec)},
    ])
    print(df_comp.to_string(index=False))

    f_sol = r_sec
    print("\n4. RESULTADO COM UNIDADE FÍSICA E ALGARISMOS SIGNIFICATIVOS:")
    print(
        f"   Fator de atrito de Darcy: f = {fmt_sig(f_sol)} (adimensional,"
        " 4 algarismos significativos)"
    )

    verificar("Problema B", f_cw, f_sol, "", "", 1e-8)

    # B.3 - Swamee-Jain vs Arbitrária
    _, h_sj = secante(f_cw, f0_sj, f0_sj * 1.1, eps=1e-8)
    # Chutes arbitrários ajustados dentro da faixa plausível
    _, h_arb = secante(f_cw, 0.03, 0.04, eps=1e-8)

    print(
        f"\nB.3 - Swamee-Jain (f0 = {f0_sj:.6f}) usou {len(h_sj)} iterações."
    )
    print(f"Chute arbitrário (f0 = 0.03/0.04) usou {len(h_arb)} iterações.")
    print(f"Economia: {len(h_arb) - len(h_sj)} iterações.")

    # B.4 / B.5 - Perda de carga e sensibilidade
    hf = f_sol * (L / D) * (V_vel**2 / (2 * g))
    hf_aprox = 0.02 * (L / D) * (V_vel**2 / (2 * g))
    erro_pct = abs(hf_aprox - hf) / hf * 100

    print(
        "\nB.4 - Perda de carga exata (Darcy-Weisbach com f ="
        f" {f_sol:.6f}): h_f = {fmt_sig(hf)} m"
    )
    print(
        "B.5 - Perda de carga aprox. (f = 0.02): h_f ="
        f" {fmt_sig(hf_aprox)} m (Erro: {erro_pct:.2f}%)"
    )


# PROBLEMA C: EQUAÇÃO DE VAN DER WAALS
def problema_C():
    print("\n" + "=" * 70)
    print("PROBLEMA C: EQUAÇÃO DE VAN DER WAALS")
    print("=" * 70)

    R, a, b, T, P = 8.314, 0.3640, 4.267e-5, 300.0, 5.0e6
    v_ideal = R * T / P

    print("\n1. DEDUÇÃO DA FUNÇÃO f(x):")
    print("   Equação de estado de Van der Waals: (P + a/v^2) * (v - b) = R * T")
    print("   Rearranjando: f(v) = (P + a/v^2) * (v - b) - R * T = 0")

    def f_vdw(v):
        return (P + a / v**2) * (v - b) - R * T

    def df_vdw(v):
        return P - a / v**2 + 2 * a * b / v**3

    print("\nC.2 - TABELAMENTO LOGARÍTMICO EM FAIXA AMPLA (1e-5 a 1e-2 m³/mol)")
    grade_log = np.geomspace(1e-5, 1e-2, 10)
    fase_I("van der Waals Log", f_vdw, grade_log, "m³/mol", "")

    print("\n3. JUSTIFICATIVA DO MÉTODO E DO CHUTE INICIAL:")
    print(
        "   - Chute Inicial: v0 = v_ideal = R*T/P ="
        f" {v_ideal:.6e} m³/mol (excelente aproximação física)."
    )
    print(
        "   - Método: Newton-Raphson. A derivada f'(v) é simples e a"
        " convergência é quadrática."
    )

    v_vdw, hist_vdw = newton(f_vdw, df_vdw, x0=v_ideal, eps=1e-10)
    erro_pct = abs(v_ideal - v_vdw) / v_vdw * 100

    print("\n4. RESULTADO COM UNIDADE FÍSICA E ALGARISMOS SIGNIFICATIVOS:")
    print(f"   Volume Gás Ideal: {v_ideal:.6e} m³/mol")
    print(f"   Volume Van der Waals: {fmt_sig(v_vdw)} m³/mol")
    print(f"   Erro do modelo de Gás Ideal: {erro_pct:.2f}%")

    verificar("Problema C", f_vdw, v_vdw, "m³/mol", "", 1e-10)

    # C.4 - Isoterma P x v
    pressoes = np.linspace(1.0, 10.0, 19) * 1e6
    v_vdw_list, v_id_list = [], []

    for p_pa in pressoes:
        f_p = lambda v, p_pa=p_pa: (p_pa + a / v**2) * (v - b) - R * T
        df_p = lambda v, p_pa=p_pa: p_pa - a / v**2 + 2 * a * b / v**3

        v_i = R * T / p_pa
        v_id_list.append(v_i * 1e3)

        r, _ = newton(f_p, df_p, x0=v_i, eps=1e-10)
        v_vdw_list.append(r * 1e3)

    plt.figure(figsize=(7, 4))
    plt.plot(v_id_list, pressoes / 1e6, "--r", label="Gás Ideal")
    plt.plot(v_vdw_list, pressoes / 1e6, "-b", label="van der Waals")
    plt.ylim(0, 10)
    plt.title("Problema C: Isoterma P x v (T = 300 K)")
    plt.xlabel("Volume Molar (L/mol)")
    plt.ylabel("Pressão P (MPa)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


# PROBLEMA D: TAXA INTERNA DE RETORNO (TIR)
def problema_D():
    print("\n" + "=" * 70)
    print("PROBLEMA D: TAXA INTERNA DE RETORNO (TIR)")
    print("=" * 70)

    print("\n1. DEDUÇÃO DA FUNÇÃO f(x):")
    print("   Valor Presente Líquido: VPL(i) = sum_k [ CF_k / (1 + i)^k ]")
    print("   A TIR é a taxa i que anula o VPL: f(i) = VPL(i) = 0")

    fluxo1 = [-1000, 300, 350, 400, 450]

    def vpl1(i):
        return sum(cf / ((1 + i) ** k) for k, cf in enumerate(fluxo1))

    fase_I(
        "TIR do Projeto 1",
        vpl1,
        [0.0, 0.05, 0.10, 0.15, 0.20, 0.25],
        "taxa",
        "R$",
    )

    print("\n3. JUSTIFICATIVA DO MÉTODO E DO CHUTE INICIAL:")
    print(
        "   - Chute/Intervalo [0.15, 0.20]: Selecionado pela Fase I (VPL(0.15) >"
        " 0 e VPL(0.20) < 0)."
    )
    print(
        "   - Método: Bissecção. Garante convergência sem dependência de"
        " derivadas."
    )

    tir1, _ = bisseccao(vpl1, 0.15, 0.20, eps=1e-8)

    print("\n4. RESULTADO COM UNIDADE FÍSICA E ALGARISMOS SIGNIFICATIVOS:")
    print(f"   TIR do Projeto 1: i = {fmt_sig(tir1 * 100)} % a.p.")

    verificar("Projeto 1", vpl1, tir1, "taxa", "R$", 1e-3)

    # D.2 - Plot VPL1
    taxas = np.linspace(0.0, 0.5, 100)
    vpl1_vals = [vpl1(i) for i in taxas]

    plt.figure(figsize=(7, 4))
    plt.plot(taxas * 100, vpl1_vals, "-b", label="VPL Projeto 1")
    plt.axhline(0, color="red", linestyle="--")
    plt.axvline(
        tir1 * 100, color="green", linestyle=":", label=f"TIR ≈ {tir1*100:.2f}%"
    )
    plt.title("Problema D: VPL x Taxa (Projeto 1)")
    plt.xlabel("Taxa i (%)")
    plt.ylabel("VPL (R$)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # D.4 - Projeto 2
    print("\nD.4 - PROJETO 2 (MÚLTIPLAS TIRs)")
    fluxo2 = [-1000, 2500, -1540]

    def vpl2(i):
        return sum(cf / ((1 + i) ** k) for k, cf in enumerate(fluxo2))

    fase_I(
        "Projeto 2",
        vpl2,
        [0.0, 0.10, 0.20, 0.30, 0.40, 0.50],
        "taxa",
        "R$",
    )

    tir2_a, _ = bisseccao(vpl2, 0.0, 0.20, eps=1e-6)
    tir2_b, _ = bisseccao(vpl2, 0.30, 0.50, eps=1e-6)

    print(
        f"\n   Projeto 2 possui duas TIRs: i1 = {fmt_sig(tir2_a * 100)} % e i2 ="
        f" {fmt_sig(tir2_b * 100)} %"
    )

    vpl2_vals = [vpl2(i) for i in taxas]
    plt.figure(figsize=(7, 4))
    plt.plot(taxas * 100, vpl2_vals, "-g", label="VPL Projeto 2")
    plt.axhline(0, color="red", linestyle="--")
    plt.scatter(
        [tir2_a * 100, tir2_b * 100],
        [0, 0],
        color="black",
        zorder=5,
        label="TIRs",
    )
    plt.title("Problema D: VPL x Taxa (Projeto 2 - Múltiplas Raízes)")
    plt.xlabel("Taxa i (%)")
    plt.ylabel("VPL (R$)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nResposta Crítica para o Relatório:")
    print(
        "Sem a Fase I, Newton pode convergir para apenas uma das taxas"
        " dependendo do chute inicial,"
    )
    print(
        "mascarando a segunda taxa e levando a tomadas de decisão financeiras"
        " equivocadas."
    )


# PROBLEMA E: EQUAÇÃO DE KEPLER
def problema_E():
    print("\n" + "=" * 70)
    print("PROBLEMA E: EQUAÇÃO DE KEPLER")
    print("=" * 70)

    print("\n1. DEDUÇÃO DA FUNÇÃO f(x):")
    print("   Equação de Kepler orbital: M = E - e * sin(E)")
    print("   f(E) = E - e * sin(E) - M = 0")

    casos = [
        {"nome": "(i)", "e": 0.10, "M": 0.5},
        {"nome": "(ii)", "e": 0.90, "M": 0.1},
        {"nome": "(iii)", "e": 0.99, "M": 0.01},
    ]

    res_casos = []
    for c in casos:
        e_val, M_val = c["e"], c["M"]
        f_k = lambda E, e=e_val, M=M_val: E - e * math.sin(E) - M
        df_k = lambda E, e=e_val: 1.0 - e * math.cos(E)

        # Chute E0 = M
        r1, h1 = newton(f_k, df_k, x0=M_val, eps=1e-8)

        # Chute melhorado E0 = M + e*sin(M)
        x0_melhor = M_val + e_val * math.sin(M_val)
        r2, h2 = newton(f_k, df_k, x0=x0_melhor, eps=1e-8)

        # Bissecção [0, pi]
        r3, h3 = bisseccao(f_k, 0.0, math.pi, eps=1e-8)

        res_casos.append({
            "Caso": c["nome"],
            "e": e_val,
            "M": M_val,
            "Iter. Newton (E0=M)": len(h1),
            "Iter. Newton (E0=Melhor)": len(h2),
            "Iter. Bissecção [0, pi]": len(h3),
        })

    print("\n3. JUSTIFICATIVA DO MÉTODO E DO CHUTE INICIAL:")
    print(
        "   - Chute Inicial Melhorado: E0 = M + e*sin(M) (expansão de primeira"
        " ordem)."
    )
    print(
        "   - Método: Newton-Raphson. Apresenta excelente desempenho para alta"
        " excentricidade quando associado a um chute otimizado."
    )

    print("\n4. RESULTADO E COMPARAÇÕES:")
    print(pd.DataFrame(res_casos).to_string(index=False))

    verificar(
        "Kepler Caso (iii)",
        lambda E: E - 0.99 * math.sin(E) - 0.01,
        r2,
        "rad",
        "",
        1e-8,
    )

    print(
        "\nAnálise E.2: À medida que e -> 1, f'(E) = 1 - e*cos(E) aproxima-se de"
        " 0 para E perto de 0,"
    )
    print(
        "diminuindo o passo de Newton e aumentando consideravelmente o número"
        " de iterações."
    )


# PROBLEMA F (BÔNUS): DEFLEXÃO DE VIGA
def problema_F():
    print("\n" + "=" * 70)
    print("PROBLEMA F (BÔNUS): DEFLEXÃO DE VIGA")
    print("=" * 70)

    L, E_mod, I_mom, w0 = 600.0, 50000.0, 30000.0, 2.5
    C = w0 / (120.0 * L * E_mod * I_mom)

    print("\n1. DEDUÇÃO DA FUNÇÃO f(x):")
    print("   O ponto de deflexão máxima ocorre onde a derivada dy/dx é nula:")
    print("   f(x) = dy/dx = C * (-5*x^4 + 6*(L^2)*(x^2) - L^4) = 0")

    def dy_dx(x):
        return C * (-5 * x**4 + 6 * (L**2) * (x**2) - L**4)

    # F.3 - Armadilha em x=0 e x=L
    dy_0 = dy_dx(0)
    dy_L = dy_dx(L)
    print(f"\nF.3 - dy/dx(0) = {dy_0:.6e} | dy/dx(L) = {dy_L:.6e}")
    print("Observação: dy/dx tem o mesmo sinal negativo em x=0 e x=L.")

    try:
        bisseccao(dy_dx, 0.0, L, eps=1e-8)
    except ValueError as e:
        print(
            f"Resultado da Bissecção em [0, L]: Sucesso no tratamento de erro"
            f" -> {e}"
        )

    print("\n3. JUSTIFICATIVA DO MÉTODO E DO CHUTE INICIAL:")
    print(
        "   - Intervalo [0.4L, 0.5L]: Isolado na Fase I interna para evitar a"
        " armadilha de extremidades com mesmo sinal."
    )
    print(
        "   - Método: Bissecção. Permite contornar a armadilha do intervalo"
        " global [0, L]."
    )

    # Solução Correta
    x_max, _ = bisseccao(dy_dx, 0.4 * L, 0.5 * L, eps=1e-8)
    y_max = C * (-(x_max**5) + 2 * (L**2) * (x_max**3) - (L**4) * x_max)

    print("\n4. RESULTADO COM UNIDADE FÍSICA E ALGARISMOS SIGNIFICATIVOS:")
    print(
        f"   Ponto de deflexão máxima: x_max = {fmt_sig(x_max)} cm (Teórico:"
        f" {fmt_sig(L/math.sqrt(5))} cm)"
    )
    print(f"   Deflexão Máxima: y(x_max) = {fmt_sig(y_max)} cm")

    verificar("Problema F (dy/dx)", dy_dx, x_max, "cm", "rad", 1e-8)

    # F.4 - Derivada Numérica e Erro vs h
    
    def y_norm(u):
        return -u**5 + 2.0*(u**3) - u

    hs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    erros_h = []
    
    # Raiz da DERIVADA dy/du = 0 (ponto de máximo em u): sqrt(1/5)
    u_max = (1.0 / 5.0) ** 0.5  # approx 0.4472136

    for h in hs:
        # A derivada numérica no ponto de máximo teórico
        dy_num = (y_norm(u_max + h) - y_norm(u_max - h)) / (2.0 * h)
        # Em u_max a derivada exata é 0, então |dy_num - 0| é o próprio erro!
        # Adicione 1e-16 para evitar log(0) no gráfico
        erros_h.append(abs(dy_num) + 1e-16)

    plt.figure(figsize=(7, 4))
    plt.loglog(hs, erros_h, "o-r")
    plt.title("Problema F.4: Erro da Derivada Numérica vs h")
    plt.xlabel("Passo h (relativo)")
    plt.ylabel("Erro Absoluto |dy_num(u_max)|")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.show()

    print("\nResposta F.4 para o Relatório:")
    print(
        "O erro não diminui indefinidamente porque, para h muito pequeno, o"
        " erro de arredondamento"
    )
    print(
        "da ponto flutuante (cancelamento catastrófico em y(x+h) - y(x-h))"
        " domina sobre o erro de truncamento."
    )


if __name__ == "__main__":
    problema_A()
    problema_B()
    problema_C()
    problema_D()
    problema_E()
    problema_F()
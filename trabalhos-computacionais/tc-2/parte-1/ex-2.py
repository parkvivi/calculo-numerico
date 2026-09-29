# EXERCÍCIO 2 - VIVI

R1 = R2 = R3 = 100
R4 = R5 = 150
R6 = R7 = R8 = 200
V1 = 10
V2 = 12

def gauss_seidel(A, b, x0, eps=1e-4, max_iter=200):
    historico = []
    x = x0.copy()

    for k in range(max_iter):
        xant = x.copy()

        for i in range(4):
            s = 0

            for j in range(4):
                if j == i:
                    continue
                s = s + A[i][j] * x[j]

            x[i] = (b[i] - s) / A[i][i]

        historico.append({"iteração": k, "solução intermediária": x.copy()})

        max_diff = max(abs(current - previous) for current, previous in zip(x, xant))
        max_x = max(abs(current) for current in x)

        if max_x == 0: 
            d = 0
        else:
            d = max_diff / max_x

        if d < eps:
            return x, historico

def main():
    A = [[R1 + R2 + R4, -R2, 0, -R4], [-R2, R2 + R3 + R5, -R5, 0], [0, -R5, R5 + R7 + R8, -R7], [-R4, 0 , -R7, R4 + R6 + R7]]
    b = [-V1, V2, 0, 0]
    x = [0, 0, 0, 0]

    x, historico = gauss_seidel(A, b, x)
    print("===== HISTÓRICO DE ITERAÇÕES =====")
    for i in historico: print(i)
    print("===== SOLUÇÃO FINAL =====")
    print(x)

if __name__ == "__main__":
    main()
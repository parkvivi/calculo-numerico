# EXERCÍCIO 1 - MATHEUS

import numpy as np

def questao_1():
    
    # Linha 1 Ciências
    # Linha 2 Engenharia
    # Linha 3 Ciência da Computação

    A = np.array([
        [2800,  200,  200], 
        [ 400,  900,    0], 
        [ 600,  100, 1500]  
    ], dtype=float)

    b = np.array([16000000, 5000000, 8000000], dtype=float)

    x = np.linalg.solve(A, b)

    faculdades = ["Ciências", "Engenharia", "Ciência da Computação"]
    
    print("Resultados do Exercicio 1: ")
    for i in range(len(faculdades)):
        print(f"Custo anual por aluno em {faculdades[i]}: ${x[i]:,.2f}")

if __name__ == "__main__":
    questao_1()
"""
Gera os sinais sintéticos de base do projeto e salva em dados/.

Cada cenário gera:
    dados/<nome>.npz  -> t, limpo, ruidoso, ruido, fs, freqs_ruido, snr_db
    dados/<nome>_limpo.wav e dados/<nome>_ruidoso.wav (para ouvir)
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from src.sinais import gerar_sinal

FS = 8000
DURACAO = 1.0
SEED = 42

CENARIOS = {
    "tom1_snr20": ([60], 20),
    "tom1_snr10": ([60], 10),
    "tom1_snr0": ([60], 0),
    "tom3_snr10": ([60, 120, 180], 10),
    "tom3_snr0": ([60, 120, 180], 0),
    "tom3_snr_neg5": ([60, 120, 180], -5),
    "tom2_nao_inteiro": ([59.7, 143.3], 5),
}


def salvar_wav(caminho, x, fs):
    x = x / np.max(np.abs(x))
    wavfile.write(caminho, fs, (x * 32767 * 0.9).astype(np.int16))


def main():
    pasta = RAIZ / "dados"
    pasta.mkdir(exist_ok=True)
    for nome, (freqs, snr_db) in CENARIOS.items():
        t, limpo, ruidoso = gerar_sinal(FS, DURACAO, freqs, snr_db, seed=SEED)
        np.savez(
            pasta / f"{nome}.npz",
            t=t, limpo=limpo, ruidoso=ruidoso, ruido=ruidoso - limpo,
            fs=FS, freqs_ruido=np.array(freqs, dtype=float), snr_db=snr_db,
        )
        salvar_wav(pasta / f"{nome}_limpo.wav", limpo, FS)
        salvar_wav(pasta / f"{nome}_ruidoso.wav", ruidoso, FS)
        print(f"{nome:18s} ruído={freqs} Hz  SNR={snr_db} dB")


if __name__ == "__main__":
    main()
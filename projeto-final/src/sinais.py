"""
Geração de sinais sintéticos.
"""

import numpy as np


def gerar_sinal(fs, duracao, freqs_ruido, snr_db, seed=0, freqs_sinal=(220.0, 440.0, 660.0, 880.0)):
    """
    Gera um sinal limpo, adiciona ruído tonal e devolve ambos.

    Parâmetros
    ----------
    fs : float
        Frequência de amostragem do sinal (Hz).
    duracao : float
        Duração do sinal (s).
    freqs_ruido : sequência de float
        Frequências (Hz) dos tons de ruído.
    snr_db : float
        SNR desejada do sinal ruidoso (dB): SNR = 10*log10(potência_limpo / potência_ruído).
    seed : int
        Semente aleatória (mesma semente gera o mesmo sinal!!!).
    freqs_sinal : sequência de float
        Frequências (Hz) do sinal limpo (diferentes das de freqs_ruido!!!).

    Retorna
    -------
    t : ndarray, tempo em segundos
    limpo : ndarray, sinal original (referência para SNR e erro)
    ruidoso : ndarray, limpo + ruído (float64)
    """
    freqs_ruido = np.atleast_1d(np.asarray(freqs_ruido, dtype=float))
    freqs_sinal = np.atleast_1d(np.asarray(freqs_sinal, dtype=float))

    if np.any(freqs_ruido <= 0) or np.any(freqs_ruido >= fs / 2):
        raise ValueError("freqs_ruido deve estar em (0, fs/2) (Nyquist).")
    if np.any(freqs_sinal >= fs / 2):
        raise ValueError("freqs_sinal deve ser menor que fs/2 (Nyquist).")

    rng = np.random.default_rng(seed)
    n = int(round(fs * duracao))
    t = np.arange(n) / fs

    limpo = np.zeros(n)
    for k, f in enumerate(freqs_sinal):
        fase = rng.uniform(0, 2 * np.pi)
        limpo += (1.0 / (k + 1)) * np.sin(2 * np.pi * f * t + fase)
    envelope = 0.7 + 0.3 * np.sin(2 * np.pi * 3.0 * t)
    limpo *= envelope
    limpo /= np.max(np.abs(limpo))

    ruido = np.zeros(n)
    for f in freqs_ruido:
        amp = rng.uniform(0.5, 1.5)
        fase = rng.uniform(0, 2 * np.pi)
        ruido += amp * np.sin(2 * np.pi * f * t + fase)

    pot_limpo = np.mean(limpo ** 2)
    pot_ruido = np.mean(ruido ** 2)
    ruido *= np.sqrt(pot_limpo / (pot_ruido * 10 ** (snr_db / 10)))

    ruidoso = limpo + ruido
    return t, limpo, ruidoso


if __name__ == "__main__":
    # teste
    t, limpo, ruidoso = gerar_sinal(8000, 1.0, [60, 120, 180], snr_db=0, seed=42)
    ruido = ruidoso - limpo
    snr = 10 * np.log10(np.mean(limpo ** 2) / np.mean(ruido ** 2))
    print(f"amostras: {len(t)}, SNR medida: {snr:.2f} dB")
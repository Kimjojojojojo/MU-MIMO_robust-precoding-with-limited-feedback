import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import gamma  # Importing the gamma function that was missing

import limited_feedback as lf
import functions as f

# Parameters
K = 4
N = 2
M = K * N
B = 10
sigma_n = 1

# T = N ** 2 * (K - 1)
T = N*(M-N)
C = (1 / math.factorial(T)) * np.prod([math.factorial(M - i) / math.factorial(N - i) for i in range(N)])
xi = gamma(1 / T) / T * C ** (-1 / T) * 2 ** (-B / T)
gamma_val = xi / N
codebook = lf.generate_semi_unitary_codebook(M, N, B)
# Simulating for rho values from 0 dB to 20 dB
rho_values_db = np.arange(-10, 31, 5)

samples = 10
iter_list = [100]

rate_samples_RMMSE = np.zeros(samples)
rate_samples_RWMMSE = np.zeros(samples)
rate_samples_MMSE = np.zeros(samples)
rate_samples_WMMSE = np.zeros(samples)

rate_avg_RMMSE = np.zeros_like(rho_values_db, dtype=float)
rate_avg_RWMMSE = np.zeros_like(rho_values_db, dtype=float)
rate_avg_MMSE = np.zeros_like(rho_values_db, dtype=float)
rate_avg_WMMSE = np.zeros_like(rho_values_db, dtype=float)
# Average over 100 samples for each rho
for iter_num in iter_list:
    print(f'Iteration {iter_num}')
    for i, rho_db in enumerate(rho_values_db):
        print(f"{rho_db}dB")

        # Generate complex Gaussian channels
        for s in range(samples):
            H = (np.random.normal(loc=0.0, scale=1, size=(K, M, N)) + 1j * np.random.normal(loc=0.0, scale=1, size=(K, M, N))) / np.sqrt(2)
            H_hat = lf.codebook_selection(codebook, H)
            #### RMMSE P_idx = 1 ####
            #### RWMMSE P_idx = 2 ####
            #### MMSE P_idx = 3 ####
            #### WMMSE P_idx = 4 ####
            rate_samples_RMMSE[s] = f.calculate_rate(H, H_hat, gamma_val, rho_db, codebook, K, M, N, B, sigma_n, iter_num, s,1)
            rate_samples_RWMMSE[s] = f.calculate_rate(H, H_hat, gamma_val, rho_db, codebook, K, M, N, B, sigma_n, iter_num,s,2)
            rate_samples_MMSE[s] = f.calculate_rate(H, H_hat, gamma_val, rho_db, codebook, K, M, N, B, sigma_n, iter_num, s, 3)
            rate_samples_WMMSE[s] = f.calculate_rate(H, H_hat, gamma_val, rho_db, codebook, K, M, N, B, sigma_n, iter_num, s, 4)

        rate_avg_RMMSE[i] = np.mean(rate_samples_RMMSE)
        rate_avg_RWMMSE[i] = np.mean(rate_samples_RWMMSE)
        rate_avg_MMSE[i] = np.mean(rate_samples_MMSE)
        rate_avg_WMMSE[i] = np.mean(rate_samples_WMMSE)




    # Plotting the result
    plt.plot(rho_values_db, rate_avg_RMMSE, label= 'RMMSE', marker='o')
    plt.plot(rho_values_db, rate_avg_RWMMSE, label= 'RWMMSE',marker='v')
    plt.plot(rho_values_db, rate_avg_MMSE, label= 'MMSE', marker='s')
    plt.plot(rho_values_db, rate_avg_WMMSE, label= 'WMMSE', marker='p')
    plt.legend()
    plt.title(f'Sum Rate vs SNR ,iter_num={iter_num}, sample_num={samples}')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Sum Rate (bps/Hz)')
    plt.grid(True)
    plt.show()

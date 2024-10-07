import numpy as np
import matplotlib.pyplot as plt

import limited_feedback as lf
import functions as f

# Parameters
K = 4
N = 2
M = K * N
B = 10
sigma_n = 1

# Simulating for rho values from 0 dB to 20 dB
rho_values_db = np.arange(-10, 31, 1)
rate_avg_RMMSE = np.zeros_like(rho_values_db, dtype=float)
rate_avg_RWMMSE = np.zeros_like(rho_values_db, dtype=float)

codebook = lf.generate_semi_unitary_codebook(M, N, B)

# Average over 100 samples for each rho
samples = 1
for i, rho_db in enumerate(rho_values_db):
    print(f"{rho_db}dB")
    #rate_samples_RMMSE = np.array([f.calculate_rate(rho_db, codebook, K, M, N, B, sigma_n, 1) for _ in range(samples)])
    #rate_avg_RMMSE[i] = np.mean(rate_samples_RMMSE)

    rate_samples_RWMMSE = np.array([f.calculate_rate(rho_db, codebook, K, M, N, B, sigma_n, 2) for _ in range(samples)])
    rate_avg_RWMMSE[i] = np.mean(rate_samples_RWMMSE)

# Plotting the result
plt.plot(rho_values_db, rate_avg_RMMSE, marker='o')
plt.plot(rho_values_db, rate_avg_RWMMSE, marker='v')
plt.title('Sum Rate vs Rho (-10 to 30 dB)')
plt.xlabel('Rho (dB)')
plt.ylabel('Sum Rate (bps/Hz)')
plt.grid(True)
plt.show()

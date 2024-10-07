import numpy as np
import math
from scipy.special import gamma  # Importing the gamma function that was missing

import limited_feedback as lf
import precoders
# Function to calculate Rate for each rho value
def calculate_rate(rho_db, codebook, K, M, N, B, sigma_n, P_idx):
    rho = 10 ** (rho_db / 10)  # Convert dB to linear scale

    # Calculate xi and gamma
    T = N ** 2 * (K - 1)
    C = (1 / math.factorial(T)) * np.prod([math.factorial(M - i) / math.factorial(N - i) for i in range(N)])
    xi = gamma(1 / T) / T * C ** (-1 / T) * 2 ** (-B / T)
    gamma_val = xi / N
    sigma = M * gamma_val / (M - N)  # sigma^2

    # Generate complex Gaussian channels
    H = (np.random.normal(loc=0.0, scale=1, size=(K, M, N)) +
         1j * np.random.normal(loc=0.0, scale=1, size=(K, M, N))) / np.sqrt(2)
    H_hat = lf.codebook_selection(codebook, H)

    #print(H-H_hat)
    # Hermitian transpose of H and H_hat
    H_H = np.transpose(H, (0, 2, 1)).conj()
    H_hat_H = np.transpose(H_hat, (0, 2, 1)).conj()

    P = np.zeros((M, K*N), dtype=complex)
    P_H = np.zeros((K*N, M), dtype=complex)

    if P_idx == 1: # RMMSE
        P, P_H = precoders.RMMSE(H_hat, gamma_val, rho, sigma_n)
    if P_idx == 2: # RWMMSE
        P, P_H = precoders.RWMMSE(H_hat, gamma_val, rho, sigma_n)


    # Calculate SINR and Rate
    SINR = np.zeros((K, N, N), dtype=complex)
    Rate = np.zeros(K)
    for k in range(K):
        tmp1 = H_H[k] @ P[k] @ P_H[k] @ H[k]
        tmp2 = np.sum([H_H[k] @ P[j] @ P_H[j] @ H[k] for j in range(K)], axis=0) - tmp1
        SINR[k] = tmp1 @ np.linalg.inv(tmp2 + sigma_n * np.eye(N))
        Rate[k] = np.real(np.log2(np.linalg.det(np.eye(N) + SINR[k])))


    return np.sum(Rate)

def diag_inv(X):
    M = np.size(X, 0)
    diag_inverse = np.zeros((M, 1),dtype=complex)
    for m in range(M):
        diag_inverse[m] = X[m][m]

    return diag_inverse

def diag(x):
    M = np.size(x, 0)
    diag = np.zeros((M, M),dtype=complex)
    for m in range(M):
        diag[m][m] = x[m]

    return diag

def blk_diag(X):
    K = np.size(X, 0)
    M = np.size(X, 1)

    blk_diag = np.zeros((M*K, M*K),dtype=complex)
    for k in range(K):
        blk_diag[k * M: (k + 1) * M:, k * M: (k + 1) * M] = X[k]
    blk_diag_H = np.transpose(blk_diag).conj()

    return blk_diag, blk_diag_H
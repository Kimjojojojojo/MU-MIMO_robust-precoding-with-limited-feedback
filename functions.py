import numpy as np

import matplotlib.pyplot as plt


import limited_feedback as lf
import precoders
# Function to calculate Rate for each rho value
def calculate_rate(H, H_hat, gamma_val, rho_db, codebook, K, M, N, B, sigma_n, iter_num,sample,P_idx):
    rho = 10 ** (rho_db / 10)  # Convert dB to linear scale

    # Calculate xi and gamma


    ########### perfect CSI setting ##########
    # H_hat = (1/ np.sqrt(M))*H
    # gamma_val = 0
    ##########################################

    #print(H-H_hat)
    # Hermitian transpose of H and H_hat
    H_H = np.transpose(H, (0, 2, 1)).conj()
    H_hat_H = np.transpose(H_hat, (0, 2, 1)).conj()

    P = np.zeros((K, M, N), dtype=complex)
    P_H = np.zeros((K, N, M), dtype=complex)


    error_list = np.zeros(iter_num)
    if P_idx == 1: # RMMSE
        P, P_H = precoders.RMMSE(H_hat, gamma_val, rho, sigma_n)
    if P_idx == 2: # RWMMSE
        P, P_H, error_list = precoders.RWMMSE(H_hat, gamma_val, rho, sigma_n, iter_num)
    if P_idx == 3: # MMSE
        gamma_val = 0
        P, P_H = precoders.RMMSE(H_hat, gamma_val, rho, sigma_n)
    if P_idx == 4: # WMMSE
        gamma_val = 0
        P, P_H, error_list = precoders.RWMMSE(H_hat, gamma_val, rho, sigma_n, iter_num)


    error_range = np.arange(0,iter_num,1)
    if rho_db % 10 == 0 and P_idx == 2 and sample == 0:
        #print(f'---------{rho_db}--------------')
        plt.plot(error_range, error_list, marker='v')
        plt.title(f'error_converge[{rho_db}dB]')
        plt.xlabel('iterations')
        plt.ylabel('MSE')
        plt.grid(True)
        plt.show()


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
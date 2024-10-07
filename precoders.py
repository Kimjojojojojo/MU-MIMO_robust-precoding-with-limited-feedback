import numpy as np
import functions as f

def RMMSE(H_hat, gamma_val, rho, sigma_n):
    H_hat_H = np.transpose(H_hat, (0, 2, 1)).conj()
    K = np.size(H_hat, 0)
    M = np.size(H_hat, 1)
    N = np.size(H_hat, 2)
# Covariance matrix R_o
    R_o = np.zeros((K, M, M))
    for k in range(K):
        R_o[k] = np.real(M * (1 - M * gamma_val / (M - N)) * H_hat[k] @ H_hat_H[k] + M * N * gamma_val / (M - N) * np.eye(M))

    R_o_sum = np.sum(R_o, axis=0)

    # Concatenating H_hat
    H_hat_concat = np.zeros((M, K * N), dtype=complex)
    for k in range(K):
        H_hat_concat[:, k * N: (k + 1) * N] = H_hat[k]

    # MMSE precoding
    delta = np.sqrt(M - M**2 * gamma_val / (M - N))
    P_RMMSE_block_bar = delta * np.linalg.inv(R_o_sum + M * sigma_n / rho * np.eye(M)) @ H_hat_concat
    beta = np.sqrt(rho/np.trace((P_RMMSE_block_bar @ np.transpose(P_RMMSE_block_bar).conj())))
    P_RMMSE_block = beta * P_RMMSE_block_bar
    #print(np.trace(P_RMMSE_block @ np.transpose(P_RMMSE_block).conj()), rho)
    P_RMMSE = np.zeros((K, M, N), dtype=complex)
    P_RMMSE_H = np.zeros((K, N, M), dtype=complex)
    for k in range(K):
        P_RMMSE[k] = P_RMMSE_block[:, k * N: (k + 1) * N]
        P_RMMSE_H[k] = np.transpose(P_RMMSE[k]).conj()

    return P_RMMSE, P_RMMSE_H

def RWMMSE(H_hat_k, gamma_val, rho, sigma_n):
    K = np.size(H_hat_k, 0)
    M = np.size(H_hat_k, 1)
    N = np.size(H_hat_k, 2)

    H_hat_k_H = np.transpose(H_hat_k, (0, 2, 1)).conj()
    H_hat = np.zeros((M, N*K),dtype=complex)
    for k in range(K):
        H_hat[:, k * N: (k + 1) * N] = H_hat_k[k]
    H_hat_H = np.transpose(H_hat).conj()

    P_k = np.zeros((K, M, N), dtype=complex)
    P_k_H = np.zeros((K, N, M), dtype=complex)
    P_init = np.zeros((M, K*N), dtype=complex)
    for k in range(K):
        P_init[:, k * N: (k + 1) * N] = H_hat_k[k]
        P_k[k] = H_hat_k[k]
        P_k_H[k] = H_hat_k_H[k]
    P_init_H = np.transpose(P_init).conj()
    P = P_init
    P_H = P_init_H

    delta = np.sqrt(M - M**2 * gamma_val / (M - N))
    eta = np.sqrt(M**2 * gamma_val / (M - N))

    iter_num = 1000
    epsilon = 10^(-3)
    P_tmp = []
    for ite in range(iter_num):
        F_k = np.zeros((K,N,N), dtype=complex)
        D_k = np.zeros((K,N,N), dtype=complex)
        Phi_k = np.zeros((K,N,N), dtype=complex)
        M_k_bar = np.zeros((K,N,N), dtype=complex)
        M_k_bar_inv = np.zeros((K,N,N), dtype=complex)

        Theta_k = (1/M)*np.ones((N,M))
        Theta = (1/M)*np.ones((M, N*K))

        for k in range(K):
            Phi_k[k] = f.diag(Theta_k @ f.diag_inv(P @ P_H))
            F_k[k] = delta ** 2 * H_hat_k_H[k] @ P @ P_H @ H_hat_k[k] + eta ** 2 * Phi_k[k] + sigma_n * np.eye(N)
            D_k[k] = delta * P_k_H[k] @ H_hat_k[k] @ np.linalg.inv(F_k[k])
            M_k_bar[k] = np.eye(N) - delta**2 * P_k_H[k] @ H_hat_k[k] @ np.linalg.inv(F_k[k]) @ H_hat_k_H[k] @ P_k[k]
            M_k_bar_inv[k] = np.linalg.inv(M_k_bar[k])
        D, D_H = f.blk_diag(D_k)
        W_bar,_ = f.blk_diag(M_k_bar_inv)
        Phi = f.diag(Theta @ f.diag_inv(D_H @ W_bar @ D))
        T = delta**2 * H_hat @ D_H @ W_bar @ D @ H_hat_H + eta**2 * Phi + sigma_n/rho * np.trace(W_bar@D@D_H) * np.eye(M)

        P_bar = np.linalg.inv(T) @ H_hat @ D_H @ W_bar
        P = np.sqrt(rho/np.trace(P_bar @ np.transpose(P_bar).conj())) * P_bar
        P_tmp.append(P)

        error = np.linalg.norm(P_tmp[ite] - P_tmp[ite - 1], 'fro')
        print(error)
        if error <= epsilon:
            break

    for k in range(K):
        P_k[k] = P[:, k * N: (k + 1) * N]
        P_k_H[k] = np.transpose(P_k[k]).conj()

    return P_k, P_k_H

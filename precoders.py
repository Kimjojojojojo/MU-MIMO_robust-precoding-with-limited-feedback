import numpy as np
import functions as f
import matplotlib.pyplot as plt

def RMMSE(H_hat, gamma_val, rho, sigma_n):
    H_hat_H = np.transpose(H_hat, (0, 2, 1)).conj()
    K = np.size(H_hat, 0)
    M = np.size(H_hat, 1)
    N = np.size(H_hat, 2)

    # Robust indicator , if 1 : robust, M : non-robust
    alpha = 0
    if gamma_val == 0:
        alpha = np.sqrt(M)
    if gamma_val > 0:
        alpha = 1

    H_hat = alpha * H_hat
# Covariance matrix R_o
    R_o = np.zeros((K, M, M), dtype=complex)
    for k in range(K):
        R_o[k] = M * (1 - M * gamma_val / (M - N)) * H_hat[k] @ H_hat_H[k] + M * N * gamma_val / (M - N) * np.eye(M)

    R_o_sum = np.sum(R_o, axis=0)

    # Concatenating H_hat to form H_hat_concat
    H_hat_concat = np.zeros((M, K * N), dtype=complex)
    for k in range(K):
        H_hat_concat[:, k * N: (k + 1) * N] = H_hat[k]  # Here M is multiplied

    # Calculate R
    R = H_hat_concat @ np.transpose(H_hat_concat).conj()


    # MMSE precoding
    P_bar = np.linalg.inv(R_o_sum + M * sigma_n / rho * np.eye(M)) @ H_hat_concat
    beta = np.sqrt(rho/np.trace((P_bar @ np.transpose(P_bar).conj())))
    P = beta * P_bar
    #print(np.trace(P_RMMSE_block @ np.transpose(P_RMMSE_block).conj()), rho)
    P_k = np.zeros((K, M, N), dtype=complex)
    P_k_H = np.zeros((K, N, M), dtype=complex)
    for k in range(K):
        P_k[k] = P[:, k * N: (k + 1) * N]
        P_k_H[k] = np.transpose(P_k[k]).conj()

    return P_k, P_k_H

def RWMMSE(H_hat_k, gamma_val, rho, sigma_n, iter_num):
    K = np.size(H_hat_k, 0)
    M = np.size(H_hat_k, 1)
    N = np.size(H_hat_k, 2)

    alpha = 0
    if gamma_val == 0:
        alpha = np.sqrt(M)
    if gamma_val > 0:
        alpha = 1

    H_hat_k = alpha * H_hat_k

    H_hat_k_H = np.transpose(H_hat_k, (0, 2, 1)).conj()
    H_hat = np.zeros((M, N*K),dtype=complex)
    for k in range(K):
        H_hat[:, k * N: (k + 1) * N] = H_hat_k[k]
    H_hat_H = np.transpose(H_hat).conj()

    P_init = np.zeros((M, K*N), dtype=complex)
    for k in range(K):
        P_init[:, k * N: (k + 1) * N] = H_hat_k[k]
    P_init_H = np.transpose(P_init).conj()

    P = np.sqrt(rho/np.trace(P_init@P_init_H)) * P_init
    P_H = np.transpose(P).conj()

    P_k = np.zeros((K, M, N), dtype=complex)
    P_k_H = np.zeros((K, N, M), dtype=complex)
    for k in range(K):
        P_k[k] = P[:, k * N: (k + 1) * N]
        P_k_H[k] = np.transpose(P_k[k]).conj()

    #print(f'total power :  {np.real(np.trace(P@P_H))},{rho}')

    delta = np.sqrt(M - M**2 * gamma_val / (M - N))
    eta = np.sqrt(M**2 * gamma_val / (M - N))

    epsilon = 10**(-1)
    P_list = np.zeros((iter_num, M, K*N), dtype=complex)
    error_list = np.zeros(iter_num)
    iter_range = np.arange(0, iter_num, 1)
    Rate_ite = np.zeros(iter_num)

    Theta_k = (1 / M) * np.ones((N, M))
    Theta = (1 / M) * np.ones((M, N * K))

    F_k = np.zeros((K, N, N), dtype=complex)
    D_k = np.zeros((K, N, N), dtype=complex)
    Phi_k = np.zeros((K, N, N), dtype=complex)
    M_k_bar = np.zeros((K, N, N), dtype=complex)
    M_k_bar_inv = np.zeros((K, N, N), dtype=complex)
    for ite in range(iter_num):
        for k in range(K):
            Phi_k[k] = f.diag(Theta_k @ f.diag_inv(P @ P_H))
            F_k[k] = (delta**2) * H_hat_k_H[k] @ P @ P_H @ H_hat_k[k] + eta ** 2 * Phi_k[k] + sigma_n * np.eye(N)
            D_k[k] = delta * P_k_H[k] @ H_hat_k[k] @ np.linalg.inv(F_k[k])
            M_k_bar[k] = np.eye(N) - delta**2 * P_k_H[k] @ H_hat_k[k] @ np.linalg.inv(F_k[k]) @ H_hat_k_H[k] @ P_k[k]
            M_k_bar_inv[k] = np.linalg.inv(M_k_bar[k])

        D, D_H = f.blk_diag(D_k)
        W_bar, _ = f.blk_diag(M_k_bar_inv)
        Phi = f.diag(Theta @ f.diag_inv(D_H @ W_bar @ D))
        T = delta**2 * H_hat @ D_H @ W_bar @ D @ H_hat_H + eta**2 * Phi + (sigma_n/rho) * np.trace(W_bar@D@D_H) * np.eye(M)

        P_bar = np.linalg.inv(T) @ H_hat @ D_H @ W_bar
        P = np.sqrt(rho/np.trace(np.transpose(P_bar).conj() @ P_bar)) * P_bar
        P_list[ite] = P

        for k in range(K):
            P_k[k] = P[:, k * N: (k + 1) * N]
            P_k_H[k] = np.transpose(P_k[k]).conj()

        # if rho == 1000:
        #     print('-----------------------------------')
        #     print(P)


        error = np.linalg.norm(P_list[ite] - P_list[ite - 1], 'fro')
        error_list[ite] = error
        # if error <= epsilon:
        #     print(f"Converged after {ite} iterations with error: {error}")
        #     break

    #     SINR = np.zeros((K, N, N), dtype=complex)
    #     Rate = np.zeros(K)
    #     for k in range(K):
    #         tmp1 = H_H[k] @ P_k[k] @ P_k_H[k] @ H[k]
    #         tmp2 = np.sum([H_H[k] @ P_k[j] @ P_k_H[j] @ H[k] for j in range(K)], axis=0) - tmp1
    #         SINR[k] = tmp1 @ np.linalg.inv(tmp2 + sigma_n * np.eye(N))
    #         Rate[k] = np.real(np.log2(np.linalg.det(np.eye(N) + SINR[k])))
    #     Rate_ite[ite] = sum(Rate)
    #
    # if rho == 1000:
    #     print(f'---------{rho}--------------')
    #     plt.plot(iter_range, Rate_ite, marker='v')
    #     plt.title(f'Sumrate[{rho}]')
    #     plt.xlabel('iterations')
    #     plt.ylabel('sumRate')
    #     plt.grid(True)
    #     plt.show()

    return P_k, P_k_H, error_list

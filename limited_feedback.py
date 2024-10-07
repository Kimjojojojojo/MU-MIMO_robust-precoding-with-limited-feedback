import numpy as np

def generate_semi_unitary_codebook(M, N, B):
    codebook_size = 2**B
    # 정규 분포에서 KxN 크기의 무작위 행렬 생성
    random_matrix = (np.random.normal(loc=0.0, scale=1, size=(codebook_size, M, N)) +
         1j * np.random.normal(loc=0.0, scale=1, size=(codebook_size, M, N))) / np.sqrt(2)

    # QR 분해를 통해 직교 행렬을 얻음
    codebooks = np.zeros((codebook_size, M, N),dtype=np.complex128)
    for b in range(codebook_size):
        Q, R = np.linalg.qr(random_matrix[b])
        # Semi-unitary codebook: KxN 행렬에서 첫 N개의 열 벡터가 직교하고, 길이가 1인 벡터들로 구성
        codebooks[b] = Q

    return codebooks

def codebook_selection(codebook, H):
    K = np.size(H, 0)
    M = np.size(H, 1)
    N = np.size(H, 2)
    H_tilde = np.zeros((K, M, M),dtype=complex)
    H_tilde_H = np.zeros((K, M, M),dtype=complex)
    for k in range(K):
        _, H_tilde[k] = np.linalg.eig(H[k] @ np.transpose(H[k]).conj())
        H_tilde_H[k] = np.transpose(H_tilde[k]).conj()

    codebook_size = len(codebook)
    #print(H_tilde)
    #print(codebook[0].shape, np.transpose(codebook[0]).conj().shape)
    distances = np.zeros((K, codebook_size))
    for k in range(K):
        for b in range(codebook_size):
            # print(np.trace(H_tilde_H[k] @ codebook[b] @
            #                                np.transpose(codebook[b]).conj() @ H_tilde[k]))
            distances[k][b] = N - np.real(np.trace(H_tilde_H[k] @ codebook[b] @
                                           np.transpose(codebook[b]).conj() @ H_tilde[k]))

    H_hat = np.zeros((K, M, N),dtype=complex)
    for k in range(K):
        i_min = np.argmin(distances[k])
        #print(distances[k][i_min])
        H_hat[k] = codebook[i_min]

    return H_hat

import numpy as np

def generate_semi_unitary_codebook(M, N, B):
    codebook_size = 2**B
    # 정규 분포에서 KxN 크기의 무작위 행렬 생성
    random_matrix = (np.random.normal(loc=0.0, scale=1, size=(codebook_size, M, N)) +
         1j * np.random.normal(loc=0.0, scale=1, size=(codebook_size, M, N))) / np.sqrt(2)
    print(random_matrix.shape)
    # QR 분해를 통해 직교 행렬을 얻음
    codebooks = np.zeros((codebook_size, M, N), dtype=np.complex128)
    for b in range(codebook_size):
        Q, R = np.linalg.qr(random_matrix[b])
        # Semi-unitary codebook: KxN 행렬에서 첫 N개의 열 벡터가 직교하고, 길이가 1인 벡터들로 구성
        codebooks[b] = Q[:,0:N]
    return codebooks

def codebook_selection(codebook, H):
    K = np.size(H, 0)
    M = np.size(H, 1)
    N = np.size(H, 2)
    H_tilde = np.zeros((K, M, N),dtype=complex)
    H_tilde_H = np.zeros((K, N, M),dtype=complex)
    for k in range(K):
        eigenvalues, eigenvectors = np.linalg.eig(H[k] @ np.transpose(H[k]).conj())
        sorted_indices = np.argsort(eigenvalues)[::-1]  # 내림차순 정렬을 위해 역순 인덱스
        sorted_eigenvalues = eigenvalues[sorted_indices]
        # 고유벡터도 같은 순서로 정렬
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        H_tilde[k] = sorted_eigenvectors[:, 0:N]
        print(H_tilde[k].shape)
        H_tilde_H[k] = np.transpose(H_tilde[k]).conj()

    codebook_size = len(codebook)
    #print(H_tilde)
    #print(codebook[0].shape, np.transpose(codebook[0]).conj().shape)
    distances = np.zeros((K, codebook_size))
    for k in range(K):
        for b in range(codebook_size):
            # print(np.trace(H_tilde_H[k] @ codebook[b] @
            #                                np.transpose(codebook[b]).conj() @ H_tilde[k]))
            distances[k][b] = N - np.real(np.trace(H_tilde_H[k] @ codebook[b] @ np.transpose(codebook[b]).conj() @ H_tilde[k]))
            if distances[k][b] < 0:
                print(distances[k][b], b)
            #print(distances[k][b])

    H_hat = np.zeros((K, M, N),dtype=complex)
    #print(distances[0])
    for k in range(K):
        i_min = np.argmin(distances[k])
        #print(distances[k][i_min])
        H_hat[k] = codebook[i_min]

        #print(distances[k][i_min])
        #print(np.linalg.norm(H_hat[k]-H[k],'fro'))

    return H_hat

import sklearn.metrics
import sklearn.neighbors
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial.distance
import numpy as np


def grid(m, dtype=np.float32):
  
    M = m**2
    x = np.linspace(0, 1, m, dtype=dtype)
    y = np.linspace(0, 1, m, dtype=dtype)
    xx, yy = np.meshgrid(x, y)
    z = np.empty((M, 2), dtype)
    z[:, 0] = xx.reshape(M)
    z[:, 1] = yy.reshape(M)
    return z


def distance_scipy_spatial(z, k=4, metric='euclidean'):
  
    d = scipy.spatial.distance.pdist(z, metric)
    d = scipy.spatial.distance.squareform(d)
   
    idx = np.argsort(d)[:, 1:k+1]
    d.sort()
    d = d[:, 1:k+1]
    return d, idx


def distance_sklearn_metrics(z, k=4, metric='euclidean'):
  
    d = sklearn.metrics.pairwise.pairwise_distances(
            z, metric=metric, n_jobs=-2)
   
    idx = np.argsort(d)[:, 1:k+1]
    d.sort()
    d = d[:, 1:k+1]
    return d, idx


def distance_lshforest(z, k=4, metric='cosine'):
  
    assert metric is 'cosine'
    lshf = sklearn.neighbors.LSHForest()
    lshf.fit(z)
    dist, idx = lshf.kneighbors(z, n_neighbors=k+1)
    assert dist.min() < 1e-10
    dist[dist < 0] = 0
    return dist, idx

# TODO: other ANNs s.a. NMSLIB, EFANNA, FLANN, Annoy, sklearn neighbors, PANN


def adjacency(dist, idx):
  
    M, k = dist.shape
    assert M, k == idx.shape
    assert dist.min() >= 0

   
    sigma2 = np.mean(dist[:, -1])**2
    dist = np.exp(- dist**2 / sigma2)

   
    I = np.arange(0, M).repeat(k)
    J = idx.reshape(M*k)
    V = dist.reshape(M*k)
    W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))
   
   
    W.setdiag(0)

   
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)

    assert W.nnz % 2 == 0
    assert np.abs(W - W.T).mean() < 1e-10
    assert type(W) is scipy.sparse.csr.csr_matrix
    return W


def replace_random_edges(A, noise_level):
  
    M, M = A.shape
    n = int(noise_level * A.nnz // 2)

    indices = np.random.permutation(A.nnz//2)[:n]
    rows = np.random.randint(0, M, n)
    cols = np.random.randint(0, M, n)
    vals = np.random.uniform(0, 1, n)
    assert len(indices) == len(rows) == len(cols) == len(vals)

    A_coo = scipy.sparse.triu(A, format='coo')
    assert A_coo.nnz == A.nnz // 2
    assert A_coo.nnz >= n
    A = A.tolil()

    for idx, row, col, val in zip(indices, rows, cols, vals):
        old_row = A_coo.row[idx]
        old_col = A_coo.col[idx]

        A[old_row, old_col] = 0
        A[old_col, old_row] = 0
        A[row, col] = 1
        A[col, row] = 1

    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


def laplacian(W, normalized=True):
  
  

   
    d = W.sum(axis=0)
    
   
   
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        d += np.spacing(np.array(0, W.dtype))
       
        d = 1 / np.sqrt(d)  
       
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D
   
    assert type(L) is scipy.sparse.csr.csr_matrix
    return L


def lmax(L, normalized=True):
  
    if normalized:
        return 2
    else:
        return scipy.sparse.linalg.eigsh(
                L, k=1, which='LM', return_eigenvectors=False)[0]


def fourier(L, algo='eigh', k=1):
  

    def sort(lamb, U):
        idx = lamb.argsort()
        return lamb[idx], U[:, idx]

    if algo is 'eig':
        lamb, U = np.linalg.eig(L.toarray())
        lamb, U = sort(lamb, U)
    elif algo is 'eigh':
        lamb, U = np.linalg.eigh(L.toarray())
    elif algo is 'eigs':
        lamb, U = scipy.sparse.linalg.eigs(L, k=k, which='SM')
        lamb, U = sort(lamb, U)
    elif algo is 'eigsh':
        lamb, U = scipy.sparse.linalg.eigsh(L, k=k, which='SM')

    return lamb, U


def plot_spectrum(L, algo='eig'):
  
   
    plt.figure(figsize=(17, 5))
    for i, lap in enumerate(L):
        lamb, U = fourier(lap, algo)
        step = 2**i
        x = range(step//2, L[0].shape[0], step)
        lb = 'L_{} spectrum in [{:1.2e}, {:1.2e}]'.format(i, lamb[0], lamb[-1])
        plt.plot(x, lamb, '.', label=lb)
    plt.legend(loc='best')
    plt.xlim(0, L[0].shape[0])
    plt.ylim(ymin=0)


def lanczos(L, X, K):
  
    M, N = X.shape
    assert L.dtype == X.dtype

    def basis(L, X, K):
       
        a = np.empty((K, N), L.dtype)
        b = np.zeros((K, N), L.dtype)
        V = np.empty((K, M, N), L.dtype)
        V[0, ...] = X / np.linalg.norm(X, axis=0)
        for k in range(K-1):
            W = L.dot(V[k, ...])
            a[k, :] = np.sum(W * V[k, ...], axis=0)
            W = W - a[k, :] * V[k, ...] - (
                    b[k, :] * V[k-1, ...] if k > 0 else 0)
            b[k+1, :] = np.linalg.norm(W, axis=0)
            V[k+1, ...] = W / b[k+1, :]
        a[K-1, :] = np.sum(L.dot(V[K-1, ...]) * V[K-1, ...], axis=0)
        return V, a, b

    def diag_H(a, b, K):
      
        H = np.zeros((K*K, N), a.dtype)
        H[:K**2:K+1, :] = a
        H[1:(K-1)*K:K+1, :] = b[1:, :]
        H.shape = (K, K, N)
        Q = np.linalg.eigh(H.T, UPLO='L')[1]
        Q = np.swapaxes(Q, 1, 2).T
        return Q

    V, a, b = basis(L, X, K)
    Q = diag_H(a, b, K)
    Xt = np.empty((K, M, N), L.dtype)
    for n in range(N):
        Xt[..., n] = Q[..., n].T.dot(V[..., n])
    Xt *= Q[0, :, np.newaxis, :]
    Xt *= np.linalg.norm(X, axis=0)
    return Xt 


def rescale_L(L, lmax=2):
  
    M, M = L.shape
    I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)
    L /= lmax / 2
    L -= I
    return L


def chebyshev(L, X, K):
  
   
    M, N = X.shape
    assert L.dtype == X.dtype

   
   
    Xt = np.empty((K, M, N), L.dtype)
   
    Xt[0, ...] = X
   
    if K > 1:
        Xt[1, ...] = L.dot(X)
   
    for k in range(2, K):
        Xt[k, ...] = 2 * L.dot(Xt[k-1, ...]) - Xt[k-2, ...]
    return Xt

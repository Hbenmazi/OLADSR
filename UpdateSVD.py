import numpy as np
import matlab.engine
import matlab

eng = matlab.engine.start_matlab()


def UpdateSVD(W):
    W = matlab.double(W.tolist())
    return np.array(eng.UpdateSVD(W))


def Update_XY(Z):
    r = Z.shape[0]
    Z_bar = Z - Z.mean(axis=1)[:, np.newaxis]
    SVD_Z = np.linalg.svd(Z_bar, full_matrices=False)
    Q = SVD_Z[2].T
    if Q.shape[1] < r:
        Q = gram_schmidt(np.c_[Q, np.ones((Q.shape[0], r - Q.shape[1]))])
    P = np.linalg.svd(np.dot(Z_bar, Z_bar.T))[0]
    Z_new = np.sqrt(Z.shape[1]) * np.dot(P, Q.T)
    return Z_new


def gram_schmidt(X):
    Q, R = np.linalg.qr(X)
    return Q

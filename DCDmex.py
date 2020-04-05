import numpy as np
from numba import njit

from scale import scale


def DCD_Phi_V(B, D, R, IDX_R, E, maxItr2):
    r, n = B.shape
    for i in range(n):
        b = B[:, i].copy().reshape((r, 1))
        idx_i = IDX_R[i, :].toarray().squeeze()
        d = D[:, idx_i]
        # obr = R[i, idx_i].toarray()
        maxR = R.max()
        minR = R.min()
        obr = scale(R[i, idx_i].toarray(), r, maxR, minR)
        e = E[:, i]
        converge = False
        no_change_count = 0
        it = 0

        while not converge:
            no_change_count = 0
            # for each bit in b
            for k in range(r):

                # calculate hat_b
                db = np.dot(d.T, b).T
                hat_b0 = np.dot((db - obr), d[k, :].T) - np.size(obr) * b[k]
                hat_b = hat_b0 - e[k]
                hat_b = np.asscalar(hat_b)

                if hat_b == 0:
                    no_change_count += 1
                elif hat_b > 0:
                    if b[k] == -1:
                        no_change_count += 1
                    else:
                        b[k] = -1
                else:
                    if b[k] == 1:
                        no_change_count += 1
                    else:
                        b[k] = 1

            if it > maxItr2 - 1 or no_change_count == r:
                converge = True
            it += 1
        B[:, i] = b.squeeze()
    return B


def DCD_U(U, Phi, Y, alpha, alpha_tran, delta_phi, gamma, maxItr2):
    r, n = U.shape
    F = delta_phi * Phi + gamma * Y
    for i in range(n):
        u = U[:, i].copy().reshape((r, 1))
        if i in alpha_tran.keys():
            id_tran_i = alpha_tran[i]['sorted_uid']
        else:
            id_tran_i = []

        converge = False

        it = 0

        while not converge:
            no_change_count = 0
            # for each bit
            for k in range(r):
                # calculate hat_u
                hat_u0 = 0
                for j in id_tran_i:
                    alpha_j = alpha[j]['alpha_u']
                    F_j = alpha[j]['sorted_uid']
                    i_idx = np.argwhere(F_j == i)
                    alpha_iu = alpha_j[i_idx]
                    hat_u0 = hat_u0 + alpha_iu * ((alpha_j * U[k, F_j]).sum() - alpha_iu * U[k, i] - Phi[k, j])
                hat_u = delta_phi * hat_u0 - 2 * F[k, i]

                if hat_u == 0:
                    no_change_count += 1
                elif hat_u > 0:
                    if u[k] == -1:
                        no_change_count += 1
                    else:
                        u[k] = -1
                else:
                    if u[k] == 1:
                        no_change_count += 1
                    else:
                        u[k] = 1

            if it > maxItr2 - 1 or no_change_count == r:
                converge = True
            it += 1
        U[:, i] = u.squeeze()
    return U

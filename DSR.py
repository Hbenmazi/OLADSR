import itertools
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from joblib import Parallel, delayed
from numpy import ndarray
from numpy.linalg import norm
from tqdm import tqdm

from UpdateSVD import UpdateSVD
from dataloader import load_data
from evaluator import Metric
from scale import scale


class DSR:
    def __init__(self, R, S, r, alpha0, beta1, beta2, beta3, maxR, minR, maxS,
                 minS, init, debug):
        self.R = R  # Rating maxtrix
        self.S = S  # Social maxtrix
        self.r = r  # #bit
        self.n, self.m = R.shape

        self.IDX_R = (self.R != 0).toarray()
        self.IDX_S = (self.S != 0).toarray()

        # tuning parameter
        self.alpha0 = alpha0
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3

        # option
        self.init = init
        self.debug = debug

        # parameter
        self.B = None
        self.D = None
        self.F = None
        self.X = None
        self.Y = None
        self.Z = None

        # max-min
        self.maxR = maxR
        self.minR = minR
        self.maxS = maxS
        self.minS = minS

    def train(self, maxItr, maxItr2, tol_init):
        """
        :param maxItr: max iteration for training and initializing
        :param maxItr2: max iteration for inner update
        :param tol_init: tolerance for initializing
        :return: B, D, F, X, Y, Z
        """
        if not self.init:
            # initialize B,D,F randomly
            self.B = np.random.randint(0, 2, (self.r, self.n), dtype=np.int8)
            self.B[self.B == 0] = -1

            self.D = np.random.randint(0, 2, (self.r, self.m), dtype=np.int8)
            self.D[self.D == 0] = -1

            self.F = np.random.randint(0, 2, (self.r, self.n), dtype=np.int8)
            self.F[self.F == 0] = -1

            # construct XYZ
            self.X = UpdateSVD(self.B)
            self.Y = UpdateSVD(self.D)
            self.Z = UpdateSVD(self.F)
        else:
            print('initialiing...')
            U, V, T, self.X, self.Y, self.Z = self.initialize(maxItr, tol_init)
            self.B = np.sign(U).astype(np.int8)
            self.B[self.B == 0] = -1
            self.D = np.sign(V).astype(np.int8)
            self.D[self.D == 0] = -1
            self.F = np.sign(T).astype(np.int8)
            self.F[self.F == 0] = -1
            print('initialiing finished')

        converge = False
        it = 1
        while not converge:
            B0 = self.B.copy()
            D0 = self.D.copy()
            F0 = self.F.copy()

            # Optimize B
            arg_user_idx = [i for i in range(self.n)]
            arg_maxItr2 = [maxItr2 for _ in range(self.n)]
            args = zip(arg_user_idx, arg_maxItr2)
            b_list = Parallel(n_jobs=1)(
                delayed(self.optimize_b)(arg) for arg in args)
            self.B = np.array(b_list).T.squeeze()
            if self.debug:
                print('UpdatedB obj:',
                      DSR.DSR_obj(self.R, self.S, self.B, self.D, self.F,
                                  self.X, self.Y, self.Z, self.r,
                                  self.maxR, self.minR, self.maxS, self.minS,
                                  self.alpha0, self.beta1, self.beta2, self.beta3))

            # Optimize D
            arg_item_idx = [i for i in range(self.m)]
            arg_maxItr2 = [maxItr2 for _ in range(self.m)]
            args = zip(arg_item_idx, arg_maxItr2)
            d_list = Parallel(n_jobs=1)(
                delayed(self.optimize_d)(arg) for arg in args)
            self.D = np.array(d_list).T.squeeze()
            if self.debug:
                print('UpdatedD obj:',
                      DSR.DSR_obj(self.R, self.S, self.B, self.D, self.F,
                                  self.X, self.Y, self.Z, self.r,
                                  self.maxR, self.minR, self.maxS, self.minS,
                                  self.alpha0, self.beta1, self.beta2, self.beta3))

            # Optimize F
            arg_user_idx = [i for i in range(self.n)]
            arg_maxItr2 = [maxItr2 for _ in range(self.n)]
            args = zip(arg_user_idx, arg_maxItr2)
            f_list = Parallel(n_jobs=1)(
                delayed(self.optimize_f)(arg) for arg in args)
            self.F = np.array(f_list).T.squeeze()
            if self.debug:
                print('UpdatedF obj:',
                      DSR.DSR_obj(self.R, self.S, self.B, self.D, self.F,
                                  self.X, self.Y, self.Z, self.r,
                                  self.maxR, self.minR, self.maxS, self.minS,
                                  self.alpha0, self.beta1, self.beta2, self.beta3))

            # update XYZ
            self.X = UpdateSVD(self.B)
            self.Y = UpdateSVD(self.D)
            self.Z = UpdateSVD(self.F)
            if self.debug:
                print('UpdatedXYZ obj:',
                      DSR.DSR_obj(self.R, self.S, self.B, self.D, self.F,
                                  self.X, self.Y, self.Z, self.r,
                                  self.maxR, self.minR, self.maxS, self.minS,
                                  self.alpha0, self.beta1, self.beta2, self.beta3))

            if it >= maxItr or (
                    (B0 == self.B).all() and (D0 == self.D).all() and (
                    F0 == self.F).all()):
                converge = True
            it += 1
        return self.B, self.D, self.F, self.X, self.Y, self.Z

    def initialize(self, maxItr, tol_init):
        U = np.random.rand(self.r, self.n) * 2 - 1
        V = np.random.rand(self.r, self.m) * 2 - 1
        T = np.random.rand(self.r, self.n) * 2 - 1
        X = UpdateSVD(U)
        Y = UpdateSVD(V)
        Z = UpdateSVD(T)

        converge = False
        it = 1
        while not converge:
            U0 = U.copy()
            V0 = V.copy()
            T0 = T.copy()
            X0 = X.copy()
            Y0 = Y.copy()
            Z0 = Z.copy()

            #  Update U
            arg_user_idx = [i for i in range(self.n)]
            arg_U = [U for _ in range(self.n)]
            arg_V = [V for _ in range(self.n)]
            arg_T = [T for _ in range(self.n)]
            arg_X = [X for _ in range(self.n)]
            args = zip(arg_user_idx, arg_U, arg_V, arg_T, arg_X)
            u_list = Parallel(n_jobs=1)(
                delayed(self.optimize_u)(arg) for arg in args)
            U = np.array(u_list).T.squeeze()
            if self.debug:
                print('UpdatedU obj:',
                      DSR.DSR_init_Obj(self.R, self.S, U, V, T,
                                       X, Y, Z, self.r,
                                       self.maxR, self.minR, self.maxS, self.minS,
                                       self.alpha0, self.beta1, self.beta2, self.beta3))

            #  Update V
            arg_item_idx = [j for j in range(self.m)]
            arg_V = [V for _ in range(self.m)]
            arg_U = [U for _ in range(self.m)]
            arg_Y = [Y for _ in range(self.m)]
            args = zip(arg_item_idx, arg_V, arg_U, arg_Y)
            v_list = Parallel(n_jobs=1)(
                delayed(self.optimize_v)(arg) for arg in args)
            V = np.array(v_list).T.squeeze()
            if self.debug:
                print('UpdatedV obj:',
                      DSR.DSR_init_Obj(self.R, self.S, U, V, T,
                                       X, Y, Z, self.r,
                                       self.maxR, self.minR, self.maxS, self.minS,
                                       self.alpha0, self.beta1, self.beta2, self.beta3))

            #  Update T
            arg_user_idx = [j for j in range(self.n)]
            arg_T = [T for _ in range(self.n)]
            arg_U = [U for _ in range(self.n)]
            arg_Z = [Z for _ in range(self.n)]
            args = zip(arg_user_idx, arg_T, arg_U, arg_Z)
            t_list = Parallel(n_jobs=1)(
                delayed(self.optimize_t)(arg) for arg in args)
            T = np.array(t_list).T.squeeze()
            if self.debug:
                print('UpdatedT obj:',
                      DSR.DSR_init_Obj(self.R, self.S, U, V, T,
                                       X, Y, Z, self.r,
                                       self.maxR, self.minR, self.maxS, self.minS,
                                       self.alpha0, self.beta1, self.beta2, self.beta3))

            #  Update XYZ
            X = UpdateSVD(U)
            Y = UpdateSVD(V)
            Z = UpdateSVD(T)
            if self.debug:
                print('UpdatedXYZ obj:',
                      DSR.DSR_init_Obj(self.R, self.S, U, V, T,
                                       X, Y, Z, self.r,
                                       self.maxR, self.minR, self.maxS, self.minS,
                                       self.alpha0, self.beta1, self.beta2, self.beta3))

            if it >= maxItr or max(norm(X - X0), norm(Y - Y0), norm(Z - Z0),
                                   norm(U - U0), norm(V - V0),
                                   norm(T - T0)) < max(self.n,
                                                       self.m) * tol_init:
                converge = True
            it += 1
        return U, V, T, X, Y, Z

    def optimize_b(self, args):
        u_idx = args[0]
        maxItr2 = args[1]

        R = scale(self.R[u_idx, self.IDX_R[u_idx, :]], self.r, self.maxR,
                  self.minR)
        scaled_S = scale(self.S[u_idx, self.IDX_S[u_idx, :]], self.r, self.maxS,
                         self.minS)

        S = self.S[u_idx, self.IDX_S[u_idx, :]]
        b = self.B[:, u_idx].copy().reshape((self.r, 1))
        d = self.D[:, self.IDX_R[u_idx, :]]
        f = self.F[:, self.IDX_S[u_idx, :]]
        x = self.X[:, u_idx]

        converge = False

        it = 0
        while not converge:
            no_change_count = 0
            # for each bit in b
            for k in range(self.r):

                # calculate hat_b
                db = np.dot(d.T, b).T
                fb = np.dot(f.T, b).T

                hat_b0 = np.dot((R - db), d[k, :].T) + R.shape[1] * b[
                    k] + self.alpha0 * (np.dot((scaled_S - fb), f[k, :].T)
                                        + S.getnnz() * b[k])

                hat_b = hat_b0 + self.beta1 * x[k]
                if isinstance(hat_b, ndarray):
                    hat_b = hat_b[0]
                else:
                    hat_b = hat_b.item(0, 0)

                if hat_b == 0:
                    no_change_count += 1
                elif hat_b > 0:
                    if b[k] == 1:
                        no_change_count += 1
                    else:
                        b[k] = 1
                else:
                    if b[k] == -1:
                        no_change_count += 1
                    else:
                        b[k] = -1

            if it > maxItr2 - 1 or no_change_count == self.r:
                converge = True
            it += 1
        return b

    def optimize_d(self, args):
        i_idx = args[0]
        maxItr2 = args[1]

        R = scale(self.R[self.IDX_R[:, i_idx], i_idx], self.r, self.maxR,
                  self.minR)
        b = self.B[:, self.IDX_R[:, i_idx]]
        d = self.D[:, i_idx].copy().reshape((self.r, 1))
        y = self.Y[:, i_idx]

        converge = False

        it = 0
        while not converge:
            no_change_count = 0
            # for each bit in r
            for k in range(self.r):

                # calculate hat_d
                bd = np.dot(b.T, d).T
                hat_d0 = np.dot((R.T - bd), b[k, :].T) + R.shape[0] * d[k]
                hat_d = hat_d0 + self.beta2 * y[k]
                hat_d = hat_d[0]

                if hat_d == 0:
                    no_change_count += 1
                elif hat_d > 0:
                    if d[k] == 1:
                        no_change_count += 1
                    else:
                        d[k] = 1
                else:
                    if d[k] == -1:
                        no_change_count += 1
                    else:
                        d[k] = -1

            if it > maxItr2 - 1 or no_change_count == self.r:
                converge = True
            it += 1
        return d

    def optimize_f(self, args):
        u_idx = args[0]
        maxItr2 = args[1]

        S = self.S[self.IDX_S[:, u_idx], u_idx]
        scaled_S = scale(self.S[self.IDX_S[:, u_idx], u_idx], self.r, self.maxS,
                         self.minS)
        b = self.B[:, self.IDX_S[:, u_idx]]
        f = self.F[:, u_idx].copy().reshape((self.r, 1))
        z = self.Z[:, u_idx]

        converge = False

        it = 0
        while not converge:
            no_change_count = 0
            # for each bit in f
            for k in range(self.r):

                # calculate hat_f
                bf = np.dot(b.T, f).T
                hat_f0 = np.dot((scaled_S.T - bf), b[k, :].T) + S.getnnz() * f[k]
                hat_f = self.alpha0 * hat_f0 + self.beta3 * z[k]
                hat_f = hat_f[0]

                if hat_f == 0:
                    no_change_count += 1
                elif hat_f > 0:
                    if f[k] == 1:
                        no_change_count += 1
                    else:
                        f[k] = 1
                else:
                    if f[k] == -1:
                        no_change_count += 1
                    else:
                        f[k] = -1

            if it > maxItr2 - 1 or no_change_count == self.r:
                converge = True
            it += 1

        return f

    def optimize_u(self, args):
        u_idx = args[0]
        U = args[1]
        V = args[2]
        T = args[3]
        X = args[4]

        Vu = V[:, self.IDX_R[u_idx, :]]
        Ru = scale(self.R[u_idx, self.IDX_R[u_idx, :]], self.r, self.maxR,
                   self.minR)

        Su = scale(self.S[u_idx, self.IDX_S[u_idx, :]], self.r, self.maxS, self.minS)
        Tu = T[:, self.IDX_S[u_idx, :]]

        if Ru.size == 0 and Su.size == 0:
            return U[:, u_idx].reshape((self.r, 1))

        VuRu = np.dot(Vu, Ru.T)
        TuSu = np.dot(Tu, Su.T)
        if VuRu.size == 0:
            VuRu = np.zeros(shape=(self.r, 1))

        if TuSu.size == 0:
            TuSu = np.zeros(shape=(self.r, 1))

        Q = Vu.dot(Vu.T) + self.alpha0 * (Tu.dot(Tu.T)) + self.beta1 * (Ru.shape[1] + Su.shape[1]) * np.eye(self.r)
        L = VuRu + self.alpha0 * TuSu + self.beta1 * X[:, u_idx].reshape((self.r, 1))

        u = np.linalg.solve(Q, L)
        return u

    def optimize_v(self, args):
        j = args[0]  # item idx
        V = args[1]
        U = args[2]
        Y = args[3]

        Uj = U[:, self.IDX_R[:, j]]
        Rj = scale(self.R[self.IDX_R[:, j], j], self.r, self.maxR, self.minR)

        if Rj.size == 0:
            return V[:, j].reshape((self.r, 1))

        UjRj = np.dot(Uj, Rj)
        if UjRj.size == 0:
            UjRj = np.zeros(shape=(self.r, 1))

        Q = np.dot(Uj, Uj.T) + self.beta2 * Rj.shape[0] * np.eye(self.r)
        L = UjRj + self.beta2 * Y[:, j].reshape((self.r, 1))
        v = np.linalg.solve(Q, L)

        return v

    def optimize_t(self, args):
        j = args[0]  # user idx
        T = args[1]
        U = args[2]
        Z = args[3]

        Uj = U[:, self.IDX_S[:, j]]
        Sj = scale(self.S[self.IDX_S[:, j], j], self.r, self.maxS, self.minS)

        if Sj.size == 0:
            return T[:, j].reshape((self.r, 1))

        UjSj = np.dot(Uj, Sj)
        if UjSj.size == 0:
            UjSj = np.zeros(shape=(self.r, 1))
        Q = self.alpha0 * np.dot(Uj, Uj.T) + self.beta3 * Sj.shape[0] * np.eye(self.r)
        L = self.alpha0 * UjSj + self.beta3 * Z[:, j].reshape((self.r, 1))
        t = np.linalg.solve(Q, L)
        return t

    @staticmethod
    def DSR_obj(R, S, B, D, F, X, Y, Z, r, maxR, minR, maxS, minS, alpha0, beta1, beta2, beta3):

        # index set
        observed_rate_idx = np.nonzero(R)
        observed_trust_idx = np.nonzero(S)

        # (R_ij - b_i * d_j)^2
        rated_R = scale(R[observed_rate_idx].getA(), r, maxR, minR)
        rated_B = B[:, observed_rate_idx[0]]
        rated_D = D[:, observed_rate_idx[1]]
        rated_BD = np.multiply(rated_B, rated_D).sum(axis=0, keepdims=True)
        obj = (rated_R - rated_BD).dot((rated_R - rated_BD).T).astype(np.float)

        # alpha0 * (S_ij - b_i * f_j)^2
        trusted_S = scale(S[observed_trust_idx].getA(), r, maxS, minS)
        trusted_B = B[:, observed_trust_idx[0]]
        trusted_F = F[:, observed_trust_idx[1]]
        trusted_BF = np.multiply(trusted_B, trusted_F).sum(axis=0, keepdims=True)
        obj += (trusted_S - trusted_BF).dot((trusted_S - trusted_BF).T).astype(
            np.float) * alpha0

        #  balance & de-correlated
        obj -= 2 * (beta1 * np.trace(B.T.dot(X)) + beta2 * np.trace(
            D.T.dot(Y)) + beta3 * np.trace(F.T.dot(Z)))
        return obj[0][0]

    @staticmethod
    def DSR_init_Obj(R, S, U, V, T, X, Y, Z, r, maxR, minR, maxS, minS, alpha0, beta1, beta2, beta3):

        # index set
        observed_rate_idx = np.nonzero(R)
        observed_trust_idx = np.nonzero(S)

        # (R_ij - b_i * d_j)^2
        rated_R = scale(R[observed_rate_idx].getA(), r, maxR, minR)
        rated_U = U[:, observed_rate_idx[0]]
        rated_V = V[:, observed_rate_idx[1]]
        rated_UV = np.multiply(rated_U, rated_V).sum(axis=0, keepdims=True)
        obj = (rated_R - rated_UV).dot((rated_R - rated_UV).T).astype(np.float)

        # alpha0 * (S_ij - U_i * T_j)^2
        trusted_S = scale(S[observed_trust_idx].getA(), r, maxS, minS)
        trusted_U = U[:, observed_trust_idx[0]]
        trusted_T = T[:, observed_trust_idx[1]]
        trusted_UT = np.multiply(trusted_U, trusted_T).sum(axis=0, keepdims=True)
        obj += (trusted_S - trusted_UT).dot((trusted_S - trusted_UT).T).astype(
            np.float) * alpha0

        #  balance & de-correlated
        obj -= 2 * (beta1 * np.trace(U.T.dot(X)) + beta2 * np.trace(
            V.T.dot(Y)) + beta3 * np.trace(T.T.dot(Z)))

        # regularize
        # obj += beta1 * norm(U) ** 2 + beta2 * norm(
        #     V) ** 2 + alpha0 * beta3 * norm(T) ** 2
        RCount = R.copy()
        SCount = S.copy()
        RCount[observed_rate_idx] = 1
        reg = ((RCount.sum(axis=1).getA().squeeze() + SCount.sum(axis=1).getA().squeeze()) * beta1 * (U ** 2)).sum() + \
              (RCount.sum(axis=0).getA().squeeze() * beta2 * (V ** 2)).sum() + \
              (SCount.sum(axis=0).getA().squeeze() * beta3 * (T ** 2)).sum()
        obj += reg
        return obj[0][0]


def grid_search_inner(pa):
    maxItr = 50
    maxItr2 = 5
    tol_init = 1e-5
    maxR = 4.5
    minR = 0.5
    maxS = 1.0
    minS = 0.0
    ndcg5_sum = 0
    for _ in range(5):
        R_train, R_test, S_bin, S_con = load_data("data/filmtrust", remove=True, user_filter=0)
        metric = Metric(R_test)
        dsr = DSR(R_train, S_bin, pa[4], pa[0], pa[1], pa[2], pa[3], maxR, minR, maxS, minS, init=False,
                  debug=False)
        dsr.train(maxItr, maxItr2, tol_init)
        ndcg5 = metric.NDCG(dsr.B, dsr.D, R_train, k=5)
        ndcg5_sum += ndcg5
    return ndcg5_sum / 5, (pa[0], pa[1], pa[2], pa[3])


def grid_search():
    alpha0s = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]
    beta1s = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    beta2s = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    beta3s = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    rs = [8, 16, 24, 32]

    logging.basicConfig(filename="DSR.txt", level="DEBUG")
    for r in rs:
        paras = list(itertools.product(*[alpha0s, beta1s, beta2s, beta3s, [r]]))
        with ProcessPoolExecutor() as pool:
            task_list = [pool.submit(grid_search_inner, para) for para in paras]
            process_results = [task.result() for task in
                               tqdm(as_completed(task_list), desc="r={}".format(r), total=len(paras))]
            metric_to_para = {re[0]: re[1] for re in process_results}
            best_metric = min(metric_to_para.keys())
            best_para = metric_to_para[best_metric]

        best_alpha0, best_beta1, best_beta2, best_beta3 = best_para

        logging.debug("DSR(r={},alpha0={},beta1={},beta2={},beta3={}):".format(r, best_alpha0,
                                                                               best_beta1, best_beta2, best_beta3))
        print("DSR(r={},alpha0={},beta1={},beta2={},beta3={}):".format(r, best_alpha0,
                                                                       best_beta1, best_beta2, best_beta3))


def single_test():
    maxItr = 50
    maxItr2 = 5
    tol_init = 1e-5
    maxR = 4.5
    minR = 0.5
    maxS = 1.0
    minS = 0.0
    r = 8
    alpha0 = 0.1
    beta1 = 0.01
    beta2 = 0.01
    beta3 = 0.1
    rmse_sum = 0
    mae_sum = 0
    for _ in range(5):
        R_train, R_test, S_bin, S_con = load_data("data/filmtrust", remove=True)
        metric = Metric(R_test)
        dsr = DSR(R_train, S_bin, r, alpha0, beta1, beta2, beta3, maxR, minR, maxS, minS, init=False,
                  debug=False)
        dsr.train(maxItr, maxItr2, tol_init)
        rmse = metric.RMSE(dsr.B, dsr.D, r=r)
        mae = metric.MAE(dsr.B, dsr.D, r=r)
        rmse_sum += rmse
        mae_sum += mae
    print(rmse_sum / 5)
    print(mae_sum / 5)


if __name__ == "__main__":
    single_test()

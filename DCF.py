import itertools
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from joblib import Parallel, delayed
from numpy.linalg import norm
from tqdm import tqdm

from UpdateSVD import UpdateSVD
from dataloader import load_data
from evaluator import Metric
from scale import scale


class DCF:
    def __init__(self, R, r, alpha, beta, maxR, minR, init, debug):
        self.R = R  # Rating maxtrix
        self.r = r  # #bit
        self.n, self.m = R.shape

        self.IDX = (self.R != 0).toarray()

        # tuning parameter
        self.alpha = alpha
        self.beta = beta
        # option
        self.init = init
        self.debug = debug

        # parameter
        self.B = None
        self.D = None
        self.X = None
        self.Y = None

        # max-min
        self.maxR = maxR
        self.minR = minR

    def train(self, maxItr, maxItr2, tol_init):
        """
        :param maxItr: max iteration for training and initializing
        :param maxItr2: max iteration for inner update
        :param tol_init: tolerance for initializing
        :return: B, D, F, X, Y, Z
        """
        if not self.init:
            # initialize B,D,F randomly
            # U = np.loadtxt('data/U.txt')
            # V = np.loadtxt('data/V.txt')

            self.B = np.random.randint(0, 2, (self.r, self.n), dtype=np.int8)

            # self.B = np.sign(U)
            self.B[self.B == 0] = -1

            self.D = np.random.randint(0, 2, (self.r, self.m), dtype=np.int8)
            # self.D = np.sign(V)
            self.D[self.D == 0] = -1

            # construct XYZ
            self.X = UpdateSVD(self.B)
            self.Y = UpdateSVD(self.D)
        else:
            print('initialiing...')
            U, V, self.X, self.Y = self.initialize(maxItr, tol_init)
            self.B = np.sign(U).astype(np.int8)
            self.B[self.B == 0] = -1
            self.D = np.sign(V).astype(np.int8)
            self.D[self.D == 0] = -1
            print('initialiing finished')

        converge = False
        it = 1
        while not converge:
            B0 = self.B.copy()
            D0 = self.D.copy()

            # Optimize B
            arg_user_idx = [i for i in range(self.n)]
            arg_maxItr2 = [maxItr2 for _ in range(self.n)]
            args = zip(arg_user_idx, arg_maxItr2)
            b_list = Parallel(n_jobs=1)(
                delayed(self.optimize_b)(arg) for arg in args)
            self.B = np.array(b_list).T.squeeze()
            if self.debug:
                print('UpdatedB obj:',
                      DCF.DCF_obj(self.R, self.B, self.D, self.X, self.Y, self.r, self.maxR, self.minR, self.alpha,
                                  self.beta))

            # Optimize D
            arg_item_idx = [i for i in range(self.m)]
            arg_maxItr2 = [maxItr2 for _ in range(self.m)]
            args = zip(arg_item_idx, arg_maxItr2)
            d_list = Parallel(n_jobs=1)(
                delayed(self.optimize_d)(arg) for arg in args)
            self.D = np.array(d_list).T.squeeze()
            if self.debug:
                print('UpdatedD obj:',
                      DCF.DCF_obj(self.R, self.B, self.D, self.X, self.Y, self.r, self.maxR, self.minR, self.alpha,
                                  self.beta))

            # update XY
            self.X = UpdateSVD(self.B)
            self.Y = UpdateSVD(self.D)
            if self.debug:
                print('UpdatedXY obj:',
                      DCF.DCF_obj(self.R, self.B, self.D, self.X, self.Y, self.r, self.maxR, self.minR, self.alpha,
                                  self.beta))

            if it >= maxItr or ((B0 == self.B).all() and (D0 == self.D).all()):
                converge = True
            it += 1
        return self.B, self.D, self.X, self.Y

    def initialize(self, maxItr, tol_init):
        U = np.random.rand(self.r, self.n) * 2 - 1
        V = np.random.rand(self.r, self.m) * 2 - 1
        X = UpdateSVD(U)
        Y = UpdateSVD(V)

        converge = False
        it = 1
        while not converge:
            U0 = U.copy()
            V0 = V.copy()
            X0 = X.copy()
            Y0 = Y.copy()

            #  Update U
            arg_user_idx = [i for i in range(self.n)]
            arg_U = [U for _ in range(self.n)]
            arg_V = [V for _ in range(self.n)]
            arg_X = [X for _ in range(self.n)]
            args = zip(arg_user_idx, arg_U, arg_V, arg_X)
            u_list = Parallel(n_jobs=1)(
                delayed(self.optimize_u)(arg) for arg in args)
            U = np.array(u_list).T.squeeze()
            if self.debug:
                print('UpdatedU init_obj:',
                      DCF.DCF_init_obj(self.R, U, V, X, Y, self.r, self.maxR, self.minR, self.alpha,
                                       self.beta))

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
                print('UpdatedV init_obj:',
                      DCF.DCF_init_obj(self.R, U, V, X, Y, self.r, self.maxR, self.minR, self.alpha,
                                       self.beta))

            #  Update XY
            X = UpdateSVD(U)
            Y = UpdateSVD(V)
            if self.debug:
                print('UpdatedXY init_obj:',
                      DCF.DCF_init_obj(self.R, U, V, X, Y, self.r, self.maxR, self.minR, self.alpha,
                                       self.beta))

            # Todo 评估init指标

            if it >= maxItr or max(norm(X - X0), norm(Y - Y0),
                                   norm(U - U0), norm(V - V0)) < max(self.n,
                                                                     self.m) * tol_init:
                converge = True
            it += 1
        return U, V, X, Y

    def optimize_b(self, args):
        u_idx = args[0]
        maxItr2 = args[1]

        R = scale(self.R[u_idx, self.IDX[u_idx, :]], self.r, self.maxR,
                  self.minR)

        b = self.B[:, u_idx].copy().reshape((self.r, 1))
        d = self.D[:, self.IDX[u_idx, :]]
        x = self.X[:, u_idx]

        converge = False

        it = 0
        while not converge:
            no_change_count = 0
            # for each bit in b
            for k in range(self.r):

                # calculate hat_b
                db = np.dot(d.T, b).T
                hat_b0 = np.dot((R - db), d[k, :].T) + R.shape[1] * b[k]
                hat_b = hat_b0 + self.alpha * x[k]
                hat_b = hat_b[0]

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

        R = scale(self.R[self.IDX[:, i_idx], i_idx], self.r, self.maxR,
                  self.minR)
        b = self.B[:, self.IDX[:, i_idx]]
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
                hat_d = hat_d0 + self.beta * y[k]
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

    def optimize_u(self, args):
        u_idx = args[0]
        U = args[1]
        V = args[2]
        X = args[3]

        Vu = V[:, self.IDX[u_idx, :]]
        Ru = scale(self.R[u_idx, self.IDX[u_idx, :]], self.r, self.maxR,
                   self.minR)

        if Ru.size == 0:
            return U[:, u_idx].reshape((self.r, 1))

        VuRu = np.dot(Vu, Ru.T)
        if VuRu.size == 0:
            VuRu = np.zeros(shape=(self.r, 1))

        Q = np.dot(Vu, Vu.T) + self.alpha * Ru.shape[1] * np.eye(self.r)
        # Q = np.dot(Vu, Vu.T) + self.alpha * np.eye(self.r)
        L = VuRu + 2 * self.alpha * X[:, u_idx].reshape((self.r, 1))

        u = np.linalg.solve(Q, L)
        return u

    def optimize_v(self, args):
        j = args[0]  # item idx
        V = args[1]
        U = args[2]
        Y = args[3]

        Uj = U[:, self.IDX[:, j]]
        Rj = scale(self.R[self.IDX[:, j], j], self.r, self.maxR, self.minR)

        if Rj.size == 0:
            return V[:, j].reshape((self.r, 1))

        UjRj = np.dot(Uj, Rj)
        if UjRj.size == 0:
            UjRj = np.zeros(shape=(self.r, 1))

        Q = np.dot(Uj, Uj.T) + self.beta * Rj.shape[0] * np.eye(self.r)
        # Q = np.dot(Uj, Uj.T) + self.zeta * np.eye(self.r)
        L = UjRj + 2 * self.beta * Y[:, j].reshape((self.r, 1))
        v = np.linalg.solve(Q, L)

        return v

    @staticmethod
    def DCF_obj(R, B, D, X, Y, r, maxR, minR, alpha, beta):

        # index set
        observed_rate_idx = np.nonzero(R)

        # (R_ij - b_i * d_j)^2
        rated_R = scale(R[observed_rate_idx].getA(), r, maxR, minR)
        # rated_R = scale(R[observed_rate_idx].getA(), dsr.r, 5.0, 1.0)
        rated_B = B[:, observed_rate_idx[0]]
        rated_D = D[:, observed_rate_idx[1]]
        rated_BD = np.multiply(rated_B, rated_D).sum(axis=0, keepdims=True)
        obj = (rated_R - rated_BD).dot((rated_R - rated_BD).T).astype(np.float)

        #  balance & de-correlated
        obj -= 2 * (alpha * np.trace(B.T.dot(X)) + beta * np.trace(
            D.T.dot(Y)))
        return obj[0][0]

    @staticmethod
    def DCF_init_obj(R, U, V, X, Y, r, maxR, minR, alpha, beta):

        m, n = R.shape
        # index set
        observed_rate_idx = np.nonzero(R)

        # (R_ij - b_i * d_j)^2
        rated_R = scale(R[observed_rate_idx].getA(), r, maxR, minR)
        rated_U = U[:, observed_rate_idx[0]]
        rated_V = V[:, observed_rate_idx[1]]
        rated_UV = np.multiply(rated_U, rated_V).sum(axis=0, keepdims=True)
        obj = (rated_R - rated_UV).dot((rated_R - rated_UV).T).astype(np.float)

        #  balance & de-correlated
        obj -= 2 * (alpha * np.trace(U.T.dot(X)) + beta * np.trace(V.T.dot(Y)))

        # regularize
        # obj += alpha * norm(U) ** 2 + zeta * norm(V) ** 2
        RCount = R.copy()
        RCount[observed_rate_idx] = 1
        reg = (RCount.sum(axis=1).getA().squeeze() * alpha * (U ** 2)).sum() + (
                RCount.sum(axis=0).getA().squeeze() * beta * (V ** 2)).sum()
        obj += reg

        return obj[0][0]


def grid_search_inner(pa):
    maxItr = 50
    maxItr2 = 5
    tol_init = 1e-5
    maxR = 4.5
    minR = 0.5
    rmse_sum = 0
    for _ in range(5):
        R_train, R_test, S_bin, S_con = load_data("data/CiaoDVD", remove=True)
        metric = Metric(R_test)
        dcf = DCF(R_train, pa[2], pa[0], pa[1], maxR, minR, init=False, debug=False)
        dcf.train(maxItr, maxItr2, tol_init)
        rmse = metric.RMSE(dcf.B, dcf.D, r=pa[2])
        rmse_sum += rmse
    return rmse_sum / 5, (pa[0], pa[1])


def grid_search():
    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]
    betas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]
    rs = [8, 16, 24, 32]
    logging.basicConfig(filename="DCF.txt", level="DEBUG")
    for r in rs:
        paras = list(itertools.product(*[alphas, betas, [r]]))
        with ProcessPoolExecutor(max_workers=1) as pool:
            task_list = [pool.submit(grid_search_inner, para) for para in paras]
            process_results = [task.result() for task in
                               tqdm(as_completed(task_list), desc="r={}".format(r), total=len(paras))]
            rmse_to_para = {re[0]: re[1] for re in process_results}
            best_rmse = min(rmse_to_para.keys())
            best_para = rmse_to_para[best_rmse]

        best_alpha, best_beta = best_para
        logging.debug("DCF(r={},alpha={},beta={}):".format(r, best_alpha, best_beta))
        print("DCF(r={},alpha={},beta={}):".format(r, best_alpha, best_beta))


def sigle_test():
    maxItr = 50
    maxItr2 = 5
    tol_init = 1e-5
    rmse_sum = 0
    mae_sum = 0
    r = 8
    alpha = 0.001
    beta = 0.0001
    for _ in range(5):
        R_train, R_test, S_bin, S_con = load_data("data/filmtrust", remove=True)
        metric = Metric(R_test)
        dcf = DCF(R_train, r, alpha, beta, maxR=4.5, minR=0.5, init=False, debug=True)
        dcf.train(maxItr, maxItr2, tol_init)
        rmse_sum += metric.RMSE(dcf.B, dcf.D, r=r)
        mae_sum += metric.MAE(dcf.B, dcf.D, r=r)
    print(rmse_sum / 5)
    print(mae_sum / 5)


if __name__ == "__main__":
    sigle_test()

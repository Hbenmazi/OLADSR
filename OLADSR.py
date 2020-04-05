import itertools
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from autograd import grad
from tqdm import tqdm

from DCDmex import DCD_Phi_V, DCD_U
from OLA import OLA
from UpdateSVD import UpdateSVD
from dataloader import load_data
from evaluator import Metric
from scale import scale


class OLADSR:
    def __init__(self, R, S, d, delta_phi, zeta, gamma, eta, Lc, lr, init, debug):
        self.R = R
        self.S = S
        self.d = d
        self.M, self.N = R.shape
        self.IDX = self.R != 0
        # self.Phi = np.random.rand(self.d, self.M)
        # self.V = np.random.rand(self.d, self.N)
        # self.U = np.random.rand(self.d, self.M)
        self.Phi = None
        self.V = None
        self.U = None
        self.X = None
        self.Y = None
        self.Z = None
        self.delta_phi = delta_phi
        self.zeta = zeta
        self.gamma = gamma
        self.eta = eta
        self.Lc = Lc
        self.alpha = {}
        self.alpha_tran = {}
        self.lr = lr
        self.friend_li = [np.nonzero(S[i, :])[1] for i in range(self.M)]
        self.init = init
        self.debug = debug

    @staticmethod
    def constructG(u, U, alpha):
        if u in alpha.keys():
            alpha_u = alpha[u]['alpha_u']
            sorted_uid = alpha[u]['sorted_uid']
            return np.sum(alpha_u * U[:, sorted_uid], axis=1)
        else:
            return np.zeros_like(U[:, 0])

    @staticmethod
    def OLADSR_obj(R, Phi, U, V, X, Y, Z, delta_phi, zeta, gamma, eta, alpha):
        # index set
        observed_rate_idx = np.nonzero(R)

        maxR = R.max()
        minR = R.min()
        d = U.shape[0]
        rated_R = scale(R[observed_rate_idx].getA(), d, maxR, minR)
        rated_Phi = Phi[:, observed_rate_idx[0]]
        rated_V = V[:, observed_rate_idx[1]]
        rated_PV = np.multiply(rated_Phi, rated_V).sum(axis=0, keepdims=True)
        term1 = np.dot((rated_R - rated_PV), (rated_R - rated_PV).T).astype(np.float)

        # term2
        G = np.array([OLADSR.constructG(u, U, alpha) for u in range(R.shape[0])]).transpose()
        term2 = delta_phi * ((Phi - G) ** 2).sum()

        # term3456
        term3456 = -2 * (delta_phi * np.trace(np.dot(U.T, Phi)) + zeta * np.trace(np.dot(Phi.T, X)) +
                         gamma * np.trace(np.dot(U.T, Y)) + eta * np.trace(np.dot(V.T, Z)))

        return (term1 + term2 + term3456)[0][0]

    def E_step(self):
        self.alpha = {}
        self.alpha_tran = {}
        for i in range(self.M):
            friend_ids = self.friend_li[i]
            num_friend = len(friend_ids)
            if num_friend > 0:
                beta = self.Lc * np.linalg.norm(self.U[:, [i]] - self.U[:, friend_ids], axis=0)
                sorted_idx = np.argsort(beta)
                beta = beta[sorted_idx]
                lbd = beta[0] + 1
                k = 0
                while k <= num_friend - 1 and lbd > beta[k]:
                    k += 1
                    sum_beta_k = beta[0:k].sum()
                    sum_beta2_k = (beta[0:k] ** 2).sum()
                    lbd = (1.0 / k) * (sum_beta_k + np.sqrt(k + sum_beta_k ** 2 - k * sum_beta2_k))
                sorted_uid = friend_ids[sorted_idx][0:k]
                alpha_u = lbd - beta[0:k]
                if alpha_u.sum() == 0:
                    alpha_u[:] = 0
                else:
                    alpha_u /= alpha_u.sum()

                self.alpha[i] = {'alpha_u': alpha_u, 'sorted_uid': sorted_uid}

                for idx, u in enumerate(sorted_uid):
                    if u not in self.alpha_tran.keys():
                        self.alpha_tran[u] = {'sorted_uid': np.array(i)[np.newaxis]}
                    else:
                        self.alpha_tran[u]['sorted_uid'] = np.append(self.alpha_tran[u]['sorted_uid'], i)
                # for idx, u in enumerate(sorted_uid):
                #     if u not in self.alpha_tran.keys():
                #         self.alpha_tran[u] = {'alpha_u': np.array(alpha_u[idx]), 'sorted_uid': np.array(i)}
                #     else:
                #         self.alpha_tran[u]['alpha_u'] = np.append(self.alpha_tran[u]['alpha_u'], alpha_u[idx],)
                #         self.alpha_tran[u]['sorted_uid'] = np.append(self.alpha_tran[u]['sorted_uid'], i)

    def M_step_init(self):
        print(OLA.OLA_obj(self.R, self.Phi, self.U, self.V, self.delta_phi, self.delta_u,
                          self.delta_v, self.alpha))
        grad_obj = grad(OLA.OLA_obj, [1, 2, 3])
        grad_Phi, grad_U, grad_V = grad_obj(self.R, self.Phi, self.U, self.V, self.delta_phi, self.delta_u,
                                            self.delta_v, self.alpha)
        self.Phi -= self.lr * grad_Phi
        self.U -= self.lr * grad_U
        self.V -= self.lr * grad_V

    def M_step(self):
        print(OLA.OLA_obj(self.R, self.Phi, self.U, self.V, self.delta_phi, self.delta_u,
                          self.delta_v, self.alpha))
        grad_obj = grad(OLA.OLA_obj, [1, 2, 3])
        grad_Phi, grad_U, grad_V = grad_obj(self.R, self.Phi, self.U, self.V, self.delta_phi, self.delta_u,
                                            self.delta_v, self.alpha)
        self.Phi -= self.lr * grad_Phi
        self.U -= self.lr * grad_U
        self.V -= self.lr * grad_V

    def train(self, maxItr, maxItr2, tol_init):
        if not self.init:
            # initialize B,D,F randomly
            self.Phi = np.random.randint(0, 2, (self.d, self.M), dtype=np.int8)
            self.Phi[self.Phi == 0] = -1

            self.V = np.random.randint(0, 2, (self.d, self.N), dtype=np.int8)
            self.V[self.V == 0] = -1

            self.U = np.random.randint(0, 2, (self.d, self.M), dtype=np.int8)
            self.U[self.U == 0] = -1

            # construct XYZ
            self.X = UpdateSVD(self.Phi)
            self.Y = UpdateSVD(self.U)
            self.Z = UpdateSVD(self.V)
        else:
            pass

        converge = False
        it = 1
        while not converge:
            # E-STEP
            self.E_step()

            # M-STEP
            Phi0 = self.Phi.copy()
            V0 = self.V.copy()
            U0 = self.U.copy()

            # Optimize Phi
            G = np.array([OLADSR.constructG(u, self.U, self.alpha) for u in range(self.M)]).transpose()
            E = self.delta_phi * (G + self.U) + self.zeta * self.X
            self.Phi = DCD_Phi_V(self.Phi, self.V, self.R, self.IDX, E, maxItr2)

            if self.debug:
                print('Updated Phi obj:',
                      OLADSR.OLADSR_obj(self.R, self.Phi, self.U, self.V, self.X, self.Y, self.Z, self.delta_phi,
                                        self.zeta, self.gamma, self.eta, self.alpha))

            # Optimize V
            self.V = DCD_Phi_V(self.V, self.Phi, self.R.T, self.IDX.T, self.eta * self.Z, maxItr2)

            if self.debug:
                print('Updated V obj:',
                      OLADSR.OLADSR_obj(self.R, self.Phi, self.U, self.V, self.X, self.Y, self.Z, self.delta_phi,
                                        self.zeta, self.gamma, self.eta, self.alpha))

            # Optimize U
            self.U = DCD_U(self.U, self.Phi, self.Y, self.alpha, self.alpha_tran, self.delta_phi, self.gamma, maxItr2)

            if self.debug:
                print('Updated U obj:',
                      OLADSR.OLADSR_obj(self.R, self.Phi, self.U, self.V, self.X, self.Y, self.Z, self.delta_phi,
                                        self.zeta, self.gamma, self.eta, self.alpha))

            # update XYZ
            self.X = UpdateSVD(self.Phi)
            self.Y = UpdateSVD(self.U)
            self.Z = UpdateSVD(self.V)

            if self.debug:
                print('Updated XYZ obj:',
                      OLADSR.OLADSR_obj(self.R, self.Phi, self.U, self.V, self.X, self.Y, self.Z, self.delta_phi,
                                        self.zeta, self.gamma, self.eta, self.alpha))

            if it >= maxItr or ((Phi0 == self.Phi).all() and (V0 == self.V).all() and (U0 == self.U).all()):
                converge = True
            it += 1


def grid_search_inner(pa):
    maxItr = 50
    maxItr2 = 5
    tol_init = 1e-5
    rmse_sum = 0
    for _ in range(5):
        R_train, R_test, S_bin, S_con = load_data("data/filmtrust", remove=True)
        metric = Metric(R_test)
        oladsr = OLADSR(R_train, S_bin, pa[5], pa[0], pa[1], pa[2], pa[3], pa[4], lr=0.0001, init=False, debug=False)
        oladsr.train(maxItr, maxItr2, tol_init)
        rmse_sum += metric.RMSE(oladsr.Phi, oladsr.V, r=pa[5])
    return rmse_sum / 5, (pa[0], pa[1], pa[2], pa[3], pa[4])


def grid_search():
    delta_phis = [1e-2, 1e-1, 1, 1e1, 1e2]
    zetas = [1e-2, 1e-1, 1, 1e1, 1e2]
    gammas = [1e-2, 1e-1, 1, 1e1, 1e2]
    etas = [1e-2, 1e-1, 1, 1e1, 1e2]
    Lcs = [1e-1, 1, 1e1, 1e2, 1e3]
    rs = [8, 16, 24, 32]

    logging.basicConfig(filename="OLADSR.txt", level="DEBUG")
    for r in rs:
        paras = list(itertools.product(*[delta_phis, zetas, gammas, etas, Lcs, [r]]))

        with ProcessPoolExecutor() as pool:
            task_list = [pool.submit(grid_search_inner, para) for para in paras]
            process_results = [task.result() for task in
                               tqdm(as_completed(task_list), desc="r={}".format(r), total=len(paras))]
            rmse_to_para = {re[0]: re[1] for re in process_results}
            best_rmse = min(rmse_to_para.keys())
            best_para = rmse_to_para[best_rmse]

        best_delta_phi, best_zeta, best_gamma, best_eta, best_Lc = best_para

        logging.debug(
            "OLADSR(r={},delta_phi={}, zeta={}, gamma={}, eta={}, Lc={},):".format(r, best_delta_phi, best_zeta,
                                                                                   best_gamma, best_eta, best_Lc))
        print("OLADSR(r={},delta_phi={}, zeta={}, gamma={}, eta={}, Lc={},):".format(r, best_delta_phi, best_zeta,
                                                                                     best_gamma, best_eta, best_Lc))


def sigle_test():
    maxItr = 50
    maxItr2 = 5
    tol_init = 1e-5
    rmse_sum = 0
    mae_sum = 0
    r = 8
    delta_phi = 10.0
    zeta = 0.1
    gamma = 100.0
    eta = 1
    Lc = 1
    for _ in range(5):
        R_train, R_test, S_bin, S_con = load_data("data/filmtrust", remove=True)
        metric = Metric(R_test)
        oladsr = OLADSR(R_train, S_bin, r, delta_phi, zeta, gamma, eta, Lc, lr=0.0001, init=False, debug=False)
        oladsr.train(maxItr, maxItr2, tol_init)
        rmse_sum += metric.RMSE(oladsr.Phi, oladsr.V, r=r)
        mae_sum += metric.MAE(oladsr.Phi, oladsr.V, r=r)
    print(rmse_sum / 5)
    print(mae_sum / 5)


if __name__ == "__main__":
    sigle_test()

import autograd.numpy as np
import tqdm
from autograd import grad
from evaluator import Metric
from dataloader import load_data
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import time


class OLA:
    def __init__(self, R, S, d, delta_phi, delta_u, delta_v, Lc, lr):
        self.R = R
        self.S = S
        self.d = d
        self.M, self.N = R.shape
        self.Phi = np.random.rand(self.d, self.M)
        self.V = np.random.rand(self.d, self.N)
        self.U = np.random.rand(self.d, self.M)
        self.delta_phi = delta_phi
        self.delta_u = delta_u
        self.delta_v = delta_v
        self.Lc = Lc
        self.alpha = {}
        self.lr = lr
        self.friend_li = [np.nonzero(S[i, :])[1] for i in range(self.M)]

    @staticmethod
    def OLA_obj(R, Phi, U, V, delta_phi, delta_u, delta_v, alpha):
        # index set
        observed_rate_idx = np.nonzero(R)

        # (R_ij - b_i * d_j)^2
        rated_R = R[observed_rate_idx].getA()
        rated_Phi = Phi[:, observed_rate_idx[0]]
        rated_V = V[:, observed_rate_idx[1]]
        rated_PV = np.multiply(rated_Phi, rated_V).sum(axis=0, keepdims=True)
        term1 = np.dot((rated_R - rated_PV), (rated_R - rated_PV).T).astype(np.float)
        term1 = 0.5 * term1

        # term2
        def construct(u):
            if u in alpha.keys():
                alpha_u = alpha[u]['alpha_u']
                sorted_uid = alpha[u]['sorted_uid']
                return np.sum(alpha_u * U[:, sorted_uid], axis=1)
            else:
                return np.zeros_like(Phi[:, 0])

        G = np.array([construct(u) for u in range(R.shape[0])]).transpose()
        term2 = 0.5 * delta_phi * ((Phi - G) ** 2).sum()

        # term3
        term3 = 0.5 * delta_phi * ((U - Phi) ** 2).sum()

        # term45
        term45 = 0.5 * (delta_u * (U ** 2).sum() + delta_v * (U ** 2).sum())
        return (term1 + term2 + term3 + term45)[0][0]

    def E_step(self):
        self.alpha = {}
        for i in range(self.M):
            friend_ids = self.friend_li[i]
            num_friend = len(friend_ids)
            if num_friend > 0:
                try:
                    beta = self.Lc * np.linalg.norm(self.U[:, [i]] - self.U[:, friend_ids], axis=0)
                except RuntimeWarning:
                    a = 10
                    print("aaaaaaaa")
                # zeta = self.Lc * euclidean_distances(self.U[:, [i]].T, self.U[:, friend_ids].T)
                # zeta = np.squeeze(zeta, axis=(0,))
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

    def M_step(self):
        print(OLA.OLA_obj(self.R, self.Phi, self.U, self.V, self.delta_phi, self.delta_u,
                          self.delta_v, self.alpha))
        grad_obj = grad(OLA.OLA_obj, [1, 2, 3])
        grad_Phi, grad_U, grad_V = grad_obj(self.R, self.Phi, self.U, self.V, self.delta_phi, self.delta_u,
                                            self.delta_v, self.alpha)
        self.Phi -= self.lr * grad_Phi
        self.U -= self.lr * grad_U
        self.V -= self.lr * grad_V

    def train(self, iteration):
        for _ in range(iteration):
            self.E_step()
            self.M_step()


if __name__ == "__main__":
    R_train, R_test, S_bin, S_con = load_data("data/Epinions", remove=True)
    ola = OLA(R_train, S_bin, d=8, delta_phi=10, delta_u=10, delta_v=10, Lc=100, lr=0.0001)
    metric = Metric(R_test)
    ola.train(iteration=150)

    fig, ax = plt.subplots()
    # metric.plot_precision_recall_curve(ola.Phi, ola.V, start_k=5, stop_k=50, ax=ax, name='OLA100PR')
    metric.plot_ROC_curve(ola.Phi, ola.V, start_k=5, stop_k=100, ax=ax, name='OLA100ROC')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.legend()
    plt.show()
    print(metric.RMSE(ola.Phi, ola.V))

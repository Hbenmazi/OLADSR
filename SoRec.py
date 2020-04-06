import math
import numpy as np
import tqdm
from numpy.random import RandomState
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from matplotlib import pyplot as plt
import time
import copy


def fx(x):
    return (x - 1) / 4


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# The derivative of sigmoid
def sigmoid_(x):
    val = sigmoid(x)
    return val * (1 - val)


class SoRec:
    def __init__(self, R, C, r=10, lr=0.01, momentum=0.8,
                 c=10, u=0.001, v=0.001, z=0.001, iters=1000, seed=None):
        """

        :param R: train set of ratings_data
        :param C: trust matrix of users
        :param lr: learning rate
        :param momentum:
        :param c:
        :param u:
        :param v:
        :param z:
        :param r:
        :param iters: iteration
        :param seed:
        """
        self.R = R
        self.C = C
        self.lambda_c = c
        self.lambda_u = u
        self.lambda_z = z
        self.lambda_v = v
        self.latent_size = r
        self.random_state = RandomState(seed)
        self.iters = iters
        self.lr = lr
        self.U = np.mat(self.random_state.rand(r, np.size(R, 0)))
        self.V = np.mat(self.random_state.rand(r, np.size(R, 1)))
        self.Z = np.mat(self.random_state.rand(r, np.size(C, 1)))
        self.momentum = momentum

    # the MAE for train set
    def train_loss(self, UVdata):
        loss = (np.fabs(4 * sigmoid(UVdata) + 1 - self.R.data)).sum()
        loss /= self.R.shape[0]
        return loss

    def train(self):
        Rindex = self.R.nonzero()
        Cindex = self.C.nonzero()
        Rdata = fx(self.R.data)
        Cdata = self.C.data
        Rnum = Rdata.shape[0]
        Cnum = Cdata.shape[0]
        UVdata = copy.deepcopy(Rdata)
        UZdata = copy.deepcopy(Cdata)
        momentum_u = np.mat(np.zeros(self.U.shape))
        momentum_v = np.mat(np.zeros(self.V.shape))
        momentum_z = np.mat(np.zeros(self.Z.shape))
        train_loss_list = []
        validate_loss_list = []
        begin = time.time()
        minloss = 5.0
        for it in (range(self.iters)):
            start = time.time()
            for k in range(Rnum):
                UVdata[k] = (self.U[:, Rindex[0][k]].T * self.V[:, Rindex[1][k]])[0][0]
            for k in range(Cnum):
                UZdata[k] = (self.U[:, Cindex[0][k]].T * self.Z[:, Cindex[1][k]])[0][0]

            UV = csr_matrix(((sigmoid(UVdata) - Rdata) * sigmoid_(UVdata), Rindex), self.R.shape)
            UZ = csr_matrix(((sigmoid(UZdata) - Cdata) * sigmoid_(UZdata), Cindex), self.C.shape)

            U = csr_matrix(self.U)
            V = csr_matrix(self.V)
            Z = csr_matrix(self.Z)

            grads_u = self.lambda_u * U + UV.dot(V.T).T + self.lambda_c * UZ.dot(Z.T).T
            grads_v = UV.T.dot(U.T).T + self.lambda_v * V
            grads_z = self.lambda_c * UZ.T.dot(U.T).T + self.lambda_z * Z
            momentum_u = self.momentum * momentum_u + grads_u
            momentum_v = self.momentum * momentum_v + grads_v
            momentum_z = self.momentum * momentum_z + grads_z

            self.U = self.U - self.lr * momentum_u
            self.V = self.V - self.lr * momentum_v
            self.Z = self.Z - self.lr * momentum_z
            trloss = self.train_loss(UVdata)
            train_loss_list.append(trloss)
            # traintxt = open("train_loss_list___.txt", 'a+')
            # valitxt = open("validate_loss_list___.txt", 'a+')
            # traintxt.write(str(trloss) + "\n")
            # valitxt.write(str(valiloss) + "\n")
            # np.savetxt("train_loss_list.txt", train_loss_list)
            # np.savetxt("validate_loss_list.txt", validate_loss_list)
            end = time.time()
            # print("iter:{}, last_train_loss:{}, validate_loss:{}, timecost:{}, have run:{}".format(it + 1, trloss,
            #                                                                                        valiloss,
            #                                                                                        end - start,
            #                                                                                        end - begin))
        return self.U, self.V, self.Z, train_loss_list, validate_loss_list

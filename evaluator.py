import math
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import ndcg_score


def normalization(x, start, end):
    """"
    归一化到区间[1,5]
    """
    assert end > start

    _range = float(np.max(x)) - float(np.min(x)) + 1e-14
    return (x - np.min(x)) / _range * (end - start) + start


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Metric:
    def __init__(self, R_test):
        self.R_test = R_test
        self.n, self.m = R_test.shape
        self.consumed_items_list = [self.R_test[i, :].indices for i in range(self.n)]

    def RMSE(self, B, D, r=None, ifsigmoid=False):
        assert r is None or ifsigmoid is False
        observed_rate_idx = np.nonzero(self.R_test)
        R_true = self.R_test[observed_rate_idx].getA().squeeze()
        R_pred = B[:, observed_rate_idx[0]] * D[:, observed_rate_idx[1]]
        R_pred = R_pred.sum(axis=0).squeeze()
        maxR = R_true.max()
        minR = R_true.min()
        if r is not None:
            R_pred = (R_pred + r) / float(2 * r)
            R_pred = R_pred * (maxR - minR) + minR

        if ifsigmoid:
            R_pred = sigmoid(R_pred)
            R_pred = R_pred * (maxR - minR) + minR

        return np.sqrt(mean_squared_error(R_true, R_pred))

    def MAE(self, B, D, r=None, ifsigmoid=False):
        assert r is None or ifsigmoid is False
        observed_rate_idx = np.nonzero(self.R_test)
        R_true = self.R_test[observed_rate_idx].getA().squeeze()
        R_pred = B[:, observed_rate_idx[0]] * D[:, observed_rate_idx[1]]
        R_pred = R_pred.sum(axis=0).squeeze()
        maxR = R_true.max()
        minR = R_true.min()
        if r is not None:
            R_pred = (R_pred + r) / float(2 * r)
            R_pred = R_pred * (maxR - minR) + minR

        if ifsigmoid:
            R_pred = sigmoid(R_pred)
            R_pred = R_pred * (maxR - minR) + minR

        return mean_absolute_error(R_true, R_pred)

    def Recall(self, B, D, k):
        num_user = B.shape[1]
        recall_sum = 0
        for i in range(num_user):
            consumed_items = self.consumed_items_list[i]
            pred_rating = np.dot(B[:, [i]].T, D).squeeze()
            topk_items = np.argpartition(pred_rating, -k)[-k:]
            tp = np.intersect1d(consumed_items, topk_items).size
            if consumed_items.size != 0:
                recall_sum += tp / consumed_items.size
        return recall_sum / num_user

    def Precision(self, B, D, k):
        num_user = B.shape[1]
        precision_sum = 0
        for i in range(num_user):
            consumed_items = self.consumed_items_list[i]
            pred_rating = np.dot(B[:, [i]].T, D).squeeze()
            topk_items = np.argpartition(pred_rating, -k)[-k:]
            tp = np.intersect1d(consumed_items, topk_items).size
            precision_sum += tp
        return precision_sum / (num_user * k)

    def PrecisionAndRecall(self, B, D, k):
        num_user = B.shape[1]
        recall_sum = 0
        precision_sum = 0
        for i in (range(num_user)):
            consumed_items = self.consumed_items_list[i]
            pred_rating = np.dot(B[:, [i]].T, D).squeeze()
            topk_items = np.argpartition(pred_rating, -k)[-k:]
            tp = np.intersect1d(consumed_items, topk_items).size
            if consumed_items.size != 0:
                recall_sum += tp / consumed_items.size
            precision_sum += tp
        precision = precision_sum / (num_user * k)
        recall = recall_sum / num_user
        return precision, recall

    def FPRAndTPR(self, B, D, k):
        num_user = B.shape[1]
        num_item = D.shape[1]
        TPR_sum = 0
        FPR_sum = 0
        for i in (range(num_user)):
            consumed_items = self.consumed_items_list[i]
            pred_rating = np.dot(B[:, [i]].T, D).squeeze()
            topk_items = np.argpartition(pred_rating, -k)[-k:]
            tp = np.intersect1d(consumed_items, topk_items).size
            fp = topk_items.size - tp
            if consumed_items.size != 0:
                TPR_sum += tp / consumed_items.size
                FPR_sum += fp / (num_item - consumed_items.size)
        FPR = FPR_sum / num_user
        TPR = TPR_sum / num_user
        return FPR, TPR

    def precision_recall_curve(self, B, D, start_k, stop_k):
        ks = list(range(start_k, stop_k + 1))
        precisions = []
        recalls = []
        for k in (ks):
            precision, recall = self.PrecisionAndRecall(B, D, k)
            precisions.append(precision)
            recalls.append(recall)
        return precisions, recalls, ks

    def ROC_curve(self, B, D, start_k, stop_k):
        ks = list(range(start_k, stop_k + 1))
        FPRs = []
        TPRs = []
        for k in (ks):
            FPR, TPR = self.FPRAndTPR(B, D, k)
            FPRs.append(FPR)
            TPRs.append(TPR)
        return FPRs, TPRs, ks

    def plot_precision_recall_curve(self, B, D, start_k, stop_k, ax, name, **kwargs):
        precision, recall, ks = self.precision_recall_curve(B, D, start_k, stop_k)
        ax.plot(recall, precision, label=name, **kwargs)

    def plot_ROC_curve(self, B, D, start_k, stop_k, ax, name, **kwargs):
        FPRs, TPRs, ks = self.ROC_curve(B, D, start_k, stop_k)
        ax.plot(FPRs, TPRs, label=name, **kwargs)
        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, 1)

    def plot_NDCG_curve(self, B, D, R_train, start_k, stop_k, ax, name, r=None, **kwargs):
        ndcgs, ks = self.NDCG_curve(B, D, R_train, start_k, stop_k, r)
        ax.plot(ks, ndcgs, label=name, **kwargs)
        ax.set_xticks(ks)

    def NDCG_curve(self, B, D, R_train, start_k, stop_k, r=None):
        ks = list(range(start_k, stop_k + 1))
        ndcgs = []
        for k in ks:
            ndcg = self.NDCG(B, D, R_train, k)
            ndcgs.append(ndcg)
        return ndcgs, ks

    def NDCG(self, B, D, R_train, k=None, batch_size=512):
        if (self.n, self.m) != (B.shape[1], D.shape[1]):
            raise ValueError("Not the same dataset")

        batch_num = math.ceil((self.n / float(batch_size)))
        ndcg_total = []
        for i in range(batch_num):
            train_batch = R_train[i * batch_size:(i + 1) * batch_size, :]
            train_batch = train_batch.toarray()
            mask = train_batch != 0

            # pred
            pred_batch = np.dot(B[:, i * batch_size:(i + 1) * batch_size].T, D)
            pred_batch = normalization(pred_batch, 1, 5)
            pred_batch[mask] = 0

            # true
            true_batch = self.R_test[i * batch_size:(i + 1) * batch_size, :].toarray()
            true_batch = normalization(true_batch, 1, 5)
            true_batch[mask] = 0
            ndcg_total.append(ndcg_score(y_true=true_batch, y_score=pred_batch, k=k))
        return np.array(ndcg_total).mean()

    def NDCG2(self, B, D, R_train, k=None, r=None):
        assert self.R_test.shape == R_train.shape
        num_user, num_item = self.R_test.shape
        maxR, minR = self.R_test.max(), self.R_test.min()
        dcg_discount = 1 / (np.log(np.arange(num_item) + 2) / np.log(2))
        discount = dcg_discount.copy()
        if k is not None:
            discount[k:] = 0

        # dcg
        dcgs = []
        discount_cumsum = np.cumsum(discount)
        for i in range(num_user):
            y_t = self.R_test[i, :].toarray().squeeze()
            y_train_idx = R_train[i, :].indices
            y_s = np.dot(B[:, i].T, D)
            if r is not None:
                y_s = (y_s + r) / float(2 * r)
                y_s = y_s * (maxR - minR) + minR
            y_s[y_train_idx] = -1e4
            dcgs.append(tie_averaged_dcg(y_t, y_s, discount_cumsum))
        dcgs = np.asarray(dcgs)

        # idcg
        idcgs = []
        for i in range(num_user):
            y_t = self.R_test[i, :].toarray()
            ranked = -np.sort(-y_t)
            idcgs.append(np.dot(discount, ranked.T)[0])
        idcgs = np.asarray(idcgs)

        all_irrelevant = idcgs == 0
        dcgs[all_irrelevant] = 0
        dcgs[~all_irrelevant] /= idcgs[~all_irrelevant]
        return np.mean(dcgs)


def tie_averaged_dcg(y_true, y_score, discount_cumsum):
    _, inv, counts = np.unique(
        - y_score, return_inverse=True, return_counts=True)
    ranked = np.zeros(len(counts))
    np.add.at(ranked, inv, y_true)
    ranked /= counts
    groups = np.cumsum(counts) - 1
    discount_sums = np.empty(len(counts))
    discount_sums[0] = discount_cumsum[groups[0]]
    discount_sums[1:] = np.diff(discount_cumsum[groups])
    return (ranked * discount_sums).sum()

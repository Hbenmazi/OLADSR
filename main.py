import json
import os
from concurrent.futures._base import as_completed
from concurrent.futures.process import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm
from DCF import DCF
from DSR import DSR
from OLA import OLA
from SoRec import SoRec
from OLADSR import OLADSR
from dataloader import load_data
from evaluator import Metric


def gen_result_inner(*arg):
    dataset, model_name, r, para = arg[0]
    assert model_name in ['DCF', 'DSR', 'OLADSE', "SoRec"]
    assert dataset in ['filmtrust', 'Epinions', 'CiaoDVD']
    maxItr = 50
    maxItr2 = 5
    tol_init = 1e-5
    times = 5

    def build_model():
        data_path = os.path.join('data', dataset)
        R_train, R_test, S_bin, S_con = load_data(data_path, remove=True)
        if dataset == 'filmtrust':
            maxR, minR = 4.5, 0.5
        else:
            maxR, minR = 5.0, 1.0
        maxS = 1.0
        minS = 0.0
        metric = Metric(R_test)
        if model_name == 'DCF':
            model = DCF(R_train, r, para['alpha'], para['beta'], maxR, minR, init=False, debug=False)

        if model_name == 'DSR':
            model = DSR(R_train, S_bin, r, para['alpha0'], para['beta1'], para['beta2'], para['beta3'], maxR, minR,
                        maxS, minS, init=False, debug=False)

        if model_name == 'OLADSR':
            model = OLADSR(R_train, S_bin, r, para['delta_phi'], para['zeta'], para['gamma'], para['eta'], para['Lc'],
                           lr=0.0001, init=False, debug=False)

        if model_name == 'SoRec':
            model = SoRec(R_train, S_con, r, para['lr'], para['momentum'], para['c'], para['u'], para['v'],
                          para['z'], iters=1000)

        return model, metric

    RMSE_total = 0
    MAE_total = 0

    pr_stratk = 5
    pr_stopk = 50
    precisions_total = np.zeros(shape=(pr_stopk - pr_stratk + 1))
    recalls_total = np.zeros(shape=(pr_stopk - pr_stratk + 1))
    pr_ks = list(range(pr_stratk, pr_stopk + 1))

    roc_stratk = 1
    roc_stopk = 100
    FPR_total = np.zeros(shape=(roc_stopk - roc_stratk + 1))
    TPR_total = np.zeros(shape=(roc_stopk - roc_stratk + 1))
    roc_ks = list(range(roc_stratk, roc_stopk + 1))

    ndcg_stratk = 1
    ndcg_stopk = 10
    ndcg_total = np.zeros(shape=(ndcg_stopk - ndcg_stratk + 1))
    ndcg_ks = list(range(ndcg_stratk, ndcg_stopk + 1))

    for _ in range(times):
        model, metric = build_model()
        if isinstance(model, SoRec):
            model.train()
        else:
            model.train(maxItr, maxItr2, tol_init)

        if isinstance(model, OLADSR):
            B, D = model.Phi, model.V
        elif isinstance(model, SoRec):
            B, D = model.U.getA(), model.V.getA()
        else:
            B, D = model.B, model.D

        # RMSE, MAE
        if isinstance(model, SoRec):
            RMSE_total += metric.RMSE(B, D, ifsigmoid=True)
            MAE_total += metric.MAE(B, D, ifsigmoid=True)
        else:
            RMSE_total += metric.RMSE(B, D, r=r)
            MAE_total += metric.MAE(B, D, r=r)

        # PR
        precisions, recalls, _ = metric.precision_recall_curve(B, D, start_k=pr_stratk,
                                                               stop_k=pr_stopk)
        precisions_total += np.array(precisions)
        recalls_total += np.array(recalls)

        # ROC
        FPRs, TPRs, _ = metric.ROC_curve(B, D, start_k=roc_stratk, stop_k=roc_stopk)
        FPR_total += np.array(FPRs)
        TPR_total += np.array(TPRs)

        # NDCG
        ndcgs, _ = metric.NDCG_curve(B, D, model.R, start_k=ndcg_stratk, stop_k=ndcg_stopk, r=r)
        ndcg_total += np.array(ndcgs)

    RMSE = RMSE_total / times
    MAE = MAE_total / times
    precisions = precisions_total / times
    recalls = recalls_total / times
    FPR = FPR_total / times
    TPR = TPR_total / times
    ndcgs = ndcg_total / times

    return RMSE, MAE, precisions, recalls, FPR, TPR, ndcgs, pr_ks, roc_ks, ndcg_ks


def gen_result(conf_filepath, result_filepath):
    with open(conf_filepath, encoding='UTF-8') as yaml_file:
        conf = yaml.safe_load(yaml_file)
    result = conf.copy()
    arg_list = []
    for dataset in conf.keys():
        for model_name in (conf[dataset].keys()):
            for bit in conf[dataset][model_name].keys():
                para = conf[dataset][model_name][bit]
                arg = (dataset, model_name, bit, para)
                arg_list.append(arg)

    with ProcessPoolExecutor() as pool:
        process_results = list(tqdm(pool.map(gen_result_inner, arg_list), desc="evaluating...", total=len(arg_list)))

    for i, arg in enumerate(arg_list):
        RMSE, MAE, precisions, recalls, FPRs, TPRs, ndcgs, pr_ks, roc_ks, ndcg_ks = process_results[i]
        dataset, model_name, bit, _ = arg
        result[dataset][model_name][bit] = {
            'RMSE': RMSE,
            'MAE': MAE,
            'precisions': list(precisions),
            'recalls': list(recalls),
            'FPRs': list(FPRs),
            'TPRs': list(TPRs),
            'ndcgs': list(ndcgs),
            'pr_ks': pr_ks,
            'roc_ks': roc_ks,
            'ndcg_ks': ndcg_ks
        }

    with open(result_filepath, 'w', encoding='UTF-8') as result_file:
        json.dump(result, result_file)


def plot(result_filepath):
    ndcg_plot(result_filepath)
    rmse_mae__plot(result_filepath)
    pr_curve_plot(result_filepath)
    roc_curve_plot(result_filepath)


def ndcg_plot(result_filepath):
    with open(result_filepath, encoding='UTF-8') as result_file:
        result = json.load(result_file)
    result = result['filmtrust']

    fig, ax = plt.subplots(ncols=4, figsize=(21, 3.5))
    # ndcg

    for model_name in result.keys():
        for idx, bit in enumerate(result[model_name].keys()):
            ax[idx].plot(result[model_name][bit]['ndcg_ks'], result[model_name][bit]['ndcgs'], label=model_name)
            ax[idx].set_xticks(result[model_name][bit]['ndcg_ks'])
            ax[idx].set_title("r={}bits".format(bit))
            ax[idx].set_xlabel('K')
            ax[idx].set_ylabel('NDCF@K')
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    plt.suptitle("NDCG")
    plt.subplots_adjust(wspace=0.25, bottom=0.20)
    plt.show()


def rmse_mae__plot(result_filepath):
    with open(result_filepath, encoding='UTF-8') as result_file:
        result = json.load(result_file)
    result = result['filmtrust']

    fig, ax = plt.subplots(ncols=4, figsize=(20, 3.5))
    # rmse
    for jdx, model_name in enumerate(result.keys()):
        for idx, bit in enumerate(result[model_name].keys()):
            x = jdx
            rmse = result[model_name][bit]['RMSE']
            mae = result[model_name][bit]['MAE']
            ax[idx].bar([x, x + 5.0], [rmse, mae], label=model_name)
            ax[idx].set_title("r={}bits".format(bit), y=-0.16)
            ax[idx].set_xticks([1, 6])
            ax[idx].set_xticklabels(['RMSE', 'MAE'])
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4)
    plt.show()


def pr_curve_plot(result_filepath):
    with open(result_filepath, encoding='UTF-8') as result_file:
        result = json.load(result_file)
    result = result['filmtrust']

    fig, ax = plt.subplots(ncols=4, figsize=(21, 3.5))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.2, hspace=None)

    for model_name in result.keys():
        for idx, bit in enumerate(result[model_name].keys()):
            ax[idx].plot(result[model_name][bit]['recalls'], result[model_name][bit]['precisions'], label=model_name)
            ax[idx].set_title("r={}bits".format(bit))
            ax[idx].set_xlabel('Recall')
            ax[idx].set_ylabel('Precision')

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    plt.suptitle("Recall-Precision")
    plt.subplots_adjust(wspace=0.25, hspace=0, bottom=0.20)
    plt.show()


def roc_curve_plot(result_filepath):
    with open(result_filepath, encoding='UTF-8') as result_file:
        result = json.load(result_file)
    result = result['filmtrust']

    fig, ax = plt.subplots(ncols=4, figsize=(21, 3.5))

    for model_name in result.keys():
        for idx, bit in enumerate(result[model_name].keys()):
            ax[idx].plot(result[model_name][bit]['FPRs'], result[model_name][bit]['TPRs'], label=model_name)
            ax[idx].set_title("r={}bits".format(bit))
            ax[idx].set_xlabel('FPRs')
            ax[idx].set_ylabel('TPRs')

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    plt.suptitle("ROC")
    plt.subplots_adjust(wspace=0.25, hspace=0, bottom=0.20)
    plt.show()


if __name__ == "__main__":
    gen_result('conf/SoRecOnly.yaml', 'gen/SoRecOnly.json')

import itertools
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from DCF import DCF
from dataloader import load_data
from evaluator import Metric


class GridSearch:
    def grid_search(self, rs, dataset, *para):
        assert dataset in ['filmtrust', 'Epinions', 'CiaoDVD']
        logging.basicConfig(filename="{}.txt".format(self.__class__.__name__), level="DEBUG")
        for r in rs:
            args_list = list(itertools.product(*para, [dataset], [r]))
            with ProcessPoolExecutor() as pool:
                task_list = [pool.submit(self.grid_search_inner, *args) for args in args_list]
                process_results = [task.result() for task in
                                   tqdm(as_completed(task_list), desc="r={}".format(r), total=len(args_list))]
                metric_to_para = {re[0]: re[1] for re in process_results}
                best_metric = min(metric_to_para.keys())
                best_para = metric_to_para[best_metric]

            best_alpha, best_beta = best_para
            logging.debug("{}(r={},alpha={},beta={}):".format(self.__class__.__name__, r, best_alpha, best_beta))
            print("{}}(r={},alpha={},beta={}):".format(self.__class__.__name__, r, best_alpha, best_beta))

    def grid_search_inner(self, *args):
        raise NotImplementedError


class DCFGridSearch(GridSearch):
    def __init__(self, times, init, metric_type, user_filter, item_filter):
        super(GridSearch, self).__init__()
        assert metric_type in ['ndcg5', 'rmse']
        self.times = times
        self.init = init
        self.metric_type = metric_type
        self.user_filter = user_filter
        self.item_filter = item_filter

    def grid_search_inner(self, *args):
        """

        :param args: (alpha,beta,dataset,r):tuple
        :return:
        """
        alpha, beta, dataset, r = args
        assert dataset in ['filmtrust', 'Epinions', 'CiaoDVD']
        maxItr = 50
        maxItr2 = 5
        tol_init = 1e-5
        if dataset == 'filmtrust':
            maxR, minR = 4.5, 0.5
        else:
            maxR, minR = 5.0, 1.0

        metric_sum = 0
        for _ in range(5):
            R_train, R_test, S_bin, S_con = load_data("data/{}".format(dataset), remove=True,
                                                      user_filter=self.user_filter, item_filter=self.item_filter)
            metric = Metric(R_test)
            dcf = DCF(R_train, r, alpha, beta, maxR, minR, init=self.init, debug=False)
            dcf.train(maxItr, maxItr2, tol_init)
            if metric == 'rmse':
                metric_value = metric.RMSE(dcf.B, dcf.D, r=5)
            elif metric == 'ndcg5':
                metric_value = metric.NDCG(dcf.B, dcf.D, R_train, k=5)
            else:
                raise ValueError('Metric {} is not supported.'.format(metric))

            metric_sum += metric_value
        return metric_sum / self.times, (alpha, beta)

import copy
import math
import numpy as np
import pandas as pd
import tqdm
from scipy import sparse


def Scale(s, scale, maxS, minS):
    s = (s - minS) / (maxS - minS)
    s = 2 * scale * s - scale
    return s


def load_data(path, frac=0.8, remove=False, user_filter=10, item_filter=10):
    # get matrix R and S
    rnames = ['user_id', 'item_id', 'rating']
    ratings = pd.read_csv(path + '/ratings.txt', sep=' ', header=None, names=rnames)
    ratings["user_id"] = ratings["user_id"].map(lambda x: x - 1)
    ratings["item_id"] = ratings["item_id"].map(lambda x: x - 1)

    snames = ['trustor_id', 'trustee_id', 'trust_value']
    social = pd.read_csv(path + '/trust.txt', sep=' ', header=None, names=snames, dtype=float)
    social["trustor_id"] = social["trustor_id"].map(lambda x: x - 1)
    social["trustee_id"] = social["trustee_id"].map(lambda x: x - 1)

    # When a user rates an item multiple times, we merge
    # them into one rating by averaging the duplicate rating scores.
    ratings = ratings.groupby(['user_id', 'item_id']).mean().reset_index()

    max_userid = ratings["user_id"].max()

    # If there exists users present in  trust.txt but not in ratings.txt
    social.drop(social[social['trustor_id'] > max_userid].index, inplace=True)
    social.drop(social[social['trustee_id'] > max_userid].index, inplace=True)

    # 去掉评分小于10个的用户和物品
    if remove:
        done = False
        while not done:
            l0 = len(ratings)
            ratings_by_user = ratings.groupby('user_id').size()
            ratings_by_user_index = ratings_by_user.index[ratings_by_user >= user_filter]
            ratings = ratings.loc[ratings['user_id'].isin(ratings_by_user_index)]
            social = social.loc[social['trustor_id'].isin(ratings_by_user_index)]
            social = social.loc[social['trustee_id'].isin(ratings_by_user_index)]

            ratings_by_item = ratings.groupby('item_id').size()
            ratings_by_item_index = ratings_by_item.index[ratings_by_item >= item_filter]
            ratings = ratings.loc[ratings['item_id'].isin(ratings_by_item_index)]

            l1 = len(ratings)
            if l1 == l0:
                done = True

        user_id_set = set(ratings['user_id'].values.tolist())
        num_user = len(user_id_set)
        toNewUserID = dict(zip(user_id_set, range(num_user)))

        item_id_set = set(ratings['item_id'].values.tolist())
        num_item = len(item_id_set)
        toNewItemID = dict(zip(item_id_set, range(num_item)))

        ratings["user_id"] = ratings["user_id"].map(lambda x: toNewUserID[x])
        ratings["item_id"] = ratings["item_id"].map(lambda x: toNewItemID[x])
        social['trustor_id'] = social['trustor_id'].map(lambda x: toNewUserID[x])
        social['trustee_id'] = social['trustee_id'].map(lambda x: toNewUserID[x])

    # train_test_split
    def typicalsamling(group):
        return group.sample(frac=frac)

    train_df = ratings.groupby('user_id', group_keys=False).apply(typicalsamling)
    test_df = ratings[~ratings.index.isin(train_df.index)].copy()

    # construct the sparse mat
    max_userid = ratings["user_id"].max()
    max_itemid = ratings["item_id"].max()
    R_train = sparse.csr_matrix((train_df['rating'], (train_df['user_id'], train_df['item_id'])),
                                shape=(max_userid + 1, max_itemid + 1))
    R_test = sparse.csr_matrix((test_df['rating'], (test_df['user_id'], test_df['item_id'])),
                               shape=(max_userid + 1, max_itemid + 1))
    S_bin = sparse.coo_matrix((social['trust_value'], (social['trustor_id'], social['trustee_id'])),
                              shape=(max_userid + 1, max_userid + 1))
    indeg = S_bin.sum(axis=0)
    outdeg = S_bin.sum(axis=1)
    S_con = copy.deepcopy(S_bin)
    for k in range(S_con.data.shape[0]):
        i = S_con.row[k]
        j = S_con.col[k]
        S_con.data[k] = math.sqrt(indeg[0, j] / (indeg[0, j] + outdeg[i, 0]))

    # print("dataset:{},users:{},items:{},ratings:{},connections:{}".format(path, max_userid + 1, max_itemid + 1,
    #                                                                       ratings.shape[0],
    #                                                                       social.shape[0]))
    return R_train, R_test, sparse.csr_matrix(S_bin, dtype=np.int8), sparse.csr_matrix(S_con)


if __name__ == "__main__":
    load_data("data/Epinions")
